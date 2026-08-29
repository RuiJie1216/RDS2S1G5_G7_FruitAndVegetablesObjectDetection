"""Train a true full-image MobileNetV2 + SSDLite object detector."""
from __future__ import annotations
import argparse, csv, json, os, random, shutil, sys, time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from config import CLASS_NAMES, DATA_ROOT, DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE, MODEL_NAME, RUNS_DIR
from detection_data import YoloDetectionDataset, collate_detection_batch, dataset_fingerprint, save_augmentation_preview
from detection_metrics import collect_predictions, compute_detection_metrics, save_evaluation_artifacts
from detector import (
    ANCHOR_PROFILES, ARCHITECTURE, CHECKPOINT_TYPE, HIGHRES_ARCHITECTURE,
    SUPPORTED_ARCHITECTURES, architecture_summary, build_detector,
    freeze_pretrained_batch_norm, load_warm_start_weights, optimizer_groups,
)

HISTORY_FIELDS = ["epoch", "phase", "train_total_loss", "train_classification_loss", "train_box_loss", "val_map_50", "val_map_50_95", "val_precision", "val_recall", "backbone_lr", "head_lr", "epoch_seconds"]

def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested if requested != "auto" else "cpu")

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); os.replace(temporary, path)

def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, device: torch.device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_detection_batch, pin_memory=device.type == "cuda", persistent_workers=workers > 0)

def training_config(args, run_dir: Path, smoke: bool) -> dict[str, Any]:
    return {"model_name": MODEL_NAME, "architecture": args.architecture, "checkpoint_type": CHECKPOINT_TYPE,
        "run_name": run_dir.name, "dataset": str(args.dataset.resolve()), "num_foreground_classes": 63,
        "internal_num_classes": 64, "image_size": args.image_size, "anchor_profile": args.anchor_profile,
        "anchor_config": deepcopy(args._anchor_config),
        "confidence_threshold": args.confidence, "ap_score_threshold": args.ap_score_threshold,
        "nms_threshold": args.nms_threshold, "max_detections": args.max_detections, "topk_candidates": args.topk_candidates,
        "positive_iou_threshold": args.positive_iou_threshold,
        "negative_to_positive_ratio": args.negative_to_positive_ratio,
        "classification_loss": args.classification_loss, "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "batch_size": args.batch_size, "phase1_epochs": args.phase1_epochs, "phase2_epochs": args.phase2_epochs,
        "backbone_lr": args.backbone_lr, "head_lr": args.head_lr, "weight_decay": args.weight_decay,
        "train_stage2": args.train_stage2, "train_full_backbone": args.train_full_backbone,
        "object_crop_probability": args.object_crop_probability,
        "evaluation_frequency": args.eval_every, "selection_metric": "val_map_50_95", "seed": args.seed,
        "num_workers": args.num_workers, "amp": args.amp, "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta, "scheduler_patience": args.scheduler_patience,
        "is_smoke_test": smoke, "resume_overrides": getattr(args, "resume_overrides", {}),
        "warm_start_source": getattr(args, "_warm_start_source", str(args.warm_start.resolve()) if args.warm_start else None),
        "test_set_accessed": False}

def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state: raise ValueError("Checkpoint is missing RNG state required for exact resume")
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None: torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])

RESUME_FIELDS = (
    ("architecture", "architecture", ("--architecture",), str),
    ("dataset", "dataset", ("--dataset",), Path), ("image_size", "image_size", ("--image-size",), int),
    ("batch_size", "batch_size", ("--batch-size",), int), ("num_workers", "num_workers", ("--num-workers",), int),
    ("phase1_epochs", "phase1_epochs", ("--phase1-epochs",), int), ("phase2_epochs", "phase2_epochs", ("--phase2-epochs",), int),
    ("backbone_lr", "backbone_lr", ("--backbone-lr",), float), ("head_lr", "head_lr", ("--head-lr",), float),
    ("weight_decay", "weight_decay", ("--weight-decay",), float), ("eval_every", "evaluation_frequency", ("--eval-every",), int),
    ("confidence", "confidence_threshold", ("--confidence",), float), ("ap_score_threshold", "ap_score_threshold", ("--ap-score-threshold",), float),
    ("nms_threshold", "nms_threshold", ("--nms-threshold",), float), ("max_detections", "max_detections", ("--max-detections",), int),
    ("topk_candidates", "topk_candidates", ("--topk-candidates",), int),
    ("positive_iou_threshold", "positive_iou_threshold", ("--positive-iou-threshold",), float),
    ("negative_to_positive_ratio", "negative_to_positive_ratio", ("--negative-to-positive-ratio",), float),
    ("classification_loss", "classification_loss", ("--classification-loss",), str),
    ("focal_gamma", "focal_gamma", ("--focal-gamma",), float),
    ("focal_alpha", "focal_alpha", ("--focal-alpha",), float),
    ("object_crop_probability", "object_crop_probability", ("--object-crop-probability",), float),
    ("train_stage2", "train_stage2", ("--train-stage2", "--no-train-stage2"), bool),
    ("train_full_backbone", "train_full_backbone", ("--train-full-backbone", "--no-train-full-backbone"), bool),
    ("early_stopping_patience", "early_stopping_patience", ("--early-stopping-patience",), int),
    ("early_stopping_min_delta", "early_stopping_min_delta", ("--early-stopping-min-delta",), float),
    ("scheduler_patience", "scheduler_patience", ("--scheduler-patience",), int), ("seed", "seed", ("--seed",), int),
    ("amp", "amp", ("--amp", "--no-amp"), bool), ("smoke_test", "is_smoke_test", ("--smoke-test",), bool),
)

def apply_resume_configuration(args, checkpoint: dict[str, Any]) -> None:
    saved = checkpoint.get("training_config")
    if not isinstance(saved, dict): raise ValueError("Checkpoint is missing training_config required for exact resume")
    explicit = getattr(args, "_explicit_options", set()); overrides = {}; unavailable = []
    for attribute, key, options, converter in RESUME_FIELDS:
        if key not in saved:
            unavailable.append(key); continue
        saved_value = converter(saved[key]); current = getattr(args, attribute)
        current_comparable = str(current.resolve()) if isinstance(current, Path) else current
        saved_comparable = str(saved_value.resolve()) if isinstance(saved_value, Path) else saved_value
        was_explicit = any(option in explicit for option in options)
        if was_explicit and current_comparable != saved_comparable:
            if not args.allow_resume_overrides:
                raise ValueError(f"Resume override for {attribute} changes {saved_value!r} to {current!r}; add --allow-resume-overrides to proceed explicitly")
            overrides[attribute] = {"saved": str(saved_value), "override": str(current)}
        else:
            setattr(args, attribute, saved_value)
    if unavailable: print(f"Resume warning: older checkpoint lacks saved fields: {', '.join(unavailable)}")
    if overrides: print("Explicit resume overrides: " + json.dumps(overrides, sort_keys=True))
    args.resume_overrides = overrides

def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, phase, class_names, config, history, best_metric, early_state):
    return {"checkpoint_format": 2, "checkpoint_type": CHECKPOINT_TYPE, "architecture": model.architecture_name, "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(), "epoch": epoch, "training_phase": phase, "class_names": class_names,
        "num_classes": len(class_names), "internal_num_classes": len(class_names)+1, "input_image_size": config["image_size"],
        "anchor_config": config["anchor_config"], "confidence_threshold": config["confidence_threshold"], "nms_threshold": config["nms_threshold"],
        "max_detections": config["max_detections"], "topk_candidates": config["topk_candidates"],
        "positive_iou_threshold": config["positive_iou_threshold"],
        "negative_to_positive_ratio": config["negative_to_positive_ratio"],
        "classification_loss": config["classification_loss"], "focal_gamma": config["focal_gamma"],
        "focal_alpha": config["focal_alpha"],
        "history": history, "dataset_fingerprint": config["dataset_fingerprint"],
        "rng_state": capture_rng_state(), "early_stopping_state": early_state, "training_config": config, "best_validation_metric": best_metric}

def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, temporary); os.replace(temporary, path)

def validate_checkpoint(checkpoint: dict[str, Any], class_names: list[str], image_size: int | None = None) -> None:
    if checkpoint.get("checkpoint_type") != CHECKPOINT_TYPE or checkpoint.get("architecture") not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Checkpoint is not a MobileNetV2 object detector; legacy crop checkpoints are rejected")
    if checkpoint.get("class_names") != class_names or checkpoint.get("num_classes") != 63:
        raise ValueError("Checkpoint class count/order does not match the 63 source categories")
    if image_size is not None and checkpoint.get("input_image_size") != image_size: raise ValueError("Checkpoint input resolution mismatch")

def load_detector_checkpoint(path: Path, device: torch.device, class_names: list[str] | None = None):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False); names = [str(n) for n in checkpoint.get("class_names", [])]
    validate_checkpoint(checkpoint, class_names or names)
    model = build_detector(
        63, int(checkpoint["input_image_size"]), False, checkpoint["anchor_config"],
        float(checkpoint.get("confidence_threshold", .25)), float(checkpoint.get("nms_threshold", .45)),
        int(checkpoint.get("max_detections", 200)), int(checkpoint.get("topk_candidates", 400)),
        str(checkpoint.get("architecture", ARCHITECTURE)),
        float(checkpoint.get("positive_iou_threshold", checkpoint.get("training_config", {}).get("positive_iou_threshold", .5))),
        float(checkpoint.get("negative_to_positive_ratio", checkpoint.get("training_config", {}).get("negative_to_positive_ratio", 3.0))),
        str(checkpoint.get("classification_loss", checkpoint.get("training_config", {}).get("classification_loss", "cross_entropy"))),
        float(checkpoint.get("focal_gamma", checkpoint.get("training_config", {}).get("focal_gamma", 2.0))),
        float(checkpoint.get("focal_alpha", checkpoint.get("training_config", {}).get("focal_alpha", .25))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval(); model._inference_device = device
    return model, checkpoint, names

def train_one_epoch(model, loader, optimizer, scaler, device, amp: bool, max_batches: int = 0):
    model.train(); freeze_pretrained_batch_norm(model)
    sums = {"classification": 0., "bbox_regression": 0., "total": 0.}; batches = 0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device) for k, v in target.items() if k in {"boxes","labels","image_id","area","iscrowd"}} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            losses = model(images, targets); total = sum(losses.values())
        if not torch.isfinite(total): raise FloatingPointError(f"Non-finite detector loss: {losses}")
        scaler.scale(total).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        scaler.step(optimizer); scaler.update(); batches += 1
        sums["classification"] += float(losses["classification"].detach()); sums["bbox_regression"] += float(losses["bbox_regression"].detach()); sums["total"] += float(total.detach())
        if max_batches and batches >= max_batches: break
    return {k: v/max(batches,1) for k,v in sums.items()}

def write_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    atomic_json(run_dir/"training_history.json", history)
    with (run_dir/"training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle, fieldnames=HISTORY_FIELDS); writer.writeheader(); writer.writerows(history)
    if history:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs=[r["epoch"] for r in history]; fig,axes=plt.subplots(1,2,figsize=(11,4))
        for key,label in (("train_total_loss","total"),("train_classification_loss","classification"),("train_box_loss","box")): axes[0].plot(epochs,[r[key] for r in history],label=label)
        axes[0].legend(); axes[0].set_title("Training losses")
        for key,label in (("val_map_50","mAP@0.5"),("val_map_50_95","mAP@0.5:0.95")): axes[1].plot(epochs,[r[key] if r[key] is not None else np.nan for r in history],label=label)
        axes[1].legend(); axes[1].set_title("Validation metrics"); fig.tight_layout(); fig.savefig(run_dir/"training_curves.png",dpi=160); plt.close(fig)

def evaluate_model(model, loader, device, class_names, output_dir: Path|None, split="val", confidence=.25, ap_score_threshold=.001, max_images=0, metadata=None):
    records=collect_predictions(model,loader,device,ap_score_threshold,max_images)
    metrics,per_class,confusion=compute_detection_metrics(records,len(class_names),.5,confidence)
    if output_dir is not None: save_evaluation_artifacts(output_dir,split,records,metrics,per_class,confusion,class_names,metadata or {})
    return metrics

def run_training(args) -> Path:
    device=resolve_device(args.device); resume_checkpoint=None; warm_start_checkpoint=None
    if args.resume:
        resume_checkpoint=torch.load(args.resume,map_location=device,weights_only=False); validate_checkpoint(resume_checkpoint,CLASS_NAMES); apply_resume_configuration(args,resume_checkpoint)
        args._anchor_config=deepcopy(resume_checkpoint["anchor_config"])
        saved_config=resume_checkpoint.get("training_config",{})
        args.anchor_profile=saved_config.get("anchor_profile","custom")
        args._warm_start_source=saved_config.get("warm_start_source")
    else:
        args._anchor_config=deepcopy(ANCHOR_PROFILES[args.anchor_profile])
    if args.warm_start:
        warm_start_checkpoint=torch.load(args.warm_start,map_location=device,weights_only=False)
        validate_checkpoint(warm_start_checkpoint,CLASS_NAMES)
    seed_everything(args.seed); smoke=args.smoke_test; stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    architecture_tag="highres_" if args.architecture==HIGHRES_ARCHITECTURE else ""
    run_dir=args.resume.resolve().parent if args.resume else RUNS_DIR/f"mobilenetv2_ssdlite_{architecture_tag}{'smoke_' if smoke else ''}{stamp}"; run_dir.mkdir(parents=True,exist_ok=True)
    train_base=YoloDetectionDataset(args.dataset,"train",True,args.object_crop_probability); val_base=YoloDetectionDataset(args.dataset,"val",False)
    if train_base.class_names != CLASS_NAMES or len(train_base.class_names)!=63: raise ValueError("Authoritative data.yaml class mapping mismatch")
    save_augmentation_preview(train_base,run_dir/"augmentation_preview.png",4)
    train_data=Subset(train_base,range(min(2,len(train_base)))) if smoke else train_base; val_data=Subset(val_base,range(min(2,len(val_base)))) if smoke else val_base
    train_loader=make_loader(train_data,min(args.batch_size,len(train_data)),True,args.num_workers,device); val_loader=make_loader(val_data,1 if smoke else args.batch_size,False,args.num_workers,device)
    config=training_config(args,run_dir,smoke); config["dataset_fingerprint"]=dataset_fingerprint(args.dataset,"train")
    if warm_start_checkpoint and warm_start_checkpoint.get("dataset_fingerprint") not in (None,config["dataset_fingerprint"]):
        raise ValueError("Warm-start checkpoint was trained on a different dataset fingerprint")
    atomic_json(run_dir/"config.json",config); atomic_json(run_dir/"class_names.json",{"class_names":CLASS_NAMES,"num_classes":63,"background_label":0})
    checkpoint=None
    if args.resume:
        model,checkpoint,_=load_detector_checkpoint(args.resume,device,CLASS_NAMES)
        validate_checkpoint(checkpoint, CLASS_NAMES, args.image_size)
        if checkpoint["dataset_fingerprint"] != config["dataset_fingerprint"]: raise ValueError("Dataset fingerprint changed; refusing exact resume")
    else:
        model=build_detector(63,args.image_size,not args.no_pretrained and warm_start_checkpoint is None,args._anchor_config,args.confidence,args.nms_threshold,args.max_detections,args.topk_candidates,args.architecture,args.positive_iou_threshold,args.negative_to_positive_ratio,args.classification_loss,args.focal_gamma,args.focal_alpha).to(device)
        if warm_start_checkpoint:
            migration=load_warm_start_weights(model,warm_start_checkpoint)
            print(f"Warm-started model weights from {args.warm_start.resolve()}: {migration}")
    atomic_json(run_dir/"architecture_summary.json",architecture_summary(model,args.image_size,device))
    total=args.phase1_epochs+args.phase2_epochs; start=int(checkpoint["epoch"])+1 if checkpoint else 1; history=list(checkpoint.get("history",[])) if checkpoint else []
    best=float(checkpoint.get("best_validation_metric",-1.)) if checkpoint else -1.; no_improve=int(checkpoint.get("early_stopping_state",{}).get("epochs_without_improvement",0)) if checkpoint else 0
    optimizer=scheduler=scaler=None; active_phase=None
    if checkpoint: restore_rng_state(checkpoint.get("rng_state"))
    for epoch in range(start,total+1):
        phase=1 if epoch<=args.phase1_epochs else 2
        if phase!=active_phase:
            optimizer=torch.optim.AdamW(optimizer_groups(model,phase,args.backbone_lr,args.head_lr,args.train_stage2,args.train_full_backbone),weight_decay=args.weight_decay)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=args.scheduler_patience,factor=.5)
            scaler=torch.amp.GradScaler("cuda",enabled=args.amp and device.type=="cuda"); active_phase=phase
            if checkpoint and int(checkpoint.get("training_phase",0))==phase:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); scheduler.load_state_dict(checkpoint["scheduler_state_dict"]); scaler.load_state_dict(checkpoint["scaler_state_dict"])
        started=time.perf_counter(); losses=train_one_epoch(model,train_loader,optimizer,scaler,device,args.amp,1 if smoke else 0)
        do_eval=smoke or epoch%args.eval_every==0 or epoch==total
        meta={**config,"checkpoint":str(run_dir/"mobilenetv2_detector_latest.pt"),"evaluation_split":"val","task_type":"object_detection","device":str(device),"test_set_accessed":False}
        metrics=evaluate_model(model,val_loader,device,CLASS_NAMES,run_dir if do_eval else None,"val",args.confidence,args.ap_score_threshold,2 if smoke else 0,meta) if do_eval else {}
        selection=metrics.get("map_50_95")
        if selection is not None: scheduler.step(selection)
        lrs={g.get("name",str(i)):g["lr"] for i,g in enumerate(optimizer.param_groups)}
        row={"epoch":epoch,"phase":phase,"train_total_loss":losses["total"],"train_classification_loss":losses["classification"],"train_box_loss":losses["bbox_regression"],"val_map_50":metrics.get("map_50"),"val_map_50_95":selection,"val_precision":metrics.get("detection_precision"),"val_recall":metrics.get("detection_recall"),"backbone_lr":lrs.get("late_backbone",0.),"head_lr":lrs.get("detection_layers",0.),"epoch_seconds":time.perf_counter()-started}
        history.append(row); write_history(run_dir,history); improved=selection is not None and selection>best+args.early_stopping_min_delta
        if improved: best=float(selection); no_improve=0
        elif selection is not None: no_improve+=1
        payload=checkpoint_payload(model,optimizer,scheduler,scaler,epoch,phase,CLASS_NAMES,config,history,best,{"epochs_without_improvement":no_improve})
        save_checkpoint(run_dir/"mobilenetv2_detector_latest.pt",payload)
        if improved:
            save_checkpoint(run_dir/"mobilenetv2_detector_best.pt",payload)
            for source_name, best_name in (
                ("predictions.json", "best_predictions.json"),
                ("val_metrics.json", "best_val_metrics.json"),
                ("per_class_metrics.csv", "best_per_class_metrics.csv"),
                ("confusion_matrix.csv", "best_confusion_matrix.csv"),
                ("precision_recall.json", "best_precision_recall.json"),
            ):
                source = run_dir / source_name
                if source.is_file():
                    shutil.copy2(source, run_dir / best_name)
        print(json.dumps(row))
        if args.early_stopping_patience>0 and no_improve>=args.early_stopping_patience: break
    return run_dir

def run_check(args):
    device=resolve_device(args.device); dataset=YoloDetectionDataset(args.dataset,"train",False); model=build_detector(63,args.image_size,False,ANCHOR_PROFILES[args.anchor_profile],topk_candidates=args.topk_candidates,architecture=args.architecture,positive_iou_threshold=args.positive_iou_threshold,negative_to_positive_ratio=args.negative_to_positive_ratio,classification_loss=args.classification_loss,focal_gamma=args.focal_gamma,focal_alpha=args.focal_alpha).to(device)
    print(json.dumps({"classes":len(dataset.class_names),"train_images":len(dataset),"architecture":architecture_summary(model,args.image_size,device),"test_set_accessed":"test" in YoloDetectionDataset.accessed_splits},indent=2))

def build_parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset",type=Path,default=DATA_ROOT); p.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    p.add_argument("--architecture",choices=SUPPORTED_ARCHITECTURES,default=ARCHITECTURE)
    p.add_argument("--image-size",type=int,default=DEFAULT_IMAGE_SIZE); p.add_argument("--batch-size",type=int,default=DEFAULT_BATCH_SIZE); p.add_argument("--num-workers",type=int,default=0)
    p.add_argument("--phase1-epochs",type=int,default=10); p.add_argument("--phase2-epochs",type=int,default=20); p.add_argument("--backbone-lr",type=float,default=1e-5); p.add_argument("--head-lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--eval-every",type=int,default=1); p.add_argument("--confidence",type=float,default=.25); p.add_argument("--ap-score-threshold",type=float,default=.001); p.add_argument("--nms-threshold",type=float,default=.45); p.add_argument("--max-detections",type=int,default=200); p.add_argument("--topk-candidates",type=int,default=400)
    p.add_argument("--positive-iou-threshold",type=float,default=.5); p.add_argument("--negative-to-positive-ratio",type=float,default=3.)
    p.add_argument("--classification-loss",choices=("cross_entropy","focal"),default="cross_entropy"); p.add_argument("--focal-gamma",type=float,default=2.); p.add_argument("--focal-alpha",type=float,default=.25)
    p.add_argument("--anchor-profile",choices=tuple(ANCHOR_PROFILES),default="default"); p.add_argument("--object-crop-probability",type=float,default=0.0)
    p.add_argument("--train-stage2",action=argparse.BooleanOptionalAction,default=False)
    p.add_argument("--train-full-backbone",action=argparse.BooleanOptionalAction,default=False)
    p.add_argument("--early-stopping-patience",type=int,default=0); p.add_argument("--early-stopping-min-delta",type=float,default=0.); p.add_argument("--scheduler-patience",type=int,default=2); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--amp",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--no-pretrained",action="store_true")
    start=p.add_mutually_exclusive_group(); start.add_argument("--resume",type=Path); start.add_argument("--warm-start",type=Path)
    p.add_argument("--allow-resume-overrides",action="store_true"); p.add_argument("--smoke-test",action="store_true"); p.add_argument("--check-only",action="store_true"); return p

def main():
    args=build_parser().parse_args(); args._explicit_options={token.split("=",1)[0] for token in sys.argv[1:] if token.startswith("--")}
    run_check(args) if args.check_only else print(f"Run directory: {run_training(args)}")
if __name__=="__main__": main()
