"""Single reproducible YOLO11m training and export pipeline.

This replaces the former baseline stamper, focal trainer, supervisor, and
finalizer scripts. The trained method has two stages:

1. ``baseline`` fine-tunes COCO-pretrained YOLO11m on the 63-class dataset.
2. ``focal`` resumes from the baseline with balanced image sampling and a
   per-class inverse-frequency focal classification loss.

The refinement therefore measures the combined effect of balanced sampling
and focal loss; it is not a focal-loss-only ablation.

Typical commands:
    python3 train_yolo.py baseline
    python3 train_yolo.py focal
    python3 train_yolo.py finalize

Or run every stage:
    python3 train_yolo.py all

Final outputs:
    models/yolo11m_focal_epoch4_best_map50_95.pt  GUI/default checkpoint
    models/yolo11m_focal_epoch16_peak_map50.pt    highest-mAP50 checkpoint
    models/yolo11m_baseline_best.pt               when baseline is retrained
    models/yolo26m_comparison_best.pt              when YOLO26 is retrained
    models/training_history.json                   compact curves and evidence
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, strip_optimizer

ROOT = Path(__file__).resolve().parent
DATASET = (
    Path(os.environ.get("YOLO_DATASET", ROOT / "LVIS_Fruits_And_Vegetables"))
    .expanduser()
    .resolve()
)
STANDARD_DATA = DATASET / "data.yaml"
BALANCED_DATA = DATASET / "data_balanced.yaml"
BALANCED_LIST = DATASET / "balanced_train.txt"
MODELS = ROOT / "models"

BASELINE_RUN = MODELS / "yolo11m_baseline"
FOCAL_RUN = MODELS / "yolo11m_focal"
YOLO26_RUN = MODELS / "yolo26m_comparison"
DEPLOY_BEST = MODELS / "yolo11m_focal_epoch4_best_map50_95.pt"
DEPLOY_PEAK = MODELS / "yolo11m_focal_epoch16_peak_map50.pt"
BASELINE_DEPLOY = MODELS / "yolo11m_baseline_best.pt"
YOLO26_DEPLOY = MODELS / "yolo26m_comparison_best.pt"
AUTHOR_MODEL = MODELS / "specialist_v3.pt"
HISTORY = MODELS / "training_history.json"

CLASS_COUNT = 63
IMAGE_SIZE = 640
BATCH_SIZE = 12
WORKERS = 4
SEED = 0
GAMMA = 1.5
ALPHA_POWER = 0.5
HISTORICAL_REPEAT_DISTRIBUTION = {4: 3612, 3: 780, 2: 515, 1: 3314}


class CustomFocalBCE(nn.Module):
    """Per-class weighted focal BCE used by the YOLO classification head."""

    def __init__(self, alpha: torch.Tensor, gamma: float = GAMMA):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha.float().reshape(1, -1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            prediction.float(), target.float(), reduction="none"
        )
        probability_for_target = torch.exp(-loss)
        alpha = self.alpha.to(prediction.device).expand_as(target)
        return alpha * (1.000001 - probability_for_target) ** self.gamma * loss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    peak50 = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    peak95 = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))

    def metrics(row: dict[str, str]) -> dict[str, float | int]:
        return {
            "epoch": int(row["epoch"]),
            "precision": float(row["metrics/precision(B)"]),
            "recall": float(row["metrics/recall(B)"]),
            "mAP50": float(row["metrics/mAP50(B)"]),
            "mAP50-95": float(row["metrics/mAP50-95(B)"]),
        }

    return {
        "epochs_logged": len(rows),
        "peak_mAP50": metrics(peak50),
        "peak_mAP50-95": metrics(peak95),
        "final": metrics(rows[-1]),
    }


def build_balanced_list() -> None:
    """Rebuild the documented 21,132-entry rare-class sampling list.

    The deleted historical list contained 8,221 unique images with repeat
    counts {4: 3612, 3: 780, 2: 515, 1: 3314}. Images are ranked by the image
    frequency of their rarest class, then assigned those same repeat quotas.
    Images are ranked by the frequency of their rarest class and assigned the
    recorded repeat quotas, producing 21,132 entries from 8,221 unique images.
    """

    examples: list[tuple[Path, set[int]]] = []
    class_image_frequency: Counter = Counter()
    for split in ("train/train", "val/val"):
        label_directory = DATASET / "labels" / split
        image_directory = DATASET / "images" / split
        for label_path in sorted(label_directory.glob("*.txt")):
            classes = {
                int(fields[0])
                for line in label_path.read_text().splitlines()
                if len(fields := line.split()) >= 5
            }
            image_path = next(
                (
                    image_directory / f"{label_path.stem}{suffix}"
                    for suffix in (".jpg", ".jpeg", ".png", ".webp")
                    if (image_directory / f"{label_path.stem}{suffix}").is_file()
                ),
                None,
            )
            if image_path is None:
                raise FileNotFoundError(f"Image for label not found: {label_path}")
            examples.append((image_path.resolve(), classes))
            class_image_frequency.update(classes)

    expected_images = sum(HISTORICAL_REPEAT_DISTRIBUTION.values())
    if len(examples) != expected_images:
        raise RuntimeError(
            f"Expected {expected_images} training images, found {len(examples)}. "
            "The historical repeat quotas only apply to this dataset version."
        )
    ranked = sorted(
        examples,
        key=lambda item: (
            min(
                (class_image_frequency[class_id] for class_id in item[1]), default=10**9
            ),
            str(item[0]),
        ),
    )
    repeated_paths: list[str] = []
    start = 0
    for repeat in (4, 3, 2, 1):
        count = HISTORICAL_REPEAT_DISTRIBUTION[repeat]
        for image_path, _classes in ranked[start : start + count]:
            repeated_paths.extend([str(image_path)] * repeat)
        start += count
    BALANCED_LIST.write_text("\n".join(repeated_paths) + "\n")
    print(
        f"created balanced list: {len(repeated_paths)} entries from "
        f"{len(examples)} unique images"
    )


def configure_dataset(balanced: bool) -> Path:
    if not STANDARD_DATA.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {STANDARD_DATA}")
    target = BALANCED_DATA if balanced else STANDARD_DATA
    config = read_yaml(target if target.is_file() else STANDARD_DATA)
    config["path"] = str(DATASET)
    if balanced:
        if not BALANCED_LIST.is_file():
            build_balanced_list()
        normalized_paths = []
        for original in BALANCED_LIST.read_text().splitlines():
            marker = "/images/"
            if marker in original:
                relative_image = original.split(marker, 1)[1]
                normalized_paths.append(str(DATASET / "images" / relative_image))
            else:
                normalized_paths.append(original)
        BALANCED_LIST.write_text("\n".join(normalized_paths) + "\n")
        config["train"] = str(BALANCED_LIST)
    else:
        config["train"] = ["images/train/train", "images/val/val"]
    config["val"] = "images/test"
    config["test"] = "images/test"
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    return target


def class_counts() -> Counter:
    counts: Counter = Counter()
    for split in ("train/train", "val/val"):
        for label_path in (DATASET / "labels" / split).glob("*.txt"):
            try:
                for line in label_path.read_text().splitlines():
                    fields = line.split()
                    if len(fields) >= 5:
                        counts[int(fields[0])] += 1
            except (OSError, ValueError):
                continue
    return counts


def install_focal_loss() -> None:
    from ultralytics.utils import loss as ultralytics_loss

    counts = class_counts()
    raw = torch.tensor(
        [counts.get(index, 1) for index in range(CLASS_COUNT)], dtype=torch.float32
    )
    alpha = (1.0 / raw) ** ALPHA_POWER
    alpha = (alpha / alpha.mean()).clamp(0.02, 4.0)
    original_init = ultralytics_loss.v8DetectionLoss.__init__

    def focal_init(self, model, tal_topk=10, tal_topk2=None):
        original_init(self, model, tal_topk, tal_topk2)
        self.bce = CustomFocalBCE(alpha, GAMMA)

    ultralytics_loss.v8DetectionLoss.__init__ = focal_init
    print(
        f"classification loss: focal gamma={GAMMA}, "
        f"alpha median={float(alpha.median()):.3f}"
    )


def common_train_arguments() -> dict[str, Any]:
    return {
        "task": "detect",
        "save": True,
        "save_period": 5,
        "plots": False,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        "device": 0,
        "workers": WORKERS,
        "seed": SEED,
        "deterministic": True,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "box": 7.5,
        "dfl": 1.5,
        "mosaic": 1.0,
        "mixup": 0.1,
        "copy_paste": 0.1,
        "close_mosaic": 10,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "translate": 0.1,
        "scale": 0.5,
        "fliplr": 0.5,
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "amp": True,
        "val": True,
    }


def train_baseline() -> Path:
    """Train or resume the initial YOLO11m baseline."""

    run = BASELINE_RUN / "train"
    last = run / "weights" / "last.pt"
    if last.is_file():
        print(f"resuming baseline: {last}")
        YOLO(str(last)).train(resume=True)
    else:
        local_base = MODELS / "yolo11m.pt"
        source = str(local_base) if local_base.is_file() else "yolo11m.pt"
        YOLO(source).train(
            data=str(configure_dataset(balanced=False)),
            project=str(BASELINE_RUN),
            name="train",
            exist_ok=True,
            epochs=600,
            time=9.0,
            patience=300,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            **common_train_arguments(),
        )
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Baseline training did not produce best.pt")
    return best


def baseline_checkpoint() -> Path:
    candidate = BASELINE_RUN / "train" / "weights" / "best.pt"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("Run `python3 train_yolo.py baseline` first.")


def train_focal() -> Path:
    """Train or resume the balanced-sampling focal refinement."""

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    install_focal_loss()
    run = FOCAL_RUN / "train"
    last = run / "weights" / "last.pt"
    if last.is_file():
        print(f"resuming focal refinement: {last}")
        YOLO(str(last)).train(resume=True)
    else:
        source = baseline_checkpoint()
        YOLO(str(source)).train(
            data=str(configure_dataset(balanced=True)),
            project=str(FOCAL_RUN),
            name="train",
            exist_ok=True,
            epochs=500,
            time=10.0,
            patience=200,
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.01,
            **common_train_arguments(),
        )
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("Focal training did not produce best.pt")
    return best


def train_yolo26() -> Path:
    """Train or resume the recorded YOLO26m comparison configuration."""

    run = YOLO26_RUN / "train"
    last = run / "weights" / "last.pt"
    if last.is_file():
        print(f"resuming YOLO26m comparison: {last}")
        YOLO(str(last)).train(resume=True)
    else:
        local_base = MODELS / "yolo26m.pt"
        source = str(local_base) if local_base.is_file() else "yolo26m.pt"
        YOLO(source).train(
            data=str(configure_dataset(balanced=False)),
            project=str(YOLO26_RUN),
            name="train",
            exist_ok=True,
            task="detect",
            save=True,
            save_period=5,
            plots=False,
            epochs=1000,
            patience=0,
            batch=4,
            imgsz=640,
            device=0,
            workers=8,
            seed=42,
            deterministic=True,
            optimizer="MuSGD",
            lr0=0.01,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            close_mosaic=0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            auto_augment="randaugment",
            erasing=0.4,
            amp=True,
            val=True,
        )
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError("YOLO26m training did not produce best.pt")
    return best


def peak_checkpoint(rows: list[dict[str, str]]) -> Path:
    peak = max(rows, key=lambda row: float(row["metrics/mAP50(B)"]))
    epoch = int(peak["epoch"])
    candidates = [
        FOCAL_RUN / "train" / "weights" / f"epoch{epoch - 1}.pt",
        FOCAL_RUN / "train" / "weights" / f"epoch{epoch}.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Saved checkpoint for peak mAP50 epoch {epoch} not found")


def stripped_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    strip_optimizer(destination)
    return destination


def measure_model(path: Path) -> dict[str, Any]:
    sys.modules["__main__"].CustomFocalBCE = CustomFocalBCE
    yolo = YOLO(str(path))
    model = yolo.model
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "GFLOPs_640": round(float(get_flops(model, IMAGE_SIZE)), 3),
        "classes": len(yolo.names),
    }


def box_iou_matrix(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if len(predicted) == 0 or len(truth) == 0:
        return np.zeros((len(predicted), len(truth)), dtype=np.float32)
    top_left = np.maximum(predicted[:, None, :2], truth[None, :, :2])
    bottom_right = np.minimum(predicted[:, None, 2:], truth[None, :, 2:])
    size = np.clip(bottom_right - top_left, 0, None)
    intersection = size[..., 0] * size[..., 1]
    predicted_area = (predicted[:, 2] - predicted[:, 0]) * (
        predicted[:, 3] - predicted[:, 1]
    )
    truth_area = (truth[:, 2] - truth[:, 0]) * (truth[:, 3] - truth[:, 1])
    return intersection / (
        predicted_area[:, None] + truth_area[None, :] - intersection + 1e-9
    )


def greedy_matches(
    overlaps: np.ndarray,
    predicted_classes: np.ndarray,
    truth_classes: np.ndarray,
    match_iou: float,
    require_same_class: bool,
) -> list[tuple[int, int, float]]:
    candidates = []
    for prediction_index in range(overlaps.shape[0]):
        for truth_index in range(overlaps.shape[1]):
            same_class = (
                predicted_classes[prediction_index] == truth_classes[truth_index]
            )
            if overlaps[prediction_index, truth_index] >= match_iou and (
                same_class or not require_same_class
            ):
                candidates.append(
                    (
                        float(overlaps[prediction_index, truth_index]),
                        prediction_index,
                        truth_index,
                    )
                )
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    matches = []
    for overlap, prediction_index, truth_index in candidates:
        if prediction_index not in used_predictions and truth_index not in used_truth:
            used_predictions.add(prediction_index)
            used_truth.add(truth_index)
            matches.append((prediction_index, truth_index, overlap))
    return matches


def load_ground_truth(
    label_directory: Path, image_path: Path, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    classes = []
    label_path = label_directory / f"{image_path.stem}.txt"
    for line in label_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(fields[0])
        center_x, center_y, box_width, box_height = map(float, fields[1:5])
        boxes.append(
            [
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            ]
        )
        classes.append(class_id)
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(classes, dtype=np.int32),
    )


def evaluate_retinanet_style_metrics(
    weight: Path,
    confidence: float,
    match_iou: float,
    nms_iou: float,
) -> dict[str, Any]:
    """Calculate fixed-threshold matched-box metrics using CUDA only."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU evaluation is forbidden")
    image_directory = DATASET / "images" / "test"
    label_directory = DATASET / "labels" / "test"
    if not image_directory.is_dir() or not label_directory.is_dir():
        raise FileNotFoundError(
            f"Test images/labels not found under external dataset: {DATASET}"
        )
    image_paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    model = YOLO(str(weight))
    true_positives = false_positives = false_negatives = 0
    localized_objects = correct_localized_classes = 0
    total_ground_truth = total_predictions = 0
    matched_ious: list[float] = []
    started = time.perf_counter()
    results = model.predict(
        str(image_directory),
        conf=confidence,
        iou=nms_iou,
        imgsz=IMAGE_SIZE,
        device=0,
        batch=1,
        half=True,
        stream=True,
        verbose=False,
    )
    for image_number, result in enumerate(results, 1):
        image_path = Path(result.path)
        height, width = result.orig_shape
        truth_boxes, truth_classes = load_ground_truth(
            label_directory, image_path, width, height
        )
        total_ground_truth += len(truth_boxes)
        if result.boxes is None:
            predicted_boxes = np.empty((0, 4), dtype=np.float32)
            predicted_classes = np.empty(0, dtype=np.int32)
        else:
            predicted_boxes = result.boxes.xyxy.cpu().numpy()
            predicted_classes = result.boxes.cls.cpu().numpy().astype(np.int32)
        total_predictions += len(predicted_boxes)
        overlaps = box_iou_matrix(predicted_boxes, truth_boxes)
        class_matches = greedy_matches(
            overlaps,
            predicted_classes,
            truth_classes,
            match_iou,
            require_same_class=True,
        )
        location_matches = greedy_matches(
            overlaps,
            predicted_classes,
            truth_classes,
            match_iou,
            require_same_class=False,
        )
        true_positives += len(class_matches)
        false_positives += len(predicted_boxes) - len(class_matches)
        false_negatives += len(truth_boxes) - len(class_matches)
        localized_objects += len(location_matches)
        matched_ious.extend(
            overlap for _prediction, _truth, overlap in location_matches
        )
        correct_localized_classes += sum(
            int(predicted_classes[prediction] == truth_classes[truth])
            for prediction, truth, _overlap in location_matches
        )
        if image_number % 30 == 0:
            print(f"{weight.name}: {image_number}/{len(image_paths)}", flush=True)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    metrics = {
        "weight": str(weight.relative_to(ROOT)),
        "weight_sha256": sha256(weight),
        "device": torch.cuda.get_device_name(0),
        "images": len(image_paths),
        "ground_truth_objects": total_ground_truth,
        "predictions": total_predictions,
        "confidence_threshold": confidence,
        "matching_iou_threshold": match_iou,
        "nms_iou_threshold": nms_iou,
        "matching": "greedy one-to-one descending IoU",
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "localized_matches": localized_objects,
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall),
        "classification_accuracy_matched": correct_localized_classes
        / localized_objects,
        "mean_iou_matched": float(np.mean(matched_ious)),
        "detection_rate_iou_0_5": localized_objects / total_ground_truth,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del results, model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def evaluate_metrics(
    weights: list[Path], confidence: float, match_iou: float, nms_iou: float
) -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU evaluation is forbidden")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    measured = {
        weight.name: evaluate_retinanet_style_metrics(
            weight, confidence, match_iou, nms_iou
        )
        for weight in weights
    }
    history = json.loads(HISTORY.read_text()) if HISTORY.is_file() else {}
    history["custom_detection_metrics"] = {
        "evaluated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": measured,
    }
    HISTORY.write_text(json.dumps(history, indent=2))
    print(json.dumps(measured, indent=2))


def snapshot_history() -> dict[str, Any]:
    """Preserve compact training evidence before generated run folders are pruned."""

    run_paths = {
        "baseline_yolo11m": BASELINE_RUN,
        "focal_refinement": FOCAL_RUN,
        "yolo26m_comparison": YOLO26_RUN,
    }
    previous = json.loads(HISTORY.read_text()) if HISTORY.is_file() else {}
    previous_runs = previous.get("runs", {})
    runs: dict[str, Any] = {}
    for name, run_root in run_paths.items():
        results_path = (
            run_root / "train" / "results.csv"
            if (run_root / "train" / "results.csv").is_file()
            else run_root / "results.csv"
        )
        args_path = (
            run_root / "train" / "args.yaml"
            if (run_root / "train" / "args.yaml").is_file()
            else run_root / "args.yaml"
        )
        rows = read_csv(results_path)
        if rows:
            runs[name] = {
                "metrics": metric_summary(rows),
                "arguments": read_yaml(args_path),
                "epoch_rows": rows,
            }
        else:
            runs[name] = previous_runs.get(name, {})
    history = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": {
            "architecture": "YOLO11m",
            "refinement": "balanced image sampling plus per-class inverse-frequency focal BCE",
            "gamma": GAMMA,
            "alpha_power": ALPHA_POWER,
            "unique_training_images": 8221,
            "balanced_list_entries": sum(1 for _ in BALANCED_LIST.open())
            if BALANCED_LIST.is_file()
            else previous.get("method", {}).get("balanced_list_entries", 21132),
            "repeat_distribution": HISTORICAL_REPEAT_DISTRIBUTION,
        },
        "runs": runs,
    }
    if DEPLOY_BEST.is_file():
        history["deployment_best"] = measure_model(DEPLOY_BEST)
    if DEPLOY_PEAK.is_file():
        history["deployment_peak_mAP50"] = measure_model(DEPLOY_PEAK)
    if BASELINE_DEPLOY.is_file():
        history["baseline_model"] = measure_model(BASELINE_DEPLOY)
    if YOLO26_DEPLOY.is_file():
        history["yolo26m_model"] = measure_model(YOLO26_DEPLOY)
    if AUTHOR_MODEL.is_file():
        history["author_model"] = measure_model(AUTHOR_MODEL)
    HISTORY.write_text(json.dumps(history, indent=2, default=str))
    print(f"saved compact training history: {HISTORY}")
    return history


def finalize(export_onnx: bool = False) -> None:
    """Create portable stripped weights from the completed focal run."""

    sys.modules["__main__"].CustomFocalBCE = CustomFocalBCE
    run = FOCAL_RUN / "train"
    best = run / "weights" / "best.pt"
    rows = read_csv(run / "results.csv")
    if not best.is_file() or not rows:
        raise FileNotFoundError(
            "A completed focal run is required before finalization."
        )
    stripped_copy(best, DEPLOY_BEST)
    stripped_copy(peak_checkpoint(rows), DEPLOY_PEAK)
    baseline_best = BASELINE_RUN / "train" / "weights" / "best.pt"
    if baseline_best.is_file():
        stripped_copy(baseline_best, BASELINE_DEPLOY)
    yolo26_best = YOLO26_RUN / "train" / "weights" / "best.pt"
    if yolo26_best.is_file():
        stripped_copy(yolo26_best, YOLO26_DEPLOY)
    snapshot_history()
    if export_onnx:
        exported = YOLO(str(DEPLOY_BEST)).export(
            format="onnx",
            imgsz=IMAGE_SIZE,
            half=True,
            dynamic=False,
            opset=17,
            simplify=True,
        )
        print(f"ONNX exported: {exported}")
    artifacts = {
        "focal_epoch4": measure_model(DEPLOY_BEST),
        "focal_epoch16": measure_model(DEPLOY_PEAK),
    }
    if BASELINE_DEPLOY.is_file():
        artifacts["baseline"] = measure_model(BASELINE_DEPLOY)
    if YOLO26_DEPLOY.is_file():
        artifacts["yolo26m"] = measure_model(YOLO26_DEPLOY)
    print(json.dumps(artifacts, indent=2))


def audit() -> None:
    if not HISTORY.is_file():
        snapshot_history()
    history = json.loads(HISTORY.read_text())
    summary: dict[str, Any] = {
        "method": history.get("method"),
        "runs": {
            name: data.get("metrics") for name, data in history.get("runs", {}).items()
        },
        "deployment_best": history.get("deployment_best"),
        "deployment_peak_mAP50": history.get("deployment_peak_mAP50"),
        "author_model": history.get("author_model"),
    }
    for key in ("baseline_model", "yolo26m_model"):
        if history.get(key) is not None:
            summary[key] = history[key]
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO11m baseline, class-balanced focal refinement, and export"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("baseline", help="train/resume the initial YOLO11m baseline")
    subparsers.add_parser("focal", help="train/resume balanced focal refinement")
    subparsers.add_parser("yolo26", help="train/resume the YOLO26m comparison")
    finalize_parser = subparsers.add_parser(
        "finalize", help="create portable final weights"
    )
    finalize_parser.add_argument("--onnx", action="store_true", help="also export ONNX")
    all_parser = subparsers.add_parser(
        "all", help="run baseline, focal, and finalization"
    )
    all_parser.add_argument("--onnx", action="store_true", help="also export ONNX")
    subparsers.add_parser("snapshot", help="save compact history from existing runs")
    subparsers.add_parser("audit", help="print preserved metrics and model information")
    metrics_parser = subparsers.add_parser(
        "metrics", help="GPU-only fixed-threshold detection metrics"
    )
    metrics_parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        default=[DEPLOY_BEST, DEPLOY_PEAK],
    )
    metrics_parser.add_argument("--confidence", type=float, default=0.25)
    metrics_parser.add_argument("--match-iou", type=float, default=0.50)
    metrics_parser.add_argument("--nms-iou", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.command == "baseline":
        print(train_baseline())
    elif args.command == "focal":
        print(train_focal())
    elif args.command == "yolo26":
        print(train_yolo26())
    elif args.command == "finalize":
        finalize(args.onnx)
    elif args.command == "all":
        train_baseline()
        train_focal()
        finalize(args.onnx)
    elif args.command == "snapshot":
        snapshot_history()
    elif args.command == "audit":
        audit()
    elif args.command == "metrics":
        evaluate_metrics(args.weights, args.confidence, args.match_iou, args.nms_iou)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
