"""
evaluation.py

Model comparison / evaluation engine for the Fruit & Vegetable
Object Detection project.

Responsibilities:
  1. Load the held-out test split (YOLO-format .txt labels + images).
  2. Run each of the three trained models (RetinaNet, YOLO11m, MobileNetV2)
     over the test set at a LOW confidence threshold, so full PR curves /
     mAP can be computed fairly across models.
  3. Compute:
       - COCO-style mAP@0.5, mAP@0.5:0.95, per-class AP (via torchmetrics)
       - Precision / Recall / F1 at a fixed operating threshold (0.5 conf)
       - A confusion matrix (greedy IoU matching at 0.5 conf / 0.5 IoU)
       - A micro-averaged PR curve per model
       - Average inference latency / FPS
  4. Save each model's result as its own JSON file, inside that model's
     own folder under 04_Trained_Model/, so results don't depend on a
     single shared cache file and can be inspected/copied independently.

This module has NO Streamlit UI calls in it (safe to import from app.py).
It can ALSO be run directly as a standalone script to perform the full
evaluation without ever starting the Streamlit app:

    python 02_Source_Code/evaluation.py            # skip models that
                                                    # already have a
                                                    # saved result
    python 02_Source_Code/evaluation.py --force    # re-run all models

Results are written to:
    04_Trained_Model/RetinaNet/eval_result.json
    04_Trained_Model/YOLO/eval_result.json
    04_Trained_Model/MobileNet_V2/eval_result.json

app.py then simply reads these three files to draw all comparison
charts - it never runs inference itself.
"""

import glob
import json
import os
import sys
import time
import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    _HAS_TORCHMETRICS = True
except ImportError:
    _HAS_TORCHMETRICS = False


# ------------------------------------------------------------------
# Shared helper (mirrors app.py's isolated_import_dir so evaluation.py
# has no import-order dependency on app.py)
# ------------------------------------------------------------------

@contextlib.contextmanager
def isolated_import_dir(directory):
    sys.path.insert(0, directory)
    stale_names = ["config", "model", "detector"]
    saved = {name: sys.modules.pop(name, None) for name in stale_names}
    try:
        yield
    finally:
        sys.path.remove(directory)
        for name in stale_names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


# ------------------------------------------------------------------
# 1. Test-set loading (YOLO-format labels)
# ------------------------------------------------------------------

def load_yolo_test_set(images_dir, labels_dir, class_names):
    """
    Returns a list of dicts:
        {"image_path": str, "width": int, "height": int,
         "boxes": [(class_id, x1, y1, x2, y2), ...]}   # absolute pixel xyxy
    Images with a label file but zero valid boxes are still included
    (they contribute to recall denominators as "no GT" images... but in
    practice this dataset should always have boxes; empty files are kept
    for completeness / debugging).
    """
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    dataset = []
    skipped_no_label = 0
    for img_path in image_paths:
        stem = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        if not os.path.exists(label_path):
            skipped_no_label += 1
            continue

        with Image.open(img_path) as im:
            width, height = im.size

        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = parts
                cls_id = int(cls_id)
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                x1 = (xc - w / 2) * width
                y1 = (yc - h / 2) * height
                x2 = (xc + w / 2) * width
                y2 = (yc + h / 2) * height
                if cls_id < 0 or cls_id >= len(class_names):
                    continue
                boxes.append((cls_id, x1, y1, x2, y2))

        dataset.append(
            {"image_path": img_path, "width": width, "height": height, "boxes": boxes}
        )

    if skipped_no_label:
        print(f"[evaluation] Skipped {skipped_no_label} images with no matching label file.")
    return dataset


# ------------------------------------------------------------------
# 2. Eval-only model loaders (override the baked-in confidence
#    threshold so all three models can be compared at the same,
#    very low operating point -> fair PR curves / mAP).
# ------------------------------------------------------------------

def load_retinanet_for_eval(base_dir, retinanet_path, num_classes, device, conf_threshold=0.001):
    retinanet_dir = os.path.join(base_dir, "RetinaNet")
    with isolated_import_dir(retinanet_dir):
        from model import build_model
        model = build_model(num_classes=num_classes, pretrained=False)

    checkpoint = torch.load(retinanet_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    # torchvision RetinaNet exposes score_thresh on the head; override it
    # directly so eval isn't limited by whatever threshold training used.
    if hasattr(model, "score_thresh"):
        model.score_thresh = conf_threshold
    elif hasattr(model, "head") and hasattr(model.head, "score_thresh"):
        model.head.score_thresh = conf_threshold
    model.to(device).eval()
    return model


def load_mobilenet_for_eval(base_dir, mobilenet_path, device, conf_threshold=0.001):
    mobilenet_dir = os.path.join(base_dir, "Mobilenet_V2")
    with isolated_import_dir(mobilenet_dir):
        from train import load_detector_checkpoint
        # class_names arg left as None -> uses names stored in checkpoint
        model, checkpoint, names = load_detector_checkpoint(Path(mobilenet_path), device)

    # Same idea as RetinaNet: try to relax the baked-in score threshold
    # so evaluation sees the full prediction set, not just the training
    # operating point.
    for attr_owner in (model, getattr(model, "head", None)):
        if attr_owner is not None and hasattr(attr_owner, "score_thresh"):
            attr_owner.score_thresh = conf_threshold
        if attr_owner is not None and hasattr(attr_owner, "confidence_threshold"):
            attr_owner.confidence_threshold = conf_threshold

    model.eval()
    return model, names


def load_yolo_for_eval(yolo_path):
    from ultralytics import YOLO
    return YOLO(yolo_path)


# ------------------------------------------------------------------
# 3. Raw (no drawing / no Streamlit) prediction functions
#    Return: list of (class_name, confidence, [x1, y1, x2, y2])
# ------------------------------------------------------------------

def predict_retinanet_raw(model, image, device, conf_threshold, class_names):
    from torchvision import transforms
    img_tensor = transforms.ToTensor()(image).to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    elapsed = time.time() - start

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    keep = scores >= conf_threshold
    preds = []
    for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
        if 0 <= label < len(class_names):
            preds.append((class_names[label], float(score), [float(v) for v in box]))
    return preds, elapsed


def predict_mobilenet_raw(model, image, device, conf_threshold, class_names):
    from torchvision import transforms
    img_tensor = transforms.ToTensor()(image).to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    elapsed = time.time() - start

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    keep = scores >= conf_threshold
    preds = []
    for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
        if 0 < label <= len(class_names):
            preds.append((class_names[label - 1], float(score), [float(v) for v in box]))
    return preds, elapsed


def predict_yolo_raw(model, image, conf_threshold):
    start = time.time()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    elapsed = time.time() - start
    result = results[0]
    preds = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        preds.append((cls_name, conf, [x1, y1, x2, y2]))
    return preds, elapsed


# ------------------------------------------------------------------
# 4. Geometry helpers
# ------------------------------------------------------------------

def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------
# 5. Per-model evaluation
# ------------------------------------------------------------------

def evaluate_model(model_name, dataset, class_names, predict_fn,
                    operating_conf=0.5, iou_match_thresh=0.5, low_conf=0.001,
                    progress_callback=None):
    """
    predict_fn(image) -> (list of (class_name, conf, box_xyxy), elapsed_seconds)
    Runs once per image at `low_conf` so results support both the full
    mAP/PR-curve computation and the fixed-threshold P/R/F1/confusion
    matrix (by filtering the same prediction list down to operating_conf).
    """
    class_to_id = {name: i for i, name in enumerate(class_names)}
    n_classes = len(class_names)

    torchmetrics_preds, torchmetrics_targets = [], []
    latencies = []

    # confusion matrix: rows/cols 0..n_classes-1 are real classes,
    # index n_classes represents "background" (FP row / FN column)
    confusion = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    BG = n_classes

    # for micro-averaged PR curve: pooled (confidence, is_tp) across all
    # classes and images, computed via greedy IoU matching per image
    pooled_conf = []
    pooled_is_tp = []
    total_gt = 0

    for i, sample in enumerate(dataset):
        image = Image.open(sample["image_path"]).convert("RGB")
        raw_preds, elapsed = predict_fn(image)
        latencies.append(elapsed)

        gt_boxes = sample["boxes"]  # (class_id, x1, y1, x2, y2)
        total_gt += len(gt_boxes)

        # ---- torchmetrics accumulation (uses ALL preds at low_conf) ----
        if _HAS_TORCHMETRICS:
            if raw_preds:
                p_boxes = torch.tensor([p[2] for p in raw_preds], dtype=torch.float32)
                p_scores = torch.tensor([p[1] for p in raw_preds], dtype=torch.float32)
                p_labels = torch.tensor(
                    [class_to_id.get(p[0], -1) for p in raw_preds], dtype=torch.int64
                )
                valid = p_labels >= 0
                p_boxes, p_scores, p_labels = p_boxes[valid], p_scores[valid], p_labels[valid]
            else:
                p_boxes = torch.zeros((0, 4))
                p_scores = torch.zeros((0,))
                p_labels = torch.zeros((0,), dtype=torch.int64)

            if gt_boxes:
                t_boxes = torch.tensor([b[1:] for b in gt_boxes], dtype=torch.float32)
                t_labels = torch.tensor([b[0] for b in gt_boxes], dtype=torch.int64)
            else:
                t_boxes = torch.zeros((0, 4))
                t_labels = torch.zeros((0,), dtype=torch.int64)

            torchmetrics_preds.append({"boxes": p_boxes, "scores": p_scores, "labels": p_labels})
            torchmetrics_targets.append({"boxes": t_boxes, "labels": t_labels})

        # ---- pooled PR-curve bookkeeping (all preds, sorted by conf, greedy match) ----
        preds_sorted = sorted(raw_preds, key=lambda p: -p[1])
        gt_used = [False] * len(gt_boxes)
        for cls_name, conf, box in preds_sorted:
            best_iou, best_j = 0.0, -1
            for j, (gt_cls, *gt_box) in enumerate(gt_boxes):
                if gt_used[j]:
                    continue
                iou = iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= iou_match_thresh and best_j != -1 and gt_boxes[best_j][0] == class_to_id.get(cls_name, -2)
            pooled_conf.append(conf)
            pooled_is_tp.append(1 if is_tp else 0)
            if is_tp:
                gt_used[best_j] = True

        # ---- confusion matrix (fixed operating threshold only) ----
        op_preds = sorted(
            [p for p in raw_preds if p[1] >= operating_conf], key=lambda p: -p[1]
        )
        gt_used_cm = [False] * len(gt_boxes)
        for cls_name, conf, box in op_preds:
            pred_id = class_to_id.get(cls_name, None)
            if pred_id is None:
                continue
            best_iou, best_j = 0.0, -1
            for j, (gt_cls, *gt_box) in enumerate(gt_boxes):
                if gt_used_cm[j]:
                    continue
                iou = iou_xyxy(box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_match_thresh and best_j != -1:
                gt_used_cm[best_j] = True
                gt_id = gt_boxes[best_j][0]
                confusion[gt_id, pred_id] += 1
            else:
                confusion[BG, pred_id] += 1  # false positive
        for used, (gt_cls, *_) in zip(gt_used_cm, gt_boxes):
            if not used:
                confusion[gt_cls, BG] += 1  # false negative (missed)

        if progress_callback:
            progress_callback(model_name, i + 1, len(dataset))

    # ---- mAP via torchmetrics ----
    map_metrics = {}
    per_class_ap = {}
    if _HAS_TORCHMETRICS and dataset:
        metric = MeanAveragePrecision(class_metrics=True)
        metric.update(torchmetrics_preds, torchmetrics_targets)
        result = metric.compute()
        map_metrics = {
            "mAP@0.5:0.95": float(result["map"]),
            "mAP@0.5": float(result["map_50"]),
            "mAP@0.75": float(result["map_75"]),
            "mAR@100": float(result["mar_100"]),
        }
        if "map_per_class" in result and "classes" in result:
            for cls_id, ap in zip(result["classes"].tolist(), result["map_per_class"].tolist()):
                if 0 <= cls_id < n_classes:
                    per_class_ap[class_names[cls_id]] = round(float(ap), 4) if ap >= 0 else None

    # ---- fixed-threshold precision/recall/F1 (from confusion matrix) ----
    tp = int(np.trace(confusion[:n_classes, :n_classes]))
    fp = int(confusion[BG, :n_classes].sum())
    fn = int(confusion[:n_classes, BG].sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # ---- pooled PR curve, sorted by descending confidence ----
    order = np.argsort(-np.array(pooled_conf)) if pooled_conf else np.array([], dtype=int)
    is_tp_sorted = np.array(pooled_is_tp)[order] if len(order) else np.array([])
    cum_tp = np.cumsum(is_tp_sorted)
    cum_fp = np.cumsum(1 - is_tp_sorted)
    pr_precision = (cum_tp / np.maximum(cum_tp + cum_fp, 1)).tolist() if len(cum_tp) else []
    pr_recall = (cum_tp / max(total_gt, 1)).tolist() if len(cum_tp) else []

    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "model_name": model_name,
        "map_metrics": map_metrics,
        "per_class_ap": per_class_ap,
        "operating_conf": operating_conf,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency_sec": avg_latency,
        "fps": (1.0 / avg_latency) if avg_latency > 0 else 0.0,
        "confusion_matrix": confusion.tolist(),
        "class_names": class_names,
        "pr_curve": {"precision": pr_precision, "recall": pr_recall},
        "num_test_images": len(dataset),
        "num_gt_boxes": total_gt,
    }


# ------------------------------------------------------------------
# 6. Orchestration - each model's result is saved to its OWN file,
#    inside that model's own folder under 04_Trained_Model/, instead
#    of one shared cache file. This lets each model be re-evaluated
#    independently and keeps results next to the checkpoint they
#    came from.
# ------------------------------------------------------------------

def get_result_paths(project_root):
    """
    Returns {model_name: json_path}, one path per model, each living
    inside that model's own checkpoint folder.
    """
    model_dir = os.path.join(project_root, "04_Trained_Model")
    return {
        "RetinaNet": os.path.join(model_dir, "RetinaNet", "eval_result.json"),
        "YOLO11m": os.path.join(model_dir, "YOLO", "eval_result.json"),
        "MobileNetV2": os.path.join(model_dir, "MobileNet_V2", "eval_result.json"),
    }


def run_full_evaluation(model_predict_fns, dataset, class_names, result_paths,
                         force_rerun=False, progress_callback=None):
    """
    model_predict_fns: dict {model_name: predict_fn(image) -> (preds, elapsed)}
    result_paths: dict {model_name: json_path}, e.g. from get_result_paths().
    Returns: {model_name: evaluate_model(...) result dict}

    A model is skipped (its existing file is loaded instead of re-run)
    if its result file already exists and force_rerun is False.
    """
    results = {}
    for model_name, predict_fn in model_predict_fns.items():
        out_path = result_paths[model_name]

        if not force_rerun and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                results[model_name] = json.load(f)
            continue

        results[model_name] = evaluate_model(
            model_name, dataset, class_names, predict_fn,
            progress_callback=progress_callback,
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results[model_name], f)

    return results


def load_cached_results(result_paths):
    """
    Load whatever per-model result files already exist, WITHOUT running
    any inference. Returns (results_dict, missing_model_names).
    This is what app.py calls - it never triggers model inference itself.
    """
    results = {}
    missing = []
    for model_name, path in result_paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                results[model_name] = json.load(f)
        else:
            missing.append(model_name)
    return results, missing


# ------------------------------------------------------------------
# 7. Standalone entry point - lets this file be run directly to
#    evaluate all three models, completely independent of app.py /
#    Streamlit.
#
#    Usage:
#        python 02_Source_Code/evaluation.py            # skip models
#                                                        # that already
#                                                        # have a saved
#                                                        # result
#        python 02_Source_Code/evaluation.py --force    # re-run all
# ------------------------------------------------------------------

def _standalone_main(force_rerun=False):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    model_dir = os.path.join(project_root, "04_Trained_Model")
    retinanet_path = os.path.join(model_dir, "RetinaNet", "retinanet_best.pt")
    yolo_path = os.path.join(model_dir, "YOLO", "yolo11m_focal_epoch4_best_map50_95.pt")
    mobilenet_path = os.path.join(model_dir, "MobileNet_V2", "mobilenetv2_detector_best.pt")
    classes_txt_path = os.path.join(base_dir, "RetinaNet", "classes.txt")

    test_images_dir = os.path.join(project_root, "03_Dataset", "LVIS_Fruits_And_Vegetables", "images", "test")
    test_labels_dir = os.path.join(project_root, "03_Dataset", "LVIS_Fruits_And_Vegetables", "labels", "test")

    with open(classes_txt_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluation] device = {device}, num_classes = {len(class_names)}")

    dataset = load_yolo_test_set(test_images_dir, test_labels_dir, class_names)
    print(f"[evaluation] test images = {len(dataset)}")

    print("[evaluation] loading models for evaluation...")
    retinanet_model = load_retinanet_for_eval(
        base_dir, retinanet_path, len(class_names), device, conf_threshold=0.001
    )
    mobilenet_model, _ = load_mobilenet_for_eval(
        base_dir, mobilenet_path, device, conf_threshold=0.001
    )
    yolo_model = load_yolo_for_eval(yolo_path)

    predict_fns = {
        "RetinaNet": lambda img: predict_retinanet_raw(
            retinanet_model, img, device, 0.001, class_names
        ),
        "YOLO11m": lambda img: predict_yolo_raw(yolo_model, img, 0.001),
        "MobileNetV2": lambda img: predict_mobilenet_raw(
            mobilenet_model, img, device, 0.001, class_names
        ),
    }

    result_paths = get_result_paths(project_root)

    def progress_cb(model_name, done, total):
        if done % 20 == 0 or done == total:
            print(f"  [{model_name}] {done}/{total}")

    for model_name, path in result_paths.items():
        if not force_rerun and os.path.exists(path):
            print(f"[evaluation] {model_name}: result already exists at {path}, "
                  f"skipping (pass --force to re-run)")

    run_full_evaluation(
        predict_fns, dataset, class_names, result_paths,
        force_rerun=force_rerun, progress_callback=progress_cb,
    )

    print("[evaluation] done. Results saved to:")
    for model_name, path in result_paths.items():
        print(f"  {model_name}: {path}")


if __name__ == "__main__":
    _standalone_main(force_rerun="--force" in sys.argv)