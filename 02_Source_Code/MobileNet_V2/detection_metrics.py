"""Class-aware detection matching, AP calculation, artifacts, and timing."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou


def _cpu_record(output: dict[str, torch.Tensor], target: dict[str, Any], confidence: float) -> dict[str, Any]:
    keep = output["scores"].detach().cpu() >= confidence
    return {
        "image_id": int(target["image_id"]),
        "source_path": str(target.get("source_path", "")),
        "pred_boxes": output["boxes"].detach().cpu()[keep],
        "pred_labels": output["labels"].detach().cpu()[keep],
        "pred_scores": output["scores"].detach().cpu()[keep],
        "gt_boxes": target["boxes"].detach().cpu(),
        "gt_labels": target["labels"].detach().cpu(),
    }


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device, score_threshold: float, max_images: int = 0):
    """Collect low-score predictions without permanently changing inference defaults."""
    model.eval(); records = []
    previous_score_threshold = float(model.score_thresh)
    model.score_thresh = float(score_threshold)
    try:
        for images, targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            records.extend(_cpu_record(output, target, score_threshold) for output, target in zip(outputs, targets))
            if max_images and len(records) >= max_images:
                return records[:max_images]
    finally:
        model.score_thresh = previous_score_threshold
    return records


def _filter_predictions(record: dict[str, Any], confidence_threshold: float) -> dict[str, Any]:
    keep = record["pred_scores"] >= confidence_threshold
    return {
        **record,
        "pred_boxes": record["pred_boxes"][keep],
        "pred_labels": record["pred_labels"][keep],
        "pred_scores": record["pred_scores"][keep],
    }


def _match(record: dict[str, Any], threshold: float, class_aware: bool = True):
    order = torch.argsort(record["pred_scores"], descending=True)
    gt_used: set[int] = set(); matches = []; false_positive = []
    ious = box_iou(record["pred_boxes"], record["gt_boxes"]) if len(record["pred_boxes"]) and len(record["gt_boxes"]) else torch.zeros((len(record["pred_boxes"]), len(record["gt_boxes"])))
    for pred_index in order.tolist():
        candidates = []
        for gt_index in range(len(record["gt_boxes"])):
            if gt_index in gt_used:
                continue
            if class_aware and int(record["pred_labels"][pred_index]) != int(record["gt_labels"][gt_index]):
                continue
            candidates.append((float(ious[pred_index, gt_index]), gt_index))
        best_iou, best_gt = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            gt_used.add(best_gt); matches.append((pred_index, best_gt, best_iou))
        else:
            false_positive.append(pred_index)
    false_negative = [i for i in range(len(record["gt_boxes"])) if i not in gt_used]
    return matches, false_positive, false_negative


def _ap_for_class(records, class_label: int, threshold: float) -> tuple[float | None, list[float], list[float]]:
    gt_by_image = {r["image_id"]: torch.where(r["gt_labels"] == class_label)[0].tolist() for r in records}
    total_gt = sum(len(indices) for indices in gt_by_image.values())
    if total_gt == 0:
        return None, [], []
    predictions = []
    for record in records:
        for index in torch.where(record["pred_labels"] == class_label)[0].tolist():
            predictions.append((float(record["pred_scores"][index]), record, index))
    predictions.sort(key=lambda row: row[0], reverse=True)
    used: dict[int, set[int]] = {record["image_id"]: set() for record in records}
    tp = []; fp = []
    for _, record, pred_index in predictions:
        image_id = record["image_id"]; candidates = gt_by_image[image_id]
        if candidates:
            ious = box_iou(record["pred_boxes"][pred_index:pred_index + 1], record["gt_boxes"][candidates])[0]
            order = torch.argsort(ious, descending=True)
            match = next((candidates[int(i)] for i in order if candidates[int(i)] not in used[image_id] and float(ious[int(i)]) >= threshold), None)
        else:
            match = None
        tp.append(1.0 if match is not None else 0.0); fp.append(0.0 if match is not None else 1.0)
        if match is not None: used[image_id].add(match)
    if not predictions:
        return 0.0, [], []
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt; precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = float(np.mean([np.max(precision[recall >= point], initial=0.0) for point in recall_points]))
    return ap, precision.tolist(), recall.tolist()


def compute_assignment_metrics(records, num_classes: int = 63, iou_threshold: float = 0.5, confidence_threshold: float = 0.25):
    assignment_records = [_filter_predictions(record, confidence_threshold) for record in records]
    tp = fp = fn = 0; matched_ious = []
    classification_correct = classification_total = 0
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for record in assignment_records:
        matches, false_pos, false_neg = _match(record, iou_threshold, class_aware=True)
        tp += len(matches); fp += len(false_pos); fn += len(false_neg)
        matched_ious.extend(match[2] for match in matches)
        object_matches, object_fp, object_fn = _match(record, iou_threshold, class_aware=False)
        for pred, gt, _ in object_matches:
            classification_total += 1
            if int(record["pred_labels"][pred]) == int(record["gt_labels"][gt]):
                classification_correct += 1
            confusion[int(record["gt_labels"][gt]) - 1, int(record["pred_labels"][pred]) - 1] += 1
        for pred in object_fp: confusion[num_classes, int(record["pred_labels"][pred]) - 1] += 1
        for gt in object_fn: confusion[int(record["gt_labels"][gt]) - 1, num_classes] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    classification_accuracy = classification_correct / classification_total if classification_total else 0.0
    detection_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "detection_precision": precision, "detection_recall": recall, "detection_f1": detection_f1,
        "classification_accuracy": classification_accuracy,
        "classification_correct": classification_correct,
        "classification_total_matched": classification_total,
        "objectness_precision": precision, "objectness_recall": recall, "objectness_f1": detection_f1,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "detection_rate_at_iou_0.5": recall, "true_positives": tp,
        "false_positives": fp, "false_negatives": fn,
        "num_ground_truth_objects": sum(len(r["gt_boxes"]) for r in records),
        "num_predictions": sum(len(r["pred_boxes"]) for r in assignment_records),
        "num_evaluated_images": len(records),
        "assignment_confidence_threshold": confidence_threshold,
    }
    return metrics, confusion


def compute_detection_metrics(records, num_classes: int = 63, iou_threshold: float = 0.5, confidence_threshold: float = 0.25):
    metrics, confusion = compute_assignment_metrics(records, num_classes, iou_threshold, confidence_threshold)
    assignment_records = [_filter_predictions(record, confidence_threshold) for record in records]
    thresholds = [round(0.50 + index * 0.05, 2) for index in range(10)]
    per_class = []; all_aps = []
    for label in range(1, num_classes + 1):
        aps = []; pr50 = ([], [])
        for threshold in thresholds:
            ap, p_curve, r_curve = _ap_for_class(records, label, threshold)
            if ap is not None: aps.append(ap)
            if threshold == 0.5: pr50 = (p_curve, r_curve)
        ap50, _, _ = _ap_for_class(records, label, 0.5)
        gt_count = sum(int((r["gt_labels"] == label).sum()) for r in records)
        pred_count = sum(int((r["pred_labels"] == label).sum()) for r in assignment_records)
        class_tp = class_fp = class_fn = 0
        for record in assignment_records:
            m, f_p, f_n = _match({**record, "pred_boxes": record["pred_boxes"][record["pred_labels"] == label], "pred_labels": record["pred_labels"][record["pred_labels"] == label], "pred_scores": record["pred_scores"][record["pred_labels"] == label], "gt_boxes": record["gt_boxes"][record["gt_labels"] == label], "gt_labels": record["gt_labels"][record["gt_labels"] == label]}, iou_threshold, True)
            class_tp += len(m); class_fp += len(f_p); class_fn += len(f_n)
        cp = class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0
        cr = class_tp / (class_tp + class_fn) if class_tp + class_fn else 0.0
        per_class.append({"class_id": label - 1, "ground_truth": gt_count, "predictions": pred_count, "precision": cp, "recall": cr, "f1": 2 * cp * cr / (cp + cr) if cp + cr else 0.0, "ap_50": ap50, "map_50_95": float(np.mean(aps)) if aps else None, "precision_curve_50": pr50[0], "recall_curve_50": pr50[1]})
        if aps: all_aps.append(aps)
    map_by_threshold = []
    for index in range(len(thresholds)):
        values = [aps[index] for aps in all_aps if len(aps) == len(thresholds)]
        map_by_threshold.append(float(np.mean(values)) if values else 0.0)
    metrics.update({"map_50": map_by_threshold[0], "map_50_95": float(np.mean(map_by_threshold)),
        "map_by_iou": dict(zip(map(str, thresholds), map_by_threshold))})
    return metrics, per_class, confusion


def serializable_predictions(records, class_names: list[str], confidence_threshold: float = 0.25):
    result = []
    for record in records:
        filtered = _filter_predictions(record, confidence_threshold)
        detections = [{"class_id": int(label) - 1, "class_name": class_names[int(label) - 1], "confidence": float(score), "bbox": [float(v) for v in box]} for box, label, score in zip(filtered["pred_boxes"].tolist(), filtered["pred_labels"].tolist(), filtered["pred_scores"].tolist())]
        result.append({"image_id": record["image_id"], "source_path": record["source_path"], "detections": detections})
    return result


def save_evaluation_artifacts(output_dir: Path, split: str, records, metrics, per_class, confusion, class_names, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"model_name": "MobileNetV2-SSDLite", "metrics": metrics, "info": metadata}
    (output_dir / f"{split}_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    confidence_threshold = float(metadata.get("confidence_threshold", 0.25))
    (output_dir / "predictions.json").write_text(json.dumps(serializable_predictions(records, class_names, confidence_threshold), indent=2), encoding="utf-8")
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["class_id", "class_name", "ground_truth", "predictions", "precision", "recall", "f1", "ap_50", "map_50_95"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in per_class: writer.writerow({key: (class_names[row["class_id"]] if key == "class_name" else row.get(key)) for key in fields})
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    pr = [{"class_id": row["class_id"], "class_name": class_names[row["class_id"]], "precision": row.pop("precision_curve_50"), "recall": row.pop("recall_curve_50")} for row in per_class]
    (output_dir / "precision_recall.json").write_text(json.dumps(pr), encoding="utf-8")
    samples = output_dir / "sample_predictions"; samples.mkdir(exist_ok=True)
    for record in records[:8]:
        if not record["source_path"] or not Path(record["source_path"]).is_file(): continue
        image = Image.open(record["source_path"]).convert("RGB"); draw = ImageDraw.Draw(image)
        filtered = _filter_predictions(record, confidence_threshold)
        for box, label, score in zip(filtered["pred_boxes"].tolist(), filtered["pred_labels"].tolist(), filtered["pred_scores"].tolist()):
            draw.rectangle(box, outline="lime", width=3); draw.text((box[0], box[1]), f"{class_names[label-1]} {score:.2f}", fill="lime")
        image.save(samples / Path(record["source_path"]).name)
    return payload


@torch.no_grad()
def inference_efficiency(model, image: torch.Tensor, device: torch.device, warmup: int = 3, iterations: int = 10):
    model.eval(); sample = [image.to(device)]
    for _ in range(warmup): model(sample)
    if device.type == "cuda": torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations): model(sample)
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / iterations
    return {"average_inference_seconds": elapsed, "fps": 1.0 / elapsed if elapsed else None, "timing_device": str(device), "timing_batch_size": 1}
