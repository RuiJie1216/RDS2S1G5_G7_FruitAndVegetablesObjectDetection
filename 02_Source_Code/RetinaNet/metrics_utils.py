"""
metrics_utils.py
=================
Evaluation metrics for the RetinaNet detector, kept in the SAME schema
as the earlier custom-CNN project (objectness_accuracy/precision/recall/
f1, classification_accuracy, mean_iou, detection_rate_at_iou_0.5) so both
models' results can sit side by side in the same model_comparison.json /
UI comparison table.

Because RetinaNet outputs real, variable-count, already-NMS'd detections
(not one-box-per-grid-cell), matching against ground truth is done with
greedy IoU matching per image instead of the old cell-by-cell comparison:

    1. Sort this image's predictions by confidence, descending.
    2. For each prediction, find the highest-IoU unmatched GT box.
    3. If that IoU >= EVAL_IOU_THRESHOLD: it's a match (a "true positive"
       for the objectness accuracy/precision/recall numbers), regardless
       of whether the predicted CLASS is correct -- classification
       correctness is then measured separately, exactly as the old
       schema's methodology does (see the old metrics_utils.py docstring).
    4. Unmatched predictions -> false positives. Unmatched GT -> false
       negatives.

This is intentionally simpler than COCO mAP (no per-class AP curve, no
confidence-ranked average precision), but stays consistent + comparable
with the CNN project's existing metric definitions.
"""

import json
import os
import time
import torch

import numpy as np

from config import CLASS_NAMES, EVAL_IOU_THRESHOLD, COMPARISON_FILE, RESULTS_DIR, MODEL_NAME
from torchvision.ops import nms as _torchvision_nms


def deduplicate_cross_class(all_preds, iou_threshold=0.6):
    """
    Remove near-duplicate detections where the SAME location was given
    DIFFERENT class labels (e.g. "apple 0.45" and "tomato 0.42" on
    almost the same box) -- these are genuine duplicates from a
    not-fully-confident classification head.

    Same-class overlapping boxes are intentionally left untouched:
    torchvision's RetinaNet already runs its own per-class NMS
    internally (config.NMS_IOU_THRESHOLD), so any same-class boxes
    that survive to the final output already represent what the model
    believes are separate object instances (e.g. several apples in a
    pile touching each other) -- suppressing them again here would
    wrongly delete real detections and tank recall.
    """
    deduped = []

    for pred in all_preds:
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        n = len(boxes)
        if n == 0:
            deduped.append(pred)
            continue

        order = sorted(range(n), key=lambda i: -scores[i])

        kept_indices = []
        kept_boxes = []
        kept_labels = []

        for i in order:
            box_i = boxes[i]
            label_i = labels[i]

            suppressed = False
            for kb, kl in zip(kept_boxes, kept_labels):
                if kl == label_i:
                    # Same class -- the model's own per-class NMS
                    # already handled this, leave it alone.
                    continue
                if iou_xyxy(box_i, kb) >= iou_threshold:
                    suppressed = True
                    break

            if not suppressed:
                kept_indices.append(i)
                kept_boxes.append(box_i)
                kept_labels.append(label_i)

        deduped.append({
            "boxes": [boxes[i] for i in kept_indices],
            "labels": [labels[i] for i in kept_indices],
            "scores": [scores[i] for i in kept_indices],
        })

    return deduped

def iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _match_one_image(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                      iou_threshold=EVAL_IOU_THRESHOLD):
    """
    Greedy IoU matching for one image. Returns per-image counts and the
    list of (iou, class_match) for matched pairs.
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    matched_gt = set()
    tp = fp = 0
    matches = []  # (iou, class_match) for every matched pair

    order = np.argsort(-np.array(pred_scores)) if num_pred > 0 else []

    for i in order:
        best_iou = 0.0
        best_j = -1
        for j in range(num_gt):
            if j in matched_gt:
                continue
            iou = iou_xyxy(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_j)
            tp += 1
            class_match = int(pred_labels[i]) == int(gt_labels[best_j])
            matches.append((best_iou, class_match))
        else:
            fp += 1

    fn = num_gt - len(matched_gt)
    return tp, fp, fn, matches

def filter_predictions_by_threshold(all_preds, score_threshold):
    filtered = []

    for pred in all_preds:
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        keep = [i for i, s in enumerate(scores) if s >= score_threshold]

        filtered.append({
            "boxes": [boxes[i] for i in keep],
            "labels": [labels[i] for i in keep],
            "scores": [scores[i] for i in keep],
        })

    return filtered

def evaluate_predictions(all_gt, all_preds, iou_threshold=EVAL_IOU_THRESHOLD):
    """
    all_gt:    list of {"boxes": [[x1,y1,x2,y2], ...], "labels": [int, ...]}, one per image
    all_preds: list of {"boxes": [...], "labels": [...], "scores": [...]}, one per image
               (already score-thresholded + NMS'd, e.g. straight from
               model.eval() forward pass)
    """
    tp = fp = fn = 0
    total_gt_objects = 0
    all_matches = []  # (iou, class_match) across every image

    for gt, pred in zip(all_gt, all_preds):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        total_gt_objects += len(gt_boxes)

        img_tp, img_fp, img_fn, img_matches = _match_one_image(
            gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_threshold
        )
        tp += img_tp
        fp += img_fp
        fn += img_fn
        all_matches.extend(img_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = precision  # no true "background cell" concept here, so
                           # objectness_accuracy here is reported as
                           # precision-equivalent (see README for caveat)

    class_correct = sum(1 for _, class_match in all_matches if class_match)
    class_total = len(all_matches)
    class_accuracy = class_correct / class_total if class_total > 0 else 0.0

    iou_values = [iou for iou, _ in all_matches]
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0

    detected_at_iou = class_correct  # matched AND correct class
    detection_rate = detected_at_iou / total_gt_objects if total_gt_objects > 0 else 0.0

    return {
        "objectness_accuracy": round(accuracy, 4),
        "objectness_precision": round(precision, 4),
        "objectness_recall": round(recall, 4),
        "objectness_f1": round(f1, 4),
        "classification_accuracy": round(class_accuracy, 4),
        "mean_iou": round(mean_iou, 4),
        f"detection_rate_at_iou_{iou_threshold}": round(detection_rate, 4),
        "num_ground_truth_objects": int(total_gt_objects),
    }


# ---------------------------------------------------------------------------
# Results logging (shared file/schema with the CNN project's comparison table)
# ---------------------------------------------------------------------------

def log_model_results(metrics, model_name=MODEL_NAME, extra_info=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    entry = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }
    if extra_info:
        entry["info"] = extra_info

    history = []
    if os.path.exists(COMPARISON_FILE):
        with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history = [h for h in history if h.get("model_name") != model_name]
    history.append(entry)

    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return entry


def load_model_results():
    if not os.path.exists(COMPARISON_FILE):
        return []
    with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []
