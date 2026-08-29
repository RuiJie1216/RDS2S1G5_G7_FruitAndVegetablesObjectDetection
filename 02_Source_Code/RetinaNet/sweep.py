"""
sweep.py
========
After training finishes, automatically sweeps SCORE_THRESHOLD across a
range of values on the BEST checkpoint, and reports the threshold that
gives the highest F1 -- so you don't have to manually re-run evaluate.py
with different --score values one at a time.

This runs inference ONCE, then filters the same predictions at many
thresholds, so every row in the table is guaranteed to come from the
same forward pass / same model.

Usage:
    python sweep.py --run RUN_NAME
"""

import argparse
import os

import torch

from config import BATCH_SIZE, MODEL_NAME
import run_manager as rm
from dataset_utils import get_dataloaders
from model import build_model
from train import run_full_evaluation, get_device
from metrics_utils import evaluate_predictions, filter_predictions_by_threshold, log_model_results


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep score thresholds on the BEST checkpoint.")
    parser.add_argument("--run", type=str, default=None,
                         help="Run name under runs/. Defaults to the latest completed run.")
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = args.run or rm.latest_completed_run()
    if run_name is None:
        print("[sweep] No completed runs found under runs/.")
        return

    device = get_device()
    model_path = rm.best_model_path(run_name)

    if not os.path.exists(model_path):
        print(f"[sweep] ERROR: No best checkpoint found at {model_path}")
        return

    print(f"[sweep] Run: {run_name}")
    print(f"[sweep] Loading model from {model_path}")

    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(
        f"[sweep] Checkpoint epoch: {checkpoint.get('epoch')}, "
        f"best_f1: {checkpoint.get('best_f1')}, "
        f"saved_because: {checkpoint.get('saved_because')}"
    )

    _, val_loader = get_dataloaders(BATCH_SIZE)

    # Run inference ONCE -- every threshold below re-filters the SAME
    # raw predictions, so results are directly comparable.
    all_gt, all_preds_raw = run_full_evaluation(model, val_loader, device)

    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'cls_acc':>10} {'mean_iou':>10}")

    results = []

    for t in THRESHOLDS:
        preds = filter_predictions_by_threshold(all_preds_raw, t)
        m = evaluate_predictions(all_gt, preds)

        print(
            f"{t:>10} "
            f"{m['objectness_precision']:>10} "
            f"{m['objectness_recall']:>10} "
            f"{m['objectness_f1']:>10} "
            f"{m['classification_accuracy']:>10} "
            f"{m['mean_iou']:>10}"
        )

        results.append((t, m))

    best_threshold, best_metrics = max(results, key=lambda r: r[1]["objectness_f1"])

    print()
    print(f"[sweep] BEST threshold: {best_threshold}")
    print(f"[sweep] BEST F1: {best_metrics['objectness_f1']}")
    print(f"[sweep] At best threshold -- precision: {best_metrics['objectness_precision']}, recall: {best_metrics['objectness_recall']}")

    log_model_results(
        best_metrics,
        model_name=MODEL_NAME,
        extra_info={
            "run_name": run_name,
            "score_threshold": best_threshold,
            "note": "auto-selected via threshold sweep (best F1)",
        },
    )
    print("\n[sweep] Best result logged to results/model_comparison.json")


if __name__ == "__main__":
    main()