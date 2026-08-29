"""
evaluate.py
===========
Stand-alone evaluation of an already-trained run's model on the
validation set. Useful for re-scoring a model (e.g. after changing
SCORE_THRESHOLD in config.py) without retraining.

IMPORTANT: This always loads retinanet_best.pt (the checkpoint with the
lowest validation loss), never retinanet.pt (the latest epoch). This is
intentional -- mixing which checkpoint gets loaded between runs is what
caused inconsistent, non-comparable results in earlier evaluations.

Usage:
    python evaluate.py                        # evaluates the latest completed run
    python evaluate.py --run RUN_NAME          # evaluates a specific run
    python evaluate.py --run RUN_NAME --score 0.3   # override the score threshold
"""

import argparse
import os

import torch

from config import BATCH_SIZE, MODEL_NAME, SCORE_THRESHOLD
import run_manager as rm
from dataset_utils import get_dataloaders
from model import build_model
from train import run_full_evaluation, get_device
from metrics_utils import evaluate_predictions, log_model_results, filter_predictions_by_threshold, deduplicate_cross_class


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RetinaNet run.")
    parser.add_argument("--run", type=str, default=None,
                         help="Run name under runs/ to evaluate. Defaults to the latest completed run.")
    parser.add_argument("--score", type=float, default=SCORE_THRESHOLD,
                         help="Score threshold to filter predictions with before scoring.")
    return parser.parse_args()


def main():
    args = parse_args()

    run_name = args.run or rm.latest_completed_run()
    if run_name is None:
        print("[evaluate] No completed runs found under runs/. Train one first with 'python train.py'.")
        return

    device = get_device()

    # Always use the BEST checkpoint (lowest val_loss). Never silently
    # fall back to the latest-epoch checkpoint -- that fallback is what
    # caused two different model states to get compared as if they were
    # the same model in earlier evaluation runs.
    model_path = rm.best_model_path(run_name)

    if not os.path.exists(model_path):
        print(f"[evaluate] ERROR: No best checkpoint found at {model_path}")
        print(f"[evaluate] This run may not have any completed epoch with a validation-loss improvement yet.")
        return

    print(f"[evaluate] Run: {run_name}")
    print(f"[evaluate] Loading model from {model_path}")
    print(f"[evaluate] Score threshold: {args.score}")

    model = build_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"[evaluate] Checkpoint epoch: {checkpoint.get('epoch')}, best_val_loss: {checkpoint.get('best_val_loss')}")

    _, val_loader = get_dataloaders(BATCH_SIZE)

    all_gt, all_preds = run_full_evaluation(model, val_loader, device)
    all_preds = filter_predictions_by_threshold(all_preds, args.score)
    #all_preds = deduplicate_cross_class(all_preds, iou_threshold=0.6)
    metrics = evaluate_predictions(all_gt, all_preds)

    print("[evaluate] Metrics:")
    for key, value in metrics.items():
        print(f"    {key}: {value}")

    log_model_results(
        metrics,
        model_name=MODEL_NAME,
        extra_info={"run_name": run_name, "score_threshold": args.score},
    )
    print("[evaluate] Metrics logged to results/model_comparison.json")


if __name__ == "__main__":
    main()