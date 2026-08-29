"""
train.py
========
Fine-tunes the pretrained RetinaNet detector on the LVIS Fruits &
Vegetables dataset.

Every run gets its own folder under runs/, so multiple runs never mix or
overwrite each other's checkpoint.

This version supports FULL training resume.

A checkpoint stores:
    - Model weights
    - Optimizer state
    - AdamW momentum / variance states
    - Scheduler state
    - Current epoch
    - Best validation loss
    - Early-stopping counter
    - Training history
    - Training configuration

If the optimizer state is compatible with the current model/environment,
it will be restored. If it is not compatible, the script will safely
fall back to a fresh optimizer while still restoring the model weights
and training progress.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 4

Start a new run without the interactive menu:
    python train.py --new

Start a new run with a custom name:
    python train.py --new --run-name my_run

Resume a specific run:
    python train.py --resume my_run

Warm-start a new run from another run's best checkpoint (replaces the
old standalone warm_start_train.py -- this is now just a flag on a
normal --new run):
    python train.py --new --run-name model_V9 --warm-start-from model_V8

Warm-start loads model_state_dict from the OTHER run's retinanet_best.pt
(strict=False, since the classification head's class_weights buffer /
any changed submodule names may not exist in the old checkpoint -- the
backbone, FPN, and box regression head still match). The optimizer and
scheduler always start FRESH for a warm-started run (never restored from
the old run), same as any other --new run -- this avoids inheriting an
old learning rate that ReduceLROnPlateau may have already decayed close
to min_lr. Warm-start only applies when starting a NEW run; it has no
effect together with --resume.
"""

import argparse
import json
import os
import time

import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import random
import numpy as np

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    FREEZE_BACKBONE_EPOCHS,
    NUM_CLASSES,
    MODEL_NAME,
    DEVICE,
    SCORE_THRESHOLD,     
    NMS_IOU_THRESHOLD,  
)

import run_manager as rm
from dataset_utils import get_dataloaders
from model import build_model, set_backbone_trainable
from metrics_utils import evaluate_predictions, log_model_results, filter_predictions_by_threshold


# ============================================================
# Checkpoint configuration
# ============================================================

CHECKPOINT_VERSION = 2
EARLY_STOP_PATIENCE = 10  # was 8 -- new anchor config needs more time to converge

# How many of the final epochs get a full F1 evaluation (not just
# val_loss) so the best checkpoint is chosen by detection F1 instead
# of val_loss alone. Running full evaluation every epoch would be too
# slow, so only the tail end -- where the model is actually close to
# converged and a true optimum is likely to appear -- gets evaluated.
F1_EVAL_LAST_N_EPOCHS = 15

# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune RetinaNet on the fruit & vegetable dataset."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Total number of epochs to train."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Training batch size."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Initial learning rate for a new run (used for the HEAD; "
             "the backbone uses 0.1x this value)."
    )

    # --------------------------------------------------------
    # Non-interactive run selection
    # --------------------------------------------------------

    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new run without the interactive menu."
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for the new run. Used together with --new."
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume an existing run without the interactive menu."
    )

    # --------------------------------------------------------
    # Warm start (replaces the old standalone warm_start_train.py)
    # --------------------------------------------------------

    parser.add_argument(
        "--warm-start-from",
        type=str,
        default=None,
        help="Run name to load initial model weights from (its "
             "retinanet_best.pt), instead of ImageNet-pretrained weights. "
             "Only applies when starting a NEW run (--new or a fresh run "
             "picked interactively); ignored with --resume."
    )

    return parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Device
# ============================================================

def get_device():
    if DEVICE == "cpu":
        return torch.device("cpu")

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return device


# ============================================================
# Run selection
# ============================================================

def choose_run(args):
    """
    Decide whether to start a new run or resume an existing run.
    """

    # --------------------------------------------------------
    # Explicit new run
    # --------------------------------------------------------

    if args.new:
        run_name = args.run_name or rm.new_run_name()

        rm.create_run(run_name)

        print(
            f"[train] Starting new run "
            f"'{run_name}' (non-interactive)."
        )

        return run_name, False

    # --------------------------------------------------------
    # Explicit resume
    # --------------------------------------------------------

    if args.resume:
        if not rm.has_checkpoint(args.resume):
            raise SystemExit(
                f"--resume '{args.resume}' has no checkpoint under "
                f"runs/{args.resume}/. "
                f"Check the run name or use --new to start fresh."
            )

        print(
            f"[train] Resuming run "
            f"'{args.resume}' (non-interactive)."
        )

        return args.resume, True

    # --------------------------------------------------------
    # Interactive mode
    # --------------------------------------------------------

    unfinished = rm.list_unfinished_runs()

    print("\n===== RetinaNet Training Runs =====")

    if unfinished:
        print("Unfinished runs found:")

        for i, run in enumerate(unfinished, start=1):
            ckpt = (
                "checkpoint saved"
                if run["has_checkpoint"]
                else "no checkpoint yet"
            )

            print(
                f"  [{i}] {run['name']} - "
                f"{run['epochs_completed']} epoch(s) completed "
                f"({ckpt})"
            )

    else:
        print("No unfinished runs found.")

    new_option = len(unfinished) + 1

    print(
        f"  [{new_option}] Start a NEW model run"
    )

    while True:
        choice = input(
            f"Select a run to continue, or "
            f"{new_option} for a new run: "
        ).strip()

        if not choice.isdigit():
            print("Please enter a number.")
            continue

        choice = int(choice)

        # ----------------------------------------------------
        # New run
        # ----------------------------------------------------

        if choice == new_option:
            run_name = input(
                "Name this run "
                "(leave blank for an auto-generated name): "
            ).strip()

            if not run_name:
                run_name = rm.new_run_name()

            rm.create_run(run_name)

            return run_name, False

        # ----------------------------------------------------
        # Resume
        # ----------------------------------------------------

        if 1 <= choice <= len(unfinished):
            return unfinished[choice - 1]["name"], True

        print("Invalid choice, try again.")


# ============================================================
# Target device helper
# ============================================================

def targets_to_device(targets, device):
    return [
        {
            key: value.to(device)
            for key, value in target.items()
        }
        for target in targets
    ]


# ============================================================
# Training epoch
# ============================================================

def run_one_epoch_train(
    model,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    scaler,
):
    """
    Run one training epoch.
    """

    model.train()

    total_loss = 0.0
    num_batches = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs} [train]",
        unit="batch",
        leave=False,
    )

    for images, targets in progress:

        images = [
            image.to(device)
            for image in images
        ]

        targets = targets_to_device(
            targets,
            device
        )

                # ----------------------------------------------------
        # Forward pass (mixed precision)
        # ----------------------------------------------------

        optimizer.zero_grad()

        with autocast("cuda"):
            loss_dict = model(
                images,
                targets
            )

            loss = sum(
                loss_dict.values()
            )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        scaler.scale(loss).backward()

        # Prevent exploding gradients.
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=10.0,
        )

        scaler.step(optimizer)
        scaler.update()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix(
            loss=f"{total_loss / num_batches:.4f}"
        )

    return total_loss / max(num_batches, 1)


# ============================================================
# Validation loss
# ============================================================

@torch.no_grad()
def run_validation_loss(
    model,
    loader,
    device,
    epoch,
    total_epochs,
):
    """
    RetinaNet only returns losses while the model is in train mode.

    We therefore temporarily keep the model in train mode but disable
    gradients. This calculates validation loss without updating weights.

    IMPORTANT: model.train() also puts BatchNorm layers into training
    mode, which means every validation forward pass would normally
    update their running_mean / running_var using validation-set
    statistics -- silently polluting the backbone's BN stats with data
    it's not supposed to learn from. The loop below explicitly forces
    every BatchNorm layer back into eval mode (frozen running stats)
    right after model.train(), while everything else in the model
    stays in train mode so RetinaNet still returns losses.
    """

    model.train()

    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()

    total_loss = 0.0
    num_batches = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{total_epochs} [val]  ",
        unit="batch",
        leave=False,
    )

    for images, targets in progress:

        images = [
            image.to(device)
            for image in images
        ]

        targets = targets_to_device(
            targets,
            device
        )

        with autocast("cuda"):
            loss_dict = model(
                images,
                targets
            )

            loss = sum(
                loss_dict.values()
            )

        total_loss += loss.item()
        num_batches += 1

        progress.set_postfix(
            loss=f"{total_loss / num_batches:.4f}"
        )

    return total_loss / max(num_batches, 1)


# ============================================================
# Full evaluation
# ============================================================

@torch.no_grad()
def run_full_evaluation(
    model,
    loader,
    device,
):
    """
    Run real RetinaNet inference and collect:
        - ground-truth boxes
        - ground-truth labels
        - predicted boxes
        - predicted labels
        - prediction scores
    """

    model.eval()

    all_gt = []
    all_preds = []

    for images, targets in tqdm(
        loader,
        desc="Evaluating",
        unit="batch",
    ):

        images = [
            image.to(device)
            for image in images
        ]

        outputs = model(images)

        for target, output in zip(
            targets,
            outputs,
        ):

            all_gt.append({
                "boxes": target["boxes"]
                    .cpu()
                    .numpy()
                    .tolist(),

                "labels": target["labels"]
                    .cpu()
                    .numpy()
                    .tolist(),
            })

            all_preds.append({
                "boxes": output["boxes"]
                    .cpu()
                    .numpy()
                    .tolist(),

                "labels": output["labels"]
                    .cpu()
                    .numpy()
                    .tolist(),

                "scores": output["scores"]
                    .cpu()
                    .numpy()
                    .tolist(),
            })

    return all_gt, all_preds


# ============================================================
# Optimizer resume helper
# ============================================================

def restore_optimizer_state(
    optimizer,
    checkpoint,
):
    """
    Restore the complete optimizer state.

    This includes AdamW:
        - exp_avg
        - exp_avg_sq
        - optimizer step
        - parameter-group information

    If the state is incompatible with the current model/environment,
    safely fall back to a fresh optimizer.
    """

    if "optimizer_state_dict" not in checkpoint:
        print(
            "[train] No optimizer state found in checkpoint."
        )

        return False

    try:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print(
            "[train] Optimizer state restored."
        )

        print(
            "[train] AdamW momentum/variance "
            "states restored."
        )

        return True

    except (
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:

        print(
            "[train] WARNING: Could not restore "
            f"optimizer state: {error}"
        )

        print(
            "[train] Model weights and training progress "
            "will still be restored."
        )

        print(
            "[train] Optimizer will start with a "
            "fresh state."
        )

        return False


# ============================================================
# Scheduler resume helper
# ============================================================

def restore_scheduler_state(
    scheduler,
    checkpoint,
):
    """
    Restore ReduceLROnPlateau scheduler state.

    This restores information such as:
        - best metric
        - bad-epoch count
        - cooldown
        - last epoch
    """

    if "scheduler_state_dict" not in checkpoint:
        print(
            "[train] No scheduler state found in checkpoint."
        )

        return False

    try:

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        print(
            "[train] Scheduler state restored."
        )

        return True

    except (
        RuntimeError,
        ValueError,
        KeyError,
    ) as error:

        print(
            "[train] WARNING: Could not restore "
            f"scheduler state: {error}"
        )

        print(
            "[train] Scheduler will start fresh."
        )

        return False


# ============================================================
# Main
# ============================================================

def main():
    
    set_seed(42)

    # ========================================================
    # Parse arguments
    # ========================================================

    args = parse_args()

    # ========================================================
    # Select run
    # ========================================================

    run_name, is_resume = choose_run(args)

    # ========================================================
    # Run paths
    # ========================================================

    run_folder = rm.run_dir(run_name)

    os.makedirs(
        run_folder,
        exist_ok=True,
    )

    model_path = rm.model_path(
        run_name
    )

    history_path = rm.history_path(
        run_name
    )

    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print()
    print(f"[train] Run: {run_name}")
    print(f"[train] Device: {device}")
    print(f"[train] Classes: {NUM_CLASSES}")
    print(
        f"[train] Target epochs: {args.epochs} "
        f"| Batch size: {args.batch_size} "
        f"| Head LR: {args.lr} "
        f"| Backbone LR: {args.lr * 0.1}"
    )

    # ========================================================
    # Data
    # ========================================================

    train_loader, val_loader = get_dataloaders(
        args.batch_size
    )

    # ========================================================
    # Model
    # ========================================================

    warm_starting = (not is_resume) and bool(args.warm_start_from)

    # If warm-starting, skip ImageNet-pretrained weights -- they're about
    # to be overwritten by the warm-start checkpoint's weights anyway.
    model = build_model(
        pretrained=not is_resume and not warm_starting
    )

    if warm_starting:

        old_ckpt_path = rm.best_model_path(args.warm_start_from)

        print()
        print(
            f"[train] Warm-starting from run "
            f"'{args.warm_start_from}': {old_ckpt_path}"
        )

        if not os.path.exists(old_ckpt_path):
            raise SystemExit(
                f"--warm-start-from '{args.warm_start_from}' has no best "
                f"checkpoint at {old_ckpt_path}. Check the run name."
            )

        old_ckpt = torch.load(
            old_ckpt_path,
            map_location=device,
            weights_only=False,
        )

        # strict=False: the classification_head's internal buffer
        # (class_weights) and any changed submodule names won't exist in
        # the old checkpoint -- everything else (backbone, FPN, box
        # regression head, conv layers of classification head) matches.
        missing, unexpected = model.load_state_dict(
            old_ckpt["model_state_dict"], strict=False
        )

        print(
            f"[train] Warm-start weights loaded. "
            f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )

        if missing:
            print(
                f"[train]   missing (expected: new class_weights buffer "
                f"etc.): {missing}"
            )
        if unexpected:
            print(
                f"[train]   unexpected: {unexpected}"
            )

    model.to(device)

    # ========================================================
    # Optimizer
    #
    # Backbone (ResNet50) uses a smaller LR than the head, since it
    # starts from pretrained ImageNet weights and shouldn't be pushed
    # around as aggressively as the randomly-initialized head.
    #
    # optimizer.param_groups[0] -> backbone
    # optimizer.param_groups[1] -> head
    # ========================================================

    backbone_params = [
        p for n, p in model.named_parameters()
        if "backbone" in n
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "backbone" not in n
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    
    scaler = GradScaler("cuda")

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # ========================================================
    # Training state
    # ========================================================

    initial_epoch = 0

    best_val_loss = float("inf")

    best_f1 = None
    
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    patience_counter = 0

    # ========================================================
    # Resume checkpoint
    # ========================================================

    if is_resume and os.path.exists(model_path):

        print()
        print(
            f"[train] Resuming from checkpoint:"
        )

        print(
            f"[train] {model_path}"
        )

        checkpoint = torch.load(
            model_path,
            map_location=device,
        )

        # ----------------------------------------------------
        # Check checkpoint version
        # ----------------------------------------------------

        checkpoint_version = checkpoint.get(
            "checkpoint_version",
            1,
        )

        print(
            f"[train] Checkpoint version: "
            f"{checkpoint_version}"
        )

        # ----------------------------------------------------
        # Restore model
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "[train] Model weights restored."
        )

        # ----------------------------------------------------
        # Restore epoch
        # ----------------------------------------------------

        initial_epoch = checkpoint.get(
            "epoch",
            0,
        )

        # ----------------------------------------------------
        # Restore best validation loss
        # ----------------------------------------------------

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        # ----------------------------------------------------
        # Restore history
        # ----------------------------------------------------

        history = checkpoint.get(
            "history",
            history,
        )

        # ----------------------------------------------------
        # Restore early stopping
        # ----------------------------------------------------

        patience_counter = checkpoint.get(
            "patience_counter",
            0,
        )
        
        best_f1 = checkpoint.get("best_f1", None)

        # ----------------------------------------------------
        # Restore optimizer
        # ----------------------------------------------------

        optimizer_restored = restore_optimizer_state(
            optimizer,
            checkpoint,
        )

        # ----------------------------------------------------
        # Restore scheduler
        # ----------------------------------------------------

        scheduler_restored = restore_scheduler_state(
            scheduler,
            checkpoint,
        )

        # ----------------------------------------------------
        # Restore learning rate(s) if optimizer could not
        # be restored.
        #
        # optimizer now has TWO param groups (backbone, head).
        # If the checkpoint's saved optimizer state also has two
        # groups, restore each group's LR individually. Otherwise
        # (e.g. an older single-group checkpoint), fall back to
        # the saved scalar LR and re-derive the backbone/head split.
        # ----------------------------------------------------

        if not optimizer_restored:

            saved_param_groups = checkpoint.get(
                "optimizer_state_dict", {}
            ).get("param_groups", [])

            if len(saved_param_groups) == len(optimizer.param_groups):

                for parameter_group, saved_group in zip(
                    optimizer.param_groups,
                    saved_param_groups,
                ):
                    parameter_group["lr"] = saved_group["lr"]

                print(
                    "[train] Restored per-group learning rates "
                    f"(backbone={optimizer.param_groups[0]['lr']:.2e}, "
                    f"head={optimizer.param_groups[1]['lr']:.2e})."
                )

            else:

                saved_lr = checkpoint.get(
                    "learning_rate",
                    None,
                )

                if saved_lr is None and saved_param_groups:
                    saved_lr = saved_param_groups[0].get("lr", None)

                if saved_lr is not None:
                    # Treat the saved value as the HEAD lr, and re-derive
                    # backbone lr using the same 0.1x ratio as a fresh run.
                    optimizer.param_groups[0]["lr"] = saved_lr * 0.1
                    optimizer.param_groups[1]["lr"] = saved_lr

                    print(
                        f"[train] Restored learning rate from a "
                        f"single-group checkpoint "
                        f"(head={saved_lr}, backbone={saved_lr * 0.1})."
                    )

        # ----------------------------------------------------
        # Print restored state
        # ----------------------------------------------------

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]

        print()
        print(
            "========== Restored Training State =========="
        )

        print(
            f"[train] Epoch: "
            f"{initial_epoch}"
        )

        print(
            f"[train] Best validation loss: "
            f"{best_val_loss:.6f}"
        )

        print(
            f"[train] Early stopping: "
            f"{patience_counter}/"
            f"{EARLY_STOP_PATIENCE}"
        )

        print(
            f"[train] Backbone LR: {backbone_lr:.2e} "
            f"| Head LR: {head_lr:.2e}"
        )

        print(
            f"[train] Optimizer restored: "
            f"{optimizer_restored}"
        )

        print(
            f"[train] Scheduler restored: "
            f"{scheduler_restored}"
        )

        print(
            "=============================================="
        )

        # ----------------------------------------------------
        # Check whether the requested epoch count has
        # already been reached
        # ----------------------------------------------------

        if initial_epoch >= args.epochs:

            print()

            print(
                f"[train] This run already completed "
                f"{initial_epoch} epochs, which is >= "
                f"--epochs {args.epochs}."
            )

            print(
                f"[train] Increase --epochs to continue, "
                f"for example:"
            )

            print(
                f"        python train.py "
                f"--epochs {initial_epoch + 10}"
            )

            return

    # ========================================================
    # Fresh run
    # ========================================================

    if not is_resume:

        print()

        if warm_starting:
            print(
                f"[train] Starting a new training run, warm-started "
                f"from '{args.warm_start_from}'."
            )
        else:
            print(
                "[train] Starting a completely new training run."
            )

        print(
            "[train] Backbone warmup will be applied."
        )

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(
        initial_epoch,
        args.epochs,
    ):

        # ----------------------------------------------------
        # Backbone warmup
        #
        # IMPORTANT:
        # This is based on the actual epoch rather than
        # is_resume, so it also works correctly after resume.
        # ----------------------------------------------------

        set_backbone_trainable(
            model,
            epoch >= FREEZE_BACKBONE_EPOCHS,
        )

        # ----------------------------------------------------
        # Epoch timer
        # ----------------------------------------------------

        start = time.time()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = run_one_epoch_train(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            args.epochs,
            scaler,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = run_validation_loss(
            model,
            val_loader,
            device,
            epoch,
            args.epochs,
        )

        elapsed = time.time() - start

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            val_loss
        )

        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        # ----------------------------------------------------
        # Print epoch result
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"- {elapsed:.1f}s "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- backbone_lr: {backbone_lr:.2e} "
            f"- head_lr: {head_lr:.2e}"
        )

        # ----------------------------------------------------
        # Log epoch
        # ----------------------------------------------------

        rm.append_log_row(
            run_name,
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
            },
        )

                # ----------------------------------------------------
        # Full F1 evaluation (only for the tail-end epochs)
        #
        # val_loss is cheap but doesn't necessarily track detection
        # F1 -- the checkpoint with the lowest val_loss is not
        # guaranteed to be the one with the best precision/recall
        # trade-off. Only the last F1_EVAL_LAST_N_EPOCHS epochs pay
        # the cost of a full evaluation pass, since the true optimum
        # is expected to be near the end of training, not the start.
        # ----------------------------------------------------

        epochs_remaining = args.epochs - (epoch + 1)
        run_f1_eval = epochs_remaining < F1_EVAL_LAST_N_EPOCHS

        epoch_f1 = None

        if run_f1_eval:

            print(
                f"[train] Epoch {epoch + 1}: running full F1 evaluation "
                f"(within last {F1_EVAL_LAST_N_EPOCHS} epochs)..."
            )

            eval_gt, eval_preds = run_full_evaluation(
                model,
                val_loader,
                device,
            )

            eval_preds = filter_predictions_by_threshold(
                eval_preds,
                SCORE_THRESHOLD,
            )

            eval_metrics = evaluate_predictions(
                eval_gt,
                eval_preds,
            )

            epoch_f1 = eval_metrics["objectness_f1"]

            print(
                f"[train] Epoch {epoch + 1} F1: {epoch_f1:.4f} "
                f"(precision={eval_metrics['objectness_precision']:.4f}, "
                f"recall={eval_metrics['objectness_recall']:.4f})"
            )

            history.setdefault("f1_eval_epoch", []).append(epoch + 1)
            history.setdefault("f1_eval_score", []).append(epoch_f1)

        # ----------------------------------------------------
        # Check improvement
        #
        # If this epoch had a full F1 evaluation, "best" is decided by
        # F1. Otherwise (early epochs, no F1 eval run), fall back to
        # val_loss so early-stopping/resume logic still has a signal
        # to work with throughout the whole run.
        # ----------------------------------------------------

        if run_f1_eval:

            improved = (
                best_f1 is None
                or epoch_f1 > best_f1
            )

            if improved:
                best_f1 = epoch_f1

        else:

            improved = (
                val_loss < best_val_loss
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        # ====================================================
        # Save latest checkpoint
        #
        # This is the IMPORTANT checkpoint for resume.
        # ====================================================

        latest_checkpoint = {
            # ------------------------------------------------
            # Checkpoint format
            # ------------------------------------------------

            "checkpoint_version":
                CHECKPOINT_VERSION,

            # ------------------------------------------------
            # Model
            # ------------------------------------------------

            "model_state_dict":
                model.state_dict(),

            # ------------------------------------------------
            # Optimizer
            #
            # Contains AdamW's:
            #     exp_avg
            #     exp_avg_sq
            #     step
            # Now with TWO param groups (backbone, head).
            # ------------------------------------------------

            "optimizer_state_dict":
                optimizer.state_dict(),

            # ------------------------------------------------
            # Scheduler
            # ------------------------------------------------

            "scheduler_state_dict":
                scheduler.state_dict(),

            # ------------------------------------------------
            # Training progress
            # ------------------------------------------------

            "epoch":
                epoch + 1,

            "best_val_loss":
                best_val_loss,

            "patience_counter":
                patience_counter,
                
            "best_f1":
                best_f1,
                
            # ------------------------------------------------
            # Training history
            # ------------------------------------------------

            "history":
                history,

            # ------------------------------------------------
            # Configuration
            # ------------------------------------------------

            "batch_size":
                args.batch_size,

            "learning_rate":
                head_lr,  # kept for backward-compat with old resume logic

            "backbone_learning_rate":
                backbone_lr,

            "initial_learning_rate":
                args.lr,

            "weight_decay":
                WEIGHT_DECAY,

            "num_classes":
                NUM_CLASSES,

            "model_name":
                MODEL_NAME,

            "freeze_backbone_epochs":
                FREEZE_BACKBONE_EPOCHS,

            "early_stop_patience":
                EARLY_STOP_PATIENCE,

            "warm_started_from":
                args.warm_start_from if warm_starting else None,
        }

        torch.save(
            latest_checkpoint,
            model_path,
        )

        print(
            f"[train] Latest checkpoint saved "
            f"at epoch {epoch + 1}."
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if improved:

            save_reason = "F1" if run_f1_eval else "val_loss"

            best_checkpoint = {
                "checkpoint_version":
                    CHECKPOINT_VERSION,

                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch + 1,

                "best_val_loss":
                    best_val_loss,

                "best_f1":
                    best_f1,

                "saved_because":
                    save_reason,

                "num_classes":
                    NUM_CLASSES,

                "model_name":
                    MODEL_NAME,
            }

            torch.save(
                best_checkpoint,
                rm.best_model_path(run_name),
            )

            if run_f1_eval:
                print(
                    f"Epoch {epoch + 1}: "
                    f"F1 improved to {epoch_f1:.4f}, "
                    f"best checkpoint saved."
                )
            else:
                print(
                    f"Epoch {epoch + 1}: "
                    f"val_loss improved to {val_loss:.4f}, "
                    f"best checkpoint saved."
                )

        else:

            print(
                f"[train] No improvement. "
                f"Early stopping: "
                f"{patience_counter}/"
                f"{EARLY_STOP_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= EARLY_STOP_PATIENCE:

            print(
                f"[train] No improvement for "
                f"{EARLY_STOP_PATIENCE} epochs, "
                f"stopping early."
            )

            break

    # ========================================================
    # Save history
    # ========================================================

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            history,
            file,
            indent=2,
        )

    print(
        f"[train] Training history saved to "
        f"{history_path}"
    )

    # ========================================================
    # Final evaluation
    # ========================================================

    print(
        "[train] Running full evaluation "
        "on the validation set..."
    )

    # --------------------------------------------------------
    # Reload BEST checkpoint
    # --------------------------------------------------------

    best_model_path = rm.best_model_path(
        run_name
    )

    if os.path.exists(
        best_model_path
    ):

        best_checkpoint = torch.load(
            best_model_path,
            map_location=device,
        )

        model.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ]
        )

        print(
            f"[train] Evaluating the BEST checkpoint "
            f"(epoch {best_checkpoint['epoch']}, "
            f"val_loss "
            f"{best_checkpoint['best_val_loss']:.4f}), "
            f"not the last epoch trained."
        )

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    all_gt, all_preds = run_full_evaluation(
        model,
        val_loader,
        device,
    )

    all_preds = filter_predictions_by_threshold(
        all_preds,
        SCORE_THRESHOLD,
    )

    metrics = evaluate_predictions(
        all_gt,
        all_preds,
    )

    # ========================================================
    # Print metrics
    # ========================================================

    print(
        "[train] Validation metrics:"
    )

    for key, value in metrics.items():

        print(
            f"    {key}: {value}"
        )

    # ========================================================
    # Log results
    # ========================================================

    log_model_results(
        metrics,
        model_name=MODEL_NAME,
        extra_info={
            "run_name":
                run_name,

            "epochs_trained":
                rm.epochs_completed(
                    run_name
                ),

            "batch_size":
                args.batch_size,

            "learning_rate":
                args.lr,

            "num_classes":
                NUM_CLASSES,
                
            "score_threshold": 
                SCORE_THRESHOLD,   
                   
            "nms_iou_threshold": 
                NMS_IOU_THRESHOLD,  
                
        },
    )

    # ========================================================
    # Mark run completed
    # ========================================================

    rm.mark_completed(
        run_name
    )

    print(
        f"[train] Run '{run_name}' "
        f"marked as COMPLETED."
    )

    print(
        "[train] Metrics logged to "
        "results/model_comparison.json"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()