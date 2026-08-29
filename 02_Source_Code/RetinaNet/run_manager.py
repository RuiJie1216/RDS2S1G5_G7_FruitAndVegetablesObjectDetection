"""
run_manager.py
==============
Keeps every training run in its own folder under runs/, so multiple runs
never overwrite each other's checkpoints, and lets train.py / evaluate.py /
app.py discover which runs exist and whether they finished.

Folder layout for a run named "run_2026-08-16_10-30-00":

    runs/
        run_2026-08-16_10-30-00/
            retinanet.pt           <- best checkpoint (model + optimizer + epoch)
            training_log.csv       <- one row per completed epoch (crash-safe)
            training_history.json  <- written ONLY when the run finishes
            COMPLETED               <- empty marker, written ONLY on success
"""

import csv
import os
import time

from config import RUNS_DIR


def run_dir(run_name):
    return os.path.join(RUNS_DIR, run_name)


def model_path(run_name):
    return os.path.join(run_dir(run_name), "retinanet.pt")


def best_model_path(run_name):
    """Separate file for the BEST val_loss checkpoint -- retinanet.pt
    always holds the LATEST epoch (for crash-safe resume), which is not
    necessarily the best one once overfitting kicks in after the loss
    stops improving."""
    return os.path.join(run_dir(run_name), "retinanet_best.pt")


def history_path(run_name):
    return os.path.join(run_dir(run_name), "training_history.json")


def log_csv_path(run_name):
    return os.path.join(run_dir(run_name), "training_log.csv")


def completed_marker_path(run_name):
    return os.path.join(run_dir(run_name), "COMPLETED")


def is_completed(run_name):
    return os.path.exists(completed_marker_path(run_name))


def mark_completed(run_name):
    with open(completed_marker_path(run_name), "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


def epochs_completed(run_name):
    path = log_csv_path(run_name)
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0
    return max(0, len(rows) - 1)  # minus header row


def append_log_row(run_name, row_dict):
    """Writes one row to training_log.csv, creating the header on first call."""
    path = log_csv_path(run_name)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def has_checkpoint(run_name):
    return os.path.exists(model_path(run_name))


def list_runs():
    if not os.path.isdir(RUNS_DIR):
        return []

    runs = []
    for name in sorted(os.listdir(RUNS_DIR)):
        path = run_dir(name)
        if not os.path.isdir(path):
            continue
        runs.append({
            "name": name,
            "completed": is_completed(name),
            "epochs_completed": epochs_completed(name),
            "has_checkpoint": has_checkpoint(name),
            "last_modified": os.path.getmtime(path),
        })

    runs.sort(key=lambda r: r["last_modified"], reverse=True)
    return runs


def list_unfinished_runs():
    return [r for r in list_runs() if not r["completed"]]


def list_completed_runs():
    return [r for r in list_runs() if r["completed"]]


def latest_completed_run():
    completed = list_completed_runs()
    return completed[0]["name"] if completed else None


def new_run_name():
    return time.strftime("run_%Y-%m-%d_%H-%M-%S")


def create_run(run_name):
    os.makedirs(run_dir(run_name), exist_ok=True)
    return run_name