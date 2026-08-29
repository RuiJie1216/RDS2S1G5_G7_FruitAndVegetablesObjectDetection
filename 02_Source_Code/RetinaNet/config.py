"""
config.py
=========
Central configuration for the Fruit & Vegetable RetinaNet Detector project.

This mirrors the folder layout / conventions of the earlier custom-CNN
project so the SAME downloaded dataset (data/LVIS_Fruits_And_Vegetables/)
and the SAME classes.txt can be reused directly -- just copy those two
into this project folder, or point DATA_ROOT / CLASSES_FILE at the old
project's copies.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Dataset paths (same layout as the earlier CNN project)
# ---------------------------------------------------------------------------
DATA_ROOT = os.path.join(BASE_DIR, "data", "LVIS_Fruits_And_Vegetables")

IMAGES_ROOT = os.path.join(DATA_ROOT, "images")
LABELS_ROOT = os.path.join(DATA_ROOT, "labels")


def _resolve_split_dir(root, split_name):
    """Some exports of this dataset nest an extra split-name folder one
    level deeper (images/train/train/*.jpg instead of images/train/*.jpg).
    Detect and handle both layouts."""
    nested = os.path.join(root, split_name, split_name)
    if os.path.isdir(nested):
        return nested
    return os.path.join(root, split_name)


TRAIN_IMAGES_DIR = _resolve_split_dir(IMAGES_ROOT, "train")
TRAIN_LABELS_DIR = _resolve_split_dir(LABELS_ROOT, "train")

VAL_IMAGES_DIR = _resolve_split_dir(IMAGES_ROOT, "val")
VAL_LABELS_DIR = _resolve_split_dir(LABELS_ROOT, "val")

TEST_IMAGES_DIR = _resolve_split_dir(IMAGES_ROOT, "test")
TEST_LABELS_DIR = _resolve_split_dir(LABELS_ROOT, "test")

# ---------------------------------------------------------------------------
# Class names (reuse the same classes.txt from the CNN project -- copy it
# into this project's root, or generate_classes.py again from data.yaml)
# ---------------------------------------------------------------------------
CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")


def _load_class_names():
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    return [f"class_{i}" for i in range(63)]


CLASS_NAMES = _load_class_names()
NUM_CLASSES = len(CLASS_NAMES)  # RetinaNet in torchvision does NOT need a
                                 # separate "background" class (it uses
                                 # per-class sigmoid, not softmax), so this
                                 # is the true number of foreground classes.

# ---------------------------------------------------------------------------
# Model / training hyper-parameters
# ---------------------------------------------------------------------------
# RetinaNet's own internal transform resizes images (keeping aspect ratio,
# with padding) -- these are MIN/MAX target sizes, not a hard square resize
# like the old CNN project. 480-640 is a reasonable range for this dataset;
# lower it if you're training on CPU / low VRAM.
MIN_SIZE = 480
MAX_SIZE = 640

BATCH_SIZE = 4          # RetinaNet + ResNet50 is much heavier than the old
                         # custom CNN -- keep this small, especially on CPU
                         # or a laptop GPU. Raise it if you have >=8GB VRAM.
EPOCHS = 40
LEARNING_RATE = 1e-4     # fine-tuning a pretrained model needs a much
                         # smaller LR than training from scratch (which
                         # used 1e-3)
WEIGHT_DECAY = 5e-4

# Only unfreeze the last N backbone layers for the first few epochs, then
# unfreeze everything -- classic transfer-learning warmup so the
# pretrained ResNet50 features aren't destroyed by early large gradients.
FREEZE_BACKBONE_EPOCHS = 5

# ---------------------------------------------------------------------------
# Inference / evaluation thresholds
# ---------------------------------------------------------------------------
SCORE_THRESHOLD = 0.35     # min confidence to keep a detection at inference
NMS_IOU_THRESHOLD = 0.3  # passed straight into torchvision's RetinaNet
                           # (it runs NMS internally, per class)
EVAL_IOU_THRESHOLD = 0.5  # IoU threshold used when scoring metrics

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_ROOT = os.environ.get("FRUIT_VEG_OUTPUT_ROOT", BASE_DIR)

RUNS_DIR = os.path.join(OUTPUT_ROOT, "runs")
RESULTS_DIR = os.path.join(OUTPUT_ROOT, "results")
COMPARISON_FILE = os.path.join(RESULTS_DIR, "model_comparison.json")

STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(OUTPUT_ROOT, "uploads")

MODEL_NAME = "RetinaNet (ResNet50-FPN, ImageNet-pretrained, fine-tuned)"

# Device selection helper (used by train.py / inference.py)
DEVICE = os.environ.get("FRUIT_VEG_DEVICE", "auto")  # "auto" | "cpu" | "cuda"
