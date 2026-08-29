"""Central paths and defaults for the MobileNetV2 full-image detector."""

from __future__ import annotations

import os
from pathlib import Path

from detection_data import read_dataset_classes


BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data" / "LVIS_Fruits_And_Vegetables"
RUNS_DIR = BASE_DIR / "runs"
RESULTS_DIR = BASE_DIR / "results"
COMPARISON_FILE = RESULTS_DIR / 'model_comparison.json'
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_NAME = "MobileNetV2-SSDLite (ImageNet-pretrained full-image detector)"
DEFAULT_IMAGE_SIZE = 320
DEFAULT_BATCH_SIZE = 4
DEFAULT_MODEL = os.environ.get("MOBILENET_DETECTOR_MODEL")


def load_class_names(dataset_root: Path = DATA_ROOT) -> list[str]:
    return read_dataset_classes(dataset_root) if (Path(dataset_root) / "data.yaml").is_file() else []


CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)
