"""Full-image YOLO detection dataset with box-safe augmentation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_dataset_classes(dataset_root: Path) -> list[str]:
    payload = yaml.safe_load((Path(dataset_root) / "data.yaml").read_text(encoding="utf-8"))
    names = payload.get("names")
    if isinstance(names, dict):
        ordered = [str(names[index]) for index in range(len(names))]
    elif isinstance(names, list):
        ordered = [str(name) for name in names]
    else:
        raise ValueError("data.yaml must contain a names mapping or list")
    if len(ordered) != 63:
        raise ValueError(f"Expected 63 source categories, found {len(ordered)}")
    return ordered


def _split_root(dataset_root: Path, kind: str, split: str) -> Path:
    requested = "val" if split == "validation" else split
    root = Path(dataset_root) / kind / requested
    nested = root / requested
    if nested.is_dir():
        root = nested
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {kind}/{requested} split: {root}")
    return root


def split_paths(dataset_root: Path, split: str) -> tuple[Path, Path]:
    if split not in {"train", "val", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return _split_root(dataset_root, "images", split), _split_root(dataset_root, "labels", split)


def dataset_fingerprint(dataset_root: Path, split: str) -> str:
    image_root, label_root = split_paths(dataset_root, split)
    digest = hashlib.sha256()
    for root in (image_root, label_root):
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


class DetectionTransform:
    def __init__(self, training: bool, object_crop_probability: float = 0.0) -> None:
        if not 0.0 <= object_crop_probability <= 1.0:
            raise ValueError("object_crop_probability must be between 0 and 1")
        self.training = training
        self.object_crop_probability = object_crop_probability
        self.color = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)

    def _object_center_crop(self, image: Image.Image, target: dict[str, torch.Tensor]):
        boxes = target["boxes"]
        if len(boxes) == 0:
            return image, target
        width, height = image.size
        scale = random.uniform(0.50, 0.90)
        crop_width = max(2, min(width, round(width * scale)))
        crop_height = max(2, min(height, round(height * scale)))
        selected = boxes[random.randrange(len(boxes))]
        center_x = float((selected[0] + selected[2]) / 2)
        center_y = float((selected[1] + selected[3]) / 2)
        left = round(max(0.0, min(width - crop_width, center_x - random.uniform(0.25, 0.75) * crop_width)))
        top = round(max(0.0, min(height - crop_height, center_y - random.uniform(0.25, 0.75) * crop_height)))
        right, bottom = left + crop_width, top + crop_height

        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        keep = (centers_x >= left) & (centers_x <= right) & (centers_y >= top) & (centers_y <= bottom)
        cropped_boxes = boxes[keep].clone()
        cropped_boxes[:, [0, 2]] = (cropped_boxes[:, [0, 2]] - left).clamp(0, crop_width)
        cropped_boxes[:, [1, 3]] = (cropped_boxes[:, [1, 3]] - top).clamp(0, crop_height)
        valid = (cropped_boxes[:, 2] > cropped_boxes[:, 0]) & (cropped_boxes[:, 3] > cropped_boxes[:, 1])
        keep_indices = torch.where(keep)[0][valid]
        cropped_boxes = cropped_boxes[valid]

        target = dict(target)
        target["boxes"] = cropped_boxes
        target["labels"] = target["labels"][keep_indices]
        target["iscrowd"] = target["iscrowd"][keep_indices]
        target["area"] = (cropped_boxes[:, 2] - cropped_boxes[:, 0]) * (cropped_boxes[:, 3] - cropped_boxes[:, 1])
        return image.crop((left, top, right, bottom)), target

    def __call__(self, image: Image.Image, target: dict[str, torch.Tensor]):
        if self.training and random.random() < self.object_crop_probability:
            image, target = self._object_center_crop(image, target)
        boxes = target["boxes"].clone()
        if self.training and random.random() < 0.5:
            image = F.hflip(image)
            width = image.width
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if self.training:
            image = self.color(image)
        target = dict(target)
        target["boxes"] = boxes
        return F.convert_image_dtype(F.pil_to_tensor(image), torch.float32), target


class YoloDetectionDataset(Dataset):
    """One sample is one complete image and every valid annotation in it."""

    accessed_splits: set[str] = set()

    def __init__(self, dataset_root: Path, split: str, training: bool = False, object_crop_probability: float = 0.0) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.split = "val" if split == "validation" else split
        if training and self.split != "train":
            raise ValueError("Random augmentation is allowed only for the train split")
        self.class_names = read_dataset_classes(self.dataset_root)
        self.image_root, self.label_root = split_paths(self.dataset_root, self.split)
        self.images = sorted(
            path for path in self.image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise RuntimeError(f"No images found under {self.image_root}")
        self.transform = DetectionTransform(training, object_crop_probability)
        self.invalid_box_count = 0
        YoloDetectionDataset.accessed_splits.add(self.split)

    def __len__(self) -> int:
        return len(self.images)

    def _label_path(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.image_root).with_suffix(".txt")
        direct = self.label_root / relative
        if direct.is_file():
            return direct
        matches = list(self.label_root.rglob(relative.name))
        return matches[0] if len(matches) == 1 else direct

    def _target(self, image_path: Path, width: int, height: int, index: int) -> dict[str, torch.Tensor]:
        boxes: list[list[float]] = []
        labels: list[int] = []
        label_path = self._label_path(image_path)
        if label_path.is_file():
            for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                fields = raw.split()
                if len(fields) < 5:
                    self.invalid_box_count += 1
                    continue
                try:
                    class_id = int(float(fields[0])); xc, yc, bw, bh = map(float, fields[1:5])
                except ValueError:
                    self.invalid_box_count += 1
                    continue
                if not 0 <= class_id < len(self.class_names):
                    raise ValueError(f"Class {class_id} outside 0..62 in {label_path}:{line_number}")
                x1 = max(0.0, min(float(width), (xc - bw / 2) * width))
                y1 = max(0.0, min(float(height), (yc - bh / 2) * height))
                x2 = max(0.0, min(float(width), (xc + bw / 2) * width))
                y2 = max(0.0, min(float(height), (yc + bh / 2) * height))
                if x2 <= x1 or y2 <= y1:
                    self.invalid_box_count += 1
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id + 1)  # torchvision SSD reserves 0 for background.
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
        return {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

    def __getitem__(self, index: int):
        image_path = self.images[index]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        target = self._target(image_path, image.width, image.height, index)
        image_tensor, target = self.transform(image, target)
        target["source_path"] = str(image_path)
        target["original_size"] = torch.tensor(image_tensor.shape[-2:], dtype=torch.int64)
        return image_tensor, target


def collate_detection_batch(batch):
    return tuple(zip(*batch))


def save_augmentation_preview(dataset: YoloDetectionDataset, output: Path, count: int = 6) -> None:
    tiles: list[Image.Image] = []
    for index in range(min(count, len(dataset))):
        tensor, target = dataset[index]
        tile = F.to_pil_image(tensor)
        draw = ImageDraw.Draw(tile)
        for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0] + 2, box[1] + 2), dataset.class_names[label - 1], fill="red")
        tile.thumbnail((420, 320))
        tiles.append(tile.copy())
    if not tiles:
        return
    width = max(tile.width for tile in tiles); height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (width * 2, height * ((len(tiles) + 1) // 2)), "white")
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % 2) * width, (i // 2) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def audit_dataset(dataset_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"num_classes": len(read_dataset_classes(dataset_root)), "splits": {}}
    for split in ("train", "val", "test"):
        images, labels = split_paths(dataset_root, split)
        result["splits"][split] = {
            "images": sum(p.suffix.lower() in IMAGE_SUFFIXES for p in images.rglob("*") if p.is_file()),
            "label_files": sum(p.suffix.lower() == ".txt" for p in labels.rglob("*") if p.is_file()),
        }
    return result
