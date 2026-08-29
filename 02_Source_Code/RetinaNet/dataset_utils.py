"""
dataset_utils.py
================
Dataset loading and augmentation for the RetinaNet detector.

Torchvision's RetinaNet performs its own internal resizing through
GeneralizedRCNNTransform. Therefore, images are loaded at their original
resolution and YOLO-format normalized boxes are converted into absolute
pixel coordinates before being passed to the model.

CHANGES (model_V6):
    - Added _small_object_zoom_crop(): a box-aware crop that zooms in
      tightly around a small object instead of cropping a random region,
      so small objects retain far more pixels after the model's internal
      resize.
    - Added per-image sample weighting + WeightedRandomSampler so images
      containing small objects are oversampled during training, without
      discarding any ground-truth boxes.
"""

import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter


_COLOR_JITTER = ColorJitter(
    brightness=0.3,
    contrast=0.3,
    saturation=0.3,
    hue=0.05,
)

# Roughly the dataset median box size (px) from earlier analysis.
# Boxes at/below this are treated as "small" for augmentation purposes.
SMALL_OBJECT_SIZE_THRESHOLD = 25.0

# Normalized-area proxy threshold used at __init__ time (before we know
# pixel dimensions), for oversampling weight computation only.
SMALL_OBJECT_NORMALIZED_THRESHOLD = 0.05


def _read_annotation(annotation_path):
    """
    Read a YOLO-format annotation file.

    Each line contains:
        class_id x_center y_center width height

    All coordinates are normalized to the range [0, 1].

    Returns:
        List of tuples:
        (class_id, x_center, y_center, width, height)
    """
    boxes = []

    if not os.path.exists(annotation_path):
        return boxes

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id, x, y, w, h = parts

            boxes.append(
                (
                    int(class_id),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                )
            )

    return boxes


def _list_pairs(images_dir, labels_dir):
    """
    Match image files with their corresponding YOLO annotation files.
    """
    extensions = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG",
    )

    image_paths = []

    for ext in extensions:
        image_paths.extend(
            glob.glob(os.path.join(images_dir, ext))
        )

    image_paths = sorted(set(image_paths))

    pairs = []

    for image_path in image_paths:
        stem = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        annotation_path = os.path.join(
            labels_dir,
            stem + ".txt",
        )

        pairs.append(
            (
                image_path,
                annotation_path,
            )
        )

    return pairs


def _random_resized_crop(
    image,
    boxes,
    labels,
    scale_range=(0.7, 1.0),
    visibility_threshold=0.2,
    min_visible_size=6.0,
    max_attempts=5,
):
    """
    Randomly crop the image and keep the resulting crop in its original
    pixel scale.

    Bounding boxes are translated and clipped to the crop boundaries.

    A partially visible object is kept when at least
    `visibility_threshold` of its original area remains visible.

    If no valid objects remain after a crop, another crop is attempted.

    Returns:
        cropped_image, cropped_boxes, cropped_labels
    """
    width, height = image.size
    image_area = width * height

    for _ in range(max_attempts):
        target_area = (
            random.uniform(*scale_range)
            * image_area
        )

        aspect_ratio = random.uniform(
            0.9,
            1.1,
        )

        crop_width = int(
            round(
                (target_area * aspect_ratio) ** 0.5
            )
        )

        crop_height = int(
            round(
                (target_area / aspect_ratio) ** 0.5
            )
        )

        if crop_width >= width or crop_height >= height:
            continue

        x0 = random.randint(
            0,
            width - crop_width,
        )

        y0 = random.randint(
            0,
            height - crop_height,
        )

        x1 = x0 + crop_width
        y1 = y0 + crop_height

        new_boxes = []
        new_labels = []

        for box, label in zip(boxes, labels):
            bx1, by1, bx2, by2 = box

            original_width = bx2 - bx1
            original_height = by2 - by1

            original_area = (
                original_width
                * original_height
            )

            if original_area <= 0:
                continue

            clipped_x1 = max(
                bx1,
                x0,
            )

            clipped_y1 = max(
                by1,
                y0,
            )

            clipped_x2 = min(
                bx2,
                x1,
            )

            clipped_y2 = min(
                by2,
                y1,
            )

            if (
                clipped_x2 <= clipped_x1
                or clipped_y2 <= clipped_y1
            ):
                continue

            visible_width = (
                clipped_x2
                - clipped_x1
            )

            visible_height = (
                clipped_y2
                - clipped_y1
            )

            visible_area = (
                visible_width
                * visible_height
            )

            visibility = (
                visible_area
                / max(original_area, 1e-6)
            )

            visible_scale = (visible_width * visible_height) ** 0.5

            if visibility < visibility_threshold or visible_scale < min_visible_size:
                continue

            new_boxes.append(
                [
                    clipped_x1 - x0,
                    clipped_y1 - y0,
                    clipped_x2 - x0,
                    clipped_y2 - y0,
                ]
            )

            new_labels.append(label)

        if len(new_boxes) == 0:
            continue

        cropped_image = image.crop(
            (
                x0,
                y0,
                x1,
                y1,
            )
        )

        return (
            cropped_image,
            new_boxes,
            new_labels,
        )

    return image, boxes, labels


def _small_object_zoom_crop(
    image,
    boxes,
    labels,
    padding_factor=3.0,
    min_crop_frac=0.30,
    visibility_threshold=0.3,
):
    """
    Pick a random SMALL box in this image and crop tightly around it
    (with some padding), so that after the model's internal resize, the
    small object occupies far more pixels than it would in the full
    original image.

    Unlike _random_resized_crop (which crops a random region regardless
    of object size), this deliberately targets small objects to give
    them more effective resolution during training.

    Falls back to _random_resized_crop if there is no small object to
    zoom into.
    """
    width, height = image.size

    sizes = []
    for box in boxes:
        bx1, by1, bx2, by2 = box
        size = ((bx2 - bx1) * (by2 - by1)) ** 0.5
        sizes.append(size)

    small_indices = [
        i for i, s in enumerate(sizes)
        if s <= SMALL_OBJECT_SIZE_THRESHOLD
    ]

    if not small_indices:
        return _random_resized_crop(image, boxes, labels)

    target_idx = random.choice(small_indices)
    bx1, by1, bx2, by2 = boxes[target_idx]
    box_w = bx2 - bx1
    box_h = by2 - by1
    center_x = (bx1 + bx2) / 2
    center_y = (by1 + by2) / 2

    crop_size = max(box_w, box_h) * padding_factor
    crop_size = max(crop_size, min(width, height) * min_crop_frac)
    crop_size = min(crop_size, min(width, height))  # never exceed the image

    x0 = int(max(0, min(center_x - crop_size / 2, width - crop_size)))
    y0 = int(max(0, min(center_y - crop_size / 2, height - crop_size)))
    x1 = int(min(width, x0 + crop_size))
    y1 = int(min(height, y0 + crop_size))

    new_boxes = []
    new_labels = []

    for box, label in zip(boxes, labels):
        obx1, oby1, obx2, oby2 = box
        original_area = max(0.0, obx2 - obx1) * max(0.0, oby2 - oby1)
        if original_area <= 0:
            continue

        clipped_x1 = max(obx1, x0)
        clipped_y1 = max(oby1, y0)
        clipped_x2 = min(obx2, x1)
        clipped_y2 = min(oby2, y1)

        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue

        visible_area = (clipped_x2 - clipped_x1) * (clipped_y2 - clipped_y1)
        visibility = visible_area / max(original_area, 1e-6)

        if visibility < visibility_threshold:
            continue

        new_boxes.append([
            clipped_x1 - x0,
            clipped_y1 - y0,
            clipped_x2 - x0,
            clipped_y2 - y0,
        ])
        new_labels.append(label)

    if len(new_boxes) == 0:
        return image, boxes, labels

    cropped_image = image.crop((x0, y0, x1, y1))
    return cropped_image, new_boxes, new_labels


class FruitVegDetectionDataset(Dataset):
    """
    Dataset for RetinaNet object detection.

    Images remain at their original resolution. RetinaNet performs the
    final resizing internally.

    Training augmentation includes:
        - Small-object-aware zoom crop (when the image has a small
          object) OR random resized crop (otherwise)
        - Horizontal flip
        - Color jitter

    Validation data is returned without augmentation.
    """

    def __init__(
        self,
        images_dir,
        labels_dir,
        augment=False,
    ):
        self.pairs = _list_pairs(
            images_dir,
            labels_dir,
        )

        if len(self.pairs) == 0:
            raise FileNotFoundError(
                f"No images found in {images_dir}. "
                "Check the dataset directory."
            )

        self.augment = augment

        # Precompute a sample weight per image: images containing at
        # least one small object get oversampled during training. This
        # only reads the (cheap) label text files, not the images.
        self.sample_weights = self._compute_sample_weights()

    def _compute_sample_weights(self):
        weights = []

        for image_path, annotation_path in self.pairs:
            raw_boxes = _read_annotation(annotation_path)

            has_small = False
            for class_id, x, y, bw, bh in raw_boxes:
                # bw/bh are normalized (0-1); we don't have pixel
                # width/height here without opening the image, so we
                # use normalized box area as a proxy for "small".
                if (bw * bh) ** 0.5 < SMALL_OBJECT_NORMALIZED_THRESHOLD:
                    has_small = True
                    break

            weights.append(2.0 if has_small else 1.0)

        return weights

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, annotation_path = self.pairs[idx]

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        raw_boxes = _read_annotation(
            annotation_path
        )

        boxes = []
        labels = []

        for (
            class_id,
            x,
            y,
            bw,
            bh,
        ) in raw_boxes:

            x_center_px = x * width
            y_center_px = y * height

            box_width_px = bw * width
            box_height_px = bh * height

            x1 = max(
                0.0,
                x_center_px
                - box_width_px / 2.0,
            )

            y1 = max(
                0.0,
                y_center_px
                - box_height_px / 2.0,
            )

            x2 = min(
                float(width),
                x_center_px
                + box_width_px / 2.0,
            )

            y2 = min(
                float(height),
                y_center_px
                + box_height_px / 2.0,
            )

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            labels.append(class_id)

        if self.augment:

            if len(boxes) > 0 and torch.rand(1).item() < 0.5:

                has_small_object = any(
                    (((bx2 - bx1) * (by2 - by1)) ** 0.5) <= SMALL_OBJECT_SIZE_THRESHOLD
                    for bx1, by1, bx2, by2 in boxes
                )

                if has_small_object and torch.rand(1).item() < 0.6:
                    (
                        image,
                        boxes,
                        labels,
                    ) = _small_object_zoom_crop(
                        image,
                        boxes,
                        labels,
                    )
                else:
                    (
                        image,
                        boxes,
                        labels,
                    ) = _random_resized_crop(
                        image,
                        boxes,
                        labels,
                        scale_range=(0.7, 1.0),
                        visibility_threshold=0.3,
                        max_attempts=5,
                    )

                width, height = image.size

            if torch.rand(1).item() < 0.5:
                image = F.hflip(image)

                flipped_boxes = []

                for (
                    x1,
                    y1,
                    x2,
                    y2,
                ) in boxes:

                    flipped_boxes.append(
                        [
                            width - x2,
                            y1,
                            width - x1,
                            y2,
                        ]
                    )

                boxes = flipped_boxes

            if torch.rand(1).item() < 0.7:
                image = _COLOR_JITTER(image)

        image_tensor = F.to_tensor(image)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32,
            )

            labels_tensor = torch.zeros(
                (0,),
                dtype=torch.int64,
            )

        else:
            boxes_tensor = torch.tensor(
                boxes,
                dtype=torch.float32,
            )

            labels_tensor = torch.tensor(
                labels,
                dtype=torch.int64,
            )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        return image_tensor, target


def _collate_fn(batch):
    """
    Keep images and targets as lists because detection images may have
    different spatial dimensions.
    """
    images, targets = zip(*batch)

    return list(images), list(targets)


def get_dataloaders(
    batch_size,
    num_workers=6,
):
    from config import (
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
    )

    train_ds = FruitVegDetectionDataset(
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        augment=True,
    )

    val_ds = FruitVegDetectionDataset(
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR,
        augment=False,
    )

    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    print(
        f"[dataset] Training images: {len(train_ds)}"
    )

    print(
        f"[dataset] Validation images: {len(val_ds)}"
    )

    n_small = sum(1 for w in train_ds.sample_weights if w > 1.0)
    print(
        f"[dataset] Images containing small objects (oversampled): {n_small}"
    )

    return train_loader, val_loader