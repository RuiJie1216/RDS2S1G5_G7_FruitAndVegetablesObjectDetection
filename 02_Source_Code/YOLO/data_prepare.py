"""
data_prepare.py
Data pre-processing: image resizing, normalization, augmentation,
label encoding, and bounding box annotation conversion.

Dataset: UEC FOOD 256 -- http://foodcam.mobi/dataset256.html

This version reads the dataset in the layout it's actually distributed in
(one sub-folder per category, each with its own bb_info.txt), NOT the old
raw_dataset/images + single bb_info.txt layout:

    row_dataset/                   <- RAW_DIR
        1/
            bb_info.txt            "img x1 y1 x2 y2" (per-line, header row first)
            1.jpg
            2.jpg
            ...
        2/
            bb_info.txt
            1.jpg
            ...
        ...
        256/
            bb_info.txt
            ...
        category.txt                "id<TAB or space>name" (header row first)
        multiple_food.txt           (not used here)

Important quirk of this dataset: the same physical photo can appear inside
several category folders (once per food item it contains), each time with
its own bb_info.txt row for that one food's bounding box, and the image
filename numbering restarts in every folder (folder 1's "5.jpg" has nothing
to do with folder 2's "5.jpg"). So samples are kept keyed by
(category_id, img_id) rather than by img_id alone -- that's what makes the
filenames below look like "3_5.jpg" (category 3, image 5) instead of "5.jpg".

This script does two things:
    1. convert_to_yolo_format()        -> builds the images/ + labels/
       directory structure YOLOv8 expects.
    2. build_classification_dataset()  -> crops out individual food items
       from the bounding boxes, for training the CNN / MobileNetV2
       classifiers (with resize, normalize, and augmentation).
"""

import os
import random
from PIL import Image, ImageEnhance

RAW_DIR = "row_dataset"
YOLO_DIR = "dataset/yolo_format"
CLS_DIR = "dataset/classification"

IMG_SIZE = 224          # CNN / MobileNet input size
TRAIN_RATIO = 0.8


def _read_category_map():
    """category.txt: each line is "class_id<sep>food_name", header row first.
    Official file uses a space, but some re-uploads use a tab -- handle both."""
    mapping = {}
    path = os.path.join(RAW_DIR, "category.txt")
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:  # skip header row
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) >= 2:
                mapping[int(parts[0])] = parts[1].strip().lower().replace(" ", "_")
    return mapping


def _category_dirs():
    """Yields (class_id, folder_path) for every numbered category folder
    that actually exists under RAW_DIR (1 .. 256)."""
    for name in sorted(os.listdir(RAW_DIR), key=lambda n: (len(n), n)):
        folder_path = os.path.join(RAW_DIR, name)
        if name.isdigit() and os.path.isdir(folder_path):
            yield int(name), folder_path


def _read_bbox_annotations(folder_path):
    """A category folder's bb_info.txt: each line "img_id x1 y1 x2 y2",
    header row first. Returns {img_id: [(x1, y1, x2, y2), ...]} -- a list
    because the same image can contain several instances of that food."""
    annotations = {}
    path = os.path.join(folder_path, "bb_info.txt")
    if not os.path.exists(path):
        return annotations
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            img_id, x1, y1, x2, y2 = parts[:5]
            annotations.setdefault(img_id, []).append(
                (int(x1), int(y1), int(x2), int(y2)))
    return annotations


def _collect_samples():
    """Walks every category folder and builds one flat list of samples:
        (unique_key, image_path, class_id, [(x1,y1,x2,y2), ...])
    unique_key is "{class_id}_{img_id}" so the same photo appearing under
    multiple categories never collides."""
    samples = []
    for class_id, folder_path in _category_dirs():
        annotations = _read_bbox_annotations(folder_path)
        for img_id, boxes in annotations.items():
            img_path = os.path.join(folder_path, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                continue
            unique_key = f"{class_id}_{img_id}"
            samples.append((unique_key, img_path, class_id, boxes))
    return samples


def _split(samples):
    shuffled = samples[:]
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * TRAIN_RATIO)
    return {"train": shuffled[:split_point], "val": shuffled[split_point:]}


def convert_to_yolo_format():
    """
    Converts bounding box annotations to YOLOv8 format:
        dataset/yolo_format/images/{train,val}/*.jpg
        dataset/yolo_format/labels/{train,val}/*.txt   (class cx cy w h, normalized 0-1)
        dataset/yolo_format/data.yaml
    """
    category_map = _read_category_map()
    class_ids = sorted(category_map)
    id_to_index = {class_id: i for i, class_id in enumerate(class_ids)}

    samples = _collect_samples()
    splits = _split(samples)

    for split, split_samples in splits.items():
        os.makedirs(f"{YOLO_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{YOLO_DIR}/labels/{split}", exist_ok=True)

        for unique_key, img_path, class_id, boxes in split_samples:
            img = Image.open(img_path)
            w, h = img.size

            dst_img_path = f"{YOLO_DIR}/images/{split}/{unique_key}.jpg"
            img.convert("RGB").save(dst_img_path)

            class_index = id_to_index[class_id]
            label_lines = []
            for (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_lines.append(f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(f"{YOLO_DIR}/labels/{split}/{unique_key}.txt", "w") as f:
                f.write("\n".join(label_lines))

    names = [category_map[i] for i in class_ids]
    yaml_content = (
        f"path: {os.path.abspath(YOLO_DIR)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: {names}\n"
    )
    with open(f"{YOLO_DIR}/data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"YOLO-format dataset written to: {YOLO_DIR} "
          f"(train={len(splits['train'])}, val={len(splits['val'])})")


def _augment(img: Image.Image) -> list:
    """Simple augmentation: horizontal flip + brightness/contrast jitter,
    used to expand the classification training set."""
    variants = [img]
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    variants.append(ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)))
    variants.append(ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)))
    return variants


def build_classification_dataset(augment: bool = True):
    """
    Crops each food item out by its bounding box and organizes the crops
    into a classification-ready directory structure:
        dataset/classification/{train,val}/{class_name}/*.jpg
    Applies resize + augmentation (normalization is handled at training
    time via ImageDataGenerator / Rescaling layers instead of on disk).
    """
    category_map = _read_category_map()
    samples = _collect_samples()
    splits = _split(samples)

    for split, split_samples in splits.items():
        for unique_key, img_path, class_id, boxes in split_samples:
            img = Image.open(img_path).convert("RGB")
            class_name = category_map.get(class_id, f"class_{class_id}")
            out_dir = f"{CLS_DIR}/{split}/{class_name}"
            os.makedirs(out_dir, exist_ok=True)

            for j, (x1, y1, x2, y2) in enumerate(boxes):
                crop = img.crop((x1, y1, x2, y2)).resize((IMG_SIZE, IMG_SIZE))
                variants = _augment(crop) if (augment and split == "train") else [crop]
                for k, v in enumerate(variants):
                    v.save(f"{out_dir}/{unique_key}_{j}_{k}.jpg")

    print(f"Classification dataset written to: {CLS_DIR}")


if __name__ == "__main__":
    if not os.path.isdir(RAW_DIR):
        raise SystemExit(
            f"Can't find '{RAW_DIR}/'. Download UEC FOOD-256 from "
            f"http://foodcam.mobi/dataset256.html and extract it so that "
            f"'{RAW_DIR}/1/', '{RAW_DIR}/2/', ..., '{RAW_DIR}/category.txt' exist."
        )
    convert_to_yolo_format()
    build_classification_dataset()