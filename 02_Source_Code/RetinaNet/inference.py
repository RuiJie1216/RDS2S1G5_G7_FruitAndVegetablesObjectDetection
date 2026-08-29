"""
inference.py
============
Runs the trained RetinaNet detector on a single image (e.g. a photo the
user uploaded through the web app), draws the predicted bounding boxes +
labels on the image, and returns a clean, JSON-friendly summary.

Because RetinaNet's own internal transform + NMS already produce clean,
well-localized, deduplicated boxes in the ORIGINAL image's coordinate
space, this file is much simpler than the old grid-decoding version --
no manual cell-to-xyxy math, no separate NMS pass needed.
"""

import colorsys
import os
import time

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

from config import CLASS_NAMES, SCORE_THRESHOLD


def _class_color(class_id, num_classes):
    hue = (class_id * 0.61803398875) % 1.0  # golden ratio spacing
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size=18):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


@torch.no_grad()
def run_detection(model, device, image_path, output_path, score_threshold=SCORE_THRESHOLD):
    """
    Runs the model on a single image file, draws detections, and saves
    the annotated image to output_path.

    Returns a dict:
        {
            "detections": [ {class_name, confidence, box_pixels}, ... ],
            "inference_time_ms": float,
            "output_image_path": output_path,
        }
    """
    model.eval()

    original = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(original).to(device)

    start = time.time()
    output = model([image_tensor])[0]  # RetinaNet resizes/pads internally
                                        # and rescales boxes back to this
                                        # image's ORIGINAL size before
                                        # returning them, so no manual
                                        # coordinate conversion is needed.
    elapsed_ms = (time.time() - start) * 1000.0

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()

    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(max(14, original.size[0] // 40))

    results = []
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x_min, y_min, x_max, y_max = [int(round(v)) for v in box]
        class_id = int(label)
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"

        color = _class_color(class_id, len(CLASS_NAMES))
        label_text = f"{class_name} {score * 100:.1f}%"

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y_min - text_h - 6)
        draw.rectangle([x_min, label_y, x_min + text_w + 8, label_y + text_h + 6], fill=color)
        draw.text((x_min + 4, label_y + 2), label_text, fill=(255, 255, 255), font=font)

        results.append({
            "class_name": class_name,
            "confidence": round(float(score) * 100, 1),
            "box_pixels": [x_min, y_min, x_max, y_max],
        })

    annotated.save(output_path)

    return {
        "detections": results,
        "inference_time_ms": round(elapsed_ms, 1),
        "output_image_path": output_path,
    }
