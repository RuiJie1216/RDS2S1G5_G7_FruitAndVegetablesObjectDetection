"""
gui_app.py
NutriScan AI -- Food Detection and Health Analysis System (GUI)

Run:
    python gui_app.py

Notes:
- Object detection priority:
    1. Custom trained YOLOv8 model (models/yolo_best.pt), if present.
    2. COCO-pretrained YOLOv8 (yolov8n.pt, auto-downloaded by ultralytics),
       filtered to food-related COCO classes. Gives REAL per-instance boxes
       (e.g. 3 separate oranges -> 3 separate boxes), no more sliding-window
       "count estimation" hacks.
    3. ImageNetSlidingDetector as last-resort fallback if ultralytics/YOLO
       can't be loaded at all (e.g. no internet to download weights).
- Same-label detections (e.g. 3x "orange") are merged into a single food
  card with quantity = number of instances found. All instance boxes are
  drawn on the image; the label text is attached to the highest-confidence
  box only, to avoid clutter.
- Nutrition data comes only from the USDA FoodData Central API (no local
  database). All foods are looked up in a single parallel batch request.
- The whole window is inside a scrollable frame so nothing gets cut off.
- Each food card shows per-unit and total (quantity-scaled) kcal/macros,
  plus a "Compare with other models" button.
"""

import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw

from nutrition_api import get_nutrition, summarize_meal

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo_best.pt")
CNN_WEIGHTS = os.path.join(MODEL_DIR, "cnn_food_classifier.h5")
MOBILENET_WEIGHTS = os.path.join(MODEL_DIR, "mobilenet_food_classifier.h5")

BOX_COLORS = ["#1D9E75", "#BA7517", "#3C3489", "#D85A30", "#185FA5"]

FOOD_KEYWORDS = [
    "pizza", "burger", "hotdog", "hot_dog", "sandwich", "burrito", "taco",
    "pretzel", "bagel", "pancake", "waffle", "pasta", "carbonara",
    "guacamole", "salad", "soup", "broccoli", "cauliflower", "cabbage",
    "mushroom", "corn", "cucumber", "artichoke", "pepper", "banana",
    "orange", "lemon", "strawberry", "pineapple", "pomegranate", "fig",
    "custard", "trifle", "ice_cream", "chocolate", "meatloaf", "steak",
    "rice", "noodle", "dough", "bread", "loaf", "cheese", "egg",
    "consomme", "potpie", "hay", "cream", "espresso", "cup", "wine",
    "plate", "menu",
]

# COCO classes we care about for the pretrained-YOLO food detector.
# Extend this list if you need more COCO food/drink categories.
COCO_FOOD_CLASSES = {
    "banana", "apple", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "sandwich",
    "wine glass", "cup", "bowl", "bottle",
}


# ---------------------------------------------------------------------------
# Detector A: real per-instance detection via COCO-pretrained YOLOv8.
# This is the preferred fallback when no custom-trained weights exist.
# Each detected instance gets its own box -> no need to "estimate" quantity,
# quantity is simply the count of same-label boxes (handled in analyze_image).
# ---------------------------------------------------------------------------
class CocoPretrainedDetector:
    """
    Uses ultralytics' COCO-pretrained YOLOv8 weights (yolov8n.pt by default).
    Weights are auto-downloaded and cached by ultralytics on first use.

    Only boxes whose class is in COCO_FOOD_CLASSES are kept. Every kept box
    is a genuine, independently detected instance (real localisation, not a
    confidence-window heuristic), so multiple identical foods in one photo
    (e.g. 3 oranges) naturally produce 3 separate boxes with quantity=1 each.
    """

    def __init__(self, weights="yolov8n.pt", conf_threshold=0.25):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf_threshold = conf_threshold

    def detect(self, pil_image):
        results = self.model.predict(
            pil_image, verbose=False, conf=self.conf_threshold
        )[0]

        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id].lower()
            if label not in COCO_FOOD_CLASSES:
                continue  # skip non-food classes (person, dining table, etc.)

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append({
                "food_key": label,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2),
                "quantity": 1,  # one real box = one real instance
            })
        return boxes


# ---------------------------------------------------------------------------
# Detector B (last-resort fallback): whole-image classification + sliding
# window. Only used if ultralytics/YOLO can't be loaded at all (e.g. no
# internet access to fetch yolov8n.pt and no local weights available).
# ---------------------------------------------------------------------------
class ImageNetSlidingDetector:
    """
    Fallback detector used only if YOLO (custom or COCO-pretrained) is
    unavailable. Classifies the whole image, localises with one sliding
    window, and *estimates* a quantity via a smaller counting window + NMS.
    This is inherently an approximation (a classifier has no real notion of
    "instances"), so prefer CocoPretrainedDetector or a custom YOLO model
    whenever possible.
    """

    WINDOW_FRACTION = 0.6
    STRIDE_FRACTION = 0.5
    CONF_THRESHOLD = 0.10
    COUNT_WINDOW_FRACTION = 0.45
    COUNT_CONF_THRESHOLD = 0.08

    def __init__(self):
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import (
            preprocess_input, decode_predictions)
        import numpy as np
        self._np = np
        self._preprocess_input = preprocess_input
        self._decode_predictions = decode_predictions
        print("Loading pretrained MobileNetV2 (ImageNet) for demo detection...")
        self.model = MobileNetV2(weights="imagenet")

    def _predict(self, crop):
        img = crop.resize((224, 224))
        arr = self._np.expand_dims(self._np.array(img), axis=0)
        arr = self._preprocess_input(arr)
        preds = self.model.predict(arr, verbose=0)
        top5 = self._decode_predictions(preds, top=5)[0]
        return {label.lower(): float(conf) for (_, label, conf) in top5}

    def _looks_like_food(self, label):
        return any(kw in label.lower() for kw in FOOD_KEYWORDS)

    def detect(self, pil_image):
        w, h = pil_image.size
        whole_scores = self._predict(pil_image)

        best_label, best_conf = None, 0.0
        for label, conf in whole_scores.items():
            if conf >= self.CONF_THRESHOLD and self._looks_like_food(label):
                if conf > best_conf:
                    best_label, best_conf = label, conf

        if best_label is None:
            top_label = max(whole_scores, key=whole_scores.get)
            return [{
                "food_key": top_label,
                "confidence": round(whole_scores[top_label], 2),
                "bbox": (0, 0, w, h),
                "quantity": 1,
            }]

        win_w = int(w * self.WINDOW_FRACTION)
        win_h = int(h * self.WINDOW_FRACTION)
        stride_x = max(1, int(win_w * self.STRIDE_FRACTION))
        stride_y = max(1, int(win_h * self.STRIDE_FRACTION))

        best_box = (0, 0, w, h)
        best_box_conf = best_conf

        x = 0
        while x + win_w <= w:
            y = 0
            while y + win_h <= h:
                crop = pil_image.crop((x, y, x + win_w, y + win_h))
                scores = self._predict(crop)
                window_conf = scores.get(best_label, 0.0)
                if window_conf > best_box_conf:
                    best_box_conf = window_conf
                    best_box = (x, y, x + win_w, y + win_h)
                y += stride_y
            x += stride_x

        quantity = self._count_items(pil_image, best_label)

        return [{
            "food_key": best_label,
            "confidence": round(best_box_conf, 2),
            "bbox": best_box,
            "quantity": quantity,
        }]

    def _count_items(self, pil_image, label):
        w, h = pil_image.size
        win_w = int(w * self.COUNT_WINDOW_FRACTION)
        win_h = int(h * self.COUNT_WINDOW_FRACTION)
        stride_x = max(1, win_w // 2)
        stride_y = max(1, win_h // 2)

        hits = []
        x = 0
        while x + win_w <= w:
            y = 0
            while y + win_h <= h:
                crop = pil_image.crop((x, y, x + win_w, y + win_h))
                scores = self._predict(crop)
                conf = scores.get(label, 0.0)
                if conf >= self.COUNT_CONF_THRESHOLD:
                    hits.append((conf, x, y, x + win_w, y + win_h))
                y += stride_y
            x += stride_x

        if not hits:
            return 1

        hits.sort(reverse=True)
        kept = []
        for hit in hits:
            _, hx1, hy1, hx2, hy2 = hit
            overlap = False
            for _, kx1, ky1, kx2, ky2 in kept:
                ix1, iy1 = max(hx1, kx1), max(hy1, ky1)
                ix2, iy2 = min(hx2, kx2), min(hy2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (hx2-hx1)*(hy2-hy1) + (kx2-kx1)*(ky2-ky1) - inter
                if union > 0 and inter / union > 0.5:
                    overlap = True
                    break
            if not overlap:
                kept.append(hit)

        return max(1, min(12, len(kept)))


class YoloDetector:
    """Wraps a custom-trained YOLOv8 model (ultralytics)."""

    def __init__(self, weights_path):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)

    def detect(self, pil_image):
        results = self.model.predict(pil_image, verbose=False)[0]
        boxes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            food_key = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append({
                "food_key": food_key,
                "confidence": round(conf, 2),
                "bbox": (x1, y1, x2, y2),
                "quantity": 1,
            })
        return boxes


def build_detector():
    # 1. Prefer a custom-trained model if you have one.
    if os.path.exists(YOLO_WEIGHTS):
        try:
            return YoloDetector(YOLO_WEIGHTS)
        except Exception as e:
            print("Failed to load custom YOLO weights:", e)

    # 2. Otherwise use COCO-pretrained YOLOv8 for real instance detection.
    try:
        return CocoPretrainedDetector()
    except Exception as e:
        print("Failed to load COCO-pretrained YOLO, falling back to demo mode:", e)

    # 3. Last resort: classification + sliding window heuristic.
    return ImageNetSlidingDetector()


# ---------------------------------------------------------------------------
# Classifier comparison: re-predict the same crop with different algorithms
# ---------------------------------------------------------------------------
def classify_with_all_models(crop_img):
    return {
        "Custom CNN": _predict_keras(CNN_WEIGHTS, crop_img),
        "MobileNetV2": _predict_keras(MOBILENET_WEIGHTS, crop_img),
    }


def _predict_keras(weights_path, crop_img):
    if not os.path.exists(weights_path):
        return None
    try:
        import numpy as np
        from tensorflow.keras.models import load_model
        model = load_model(weights_path)
        img = crop_img.resize((128, 128))
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        idx = int(preds.argmax())
        return str(idx), round(float(preds[idx]), 2)
    except Exception as e:
        print("Classification failed:", e)
        return None


# ---------------------------------------------------------------------------
# Nutrition scaling helpers
# ---------------------------------------------------------------------------
def _scale_info_by_quantity(info: dict, quantity: int) -> dict:
    if quantity <= 1:
        return info
    scaled = dict(info)
    for key in ("calories", "protein", "fat", "carbs"):
        if key in scaled:
            scaled[key] = round(scaled[key] * quantity, 1)
    return scaled


def _merge_same_label_detections(raw_detections):
    """
    Merge detections that share the same food_key into a single entry with:
      - quantity = number of instances found
      - bbox / confidence = from the highest-confidence instance (used as
        the "representative" box for the label text)
      - _all_boxes = every instance's bbox, so all of them can still be
        drawn on the image.

    This lets a real per-instance detector (CocoPretrainedDetector /
    YoloDetector) report N independent boxes for N independent food items,
    while the rest of the app (nutrition lookup, food cards, summary) keeps
    working with "one food_key -> one quantity" as before.
    """
    merged = {}
    for d in raw_detections:
        key = d["food_key"]
        if key not in merged:
            merged[key] = {
                **d,
                "quantity": d.get("quantity", 1),
                "_all_boxes": [d["bbox"]],
            }
        else:
            merged[key]["quantity"] += d.get("quantity", 1)
            merged[key]["_all_boxes"].append(d["bbox"])
            if d["confidence"] > merged[key]["confidence"]:
                merged[key]["confidence"] = d["confidence"]
                merged[key]["bbox"] = d["bbox"]
    return list(merged.values())


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class NutriScanApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NutriScan AI - Food Detection and Health Analysis")
        self.geometry("1180x760")
        self.minsize(900, 600)
        self.configure(fg_color="#F1EFE8")

        self.detector = build_detector()
        self.current_image = None
        self.current_detections = []
        self.current_summary = None

        self._build_header()
        self._build_scrollable_body()

    def _build_header(self):
        header = ctk.CTkFrame(self, height=64, fg_color="white", corner_radius=12)
        header.pack(fill="x", padx=16, pady=(16, 8))

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", padx=16, pady=10)
        ctk.CTkLabel(title_box, text="NutriScan AI",
                      font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(title_box, text="Food detection - nutrition analysis - health advice",
                      font=ctk.CTkFont(size=12), text_color="#5F5E5A").pack(anchor="w")

        ctk.CTkButton(header, text="Upload photo", width=110,
                       command=self.upload_image).pack(side="right", padx=16, pady=14)
        ctk.CTkButton(header, text="Re-analyze", width=110,
                       command=self.analyze_image).pack(side="right", padx=6, pady=14)

    def _build_scrollable_body(self):
        self.scroll_container = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_container.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.scroll_container.grid_columnconfigure(0, weight=3)
        self.scroll_container.grid_columnconfigure(1, weight=2)

        self.image_panel = ctk.CTkFrame(self.scroll_container, fg_color="white", corner_radius=12)
        self.image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        self.image_label = ctk.CTkLabel(self.image_panel, text="Upload a food photo to begin",
                                          font=ctk.CTkFont(size=14), text_color="#888780")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=12)

        right_panel = ctk.CTkFrame(self.scroll_container, fg_color="transparent")
        right_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        ctk.CTkLabel(right_panel, text="Detected items",
                      font=ctk.CTkFont(size=13), text_color="#5F5E5A").pack(anchor="w", pady=(0, 6))
        self.food_list_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        self.food_list_frame.pack(fill="both", expand=True)

        self.summary_frame = ctk.CTkFrame(self.scroll_container, fg_color="white", corner_radius=12)
        self.summary_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ctk.CTkLabel(self.summary_frame,
                      text="Upload and analyze a photo to see the meal's nutrition summary here",
                      font=ctk.CTkFont(size=13), text_color="#888780").pack(padx=16, pady=20)

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.current_image = Image.open(path).convert("RGB")
        self.analyze_image()

    def analyze_image(self):
        if self.current_image is None:
            return

        raw_detections = self.detector.detect(self.current_image)

        # Real per-instance detectors (CocoPretrainedDetector / YoloDetector)
        # can return several boxes with the same food_key (e.g. 3 oranges).
        # Merge those into one card per unique food_key, quantity = count.
        self.current_detections = _merge_same_label_detections(raw_detections)

        detected = [{"food_key": d["food_key"], "confidence": d["confidence"]}
                    for d in self.current_detections]
        self.current_summary = summarize_meal(detected)

        self._render_image_with_boxes()
        self._render_food_list()
        self._render_summary()

    def _render_image_with_boxes(self):
        img = self.current_image.copy()
        draw = ImageDraw.Draw(img)
        for i, (det, info) in enumerate(zip(self.current_detections, self.current_summary["items"])):
            color = BOX_COLORS[i % len(BOX_COLORS)]
            qty = det.get("quantity", 1)
            boxes_to_draw = det.get("_all_boxes", [det["bbox"]])

            # Draw every instance box (e.g. all 3 oranges get outlined).
            for bx in boxes_to_draw:
                draw.rectangle(bx, outline=color, width=3)

            # Attach the text label to only the representative (highest
            # confidence) box so labels don't overlap when quantity > 1.
            x1, y1, x2, y2 = det["bbox"]
            qty_str = f" ×{qty}" if qty > 1 else ""
            label = f'{info["display_name"]}{qty_str} {int(det["confidence"]*100)}%'
            draw.rectangle([x1, max(0, y1 - 20), x1 + 8 * len(label), y1], fill=color)
            draw.text((x1 + 4, max(0, y1 - 19)), label, fill="white")

        max_w, max_h = 640, 560
        ratio = min(max_w / img.width, max_h / img.height, 1.0)
        display_img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        ctk_img = ctk.CTkImage(light_image=display_img, size=display_img.size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    def _render_food_list(self):
        for widget in self.food_list_frame.winfo_children():
            widget.destroy()

        for det, info in zip(self.current_detections, self.current_summary["items"]):
            qty = det.get("quantity", 1)
            scaled_info = _scale_info_by_quantity(info, qty)
            self._build_food_card(self.food_list_frame, det, info, scaled_info)

    def _build_food_card(self, parent, det, info, scaled_info):
        qty = det.get("quantity", 1)
        card = ctk.CTkFrame(parent, fg_color="white", corner_radius=12)
        card.pack(fill="x", pady=6)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(10, 2))

        name_text = info["display_name"]
        if qty > 1:
            name_text += f"  ×{qty}"
        ctk.CTkLabel(top, text=name_text,
                      font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")

        kcal_frame = ctk.CTkFrame(top, fg_color="transparent")
        kcal_frame.pack(side="right")
        ctk.CTkLabel(kcal_frame,
                      text=f'{int(scaled_info["calories"])} kcal',
                      fg_color="#EAF3DE", text_color="#173404", corner_radius=6,
                      font=ctk.CTkFont(size=11)).pack(side="left", ipadx=6, ipady=2)
        if qty > 1:
            ctk.CTkLabel(kcal_frame,
                          text=f'  ({int(info["calories"])} kcal each)',
                          font=ctk.CTkFont(size=10), text_color="#888780").pack(side="left")

        if qty > 1:
            macro_text = (
                f'Protein {scaled_info["protein"]}g · Fat {scaled_info["fat"]}g · '
                f'Carbs {scaled_info["carbs"]}g  '
                f'(per unit: P {info["protein"]}g / F {info["fat"]}g / C {info["carbs"]}g) · '
                f'Source: {info["source"]}'
            )
        else:
            macro_text = (
                f'Protein {info["protein"]}g · Fat {info["fat"]}g · '
                f'Carbs {info["carbs"]}g · Source: {info["source"]}'
            )
        ctk.CTkLabel(card, text=macro_text,
                      font=ctk.CTkFont(size=11), text_color="#888780",
                      anchor="w", wraplength=300, justify="left").pack(fill="x", padx=12)

        if info["is_composite"] and info["ingredients"]:
            ing_text = ", ".join(info["ingredients"])
            ctk.CTkLabel(card, text=f'Ingredients: {ing_text}', wraplength=280,
                          font=ctk.CTkFont(size=11), text_color="#5F5E5A",
                          anchor="w", justify="left").pack(fill="x", padx=12, pady=(4, 0))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(6, 10))
        result_label = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=11),
                                      text_color="#5F5E5A", anchor="w", justify="left",
                                      wraplength=280)
        ctk.CTkButton(btn_row, text="Compare with other models", height=26,
                       font=ctk.CTkFont(size=11),
                       command=lambda: self._compare_algorithms(det, info, result_label)
                       ).pack(side="left")
        result_label.pack(fill="x", padx=12, pady=(0, 10))

    def _compare_algorithms(self, det, info, label_widget):
        x1, y1, x2, y2 = det["bbox"]
        crop = self.current_image.crop((x1, y1, x2, y2))
        results = classify_with_all_models(crop)

        qty = det.get("quantity", 1)
        qty_str = f" ×{qty}" if qty > 1 else ""
        lines = [f'YOLOv8 / demo detector: {info["display_name"]}{qty_str} ({int(det["confidence"]*100)}%)']
        for name, res in results.items():
            if res is None:
                lines.append(f'{name}: model not trained / weights not found')
            else:
                pred_key, conf = res
                lines.append(f'{name}: class {pred_key} ({int(conf*100)}%)')
        label_widget.configure(text="\n".join(lines))

    def _render_summary(self):
        for widget in self.summary_frame.winfo_children():
            widget.destroy()

        summary = self.current_summary

        scaled_items = [
            _scale_info_by_quantity(info, det.get("quantity", 1))
            for det, info in zip(self.current_detections, summary["items"])
        ]
        totals = {
            "calories": sum(s["calories"] for s in scaled_items),
            "protein":  round(sum(s["protein"]  for s in scaled_items), 1),
            "carbs":    round(sum(s["carbs"]     for s in scaled_items), 1),
            "fat":      round(sum(s["fat"]       for s in scaled_items), 1),
        }

        ctk.CTkLabel(self.summary_frame, text="Meal summary",
                      font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=16, pady=(14, 8))

        stats = ctk.CTkFrame(self.summary_frame, fg_color="transparent")
        stats.pack(fill="x", padx=16)
        stat_items = [
            ("Calories", f'{int(totals["calories"])} kcal'),
            ("Protein",  f'{totals["protein"]:.1f} g'),
            ("Carbs",    f'{totals["carbs"]:.1f} g'),
            ("Fat",      f'{totals["fat"]:.1f} g'),
        ]
        for name, value in stat_items:
            box = ctk.CTkFrame(stats, fg_color="#F1EFE8", corner_radius=8)
            box.pack(side="left", expand=True, fill="x", padx=4)
            ctk.CTkLabel(box, text=name, font=ctk.CTkFont(size=11),
                          text_color="#888780").pack(pady=(8, 0))
            ctk.CTkLabel(box, text=value, font=ctk.CTkFont(size=17, weight="bold")).pack(pady=(0, 8))

        verdict_color = {"balanced and healthy": "#EAF3DE",
                          "acceptable, some caution advised": "#FAEEDA",
                          "not very healthy": "#FCEBEB"}
        text_color = {"balanced and healthy": "#173404",
                      "acceptable, some caution advised": "#412402",
                      "not very healthy": "#501313"}
        v = summary["verdict"]
        banner = ctk.CTkFrame(self.summary_frame, fg_color=verdict_color.get(v, "#F1EFE8"),
                                corner_radius=8)
        banner.pack(fill="x", padx=16, pady=(10, 16))
        ctk.CTkLabel(banner, text=f'{v.capitalize()} - health score {summary["score"]}/10',
                      font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=text_color.get(v, "#2C2C2A")).pack(anchor="w", padx=12, pady=(10, 2))
        suit = ", ".join(summary["suitable_for"])
        caution = ", ".join(summary["caution_for"]) if summary["caution_for"] else "no particular concerns"
        ctk.CTkLabel(banner, text=f'Suitable for: {suit}\nCaution for: {caution}',
                      font=ctk.CTkFont(size=11), text_color=text_color.get(v, "#2C2C2A"),
                      justify="left", anchor="w").pack(anchor="w", padx=12, pady=(0, 10))


if __name__ == "__main__":
    app = NutriScanApp()
    app.mainloop()