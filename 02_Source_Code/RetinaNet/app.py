"""
app.py
======
Flask web application. Lets a user upload a photo, runs it through the
fine-tuned RetinaNet detector, shows the photo with bounding boxes +
labels drawn on it, and displays the model's recorded evaluation metrics
so this RetinaNet model can be compared against other models (e.g. the
teammates' YOLO / MobileNet) later.

Usage:
    python app.py                    # uses the latest completed run
    python app.py --run model_V6     # uses a specific run
Then open http://127.0.0.1:5000 in a browser.
"""

import argparse
import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

from config import UPLOAD_DIR, MODEL_NAME, CLASS_NAMES
import run_manager as rm
from metrics_utils import load_model_results
from collections import defaultdict
# NOTE: torch, model.py and inference.py are imported lazily inside
# get_model()/predict() below, so the app can still start and show the
# upload page even if no run has finished training yet.

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)
app.secret_key = "retinanet-fruit-veg-dev-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

os.makedirs(UPLOAD_DIR, exist_ok=True)

_model = None
_device = None
_model_load_error = None
_active_run_name = None
_requested_run_name = None  # set from --run at startup, if given


def get_model():
    """Lazily load the model from a completed run, once, on first request."""
    global _model, _device, _model_load_error, _active_run_name

    if _model is not None or _model_load_error is not None:
        return _model

    run_name = _requested_run_name or rm.latest_completed_run()

    if run_name is None:
        _model_load_error = (
            "No completed training run found under runs/. "
            "Run 'python train.py' and let it finish (see README.md)."
        )
        return None

    model_file = rm.best_model_path(run_name) if os.path.exists(rm.best_model_path(run_name)) else rm.model_path(run_name)
    if not os.path.exists(model_file):
        _model_load_error = (
            f"Run '{run_name}' is marked completed but its checkpoint is missing at {model_file}."
        )
        return None

    try:
        import torch
        from model import build_model
        from train import get_device

        _device = get_device()
        _model = build_model(pretrained=False)
        checkpoint = torch.load(model_file, map_location=_device)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.to(_device)
        _model.eval()
        _active_run_name = run_name
    except Exception as exc:  # noqa: BLE001
        _model_load_error = f"Failed to load model: {exc}"

    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


def get_metrics_context():
    all_results = load_model_results()
    current = next((r for r in reversed(all_results) if r["model_name"] == MODEL_NAME), None)
    others = [r for r in all_results if r["model_name"] != MODEL_NAME]
    return current, others


@app.route("/")
def index():
    model = get_model()
    current_metrics, other_models = get_metrics_context()
    return render_template(
        "index.html",
        model_ready=model is not None,
        model_error=_model_load_error,
        num_classes=len(CLASS_NAMES),
        model_name=MODEL_NAME,
        active_run_name=_active_run_name,
        current_metrics=current_metrics,
        other_models=other_models,
    )


def summarize_detections(detections):
    """
    Groups raw per-box detections by class name, e.g. turns:
        [{"class_name": "apple", "confidence": 87.3}, {"class_name": "apple", "confidence": 82.1}, ...]
    into:
        [{"class_name": "apple", "count": 2, "min_confidence": 82.1, "max_confidence": 87.3}, ...]
    Sorted by count (most frequent class first).
    """
    groups = defaultdict(list)
    for det in detections:
        groups[det["class_name"]].append(det["confidence"])

    summary = []
    for class_name, confidences in groups.items():
        summary.append({
            "class_name": class_name,
            "count": len(confidences),
            "min_confidence": round(min(confidences), 1),
            "max_confidence": round(max(confidences), 1),
        })

    summary.sort(key=lambda g: g["count"], reverse=True)
    return summary

def summarize_detections(detections):
    """
    Groups raw per-box detections by class name, e.g. turns:
        [{"class_name": "apple", "confidence": 87.3}, {"class_name": "apple", "confidence": 82.1}, ...]
    into:
        [{"class_name": "apple", "count": 2, "min_confidence": 82.1, "max_confidence": 87.3, "avg_confidence": 84.7}, ...]
    Sorted by count (most frequent class first).
    """
    groups = defaultdict(list)
    for det in detections:
        groups[det["class_name"]].append(det["confidence"])

    summary = []
    for class_name, confidences in groups.items():
        summary.append({
            "class_name": class_name,
            "count": len(confidences),
            "min_confidence": round(min(confidences), 1),
            "max_confidence": round(max(confidences), 1),
            "avg_confidence": round(sum(confidences) / len(confidences), 1),
        })

    summary.sort(key=lambda g: g["count"], reverse=True)
    return summary

@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        flash(_model_load_error or "Model is not available.")
        return redirect(url_for("index"))

    if "photo" not in request.files:
        flash("No file was uploaded.")
        return redirect(url_for("index"))

    file = request.files["photo"]
    if file.filename == "":
        flash("No file was selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PNG, JPG, JPEG, WEBP or BMP image.")
        return redirect(url_for("index"))

    unique_id = uuid.uuid4().hex[:10]
    ext = file.filename.rsplit(".", 1)[1].lower()
    input_filename = f"{unique_id}_input.{ext}"
    output_filename = f"{unique_id}_result.jpg"

    input_path = os.path.join(UPLOAD_DIR, input_filename)
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    file.save(input_path)

    try:
        from inference import run_detection
        result = run_detection(model, _device, input_path, output_path)
    except Exception as exc:  # noqa: BLE001
        flash(f"Something went wrong while processing the image: {exc}")
        return redirect(url_for("index"))

    detection_summary = summarize_detections(result["detections"])
    current_metrics, other_models = get_metrics_context()

    return render_template(
        "result.html",
        result_image=url_for("uploaded_file", filename=output_filename),
        detections=result["detections"],
        detection_summary=detection_summary,
        inference_time_ms=result["inference_time_ms"],
        model_name=MODEL_NAME,
        active_run_name=_active_run_name,
        current_metrics=current_metrics,
        other_models=other_models,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FreshScan AI web app.")
    parser.add_argument("--run", type=str, default=None,
                         help="Run name under runs/ to serve. Defaults to the latest completed run.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _requested_run_name = args.run
    app.run(debug=True, host="127.0.0.1", port=5000)