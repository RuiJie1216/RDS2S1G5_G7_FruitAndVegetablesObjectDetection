"""
Fruit and Vegetable Object Detection - Multi-Model Comparison UI (Streamlit)

Two pages:
  1. Image / Video Test  - two tabs:
       - Upload Image/Video: upload a still image (runs through all three
         models, one per tab) OR a video file (all three models process it
         in parallel in the background; once each is done, the finished
         annotated video is re-encoded to H.264 for browser playback and
         shown with a detection table that updates live as you play the
         video, synced to playback time).
       - Live Webcam: real-time detection from the browser webcam via
         streamlit-webrtc, with a live-updating detection table beside it.
  2. Model Comparison    - reads the per-model evaluation results that
     were already computed by running evaluation.py directly, and
     compares mAP / P-R-F1 / PR curves / confusion matrix / inference
     speed. This page never runs inference itself; it only reads JSON.

Run with:
    streamlit run 02_Source_Code/app.py

To (re)generate evaluation results first, run:
    python 02_Source_Code/evaluation.py

Additional dependencies for video/webcam support:
    pip install streamlit-webrtc av opencv-python

Additional SYSTEM dependency (not pip-installable) for browser-playable
video output:
    ffmpeg must be installed and on PATH (apt-get install ffmpeg /
    brew install ffmpeg / add to packages.txt on Streamlit Cloud).
"""

import os
import time
import json
import base64
import tempfile
import threading
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models.detection import retinanet_resnet50_fpn_v2
import streamlit as st
import sys
import contextlib
from pathlib import Path
import imageio_ffmpeg

import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import evaluation


@contextlib.contextmanager
def isolated_import_dir(directory):
    """
    Temporarily prepends `directory` to sys.path and clears any cached
    'config' / 'model' / 'detector' modules first, since RetinaNet and
    MobileNetV2 each ship their own config.py and would otherwise
    collide through Python's global module cache (sys.modules).

    NOTE: sys.path / sys.modules are PROCESS-GLOBAL, not thread-local.
    When RetinaNet and MobileNetV2 load concurrently on different threads
    (as they do when all three models process a video in parallel), two
    threads mutating sys.modules/sys.path at the same time can stomp on
    each other and cause spurious import failures. Callers of this
    context manager MUST hold `_model_load_lock` (see below) so only one
    thread is ever inside this block at a time.
    """
    sys.path.insert(0, directory)
    stale_names = ["config", "model", "detector"]
    saved = {name: sys.modules.pop(name, None) for name in stale_names}
    try:
        yield
    finally:
        sys.path.remove(directory)
        for name in stale_names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


# Serializes any model-loading code that touches sys.path / sys.modules
# (load_retinanet, load_mobilenet). Prevents the race condition that was
# causing RetinaNet to fail when all three models loaded at once in
# separate threads. Loading is fast and only happens once per session
# (st.cache_resource), so this lock does not hurt the parallel *inference*
# that happens afterwards.
_model_load_lock = threading.Lock()

# ------------------------------------------------------------------
# Config - paths are resolved relative to this script's location,
# so it works regardless of the working directory `streamlit run`
# is launched from.
# ------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../02_Source_Code
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../<project root>

MODEL_DIR = os.path.join(PROJECT_ROOT, "04_Trained_Model")
RETINANET_PATH = os.path.join(MODEL_DIR, "RetinaNet", "retinanet_best.pt")
YOLO_PATH = os.path.join(MODEL_DIR, "YOLO", "yolo11m_focal_epoch4_best_map50_95.pt")
MOBILENET_PATH = os.path.join(MODEL_DIR, "MobileNet_V2", "mobilenetv2_detector_best.pt")
CLASSES_TXT_PATH = os.path.join(BASE_DIR, "RetinaNet", "classes.txt")

TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "03_Dataset", "LVIS_Fruits_And_Vegetables", "images", "test")
TEST_LABELS_DIR = os.path.join(PROJECT_ROOT, "03_Dataset", "LVIS_Fruits_And_Vegetables", "labels", "test")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


CLASS_NAMES = load_class_names(CLASSES_TXT_PATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Fruit and Vegetable Object Detection", layout="wide")

# ------------------------------------------------------------------
# Model loaders (cached so each model loads only once per session)
# Used by image, video, and webcam inference.
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading RetinaNet...")
def load_retinanet():
    with _model_load_lock:
        retinanet_dir = os.path.join(BASE_DIR, "RetinaNet")
        with isolated_import_dir(retinanet_dir):
            from model import build_model

            num_classes = len(CLASS_NAMES)
            model = build_model(num_classes=num_classes, pretrained=False)

        checkpoint = torch.load(RETINANET_PATH, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        return model


@st.cache_resource(show_spinner="Loading YOLO11m...")
def load_yolo():
    from ultralytics import YOLO

    return YOLO(YOLO_PATH)


@st.cache_resource(show_spinner="Loading MobileNetV2...")
def load_mobilenet():
    with _model_load_lock:
        mobilenet_dir = os.path.join(BASE_DIR, "Mobilenet_V2")
        with isolated_import_dir(mobilenet_dir):
            from train import load_detector_checkpoint

            model, checkpoint, class_names = load_detector_checkpoint(Path(MOBILENET_PATH), DEVICE)

        model.eval()
        return model


def load_model_by_name(model_choice: str):
    if model_choice == "YOLO11m":
        return load_yolo()
    elif model_choice == "RetinaNet":
        return load_retinanet()
    else:
        return load_mobilenet()


def get_run_fn(model_choice: str):
    if model_choice == "YOLO11m":
        return run_yolo
    elif model_choice == "RetinaNet":
        return run_retinanet
    else:
        return run_mobilenet


# ------------------------------------------------------------------
# NOTE: the eval-mode model loaders (load_retinanet_eval, etc.) and
# load_test_dataset() used to live here so the Model Comparison page
# could run inference itself. That responsibility has moved entirely
# to evaluation.py, which is run standalone from the command line:
#
#     python 02_Source_Code/evaluation.py
#
# This app now only ever READS the resulting JSON files - it never
# loads a model or runs inference for the comparison page.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Inference helpers (used by image, video, and webcam)
# ------------------------------------------------------------------


def run_retinanet(model, image: Image.Image, conf_threshold: float):
    img_tensor = transforms.ToTensor()(image).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    elapsed = time.time() - start

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    keep = scores >= conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cls_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(y1 - 12, 0)), f"{cls_name} {score:.2f}", fill="red")
        detections.append(
            {
                "class": cls_name,
                "confidence": round(float(score), 4),
                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            }
        )

    return result_img, detections, elapsed


def run_yolo(model, image: Image.Image, conf_threshold: float):
    start = time.time()
    results = model.predict(image, conf=conf_threshold, verbose=False)
    elapsed = time.time() - start

    result = results[0]
    result_img = Image.fromarray(result.plot()[:, :, ::-1])  # BGR -> RGB

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            {
                "class": cls_name,
                "confidence": round(conf, 4),
                "box": [round(v, 1) for v in (x1, y1, x2, y2)],
            }
        )

    return result_img, detections, elapsed


def run_mobilenet(model, image: Image.Image, conf_threshold: float):
    img_tensor = transforms.ToTensor()(image).to(DEVICE)
    start = time.time()
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    elapsed = time.time() - start

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    keep = scores >= conf_threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    result_img = image.copy()
    draw = ImageDraw.Draw(result_img)
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        # SSD reserves label 0 for background, same convention as detection_data.py (class_id + 1)
        cls_name = CLASS_NAMES[label - 1] if 0 < label <= len(CLASS_NAMES) else str(label)
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, max(y1 - 12, 0)), f"{cls_name} {score:.2f}", fill="lime")
        detections.append(
            {
                "class": cls_name,
                "confidence": round(float(score), 4),
                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            }
        )

    return result_img, detections, elapsed


# ------------------------------------------------------------------
# Video processing - reads the uploaded video frame-by-frame with
# OpenCV, runs ONE selected model on each frame, writes an annotated
# copy to disk with cv2.VideoWriter, and also records the detections
# for every frame so the finished video can be paired with a table
# that updates live as it's played back (synced via JS timeupdate).
#
# NOTE: cv2.VideoWriter with fourcc "mp4v" produces an MPEG-4 Part 2
# stream inside an .mp4 container. That container/codec combo is NOT
# reliably playable by browser <video> tags (Chrome/Firefox/Safari all
# expect H.264/AVC for mp4). The file is written correctly, but the
# <video> element just shows a black box with a non-functional play
# button because it can't decode it. We fix this by re-encoding the
# OpenCV output to H.264 with ffmpeg (see reencode_to_h264 below)
# before ever handing the file to render_synced_video_with_table.
# ------------------------------------------------------------------


def process_video(input_path, output_path, model_choice, conf_threshold, progress_callback=None):
    model = load_model_by_name(model_choice)
    run_fn = get_run_fn(model_choice)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # may be 0/unreliable for some containers

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open the output video writer (codec issue).")

    frame_idx = 0
    per_frame_detections = []  # per_frame_detections[i] = detections list for frame i
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            result_img, detections, _ = run_fn(model, pil_img, conf_threshold)
            per_frame_detections.append(detections)

            result_bgr = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
            writer.write(result_bgr)

            frame_idx += 1
            if progress_callback is not None and total_frames > 0:
                progress_callback(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()
        writer.release()

    return frame_idx, fps, per_frame_detections


def reencode_to_h264(input_path: str) -> str:
    """
    Re-encodes an OpenCV-written mp4 (mp4v codec, not browser-playable)
    into H.264 with yuv420p pixel format and +faststart, which browsers
    can actually decode and play inline. Uses the ffmpeg binary bundled
    by imageio-ffmpeg (pip-installable, no system ffmpeg required).
    Falls back to the original file (with a Streamlit warning) if the
    conversion fails, so it degrades gracefully instead of crashing.
    """
    output_path = input_path.replace(".mp4", "_h264.mp4")
    if output_path == input_path:
        output_path = input_path + "_h264.mp4"

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
            capture_output=True,
        )
        return output_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        st.warning(
            f"Could not re-encode video to H.264. Falling back to the raw "
            f"output file, which may not play in your browser. Error: {e}"
        )
        return input_path


def render_synced_video_with_table(video_path, fps, per_frame_detections, height=420):
    """
    Embeds the FINISHED annotated video (expected to already be H.264,
    see reencode_to_h264) next to a detection table that updates live as
    the video plays, synced to the video's own playback time via the JS
    `timeupdate` event (no server round-trip needed).
    """
    with open(video_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode("utf-8")

    dets_json = json.dumps(per_frame_detections)

    html = f"""
    <div style="display:flex; gap:16px; align-items:flex-start;">
      <video id="vid_{id(video_path)}" controls style="width:58%; border-radius:8px; background:#000;">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
      </video>
      <div style="width:42%; max-height:{height}px; overflow-y:auto; font-family:sans-serif; font-size:13px; border:1px solid #333; border-radius:8px; padding:8px;">
        <div id="frame_label_{id(video_path)}" style="font-weight:bold; margin-bottom:6px;">Frame: 0</div>
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr style="text-align:left; border-bottom:1px solid #666;">
              <th style="padding:4px;">Class</th>
              <th style="padding:4px;">Confidence</th>
              <th style="padding:4px;">Box</th>
            </tr>
          </thead>
          <tbody id="det_table_{id(video_path)}"></tbody>
        </table>
      </div>
    </div>
    <script>
      (function() {{
        const fps = {fps};
        const dets = {dets_json};
        const video = document.getElementById("vid_{id(video_path)}");
        const tbody = document.getElementById("det_table_{id(video_path)}");
        const label = document.getElementById("frame_label_{id(video_path)}");

        function renderFrame(idx) {{
          idx = Math.max(0, Math.min(idx, dets.length - 1));
          label.textContent = "Frame: " + idx + " / " + (dets.length - 1);
          const rows = dets[idx] || [];
          if (rows.length === 0) {{
            tbody.innerHTML = '<tr><td colspan="3" style="padding:4px; color:#888;">No detections</td></tr>';
            return;
          }}
          tbody.innerHTML = rows.map(d =>
            `<tr style="border-bottom:1px solid #333;">
               <td style="padding:4px;">${{d.class}}</td>
               <td style="padding:4px;">${{d.confidence.toFixed(2)}}</td>
               <td style="padding:4px;">${{d.box.join(", ")}}</td>
             </tr>`
          ).join("");
        }}

        video.addEventListener("timeupdate", () => {{
          const idx = Math.floor(video.currentTime * fps);
          renderFrame(idx);
        }});

        renderFrame(0);
      }})();
    </script>
    """
    st.components.v1.html(html, height=height + 40, scrolling=False)


# ------------------------------------------------------------------
# Live Webcam - real-time detection via streamlit-webrtc.
#
# webrtc_streamer captures frames from the browser's camera and calls
# our VideoProcessor's recv() for every single frame. We run whichever
# model the user picked and return the annotated frame to be displayed,
# and stash the latest detections so the main thread can poll them into
# a live-updating table next to the video.
#
# NOTE on screen-sharing: streamlit-webrtc only supports the browser
# camera (getUserMedia), not screen capture (getDisplayMedia) - that is
# a limitation of the library itself. To run detection on a screen
# recording, record your screen with your OS's built-in recorder and
# upload the resulting file in the "Upload Image/Video" tab; it goes
# through the exact same parallel processing + synced-table pipeline.
#
# NOTE: model_choice / conf_threshold are captured via the factory
# closure at the moment the WebRTC connection is (re)started. Changing
# the dropdown/slider mid-stream will only take effect once Streamlit
# reruns and rebuilds the connection - this is a streamlit-webrtc
# behavior, not something this code can hot-patch mid-frame.
# ------------------------------------------------------------------


class DetectionVideoProcessor(VideoProcessorBase):
    def __init__(self, model_choice: str, conf_threshold: float):
        self.model_choice = model_choice
        self.conf_threshold = conf_threshold
        self.model = None  # lazily loaded on first frame, inside the worker thread
        self.latest_detections = []  # read by main thread for the live panel

    def _ensure_model(self):
        if self.model is None:
            self.model = load_model_by_name(self.model_choice)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self._ensure_model()
        img = frame.to_image().convert("RGB")  # av.VideoFrame -> PIL.Image
        run_fn = get_run_fn(self.model_choice)

        try:
            result_img, detections, _ = run_fn(self.model, img, self.conf_threshold)
            self.latest_detections = detections
        except Exception:
            # If a single frame fails to process, show the raw frame rather
            # than crashing the whole webcam stream.
            result_img = img

        return av.VideoFrame.from_image(result_img)


def render_live_webcam_tab():
    st.caption(
        "Runs inference on your webcam feed in real time, with a live-updating "
        "detection table beside it. YOLO11m is the fastest and most likely to "
        "keep up with live video; RetinaNet and MobileNetV2 may lag on slower "
        "GPUs/CPUs since every frame is a full forward pass."
    )

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Model", ["YOLO11m", "RetinaNet", "MobileNetV2"], key="live_model_choice")
    with col2:
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05, key="live_conf_threshold")

    st.info(
        "Change settings above, then click Start again to apply them to a new session. "
        "To run detection on a screen recording instead of the camera, record your screen "
        "with your OS's screen recorder and upload it in the 'Upload Image/Video' tab - "
        "the same parallel processing and live table apply there."
    )

    video_col, panel_col = st.columns([2, 1])
    with video_col:
        ctx = webrtc_streamer(
            key="live-detect",
            video_processor_factory=lambda: DetectionVideoProcessor(model_choice, conf_threshold),
            media_stream_constraints={"video": True, "audio": False},
        )

    with panel_col:
        st.markdown("**Live detections**")
        table_slot = st.empty()
        if ctx.state.playing:
            # Poll the processor's latest detections and refresh the panel
            # while the stream is running.
            while ctx.state.playing:
                if ctx.video_processor is not None:
                    dets = ctx.video_processor.latest_detections
                    table_slot.dataframe(
                        dets if dets else [{"info": "No detections yet"}],
                        use_container_width=True,
                    )
                time.sleep(0.3)
        else:
            table_slot.info("Click Start to begin.")


# ------------------------------------------------------------------
# Upload Image/Video tab
# ------------------------------------------------------------------


def render_upload_tab():
    st.caption(
        "Upload an image (runs through all three models) or a video (all three models "
        "process it in parallel in the background; once each finishes, its annotated "
        "video appears below with a detection table that updates live as you play it)."
    )

    uploaded_file = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"],
        key="upload_file",
    )
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05, key="upload_conf_threshold")

    if uploaded_file is None:
        st.info("Upload an image or video to see detection results.")
        return

    file_ext = Path(uploaded_file.name).suffix.lower()

    if file_ext in IMAGE_EXTENSIONS:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        tab1, tab2, tab3 = st.tabs(["RetinaNet", "YOLO11m", "MobileNetV2"])

        with tab1:
            st.subheader("RetinaNet Detection Result")
            try:
                model = load_retinanet()
                result_img, detections, elapsed = run_retinanet(model, image, conf_threshold)
                st.image(result_img, use_container_width=True)
                st.write(f"Inference time: {elapsed:.3f}s")
                if detections:
                    st.table(detections)
                else:
                    st.info("No detections above threshold.")
            except Exception as e:
                st.error(f"Failed to run RetinaNet: {e}")

        with tab2:
            st.subheader("YOLO11m Detection Result")
            try:
                model = load_yolo()
                result_img, detections, elapsed = run_yolo(model, image, conf_threshold)
                st.image(result_img, use_container_width=True)
                st.write(f"Inference time: {elapsed:.3f}s")
                if detections:
                    st.table(detections)
                else:
                    st.info("No detections above threshold.")
            except Exception as e:
                st.error(f"Failed to run YOLO11m: {e}")

        with tab3:
            st.subheader("MobileNetV2 (SSDLite) Detection Result")
            try:
                model = load_mobilenet()
                result_img, detections, elapsed = run_mobilenet(model, image, conf_threshold)
                st.image(result_img, use_container_width=True)
                st.write(f"Inference time: {elapsed:.3f}s")
                if detections:
                    st.table(detections)
                else:
                    st.info("No detections above threshold.")
            except Exception as e:
                st.error(f"Failed to run MobileNetV2: {e}")

    elif file_ext in VIDEO_EXTENSIONS:
        st.video(uploaded_file)
        st.warning(
            "All three models process the video in parallel in the background "
            "(one model finishing doesn't wait for the others). Once each model "
            "is done, its annotated video is re-encoded for browser playback and "
            "appears below with a detection table that updates live as you play it back."
        )

        if st.button("Run detection on video (all 3 models)", key="run_video_button"):
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_in:
                tmp_in.write(uploaded_file.read())
                input_path = tmp_in.name

            models = ["YOLO11m", "RetinaNet", "MobileNetV2"]
            progress = {m: 0.0 for m in models}
            status = {m: "Starting..." for m in models}
            done = {m: False for m in models}
            errors = {m: None for m in models}
            results = {m: None for m in models}  # (num_frames, fps, per_frame_detections)
            output_paths = {m: f"{input_path}_{m}_annotated.mp4" for m in models}

            def worker(model_choice):
                try:
                    def pcb(frac):
                        progress[model_choice] = frac

                    n, fps, dets = process_video(
                        input_path, output_paths[model_choice], model_choice,
                        conf_threshold, progress_callback=pcb,
                    )
                    results[model_choice] = (n, fps, dets)
                    progress[model_choice] = 1.0
                    status[model_choice] = f"Done - {n} frames"
                except Exception as e:
                    errors[model_choice] = str(e)
                    status[model_choice] = "Failed"
                finally:
                    done[model_choice] = True

            # Each model runs on its own thread, fully independently - one
            # finishing early does not block or wait for the others.
            threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in models]
            for t in threads:
                t.start()

            cols = st.columns(3)
            bars, texts = {}, {}
            for m, col in zip(models, cols):
                with col:
                    st.markdown(f"**{m}**")
                    bars[m] = st.progress(0.0)
                    texts[m] = st.empty()

            while not all(done[m] for m in models):
                for m in models:
                    bars[m].progress(min(progress[m], 1.0))
                    texts[m].text(status[m] if done[m] else f"{progress[m] * 100:.0f}%")
                time.sleep(0.3)

            for m in models:
                bars[m].progress(1.0)
                texts[m].text(status[m])

            for t in threads:
                t.join()

            st.markdown("---")
            st.markdown("### Results - annotated video with live-synced detection table")
            for m in models:
                st.markdown(f"#### {m}")
                if errors[m]:
                    st.error(errors[m])
                    continue
                if results[m] is None:
                    continue
                n, fps, dets = results[m]

                # Re-encode the OpenCV mp4v output to H.264 so the <video>
                # tag in render_synced_video_with_table can actually play
                # it (this is the fix for the black/unplayable video box).
                playable_path = reencode_to_h264(output_paths[m])

                render_synced_video_with_table(playable_path, fps, dets)
                with open(playable_path, "rb") as f:
                    st.download_button(
                        f"Download {m} annotated video", f,
                        file_name=f"{m}_annotated_{uploaded_file.name}",
                        mime="video/mp4", key=f"dl_{m}",
                    )

            if os.path.exists(input_path):
                os.remove(input_path)
            # output_paths / h264-reencoded paths are intentionally left on
            # disk so the embedded <video> tags above keep serving them
            # during this session.

    else:
        st.error(f"Unsupported file type: {file_ext}")


# ------------------------------------------------------------------
# Model Comparison page
#
# This page ONLY reads the per-model JSON result files produced by
# running `python 02_Source_Code/evaluation.py` on the command line.
# It never loads a model or runs inference - so opening this page is
# always instant, and results are identical no matter who generated
# them or when.
# ------------------------------------------------------------------


def render_comparison_page():
    st.subheader("Three-Model Evaluation Comparison")
    st.caption(f"Test set: `{TEST_IMAGES_DIR}`")

    if not evaluation._HAS_TORCHMETRICS:
        st.warning(
            "torchmetrics is not installed - mAP / per-class AP could not be "
            "computed when evaluation.py was run. Install it with "
            "`pip install torchmetrics` and re-run evaluation.py if you need "
            "those numbers."
        )

    result_paths = evaluation.get_result_paths(PROJECT_ROOT)
    results, missing = evaluation.load_cached_results(result_paths)

    if missing:
        st.warning(
            "No evaluation result found yet for: **" + ", ".join(missing) + "**.\n\n"
            "Run this from the command line first, then reload this page:\n\n"
            "`python 02_Source_Code/evaluation.py`"
        )
        st.caption("Expected result file locations:")
        for name in missing:
            st.code(result_paths[name])
        if not results:
            return

    st.session_state["eval_results"] = results

    model_names = list(results.keys())

    # ---- 1. Overall metrics: grouped bar chart ----
    st.markdown("### Overall Metrics Comparison")
    metric_rows = []
    for name in model_names:
        r = results[name]
        row = {
            "Model": name,
            "mAP@0.5": r["map_metrics"].get("mAP@0.5"),
            "mAP@0.5:0.95": r["map_metrics"].get("mAP@0.5:0.95"),
            "Precision@0.5conf": r["precision"],
            "Recall@0.5conf": r["recall"],
            "F1@0.5conf": r["f1"],
        }
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows)
    st.dataframe(metrics_df, use_container_width=True)

    melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig_bar = px.bar(
        melted, x="Metric", y="Value", color="Model", barmode="group",
        title="Overall Metrics Comparison (Grouped Bar Chart)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- 2. Inference speed ----
    st.markdown("### Inference Speed Comparison")
    speed_df = pd.DataFrame(
        [{"Model": n, "Avg Latency (ms)": results[n]["avg_latency_sec"] * 1000,
          "FPS": results[n]["fps"]} for n in model_names]
    )
    col1, col2 = st.columns(2)
    with col1:
        fig_latency = px.bar(speed_df, x="Model", y="Avg Latency (ms)",
                              color="Model", title="Average Inference Latency (ms/image, lower is better)")
        st.plotly_chart(fig_latency, use_container_width=True)
    with col2:
        fig_fps = px.bar(speed_df, x="Model", y="FPS", color="Model",
                          title="FPS (higher is better)")
        st.plotly_chart(fig_fps, use_container_width=True)

    # ---- 3. PR curve overlay ----
    st.markdown("### PR Curve (Micro-Averaged Across All Classes)")
    fig_pr = go.Figure()
    for name in model_names:
        pr = results[name]["pr_curve"]
        fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"],
                                     mode="lines", name=name))
    fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision",
                          title="Precision-Recall Curve", xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig_pr, use_container_width=True)

    # ---- 4. Per-class AP ----
    st.markdown("### Per-Class AP Breakdown")
    per_class_rows = []
    for name in model_names:
        for cls_name, ap in results[name]["per_class_ap"].items():
            ap_value = 0.0 if ap is None else ap
            per_class_rows.append({"Model": name, "Class": cls_name, "AP": ap_value})
    if per_class_rows:
        pc_df = pd.DataFrame(per_class_rows)
        pivot = pc_df.pivot(index="Class", columns="Model", values="AP").fillna(0.0)
        st.dataframe(pivot.sort_values(by=model_names[0], ascending=False), use_container_width=True)

        sel_model = st.selectbox("View worst-performing classes for a model (Top 15)", model_names, key="worst_cls_model")

        model_df = pc_df[pc_df["Model"] == sel_model]
        nonzero = model_df[model_df["AP"] > 0]
        zero_classes = sorted(model_df.loc[model_df["AP"] == 0, "Class"].tolist())

        if not nonzero.empty:
            worst = nonzero.sort_values("AP").head(15)
            fig_worst = px.bar(worst, x="AP", y="Class", orientation="h",
                                title=f"{sel_model} - 15 Lowest-AP Classes (AP > 0)")
            st.plotly_chart(fig_worst, use_container_width=True)
        else:
            st.info(f"Every class for {sel_model} has AP = 0 - nothing to plot above zero.")

        st.markdown(f"**Classes with AP = 0 for {sel_model}** ({len(zero_classes)} of {len(model_df)})")
        if zero_classes:
            st.dataframe(pd.DataFrame({"Class": zero_classes}), use_container_width=True)
        else:
            st.success("No zero-AP classes for this model.")
    else:
        st.info("torchmetrics is not installed, so per-class AP is unavailable.")

    # ---- 5. Confusion matrix ----
    st.markdown("### Confusion Matrix")
    sel_cm_model = st.selectbox("Select model", model_names, key="cm_model")
    cm_data = results[sel_cm_model]
    cm = np.array(cm_data["confusion_matrix"])
    labels = cm_data["class_names"] + ["background"]

    # Full matrix is large (63+1 classes) - show top confused pairs
    # plus an optional full heatmap.
    off_diag = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                off_diag.append({"Ground Truth": labels[i], "Predicted As": labels[j], "Count": int(cm[i, j])})
    off_diag_df = pd.DataFrame(off_diag).sort_values("Count", ascending=False)

    st.write(f"Operating threshold (decision boundary): confidence >= {cm_data['operating_conf']}")
    st.markdown("**Most Common Confusions / Misses / False Positives (Top 20)**")
    st.dataframe(off_diag_df.head(20), use_container_width=True)

    if st.checkbox("Show full confusion matrix heatmap (63x63, may be large)", key="show_full_cm"):
        fig_cm = px.imshow(
            cm, x=labels, y=labels, labels=dict(x="Predicted", y="Ground Truth", color="Count"),
            title=f"{sel_cm_model} Confusion Matrix", aspect="auto",
        )
        st.plotly_chart(fig_cm, use_container_width=True)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

st.title("Fruit and Vegetable Object Detection - Model Comparison")

page = st.sidebar.radio("Page", ["🖼️ Image / Video Test", "📊 Model Comparison"])

if page == "🖼️ Image / Video Test":
    tab_upload, tab_webcam = st.tabs(["📁 Upload Image/Video", "🎥 Live Webcam"])

    with tab_upload:
        render_upload_tab()

    with tab_webcam:
        render_live_webcam_tab()

else:
    render_comparison_page()