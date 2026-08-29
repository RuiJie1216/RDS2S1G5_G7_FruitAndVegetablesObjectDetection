"""Native desktop interface for the project's best YOLO fruit detector.

This application is intentionally independent from ``gui_app.py``. It keeps
static images, an OpenCV webcam with exposure control, and portal/MSS desktop
screen sharing in one Tkinter interface using the focal-loss YOLO11m model.

Run the interface:
    python3 02_Source_Code/YOLO/yolo_native_app.py

Run a headless model check:
    python3 02_Source_Code/YOLO/yolo_native_app.py --self-test path/to/image.jpg
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import secrets
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def _screen_portal_helper_main(arguments: list[str]) -> int:
    """Run the Wayland portal/PipeWire bridge using the system Python.

    This function intentionally executes before imports such as Torch and
    Ultralytics, which are installed in the project's Python environment but
    not necessarily in the operating system Python.
    """

    helper_parser = argparse.ArgumentParser(add_help=False)
    helper_parser.add_argument("--screen-portal-helper", action="store_true")
    helper_parser.add_argument("--portal-fps", type=int, default=15)
    helper_parser.add_argument("--portal-check", action="store_true")
    helper_args = helper_parser.parse_args(arguments)
    try:
        import dbus
        import gi
        from dbus.mainloop.glib import DBusGMainLoop

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        if helper_args.portal_check:
            Gst.init(None)
            required = (
                "pipewiresrc",
                "videoconvert",
                "videorate",
                "jpegenc",
                "appsink",
            )
            missing = [
                name for name in required if Gst.ElementFactory.find(name) is None
            ]
            if missing:
                raise RuntimeError(f"Missing GStreamer elements: {', '.join(missing)}")
            print("portal capture dependencies OK")
            return 0

        portal_bus_name = "org.freedesktop.portal.Desktop"
        portal_path = "/org/freedesktop/portal/desktop"
        screencast_interface = "org.freedesktop.portal.ScreenCast"
        request_interface = "org.freedesktop.portal.Request"
        session_interface = "org.freedesktop.portal.Session"

        def wait_for_response(bus, request_path):
            state = {}
            loop = GLib.MainLoop()
            request = bus.get_object(portal_bus_name, request_path)

            def responded(response, results):
                state["response"] = int(response)
                state["results"] = dict(results)
                loop.quit()

            request.connect_to_signal(
                "Response", responded, dbus_interface=request_interface
            )
            loop.run()
            if state.get("response") != 0:
                raise RuntimeError("Screen sharing was cancelled or denied.")
            return state["results"]

        DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        portal = bus.get_object(portal_bus_name, portal_path)
        screencast = dbus.Interface(portal, dbus_interface=screencast_interface)
        token = secrets.token_hex(8)
        create_options = dbus.Dictionary(
            {
                "handle_token": dbus.String(f"yolo_create_{token}"),
                "session_handle_token": dbus.String(f"yolo_session_{token}"),
            },
            signature="sv",
        )
        created = wait_for_response(bus, screencast.CreateSession(create_options))
        session_path = str(created["session_handle"])
        try:
            select_options = dbus.Dictionary(
                {
                    "handle_token": dbus.String(f"yolo_select_{token}"),
                    "types": dbus.UInt32(3),
                    "multiple": dbus.Boolean(False),
                    "cursor_mode": dbus.UInt32(2),
                },
                signature="sv",
            )
            wait_for_response(
                bus, screencast.SelectSources(session_path, select_options)
            )
            start_options = dbus.Dictionary(
                {"handle_token": dbus.String(f"yolo_start_{token}")},
                signature="sv",
            )
            started = wait_for_response(
                bus, screencast.Start(session_path, "", start_options)
            )
            streams = started.get("streams", [])
            if not streams:
                raise RuntimeError("The screen portal returned no video stream.")
            node_id = int(streams[0][0])
            remote = screencast.OpenPipeWireRemote(
                session_path, dbus.Dictionary({}, signature="sv")
            )
            pipewire_fd = remote.take()

            Gst.init(None)
            description = (
                f"pipewiresrc fd={pipewire_fd} path={node_id} "
                "do-timestamp=true keepalive-time=1000 ! "
                f"videoconvert ! videorate ! video/x-raw,format=RGB,"
                f"framerate={max(1, min(30, helper_args.portal_fps))}/1 ! "
                "jpegenc quality=88 ! "
                "appsink name=frames max-buffers=1 drop=true sync=false"
            )
            pipeline = Gst.parse_launch(description)
            sink = pipeline.get_by_name("frames")
            running = True

            def stop(_signum, _frame):
                nonlocal running
                running = False
                pipeline.set_state(Gst.State.NULL)

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Could not start the PipeWire capture pipeline.")
            output = sys.stdout.buffer
            try:
                while running:
                    sample = sink.emit("pull-sample")
                    if sample is None:
                        if running:
                            raise RuntimeError("The PipeWire screen stream ended.")
                        break
                    buffer = sample.get_buffer()
                    ok, mapped = buffer.map(Gst.MapFlags.READ)
                    if not ok:
                        continue
                    try:
                        payload = bytes(mapped.data)
                    finally:
                        buffer.unmap(mapped)
                    output.write(struct.pack(">I", len(payload)))
                    output.write(payload)
                    output.flush()
            finally:
                pipeline.set_state(Gst.State.NULL)
        finally:
            try:
                session = bus.get_object(portal_bus_name, session_path)
                dbus.Interface(session, dbus_interface=session_interface).Close()
            except Exception as close_error:  # noqa: BLE001 - helper is exiting
                print(
                    f"portal session close warning: {close_error}",
                    file=sys.stderr,
                    flush=True,
                )
        return 0
    except Exception as exc:  # noqa: BLE001 - report to the parent application
        print(f"portal capture error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__" and "--screen-portal-helper" in sys.argv:
    raise SystemExit(_screen_portal_helper_main(sys.argv[1:]))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mss import MSS
from PIL import Image, ImageDraw, ImageFont, ImageTk
from torch import nn
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DEPLOY_MODEL = (
    PROJECT_ROOT
    / "04_Trained_Model"
    / "YOLO"
    / "yolo11m_focal_epoch4_best_map50_95.pt"
)
LEGACY_MODEL = APP_DIR / "models" / "yolo_best.pt"
DEFAULT_MODEL = DEPLOY_MODEL if DEPLOY_MODEL.is_file() else LEGACY_MODEL
SUPPORTED_IMAGES = (
    ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
)


class CustomFocalBCE(nn.Module):
    """Compatibility class required by the saved training checkpoint.

    The loss is not executed at inference time.  The original checkpoint was
    saved while this class lived in ``__main__``, so PyTorch must be able to
    resolve the same symbol while deserializing it.
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 1.5):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha.float().reshape(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            pred.float(), target.float(), reduction="none"
        )
        probability = torch.exp(-loss)
        alpha = self.alpha.to(pred.device).expand_as(target)
        return alpha * (1.000001 - probability) ** self.gamma * loss


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass
class InferenceFrame:
    image: Image.Image
    detections: list[Detection]
    inference_ms: float
    source_size: tuple[int, int]


class ScreenSource(Protocol):
    label: str

    def grab(self) -> Image.Image: ...

    def close(self) -> None: ...


class MssScreenSource:
    def __init__(self, screen_index: int):
        self.capture = MSS()
        screens = self.capture.monitors
        if len(screens) <= 1:
            self.capture.close()
            raise RuntimeError("No physical screen was found.")
        if not 1 <= screen_index < len(screens):
            self.capture.close()
            raise RuntimeError(
                f"Screen {screen_index} is unavailable; choose 1-{len(screens) - 1}."
            )
        self.screen = screens[screen_index]
        output = self.screen.get("output", f"screen {screen_index}")
        self.label = f"MSS / {output} / {self.screen['width']}x{self.screen['height']}"

    def grab(self) -> Image.Image:
        shot = self.capture.grab(self.screen)
        return Image.frombytes("RGB", shot.size, shot.rgb)

    def close(self) -> None:
        self.capture.close()


class PortalScreenSource:
    label = "WAYLAND PORTAL / SHARED SOURCE"

    def __init__(self, application_file: Path):
        system_python = Path("/usr/bin/python3")
        if not system_python.is_file():
            raise RuntimeError("Wayland screen sharing requires /usr/bin/python3.")
        self.process = subprocess.Popen(
            [
                str(system_python),
                str(application_file),
                "--screen-portal-helper",
                "--portal-fps",
                "15",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def _read_exact(self, length: int) -> bytes:
        if self.process.stdout is None:
            raise RuntimeError("Screen portal has no video output.")
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self.process.stdout.read(length - len(chunks))
            if not chunk:
                error = ""
                if self.process.stderr is not None:
                    error = self.process.stderr.read().decode("utf-8", errors="replace")
                raise RuntimeError(error.strip() or "The screen-sharing stream ended.")
            chunks.extend(chunk)
        return bytes(chunks)

    def grab(self) -> Image.Image:
        payload_length = struct.unpack(">I", self._read_exact(4))[0]
        if not 100 <= payload_length <= 64 * 1024 * 1024:
            raise RuntimeError("The screen portal returned an invalid frame.")
        with Image.open(io.BytesIO(self._read_exact(payload_length))) as opened:
            return opened.convert("RGB")

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=1.5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=1.0)


def _friendly_label(raw: str) -> str:
    """Use the first dataset alias and normalize its display casing."""

    return raw.split("/", 1)[0].strip().replace("_", " ").title()


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_image(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Draw restrained black-and-white boxes without Ultralytics' bright palette."""

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    scale = max(1.0, min(output.size) / 700)
    line_width = max(2, round(2.2 * scale))
    label_font = _font(max(11, round(13 * scale)), bold=True)

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        box = (round(x1), round(y1), round(x2), round(y2))
        # A white halo keeps the monochrome box visible on dark produce.
        draw.rectangle(box, outline="#f4f4f1", width=line_width + 2)
        draw.rectangle(box, outline="#151515", width=line_width)

        caption = f"{detection.label}  {detection.confidence:.0%}"
        text_box = draw.textbbox((0, 0), caption, font=label_font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        pad_x, pad_y = max(5, line_width * 2), max(3, line_width)
        top = max(0, round(y1) - text_h - pad_y * 2 - 2)
        label_box = (
            round(x1),
            top,
            min(output.width, round(x1) + text_w + pad_x * 2),
            top + text_h + pad_y * 2,
        )
        draw.rounded_rectangle(label_box, radius=4, fill="#151515")
        draw.text(
            (label_box[0] + pad_x, label_box[1] + pad_y - 1),
            caption,
            font=label_font,
            fill="#f4f4f1",
        )
    return output


class YoloDetector:
    """Small inference wrapper shared by the GUI and command-line self-test."""

    def __init__(self, model_path: Path):
        model_path = model_path.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {model_path}")

        # Make historical checkpoints produced by train_yolo.py portable when this module
        # is imported instead of launched as the main script.
        sys.modules["__main__"].CustomFocalBCE = CustomFocalBCE

        self.path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(model_path))
        self.names = self.model.names
        self.lock = threading.Lock()

    @property
    def device_label(self) -> str:
        if self.device.startswith("cuda"):
            return torch.cuda.get_device_name(0)
        return "CPU"

    def predict(self, image: Image.Image, confidence: float) -> InferenceFrame:
        source = image.convert("RGB")
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            result = self.model.predict(
                source=source,
                conf=float(confidence),
                imgsz=640,
                device=self.device,
                verbose=False,
            )[0]
        wall_ms = (time.perf_counter() - started) * 1000

        detections: list[Detection] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=_friendly_label(str(self.names[class_id])),
                        confidence=float(box.conf.item()),
                        box=tuple(float(value) for value in box.xyxy[0].tolist()),
                    )
                )

        speed = getattr(result, "speed", {}) or {}
        inference_ms = float(speed.get("inference", wall_ms))
        return InferenceFrame(
            image=annotate_image(source, detections),
            detections=detections,
            inference_ms=inference_ms,
            source_size=source.size,
        )


class YoloNativeApp:
    BG = "#e9e9e6"
    PANEL = "#f7f7f4"
    INK = "#171717"
    MUTED = "#6f706d"
    LINE = "#c9c9c4"
    WHITE = "#f7f7f4"
    DARK_2 = "#252525"

    def __init__(
        self,
        root: Any,
        model_path: Path,
        camera_index: int = 0,
        screen_index: int = 1,
    ):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.model_path = model_path
        self.camera_index = camera_index
        self.screen_index = screen_index
        self.detector: YoloDetector | None = None
        self.current_confidence = 0.25
        self.auto_exposure = True
        self.exposure_bias = 0.0
        self.source_image: Image.Image | None = None
        self.source_path: Path | None = None
        self.latest_frame: InferenceFrame | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.camera_thread: threading.Thread | None = None
        self.camera_stop = threading.Event()
        self.camera_active = False
        self.screen_thread: threading.Thread | None = None
        self.screen_stop = threading.Event()
        self.screen_source: ScreenSource | None = None
        self.screen_active = False
        self.closing = False
        self.static_job_id = 0
        self.static_debounce: str | None = None
        self.ui_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=3)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="yolo-image"
        )
        self.last_camera_arrival: float | None = None
        self.smoothed_fps = 0.0

        self.root.title("ORCHARD / YOLO DETECTOR")
        self.root.geometry("1360x820")
        self.root.minsize(1040, 680)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.status_var = tk.StringVar(value="Loading focal YOLO11m...")
        self.confidence_var = tk.DoubleVar(value=self.current_confidence)
        self.confidence_text = tk.StringVar(value="25%")
        self.auto_exposure_var = tk.BooleanVar(value=self.auto_exposure)
        self.exposure_bias_var = tk.DoubleVar(value=self.exposure_bias)
        self.exposure_text = tk.StringVar(value="+0.0 EV")
        self.source_var = tk.StringVar(value="NO SOURCE")
        self.mode_var = tk.StringVar(value="STANDBY")
        self.metric_objects = tk.StringVar(value="00")
        self.metric_classes = tk.StringVar(value="00")
        self.metric_latency = tk.StringVar(value="--")
        self.metric_fps = tk.StringVar(value="--")

        self._configure_styles()
        self._build_ui()
        self._set_controls_enabled(False)
        self._show_placeholder(
            "MODEL INITIALIZING", "The best focal-loss checkpoint is loading"
        )

        self.root.bind("<Control-o>", lambda _event: self.open_image())
        self.root.bind("<Control-s>", lambda _event: self.save_result())
        self.root.bind("<Control-d>", lambda _event: self.toggle_screen())
        self.root.bind("<space>", lambda _event: self.toggle_camera())
        self.root.after(30, self._poll_ui_queue)
        threading.Thread(
            target=self._load_model, daemon=True, name="yolo-loader"
        ).start()

    def _configure_styles(self) -> None:
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        style.configure(
            "Mono.Horizontal.TScale",
            background=self.PANEL,
            troughcolor=self.LINE,
            bordercolor=self.PANEL,
            lightcolor=self.INK,
            darkcolor=self.INK,
            sliderthickness=14,
            gripcount=0,
        )

    def _build_ui(self) -> None:
        tk = self.tk
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg=self.BG, height=82)
        header.grid(row=0, column=0, sticky="ew", padx=30, pady=(20, 14))
        header.grid_columnconfigure(1, weight=1)
        tk.Label(
            header,
            text="ORCHARD / 11M",
            bg=self.INK,
            fg=self.WHITE,
            font=("DejaVu Sans Mono", 11, "bold"),
            padx=16,
            pady=9,
        ).grid(row=0, column=0, sticky="w")
        title_block = tk.Frame(header, bg=self.BG)
        title_block.grid(row=0, column=1, sticky="w", padx=18)
        tk.Label(
            title_block,
            text="FRUIT + VEGETABLE DETECTION",
            bg=self.BG,
            fg=self.INK,
            font=("DejaVu Sans", 18, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="native vision console  /  class-balanced focal YOLO",
            bg=self.BG,
            fg=self.MUTED,
            font=("DejaVu Sans Mono", 9),
        ).pack(anchor="w", pady=(3, 0))
        self.header_status = tk.Label(
            header,
            textvariable=self.status_var,
            bg=self.BG,
            fg=self.MUTED,
            font=("DejaVu Sans Mono", 9),
            anchor="e",
        )
        self.header_status.grid(row=0, column=2, sticky="e")

        body = tk.Frame(self.root, bg=self.BG)
        body.grid(row=1, column=0, sticky="nsew", padx=30)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        self._build_controls(body)
        self._build_viewport(body)
        self._build_results(body)

        footer = tk.Frame(self.root, bg=self.BG, height=44)
        footer.grid(row=2, column=0, sticky="ew", padx=30, pady=(12, 18))
        footer.grid_columnconfigure(1, weight=1)
        tk.Label(
            footer,
            textvariable=self.mode_var,
            bg=self.BG,
            fg=self.INK,
            font=("DejaVu Sans Mono", 9, "bold"),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            footer,
            text="CTRL+O OPEN   SPACE CAMERA   CTRL+D SCREEN   CTRL+S EXPORT",
            bg=self.BG,
            fg=self.MUTED,
            font=("DejaVu Sans Mono", 8),
        ).grid(row=0, column=1, sticky="e")

    def _build_controls(self, parent: Any) -> None:
        tk = self.tk
        panel = tk.Frame(parent, bg=self.PANEL, width=230, padx=20, pady=22)
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        panel.grid_propagate(False)

        self._section_label(panel, "01 / INPUT").pack(anchor="w")
        self.open_button = self._button(panel, "OPEN IMAGE", self.open_image, dark=True)
        self.open_button.pack(fill="x", pady=(14, 8))
        self.camera_button = self._button(panel, "START CAMERA", self.toggle_camera)
        self.camera_button.pack(fill="x", pady=(0, 8))
        self.screen_button = self._button(panel, "START SCREEN", self.toggle_screen)
        self.screen_button.pack(fill="x")

        tk.Frame(panel, bg=self.LINE, height=1).pack(fill="x", pady=24)
        row = tk.Frame(panel, bg=self.PANEL)
        row.pack(fill="x")
        self._section_label(row, "02 / CONFIDENCE").pack(side="left")
        tk.Label(
            row,
            textvariable=self.confidence_text,
            bg=self.PANEL,
            fg=self.INK,
            font=("DejaVu Sans Mono", 10, "bold"),
        ).pack(side="right")
        self.confidence_scale = self.ttk.Scale(
            panel,
            from_=0.05,
            to=0.95,
            variable=self.confidence_var,
            command=self._confidence_changed,
            style="Mono.Horizontal.TScale",
        )
        self.confidence_scale.pack(fill="x", pady=(18, 4))
        ticks = tk.Frame(panel, bg=self.PANEL)
        ticks.pack(fill="x")
        tk.Label(ticks, text="05", **self._micro_label()).pack(side="left")
        tk.Label(ticks, text="95", **self._micro_label()).pack(side="right")

        tk.Frame(panel, bg=self.LINE, height=1).pack(fill="x", pady=18)
        exposure_row = tk.Frame(panel, bg=self.PANEL)
        exposure_row.pack(fill="x")
        self._section_label(exposure_row, "03 / EXPOSURE").pack(side="left")
        tk.Label(
            exposure_row,
            textvariable=self.exposure_text,
            bg=self.PANEL,
            fg=self.INK,
            font=("DejaVu Sans Mono", 9, "bold"),
        ).pack(side="right")
        self.auto_exposure_check = tk.Checkbutton(
            panel,
            text="AUTO CAMERA EXPOSURE",
            variable=self.auto_exposure_var,
            command=self._auto_exposure_changed,
            bg=self.PANEL,
            fg=self.INK,
            activebackground=self.PANEL,
            activeforeground=self.INK,
            selectcolor=self.PANEL,
            disabledforeground="#92928d",
            highlightthickness=0,
            borderwidth=0,
            anchor="w",
            font=("DejaVu Sans Mono", 8),
            padx=0,
            pady=0,
        )
        self.auto_exposure_check.pack(fill="x", pady=(10, 5))
        self.exposure_scale = self.ttk.Scale(
            panel,
            from_=-2.0,
            to=2.0,
            variable=self.exposure_bias_var,
            command=self._exposure_changed,
            style="Mono.Horizontal.TScale",
        )
        self.exposure_scale.pack(fill="x", pady=(4, 2))
        exposure_ticks = tk.Frame(panel, bg=self.PANEL)
        exposure_ticks.pack(fill="x")
        tk.Label(exposure_ticks, text="-2 DARK", **self._micro_label()).pack(
            side="left"
        )
        tk.Label(exposure_ticks, text="+2 BRIGHT", **self._micro_label()).pack(
            side="right"
        )

        tk.Frame(panel, bg=self.LINE, height=1).pack(fill="x", pady=18)
        self._section_label(panel, "04 / MODEL").pack(anchor="w")
        model_lines = (
            "YOLO11m + FOCAL\n"
            "63 categories\n"
            "epoch 04 / best mAP95\n\n"
            "mAP50       0.3456\n"
            "mAP50-95    0.2173\n"
            "precision   0.4656\n"
            "recall      0.3546"
        )
        tk.Label(
            panel,
            text=model_lines,
            justify="left",
            anchor="nw",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("DejaVu Sans Mono", 8),
            pady=10,
        ).pack(fill="x")

        spacer = tk.Frame(panel, bg=self.PANEL)
        spacer.pack(fill="both", expand=True)
        self.save_button = self._button(panel, "EXPORT RESULT", self.save_result)
        self.save_button.pack(fill="x")

    def _build_viewport(self, parent: Any) -> None:
        tk = self.tk
        frame = tk.Frame(parent, bg=self.INK, padx=1, pady=1)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        top = tk.Frame(frame, bg=self.INK, height=42)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(1, weight=1)
        tk.Label(
            top,
            text="VIEW / 01",
            bg=self.INK,
            fg=self.WHITE,
            font=("DejaVu Sans Mono", 9, "bold"),
            padx=14,
            pady=12,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            top,
            textvariable=self.source_var,
            bg=self.INK,
            fg="#aaa9a5",
            font=("DejaVu Sans Mono", 8),
            padx=14,
        ).grid(row=0, column=1, sticky="e")

        self.viewport = tk.Label(
            frame,
            bg="#111111",
            fg="#b7b7b2",
            text="",
            font=("DejaVu Sans Mono", 10),
            compound="center",
        )
        self.viewport.grid(row=1, column=0, sticky="nsew")
        self.viewport.bind("<Configure>", lambda _event: self._render_latest())

    def _build_results(self, parent: Any) -> None:
        tk = self.tk
        panel = tk.Frame(parent, bg=self.INK, width=310, padx=18, pady=18)
        panel.grid(row=0, column=2, sticky="ns", padx=(14, 0))
        panel.grid_propagate(False)

        tk.Label(
            panel,
            text="DETECTION OUTPUT",
            bg=self.INK,
            fg=self.WHITE,
            font=("DejaVu Sans Mono", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            panel,
            text="live inference ledger",
            bg=self.INK,
            fg="#969691",
            font=("DejaVu Sans Mono", 8),
        ).pack(anchor="w", pady=(4, 18))

        metrics = tk.Frame(panel, bg=self.INK)
        metrics.pack(fill="x")
        self._metric(metrics, "OBJECTS", self.metric_objects, 0, 0)
        self._metric(metrics, "CLASSES", self.metric_classes, 0, 1)
        self._metric(metrics, "INFER MS", self.metric_latency, 1, 0)
        self._metric(metrics, "VIEW FPS", self.metric_fps, 1, 1)

        tk.Frame(panel, bg="#484845", height=1).pack(fill="x", pady=18)
        self.results_text = tk.Text(
            panel,
            bg=self.INK,
            fg="#deded9",
            insertbackground=self.WHITE,
            borderwidth=0,
            highlightthickness=0,
            font=("DejaVu Sans Mono", 9),
            wrap="none",
            state="disabled",
            cursor="arrow",
            padx=0,
            pady=0,
        )
        self.results_text.pack(fill="both", expand=True)
        self.results_text.tag_configure("muted", foreground="#858580")
        self.results_text.tag_configure("bright", foreground=self.WHITE)
        self.results_text.tag_configure("rule", foreground="#555550")
        self._write_results([])

    def _section_label(self, parent: Any, text: str) -> Any:
        return self.tk.Label(
            parent,
            text=text,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("DejaVu Sans Mono", 8, "bold"),
        )

    def _micro_label(self) -> dict[str, Any]:
        return {
            "bg": self.PANEL,
            "fg": self.MUTED,
            "font": ("DejaVu Sans Mono", 7),
        }

    def _button(self, parent: Any, text: str, command: Any, dark: bool = False) -> Any:
        return self.tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.INK if dark else self.PANEL,
            fg=self.WHITE if dark else self.INK,
            activebackground=self.DARK_2 if dark else "#ddddda",
            activeforeground=self.WHITE if dark else self.INK,
            disabledforeground="#92928d",
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.INK,
            highlightcolor=self.INK,
            font=("DejaVu Sans Mono", 9, "bold"),
            padx=12,
            pady=10,
            cursor="hand2",
        )

    def _metric(
        self, parent: Any, label: str, variable: Any, row: int, column: int
    ) -> None:
        frame = self.tk.Frame(parent, bg=self.DARK_2, padx=10, pady=9)
        frame.grid(row=row, column=column, sticky="nsew", padx=(0, 6), pady=(0, 6))
        parent.grid_columnconfigure(column, weight=1)
        self.tk.Label(
            frame,
            text=label,
            bg=self.DARK_2,
            fg="#8f8f8a",
            font=("DejaVu Sans Mono", 7),
        ).pack(anchor="w")
        self.tk.Label(
            frame,
            textvariable=variable,
            bg=self.DARK_2,
            fg=self.WHITE,
            font=("DejaVu Sans Mono", 14, "bold"),
        ).pack(anchor="w", pady=(2, 0))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.open_button.configure(state=state)
        self.camera_button.configure(state=state)
        self.screen_button.configure(state=state)
        self.confidence_scale.configure(state=state)
        self.auto_exposure_check.configure(state=state)
        self.exposure_scale.configure(state=state)
        self.save_button.configure(
            state="normal" if enabled and self.latest_frame else "disabled"
        )

    def _load_model(self) -> None:
        try:
            detector = YoloDetector(self.model_path)
            self._queue_event("model_ready", detector)
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self._queue_event("error", f"Could not load model\n{exc}")

    def _confidence_changed(self, value: str) -> None:
        self.current_confidence = float(value)
        self.confidence_text.set(f"{self.current_confidence:.0%}")
        if self.source_image is not None and not self.camera_active:
            if self.static_debounce is not None:
                self.root.after_cancel(self.static_debounce)
            self.static_debounce = self.root.after(280, self._submit_static_inference)

    def _auto_exposure_changed(self) -> None:
        self.auto_exposure = bool(self.auto_exposure_var.get())
        self._update_camera_source_label()
        if self.camera_active:
            mode = "automatic" if self.auto_exposure else "locked"
            self.status_var.set(f"Camera exposure {mode}")

    def _exposure_changed(self, value: str) -> None:
        self.exposure_bias = float(value)
        self.exposure_text.set(f"{self.exposure_bias:+.1f} EV")
        self._update_camera_source_label()

    def _update_camera_source_label(self) -> None:
        if not self.camera_active:
            return
        exposure_mode = "AUTO" if self.auto_exposure else "LOCK"
        self.source_var.set(
            f"CAMERA {self.camera_index:02d} / {exposure_mode} / "
            f"{self.exposure_bias:+.1f} EV"
        )

    def open_image(self) -> None:
        from tkinter import filedialog, messagebox

        if self.detector is None:
            return
        if self.camera_active:
            self.stop_camera()
        if self.screen_active:
            self.stop_screen()
        filename = filedialog.askopenfilename(
            title="Open an image", filetypes=SUPPORTED_IMAGES
        )
        if not filename:
            return
        try:
            with Image.open(filename) as opened:
                self.source_image = opened.convert("RGB")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Image error", f"Could not open that image.\n\n{exc}")
            return

        self.source_path = Path(filename)
        source_name = self.source_path.name.upper()
        if len(source_name) > 48:
            source_name = "..." + source_name[-45:]
        self.source_var.set(source_name)
        self.mode_var.set("IMAGE / ANALYZING")
        self.latest_frame = InferenceFrame(
            image=self.source_image.copy(),
            detections=[],
            inference_ms=0,
            source_size=self.source_image.size,
        )
        self._render_latest()
        self._submit_static_inference()

    def _submit_static_inference(self) -> None:
        self.static_debounce = None
        if self.source_image is None or self.detector is None or self.camera_active:
            return
        self.static_job_id += 1
        job_id = self.static_job_id
        source = self.source_image.copy()
        confidence = self.current_confidence
        self.status_var.set(f"Scanning at {confidence:.0%} confidence...")

        future = self.executor.submit(self.detector.predict, source, confidence)

        def completed(done: Any) -> None:
            try:
                self._queue_event("static_result", (job_id, done.result()))
            except Exception as exc:  # noqa: BLE001 - worker boundary
                self._queue_event("error", f"Image inference failed\n{exc}")

        future.add_done_callback(completed)

    def toggle_camera(self) -> None:
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        if self.detector is None or self.camera_active:
            return
        if self.screen_active:
            self.stop_screen()
        self.static_job_id += 1
        self.source_image = None
        self.source_path = None
        self.latest_frame = None
        self.camera_active = True
        self.camera_stop.clear()
        self.last_camera_arrival = None
        self.smoothed_fps = 0.0
        self.camera_button.configure(text="STOP CAMERA")
        self.open_button.configure(state="disabled")
        self.screen_button.configure(state="disabled")
        self._update_camera_source_label()
        self.mode_var.set("CAMERA / CONNECTING")
        self.status_var.set(f"Opening camera {self.camera_index}...")
        self._show_placeholder("CAMERA CONNECTING", "Waiting for the first frame")
        self.camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="yolo-camera"
        )
        self.camera_thread.start()

    def _camera_loop(self) -> None:
        capture = cv2.VideoCapture(self.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            self._queue_event(
                "camera_error",
                f"Camera {self.camera_index} is unavailable or permission was denied.",
            )
            return

        try:
            backend_name = capture.getBackendName()
        except cv2.error:
            backend_name = "CAMERA"
        applied_auto_exposure: bool | None = None
        self._queue_event("camera_open", backend_name)
        failures = 0
        try:
            while not self.camera_stop.is_set() and not self.closing:
                if applied_auto_exposure != self.auto_exposure:
                    self._set_hardware_auto_exposure(capture, self.auto_exposure)
                    applied_auto_exposure = self.auto_exposure
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= 12:
                        self._queue_event(
                            "camera_error", "The camera stopped returning frames."
                        )
                        return
                    continue
                failures = 0
                bias = self.exposure_bias
                frame = self._apply_exposure_bias(frame, bias)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                source = Image.fromarray(rgb)
                if self.detector is None:
                    return
                try:
                    result = self.detector.predict(source, self.current_confidence)
                except Exception as exc:  # noqa: BLE001 - worker boundary
                    self._queue_event("camera_error", f"Camera inference failed: {exc}")
                    return
                self._queue_event("camera_result", result)
        finally:
            capture.release()

    @staticmethod
    def _apply_exposure_bias(frame: np.ndarray, bias: float) -> np.ndarray:
        """Apply a photographic EV-style gain to a BGR camera frame."""

        if abs(bias) < 0.05:
            return frame
        gain = 2.0**bias
        return np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    @staticmethod
    def _set_hardware_auto_exposure(capture: cv2.VideoCapture, enabled: bool) -> None:
        """Best-effort auto-exposure control for common laptop-camera backends.

        OpenCV uses different values for V4L2 and Windows camera backends. When
        manual mode is selected, the current exposure is restored after
        disabling auto mode so the camera locks near its current brightness.
        The software EV bias remains available when a driver ignores this call.
        """

        try:
            backend = capture.getBackendName().upper()
        except cv2.error:
            backend = ""
        current_exposure = capture.get(cv2.CAP_PROP_EXPOSURE)
        if "V4L" in backend:
            auto_value, manual_value = 0.75, 0.25
        else:
            auto_value, manual_value = 1.0, 0.0
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value if enabled else manual_value)
        if not enabled and np.isfinite(current_exposure):
            capture.set(cv2.CAP_PROP_EXPOSURE, current_exposure)

    def stop_camera(self) -> None:
        if not self.camera_active:
            return
        self.camera_active = False
        self.camera_stop.set()
        self.camera_button.configure(text="START CAMERA")
        self.open_button.configure(state="normal" if self.detector else "disabled")
        self.screen_button.configure(state="normal" if self.detector else "disabled")
        self.mode_var.set("CAMERA / STOPPED")
        self.status_var.set("Camera stopped")
        self.source_var.set("NO SOURCE")
        self._show_placeholder(
            "CAMERA OFFLINE", "Open an image or restart the live feed"
        )

    def toggle_screen(self) -> None:
        if self.screen_active:
            self.stop_screen()
        else:
            self.start_screen()

    def start_screen(self) -> None:
        if self.detector is None or self.screen_active:
            return
        if self.camera_active:
            self.stop_camera()
        self.static_job_id += 1
        self.source_image = None
        self.source_path = None
        self.latest_frame = None
        self.screen_active = True
        self.screen_stop.clear()
        self.last_camera_arrival = None
        self.smoothed_fps = 0.0
        self.screen_button.configure(text="STOP SCREEN")
        self.open_button.configure(state="disabled")
        self.camera_button.configure(state="disabled")
        self.source_var.set("AWAITING SCREEN SHARE")
        self.mode_var.set("SCREEN / SELECTING SOURCE")
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            self.status_var.set("Choose a screen or window in the system dialog...")
        else:
            self.status_var.set(f"Opening screen {self.screen_index}...")
        self._show_placeholder(
            "SELECT A SCREEN", "Waiting for desktop capture permission"
        )
        self.screen_thread = threading.Thread(
            target=self._screen_loop, daemon=True, name="yolo-screen"
        )
        self.screen_thread.start()

    def _new_screen_source(self) -> ScreenSource:
        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            return PortalScreenSource(Path(__file__).resolve())
        return MssScreenSource(self.screen_index)

    def _screen_loop(self) -> None:
        source = None
        try:
            source = self._new_screen_source()
            self.screen_source = source
            announced = False
            while not self.screen_stop.is_set() and not self.closing:
                started = time.perf_counter()
                image = source.grab()
                if self.screen_stop.is_set():
                    break
                if not announced:
                    self._queue_event("screen_open", source.label)
                    announced = True
                if self.detector is None:
                    return
                result = self.detector.predict(image, self.current_confidence)
                self._queue_event("screen_result", result)
                remaining = (1.0 / 8.0) - (time.perf_counter() - started)
                if remaining > 0:
                    self.screen_stop.wait(remaining)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if not self.screen_stop.is_set() and not self.closing:
                self._queue_event("screen_error", str(exc))
        finally:
            if source is not None:
                source.close()
            self.screen_source = None

    def stop_screen(self) -> None:
        if not self.screen_active:
            return
        self.screen_active = False
        self.screen_stop.set()
        source = self.screen_source
        if isinstance(source, PortalScreenSource):
            source.close()
        self.screen_button.configure(text="START SCREEN")
        self.open_button.configure(state="normal" if self.detector else "disabled")
        self.camera_button.configure(state="normal" if self.detector else "disabled")
        self.mode_var.set("SCREEN / STOPPED")
        self.status_var.set("Screen capture stopped")
        self.source_var.set("NO SOURCE")
        self._show_placeholder(
            "SCREEN OFFLINE", "Open an image, camera, or new screen share"
        )

    def save_result(self) -> None:
        from tkinter import filedialog, messagebox

        if self.latest_frame is None:
            return
        if self.source_path:
            stem = self.source_path.stem
        elif self.screen_active:
            stem = "screen_capture"
        else:
            stem = "camera_capture"
        filename = filedialog.asksaveasfilename(
            title="Export annotated result",
            initialfile=f"{stem}_detected.jpg",
            defaultextension=".jpg",
            filetypes=(("JPEG image", "*.jpg"), ("PNG image", "*.png")),
        )
        if not filename:
            return
        try:
            extension = Path(filename).suffix.lower()
            if extension == ".png":
                self.latest_frame.image.save(filename, format="PNG")
            else:
                self.latest_frame.image.save(filename, format="JPEG", quality=95)
            self.status_var.set(f"Exported {Path(filename).name}")
        except OSError as exc:
            messagebox.showerror("Export error", f"Could not save the result.\n\n{exc}")

    def _queue_event(self, kind: str, payload: Any) -> None:
        if self.closing:
            return
        live_result = kind in {"camera_result", "screen_result"}
        if live_result and self.ui_queue.full():
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.ui_queue.put_nowait((kind, payload))
        except queue.Full:
            if not live_result:
                self.ui_queue.put((kind, payload))

    def _poll_ui_queue(self) -> None:
        if self.closing:
            return
        try:
            for _ in range(4):
                kind, payload = self.ui_queue.get_nowait()
                self._handle_event(kind, payload)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_ui_queue)

    def _handle_event(self, kind: str, payload: Any) -> None:
        from tkinter import messagebox

        if kind == "model_ready":
            self.detector = payload
            self._set_controls_enabled(True)
            self.status_var.set(f"Ready / {self.detector.device_label}")
            self.mode_var.set("READY / BEST.PT")
            self._show_placeholder(
                "READY FOR INPUT", "Open an image, camera, or shared screen"
            )
        elif kind == "static_result":
            job_id, frame = payload
            if job_id != self.static_job_id or self.camera_active or self.screen_active:
                return
            self._present_frame(frame, camera=False)
            self.mode_var.set("IMAGE / COMPLETE")
            self.status_var.set(
                f"{len(frame.detections)} objects / {frame.inference_ms:.1f} ms inference"
            )
        elif kind == "camera_open":
            if self.camera_active:
                backend = str(payload).upper()[:12]
                self.mode_var.set(f"CAMERA / LIVE / {backend}")
                self.status_var.set(f"Camera {self.camera_index} live")
        elif kind == "camera_result":
            if self.camera_active:
                self._present_frame(payload, camera=True)
        elif kind == "camera_error":
            was_active = self.camera_active
            self.stop_camera()
            if was_active:
                messagebox.showerror("Camera error", str(payload))
        elif kind == "screen_open":
            if self.screen_active:
                self.source_var.set(str(payload).upper()[:58])
                self.mode_var.set("SCREEN / LIVE")
                self.status_var.set("Screen share active")
        elif kind == "screen_result":
            if self.screen_active:
                self._present_frame(payload, camera=True)
        elif kind == "screen_error":
            was_active = self.screen_active
            self.stop_screen()
            if was_active:
                messagebox.showerror("Screen capture error", str(payload))
        elif kind == "error":
            self.status_var.set("ERROR / see message")
            self.mode_var.set("MODEL / ERROR")
            self._show_placeholder("MODEL ERROR", str(payload))
            messagebox.showerror("YOLO error", str(payload))

    def _present_frame(self, frame: InferenceFrame, camera: bool) -> None:
        self.latest_frame = frame
        self.metric_objects.set(f"{len(frame.detections):02d}")
        self.metric_classes.set(f"{len({d.label for d in frame.detections}):02d}")
        self.metric_latency.set(f"{frame.inference_ms:.1f}")
        if camera:
            now = time.perf_counter()
            if self.last_camera_arrival is not None:
                instantaneous = 1.0 / max(0.001, now - self.last_camera_arrival)
                self.smoothed_fps = (
                    instantaneous
                    if self.smoothed_fps == 0
                    else self.smoothed_fps * 0.82 + instantaneous * 0.18
                )
            self.last_camera_arrival = now
            self.metric_fps.set(
                f"{self.smoothed_fps:.1f}" if self.smoothed_fps else "--"
            )
            self.status_var.set(
                f"Live / {len(frame.detections)} objects / {frame.inference_ms:.1f} ms"
            )
        else:
            self.metric_fps.set("--")
        self._write_results(frame.detections)
        self.save_button.configure(state="normal")
        self._render_latest()

    def _render_latest(self) -> None:
        if self.latest_frame is None:
            return
        width = max(200, self.viewport.winfo_width() - 24)
        height = max(180, self.viewport.winfo_height() - 24)
        image = self.latest_frame.image.copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(image)
        self.viewport.configure(image=self.photo_image, text="")

    def _show_placeholder(self, headline: str, detail: str) -> None:
        self.photo_image = None
        self.viewport.configure(
            image="",
            text=f"+----------------------------+\n"
            f"| {headline:^26} |\n"
            f"+----------------------------+\n\n{detail}",
        )
        self.metric_objects.set("00")
        self.metric_classes.set("00")
        self.metric_latency.set("--")
        self.metric_fps.set("--")
        self._write_results([])
        self.save_button.configure(state="disabled")

    def _write_results(self, detections: list[Detection]) -> None:
        text = self.results_text
        text.configure(state="normal")
        text.delete("1.0", "end")
        if not detections:
            text.insert("end", "NO OBJECTS ABOVE THRESHOLD\n", "muted")
            text.insert(
                "end",
                "\nAdjust confidence or provide\na clearer view of the produce.\n",
                "muted",
            )
        else:
            text.insert("end", " ID  LABEL                 CONF\n", "muted")
            text.insert("end", " --------------------------------\n", "rule")
            ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
            for index, detection in enumerate(ordered[:15], start=1):
                label = detection.label[:19]
                line = f" {index:02d}  {label:<19} {detection.confidence:>5.1%}\n"
                text.insert("end", line, "bright")
            if len(ordered) > 15:
                text.insert("end", f"\n + {len(ordered) - 15} more objects\n", "muted")

            counts = Counter(detection.label for detection in detections)
            text.insert("end", "\n SUMMARY\n", "muted")
            text.insert("end", " --------------------------------\n", "rule")
            for label, count in counts.most_common(10):
                text.insert("end", f" {label[:21]:<21} x{count:02d}\n", "bright")
        text.configure(state="disabled")

    def close(self) -> None:
        self.closing = True
        self.camera_active = False
        self.camera_stop.set()
        self.screen_active = False
        self.screen_stop.set()
        if isinstance(self.screen_source, PortalScreenSource):
            self.screen_source.close()
        if self.static_debounce is not None:
            self.root.after_cancel(self.static_debounce)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def run_self_test(model_path: Path, image_path: Path, confidence: float) -> int:
    if not image_path.is_file():
        print(
            json.dumps(
                {"ok": False, "error": f"Image not found: {image_path}"}, indent=2
            )
        )
        return 2
    try:
        detector = YoloDetector(model_path)
        with Image.open(image_path) as opened:
            frame = detector.predict(opened.convert("RGB"), confidence)
        output = {
            "ok": True,
            "model": str(detector.path),
            "device": detector.device_label,
            "class_count": len(detector.names),
            "confidence": confidence,
            "image_size": list(frame.source_size),
            "inference_ms": round(frame.inference_ms, 3),
            "detections": [
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 4),
                    "box": [round(value, 2) for value in detection.box],
                }
                for detection in frame.detections
            ],
        }
        print(json.dumps(output, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - command-line error boundary
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native YOLO fruit and vegetable detector"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="YOLO checkpoint (default: the best focal-loss YOLO11m weight)",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--screen", type=int, default=1, help="MSS screen index used on X11"
    )
    parser.add_argument(
        "--self-test",
        type=Path,
        metavar="IMAGE",
        help="Run one headless inference and print JSON instead of opening the GUI",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence used by --self-test (0.05 to 0.95)",
    )
    args = parser.parse_args()
    if not 0.05 <= args.confidence <= 0.95:
        parser.error("--confidence must be between 0.05 and 0.95")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test is not None:
        return run_self_test(args.model, args.self_test, args.confidence)

    import tkinter as tk

    root = tk.Tk()
    YoloNativeApp(root, args.model, args.camera, args.screen)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
