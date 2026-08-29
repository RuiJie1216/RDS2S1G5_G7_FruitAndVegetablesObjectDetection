# Insurance Scam Detection — Model Comparison UI

A Streamlit app that runs an uploaded claim image through three trained
models — **RetinaNet**, **YOLO11m**, and **MobileNetV2** — and shows each
model's result in its own tab.

## Folder structure expected

```
Trained_Model/
├── RetinaNet/
│   └── retinanet_best.pt
├── YOLO/
│   └── yolo11m_focal_epoch4_best_map50_95.pt
└── Mobilenet_V2/
    └── mobilenetv2_detector_best.pt
app.py
requirements.txt
```

Place `Trained_Model/` in the same directory as `app.py` (or update the
`MODEL_DIR` / `*_PATH` variables at the top of `app.py` to point elsewhere).

## Installation

```bash
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
## Windows
venv\Scripts\activate       #Mac/Linux: source venv/bin/activate      

## If using Visual Studio Code, use below command to enter venv stage
.\venv\Scripts\Activate.ps1
pip install -r 05_Installation_and_User_Guide/requirements.txt
```

Requires Python 3.9+ and, ideally, a CUDA-capable GPU (the app falls back
to CPU automatically if none is found).

## Running the app

```bash
streamlit run 02_Source_Code/app.py
```

This opens the UI in your browser (default: `http://localhost:8501`).

## Usage

1. Upload a claim image (`.jpg`, `.jpeg`, or `.png`).
2. Adjust the confidence threshold in the sidebar if needed (applies to
   RetinaNet and YOLO11m).
3. Switch between the **RetinaNet**, **YOLO11m**, and **MobileNetV2** tabs
   to compare each model's output — detected boxes, class labels, and
   confidence scores.

## Configuration

At the top of `app.py`:

| Variable | Purpose |
|---|---|
| `MODEL_DIR`, `RETINANET_PATH`, `YOLO_PATH`, `MOBILENET_PATH` | Paths to each model's checkpoint file |
| `CLASS_NAMES` | List of class labels, in the same order used during training |
| `DEVICE` | Auto-detects GPU (`cuda`) vs CPU |

## Notes / assumptions

- **RetinaNet** is loaded as torchvision's `retinanet_resnet50_fpn`. If your
  checkpoint was trained with a different implementation, update
  `load_retinanet()` accordingly.
- **YOLO11m** is loaded directly via `ultralytics.YOLO(path)` — no
  architecture assumptions needed.
- **MobileNetV2** is treated as a classifier (backbone + linear head)
  rather than an object detector, since torchvision has no built-in
  MobileNetV2 detection head. If your model actually outputs bounding
  boxes, `run_mobilenet()` needs to be rewritten to match.

## Troubleshooting

- **`FileNotFoundError` for a model path** — check that the folder
  structure above matches, or update the path variables in `app.py`.
- **`RuntimeError` on `load_state_dict`** — the model architecture defined
  in code doesn't match the checkpoint; confirm the architecture used
  during training.
- **Ultralytics import error** — run `pip install ultralytics` (included
  in `requirements.txt`).
