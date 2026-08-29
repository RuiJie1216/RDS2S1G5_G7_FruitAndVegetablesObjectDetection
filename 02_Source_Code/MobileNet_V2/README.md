# MobileNetV2-SSDLite Fruit and Vegetable Detector

This project is a true multi-class object detector. It consumes a complete RGB image and predicts zero or more bounding boxes, class IDs, class names, and confidence scores. Ground-truth YOLO boxes are used only as training/evaluation targets—never to crop inference inputs.

## Architecture

`torchvision.models.mobilenet_v2` with `MobileNet_V2_Weights.IMAGENET1K_V2` is the principal backbone. Intermediate MobileNetV2 maps at 1/8, 1/16, and 1/32 input resolution feed three additional depthwise-separable stride-2 blocks. A public torchvision `SSDLiteHead` predicts 64 logits (background plus 63 foreground classes) and four box offsets per default box. Public `SSD` and `DefaultBoxGenerator` code performs matching, hard-negative mining, MultiBox classification/regression loss, decoding, clipping, confidence filtering, and NMS.

At 320×320 the six maps are normally 40×40 (32 channels), 20×20 (96), 10×10 (1280), 5×5 (512), 3×3 (256), and 2×2 (256). Small default boxes begin at 4% of the input size. Use `--image-size 512` for a higher-resolution experiment without changing code.

## Dataset and labels

The loader reads full images plus all YOLO rows (`class_id x_center y_center width height`) and converts normalised coordinates to clipped absolute `xyxy` boxes. Only non-finite/malformed or non-positive-area boxes are rejected; there is no 16-pixel minimum. Random horizontal flip updates boxes, and colour jitter is photometric only. Validation/test transforms are deterministic.

`data/LVIS_Fruits_And_Vegetables/data.yaml` is authoritative. All original 63 case-sensitive IDs are preserved, including separate `Strawberry`/`strawberry` and `Tomato`/`tomato`. Dataset ID 0 maps to internal detector label 1; label 0 is background. Physical splits remain train, val, and test. Training and smoke tests instantiate only train and val.

## Environment and checks

```powershell
cd D:\ai
.\.venv\Scripts\python.exe train.py --check-only --device cuda
.\.venv\Scripts\python.exe test.py --device cuda
```

The test performs full-image dataset checks, class-aware metric tests, one forward/backward detector batch, and eval-output validation without opening test data.

## Training and resume

```powershell
.\.venv\Scripts\python.exe train.py --device cuda --batch-size 4 --image-size 320 --phase1-epochs 10 --phase2-epochs 20
.\.venv\Scripts\python.exe train.py --device cuda --resume runs\<run>\mobilenetv2_detector_latest.pt
```

Phase 1 freezes MobileNetV2 and trains the extra feature blocks plus SSDLite head. Phase 2 unfreezes the late MobileNetV2 stage with a lower backbone LR. AMP, gradient clipping, ReduceLROnPlateau, exact-state checkpoints, deterministic seed, and optional early stopping are supported. Early stopping is disabled by default; opt in with `--early-stopping-patience N`.

Before formal training, run the isolated smoke workflow:

```powershell
.\.venv\Scripts\python.exe train.py --smoke-test --device cuda --batch-size 2 --phase1-epochs 1 --phase2-epochs 0
```

Each run writes `config.json`, `class_names.json`, `architecture_summary.json`, `augmentation_preview.png`, history CSV/JSON, curves, latest/best checkpoints, validation metrics, per-class metrics, PR data, predictions, a confusion matrix, and annotated validation samples. Best selection uses validation mAP@0.5:0.95.

## Evaluation

Validation is safe for model selection. Test evaluation is explicit only:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model runs\<run>\mobilenetv2_detector_best.pt --split val --device cuda
.\.venv\Scripts\python.exe evaluate.py --model runs\<run>\mobilenetv2_detector_best.pt --split test --device cuda
```

Artifacts include class-aware Precision, Recall, F1 at IoU 0.5, mean IoU of class-correct true positives, Detection Rate @0.5, mAP@0.5, mAP@0.5:0.95, per-class AP/PR, prediction JSON, parameter counts, checkpoint bytes, and warmed batch-1 inference time/FPS. AP uses predictions retained at `--ap-score-threshold 0.001`; assignment Precision/Recall/F1 and normal inference use `--confidence 0.25`. No classification metric is relabelled as detection performance.

Exact resume restores Python, NumPy, Torch, and CUDA RNG state and reuses saved phase boundaries and important hyperparameters. A conflicting explicitly supplied option is rejected unless `--allow-resume-overrides` is supplied; accepted overrides are printed and saved in the new configuration. Pretrained MobileNetV2 BatchNorm running statistics stay frozen in both phases for stability with small detector batches.

## Full-image inference and interfaces

```powershell
.\.venv\Scripts\python.exe inference.py example.jpg --model runs\<run>\mobilenetv2_detector_best.pt --device cuda --confidence 0.25 --nms-threshold 0.45
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe gui_app.py
```

CLI JSON and both interfaces show all accepted classes, confidences, and original-image `xyxy` coordinates and save an annotated full image. Set `MOBILENET_DETECTOR_MODEL` to select the default checkpoint.

## Comparison and limitations

```powershell
.\.venv\Scripts\python.exe run_manager.py --split val --include-results path\to\other_model_results.json
```

Comparison exports treat MobileNetV2-SSDLite, RetinaNet, and YOLO as object detectors only when actual localisation metrics exist. Historical crop-classifier results may remain on disk but are ignored by automatic detector checkpoint discovery and marked legacy when imported.

No trained detector metric is included in source control. A smoke checkpoint validates plumbing but is not a formal result. Full training is computationally substantial, and 320×320 may remain challenging for very small objects; use the supported 512 experiment and report its resolution rather than mixing results.
