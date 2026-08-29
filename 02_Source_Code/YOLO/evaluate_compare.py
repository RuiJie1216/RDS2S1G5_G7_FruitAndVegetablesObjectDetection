"""
evaluate_compare.py
Compares the three algorithms (CNN / MobileNetV2 / YOLOv8):
- CNN, MobileNetV2: accuracy / precision / recall / F1 on the classification
  validation set.
- YOLOv8: mAP50 / mAP50-95 / precision / recall via ultralytics' own val().
Writes a markdown comparison report (report_comparison.md) you can paste
straight into an assignment writeup.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

DATA_DIR = "dataset/classification"
IMG_SIZE = (128, 128)


def evaluate_keras_model(model_path, name):
    model = tf.keras.models.load_model(model_path)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val", image_size=IMG_SIZE, batch_size=32, shuffle=False)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    print(f"\n[{name}] accuracy={acc:.4f} precision={precision:.4f} "
          f"recall={recall:.4f} f1={f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1}


def evaluate_yolo(weights_path="models/yolo_best.pt", data_yaml="dataset/yolo_format/data.yaml"):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)

    result = {
        "model": "YOLOv8",
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    print(f"\n[YOLOv8] mAP50={result['mAP50']:.4f} mAP50-95={result['mAP50-95']:.4f} "
          f"precision={result['precision']:.4f} recall={result['recall']:.4f}")
    return result


def write_report(cnn_result, mobilenet_result, yolo_result):
    report = f"""# Algorithm Comparison Report

## 1. Classification algorithms (CNN vs MobileNetV2)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Custom CNN | {cnn_result['accuracy']:.4f} | {cnn_result['precision']:.4f} | {cnn_result['recall']:.4f} | {cnn_result['f1']:.4f} |
| MobileNetV2 (transfer learning) | {mobilenet_result['accuracy']:.4f} | {mobilenet_result['precision']:.4f} | {mobilenet_result['recall']:.4f} | {mobilenet_result['f1']:.4f} |

## 2. Object detection (YOLOv8)

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLOv8 | {yolo_result['mAP50']:.4f} | {yolo_result['mAP50-95']:.4f} | {yolo_result['precision']:.4f} | {yolo_result['recall']:.4f} |

## 3. Method comparison and analysis

**Custom CNN**
- Advantages: simple architecture, few parameters, trained fully from
  scratch -- good for demonstrating core CNN concepts (convolution,
  pooling, fully-connected layers).
- Disadvantages: no pretrained knowledge, so it overfits more easily on
  small datasets and typically scores lower than transfer learning.
- Characteristics: single-label classification only -- cannot locate a
  food within the image or handle multiple foods in one photo.

**MobileNetV2 (transfer learning)**
- Advantages: ImageNet-pretrained weights mean faster convergence and
  usually higher accuracy; lightweight enough for mobile/edge deployment.
- Disadvantages: fine-tuning with too high a learning rate can damage the
  pretrained features; still just a classifier, so it can't localize food
  either.
- Characteristics: works well as a secondary classifier to double-check
  crops that YOLO has already located.

**YOLOv8 (object detection)**
- Advantages: performs localization (bounding box) and classification
  together, detects multiple foods in a single photo, and runs fast --
  matching the real-world use case of a plate with several foods on it.
- Disadvantages: needs bounding-box-annotated data, which costs more to
  label than plain classification data; small or overlapping foods are
  more likely to be missed or misidentified.
- Characteristics: the core detector for this system; the CNN and
  MobileNetV2 classifiers exist to compare and cross-check results, not
  to replace it.

## Conclusion

For "detect multiple foods in one photo and analyze their nutrition",
YOLOv8's detection capability is essential -- a pure classifier cannot
localize food. The CNN vs MobileNetV2 comparison illustrates the
difference between training from scratch and transfer learning on a
small dataset: MobileNetV2 usually wins on accuracy and training
efficiency, while the custom CNN is more useful for explaining how a
convolutional network works from first principles.
"""
    with open("report_comparison.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\nComparison report written to: report_comparison.md")


def main():
    cnn_result = evaluate_keras_model("models/cnn_food_classifier.h5", "Custom CNN")
    mobilenet_result = evaluate_keras_model("models/mobilenet_food_classifier.h5", "MobileNetV2")
    yolo_result = evaluate_yolo()
    write_report(cnn_result, mobilenet_result, yolo_result)


if __name__ == "__main__":
    main()
