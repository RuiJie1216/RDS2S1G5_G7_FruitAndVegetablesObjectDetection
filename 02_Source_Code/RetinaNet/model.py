"""
model.py
========
Builds a RetinaNet (ResNet50 backbone + FPN) detector, starting from
ImageNet+COCO-pretrained weights and replacing only the classification
head so it predicts our NUM_CLASSES fruit/vegetable categories instead
of COCO's 80 classes.

V7 changes:
    - Anchor sizes scaled up ~1.33x to match the actual input resolution
      (MIN_SIZE/MAX_SIZE = 640/800, up from the original 480/640 baseline
      these anchors were originally sized for). NOTE: resolution was kept
      at 640/800 (not raised to 800/960), so the scale factor here is
      tied to config.MIN_SIZE/MAX_SIZE -- if those ever change again,
      these anchor sizes need to be rescaled to match.
    - fg_iou_thresh / bg_iou_thresh lowered from the torchvision
      defaults (0.5 / 0.4) to 0.4 / 0.3, since this dataset has a lot
      of small objects where anchors rarely reach IoU 0.5 with the
      ground-truth box -- the default threshold was starving training
      of positive anchor signal.
    - detections_per_img / topk_candidates raised, since validation
      images can contain many densely-packed objects and the
      torchvision defaults (300 / 1000) risked dropping correct
      candidates before NMS even ran.
"""

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial
import os
import config
from config import NUM_CLASSES, MIN_SIZE, MAX_SIZE

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import sigmoid_focal_loss

# Resolve relative to this file's own directory, not the current
# working directory - otherwise this breaks whenever model.py is
# imported from a script running outside RetinaNet/ (e.g. app.py or
# evaluation.py launched from 02_Source_Code/ or the project root).
CLASS_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "class_weights.pt")

class WeightedRetinaNetClassificationHead(RetinaNetClassificationHead):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("class_weights", class_weights)

    def compute_loss(self, targets, head_outputs, matched_idxs):
        cls_logits = head_outputs["cls_logits"]
        losses = []

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(
            targets, cls_logits, matched_idxs
        ):
            foreground_idxs = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs.numel()

            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs]],
            ] = 1.0

            valid_idxs = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            loss = sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs],
                gt_classes_target[valid_idxs],
                reduction="none",
            )
            loss = loss * self.class_weights.unsqueeze(0)

            losses.append(loss.sum() / max(1, num_foreground))

        return sum(losses) / len(targets)

def build_model(num_classes=NUM_CLASSES, pretrained=True):

    weights = (
        torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        if pretrained else None
    )

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=weights,
        min_size=MIN_SIZE,
        max_size=MAX_SIZE,
        score_thresh=0.01,
        nms_thresh=config.NMS_IOU_THRESHOLD,
        fg_iou_thresh=0.4,
        bg_iou_thresh=0.3,
        detections_per_img=500,
        topk_candidates=2000,
    )

    # Original anchors were sized for 480/640 input. Input stays at
    # 640/800 (~1.33x larger than that baseline), so anchors are scaled
    # up ~1.33x -- NOT 1.5x -- to match the pixel size objects actually
    # appear at after resize.
    anchor_sizes = (
        (8, 10, 13),
        (16, 20, 25),
        (32, 40, 50),
        (64, 80, 101),
        (128, 171, 256),
    )
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model.anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors

    if os.path.exists(CLASS_WEIGHTS_PATH):
        class_weights = torch.load(CLASS_WEIGHTS_PATH)
        print(f"[model] Loaded class weights from {CLASS_WEIGHTS_PATH}")
    else:
        class_weights = torch.ones(num_classes)
        print(f"[model] WARNING: {CLASS_WEIGHTS_PATH} not found, using uniform weights")

    model.head.classification_head = WeightedRetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32),
        class_weights=class_weights,
    )

    return model


def set_backbone_trainable(model, trainable):
    """
    Freezes/unfreezes the ResNet50 backbone. Used for a short warmup
    period at the start of training (see config.FREEZE_BACKBONE_EPOCHS):
    keeping the pretrained backbone frozen for the first few epochs
    stops the randomly-initialized new classification head from sending
    large, destructive gradients back into the pretrained features before
    it has learned anything sensible.
    """
    for param in model.backbone.parameters():
        param.requires_grad = trainable


if __name__ == "__main__":
    m = build_model()
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")