"""MobileNetV2 multi-scale backbone connected to torchvision's public SSD stack."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as NF
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteHead

ARCHITECTURE = "mobilenetv2_ssdlite_v1"
HIGHRES_ARCHITECTURE = "mobilenetv2_ssdlite_highres_v2"
SUPPORTED_ARCHITECTURES = (ARCHITECTURE, HIGHRES_ARCHITECTURE)
CHECKPOINT_TYPE = "mobilenetv2_object_detector"
DEFAULT_ANCHORS = {
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "scales": [0.04, 0.10, 0.20, 0.38, 0.58, 0.78, 0.96],
    "clip": True,
}
SMALL_OBJECT_ANCHORS = {
    "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "scales": [0.02, 0.06, 0.13, 0.25, 0.42, 0.62, 0.82],
    "clip": True,
}
HIGHRES_SMALL_OBJECT_ANCHORS = {
    "aspect_ratios": [[2], [2], [2, 3], [2, 3], [2, 3], [2], [2]],
    "scales": [0.01, 0.02, 0.06, 0.13, 0.25, 0.42, 0.62, 0.82],
    "clip": True,
}
ANCHOR_PROFILES = {
    "default": DEFAULT_ANCHORS,
    "small-object": SMALL_OBJECT_ANCHORS,
    "small-object-highres": HIGHRES_SMALL_OBJECT_ANCHORS,
}


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super().__init__(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        )


class MobileNetV2MultiScaleBackbone(nn.Module):
    out_channels = [32, 96, 1280, 512, 256, 256]

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        features = mobilenet_v2(weights=weights).features
        self.stage1 = nn.Sequential(*features[:7])
        self.stage2 = nn.Sequential(*features[7:14])
        self.stage3 = nn.Sequential(*features[14:])
        self.extras = nn.ModuleList([
            DepthwiseSeparable(1280, 512), DepthwiseSeparable(512, 256), DepthwiseSeparable(256, 256)
        ])

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        x = self.stage1(x); outputs["0"] = x
        x = self.stage2(x); outputs["1"] = x
        x = self.stage3(x); outputs["2"] = x
        for index, layer in enumerate(self.extras, 3):
            x = layer(x); outputs[str(index)] = x
        return outputs


class MobileNetV2HighResBackbone(nn.Module):
    """Expose a stride-4 MobileNetV2 feature before the existing SSD scales."""

    out_channels = [64, 32, 96, 1280, 512, 256, 256]

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        features = mobilenet_v2(weights=weights).features
        self.stage0 = nn.Sequential(*features[:4])
        self.stage1 = nn.Sequential(*features[4:7])
        self.stage2 = nn.Sequential(*features[7:14])
        self.stage3 = nn.Sequential(*features[14:])
        self.highres_fusion = nn.ModuleDict({
            "detail": nn.Sequential(nn.Conv2d(24, 64, 1, bias=False), nn.BatchNorm2d(64)),
            "semantic": nn.Sequential(nn.Conv2d(32, 64, 1, bias=False), nn.BatchNorm2d(64)),
            "smooth": DepthwiseSeparable(64, 64, stride=1),
        })
        self.highres_activation = nn.ReLU6(inplace=True)
        self.extras = nn.ModuleList([
            DepthwiseSeparable(1280, 512), DepthwiseSeparable(512, 256), DepthwiseSeparable(256, 256)
        ])

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        detail = self.stage0(x)
        x = self.stage1(detail)
        semantic = NF.interpolate(self.highres_fusion["semantic"](x), size=detail.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.highres_activation(self.highres_fusion["detail"](detail) + semantic)
        outputs["0"] = self.highres_fusion["smooth"](fused)
        outputs["1"] = x
        x = self.stage2(x); outputs["2"] = x
        x = self.stage3(x); outputs["3"] = x
        for index, layer in enumerate(self.extras, 4):
            x = layer(x); outputs[str(index)] = x
        return outputs


class ConfigurableLossSSD(SSD):
    """SSD with an opt-in softmax focal classification loss.

    Cross-entropy remains the default so existing checkpoints and runs keep
    their original behavior. Focal loss is applied after SSD's standard hard
    negative selection and therefore does not change matching or inference.
    """

    def __init__(
        self,
        *args: Any,
        classification_loss: str = "cross_entropy",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if classification_loss not in {"cross_entropy", "focal"}:
            raise ValueError(f"Unsupported classification loss: {classification_loss}")
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        if not 0 <= focal_alpha <= 1:
            raise ValueError("focal_alpha must be between 0 and 1")
        self.classification_loss_name = classification_loss
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if self.classification_loss_name == "cross_entropy":
            return super().compute_loss(targets, head_outputs, anchors, matched_idxs)

        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]
        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        for target, box_prediction, anchors_per_image, matched in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            foreground = torch.where(matched >= 0)[0]
            foreground_matches = matched[foreground]
            num_foreground += foreground_matches.numel()
            target_regression = self.box_coder.encode_single(
                target["boxes"][foreground_matches], anchors_per_image[foreground]
            )
            bbox_loss.append(
                NF.smooth_l1_loss(box_prediction[foreground], target_regression, reduction="sum")
            )
            target_classes = torch.zeros(
                (box_prediction.size(0),), dtype=target["labels"].dtype, device=target["labels"].device
            )
            target_classes[foreground] = target["labels"][foreground_matches]
            cls_targets.append(target_classes)

        bbox_loss_tensor = torch.stack(bbox_loss)
        cls_targets_tensor = torch.stack(cls_targets)
        num_classes = cls_logits.size(-1)
        cross_entropy = NF.cross_entropy(
            cls_logits.view(-1, num_classes), cls_targets_tensor.view(-1), reduction="none"
        ).view(cls_targets_tensor.size())
        probability_of_target = torch.exp(-cross_entropy)
        foreground = cls_targets_tensor > 0
        alpha = torch.where(
            foreground,
            cross_entropy.new_tensor(self.focal_alpha),
            cross_entropy.new_tensor(1.0 - self.focal_alpha),
        )
        classification_loss = alpha * (1.0 - probability_of_target).pow(self.focal_gamma) * cross_entropy

        num_negative = self.neg_to_pos_ratio * foreground.sum(1, keepdim=True)
        negative_loss = classification_loss.detach().clone()
        negative_loss[foreground] = -float("inf")
        _, sorted_indices = negative_loss.sort(1, descending=True)
        background = sorted_indices.sort(1)[1] < num_negative
        normalizer = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss_tensor.sum() / normalizer,
            "classification": (
                classification_loss[foreground].sum() + classification_loss[background].sum()
            ) / normalizer,
        }


def build_detector(
    num_foreground_classes: int = 63,
    image_size: int = 320,
    pretrained: bool = True,
    anchor_config: dict[str, Any] | None = None,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    detections_per_img: int = 200,
    topk_candidates: int = 400,
    architecture: str = ARCHITECTURE,
    positive_iou_threshold: float = 0.5,
    negative_to_positive_ratio: float = 3.0,
    classification_loss: str = "cross_entropy",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> SSD:
    if image_size < 256:
        raise ValueError("Detector image size must be at least 256")
    anchors_cfg = dict(DEFAULT_ANCHORS if anchor_config is None else anchor_config)
    if architecture == ARCHITECTURE:
        backbone = MobileNetV2MultiScaleBackbone(pretrained=pretrained)
    elif architecture == HIGHRES_ARCHITECTURE:
        backbone = MobileNetV2HighResBackbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported detector architecture: {architecture}")
    if len(anchors_cfg["aspect_ratios"]) != len(backbone.out_channels):
        raise ValueError(f"Anchor profile has {len(anchors_cfg['aspect_ratios'])} feature maps but {architecture} requires {len(backbone.out_channels)}")
    generator = DefaultBoxGenerator(
        aspect_ratios=anchors_cfg["aspect_ratios"], scales=anchors_cfg["scales"],
        clip=bool(anchors_cfg.get("clip", True)),
    )
    head = SSDLiteHead(
        backbone.out_channels, generator.num_anchors_per_location(), num_foreground_classes + 1,
        norm_layer=nn.BatchNorm2d,
    )
    model = ConfigurableLossSSD(
        backbone, generator, (image_size, image_size), num_foreground_classes + 1,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], head=head,
        score_thresh=score_thresh, nms_thresh=nms_thresh,
        detections_per_img=detections_per_img, topk_candidates=topk_candidates,
        iou_thresh=positive_iou_threshold, neg_to_pos_ratio=negative_to_positive_ratio,
        classification_loss=classification_loss, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
    )
    model.architecture_name = architecture
    model.detector_image_size = image_size
    model.anchor_config = anchors_cfg
    model.training_positive_iou_threshold = positive_iou_threshold
    model.training_negative_to_positive_ratio = negative_to_positive_ratio
    return model


def load_warm_start_weights(model: SSD, checkpoint: dict[str, Any]) -> dict[str, int]:
    """Load a same-architecture checkpoint or migrate v1 scales into HighRes v2."""
    source_architecture = str(checkpoint.get("architecture", ARCHITECTURE))
    target_architecture = str(model.architecture_name)
    source_state = checkpoint["model_state_dict"]
    if source_architecture == target_architecture:
        model.load_state_dict(source_state)
        return {"copied_tensors": len(source_state), "initialized_tensors": 0}
    if source_architecture != ARCHITECTURE or target_architecture != HIGHRES_ARCHITECTURE:
        raise ValueError(f"Unsupported warm-start migration: {source_architecture} -> {target_architecture}")

    target_state = model.state_dict()
    copied = 0
    for source_key, value in source_state.items():
        target_key = source_key
        if source_key.startswith("backbone.stage1."):
            suffix = source_key.removeprefix("backbone.stage1.")
            index_text, remainder = suffix.split(".", 1)
            index = int(index_text)
            target_key = f"backbone.stage0.{index}.{remainder}" if index < 4 else f"backbone.stage1.{index - 4}.{remainder}"
        elif source_key.startswith("head.classification_head.module_list.") or source_key.startswith("head.regression_head.module_list."):
            parts = source_key.split(".")
            parts[3] = str(int(parts[3]) + 1)
            target_key = ".".join(parts)
        if target_key in target_state and target_state[target_key].shape == value.shape:
            target_state[target_key] = value
            copied += 1
    model.load_state_dict(target_state)
    return {"copied_tensors": copied, "initialized_tensors": len(target_state) - copied}


def configure_training_phase(
    model: SSD, phase: int, train_stage2: bool = False, train_full_backbone: bool = False
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.extras.parameters():
        parameter.requires_grad = True
    if hasattr(model.backbone, "highres_fusion"):
        for parameter in model.backbone.highres_fusion.parameters():
            parameter.requires_grad = True
    if phase == 2:
        if train_full_backbone:
            early_stages = [model.backbone.stage1, model.backbone.stage2]
            if hasattr(model.backbone, "stage0"):
                early_stages.insert(0, model.backbone.stage0)
            for stage in early_stages:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
        elif train_stage2:
            for parameter in model.backbone.stage2.parameters():
                parameter.requires_grad = True
        for parameter in model.backbone.stage3.parameters():
            parameter.requires_grad = True
    elif phase != 1:
        raise ValueError("phase must be 1 or 2")


def freeze_pretrained_batch_norm(model: SSD) -> None:
    """Keep ImageNet BatchNorm statistics fixed for small detection batches.

    ``model.train()`` recursively enables every BatchNorm module, regardless of
    ``requires_grad``. Reapply eval mode to the original MobileNetV2 stages
    after every train-mode transition. BatchNorm in the new extras/head remains
    trainable.
    """
    stages = [model.backbone.stage1, model.backbone.stage2, model.backbone.stage3]
    if hasattr(model.backbone, "stage0"):
        stages.insert(0, model.backbone.stage0)
    for stage in stages:
        for module in stage.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def optimizer_groups(
    model: SSD, phase: int, backbone_lr: float, head_lr: float,
    train_stage2: bool = False, train_full_backbone: bool = False,
):
    configure_training_phase(model, phase, train_stage2, train_full_backbone)
    groups = []
    if phase == 2:
        if train_full_backbone:
            backbone_modules = tuple(
                module for module in (
                    getattr(model.backbone, "stage0", None), model.backbone.stage1,
                    model.backbone.stage2, model.backbone.stage3,
                ) if module is not None
            )
        else:
            backbone_modules = (model.backbone.stage2, model.backbone.stage3) if train_stage2 else (model.backbone.stage3,)
        groups.append({"params": [p for module in backbone_modules for p in module.parameters() if p.requires_grad], "lr": backbone_lr, "name": "late_backbone"})
    new_modules = [model.backbone.extras, model.head]
    if hasattr(model.backbone, "highres_fusion"):
        new_modules.append(model.backbone.highres_fusion)
    new_parameters = [p for module in new_modules for p in module.parameters() if p.requires_grad]
    groups.append({"params": new_parameters, "lr": head_lr, "name": "detection_layers"})
    return groups


@torch.no_grad()
def architecture_summary(model: SSD, image_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    tensor = torch.zeros(1, 3, image_size, image_size, device=device)
    features = model.backbone(tensor)
    anchors_per_location = model.anchor_generator.num_anchors_per_location()
    feature_rows = []
    for (name, feature), anchors in zip(features.items(), anchors_per_location):
        feature_rows.append({"name": name, "shape": list(feature.shape), "anchors_per_location": anchors})
    head = model.head(list(features.values()))
    if head["bbox_regression"].shape[-1] != 4:
        raise AssertionError("SSD box regression must end in four offsets")
    if head["cls_logits"].shape[-1] != 64:
        raise AssertionError("SSD classification output must contain background + 63 classes")
    return {
        "input_shape": list(tensor.shape), "features": feature_rows,
        "classification_shape": list(head["cls_logits"].shape),
        "bbox_regression_shape": list(head["bbox_regression"].shape),
    }
