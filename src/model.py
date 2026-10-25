from __future__ import annotations

from typing import Literal
import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights

from .labels import LABELS

# TorchXRayVision's DenseNet121 NIH checkpoint uses this output order.
# It matches the repository's 14 targets, but the order differs.
_XRV_TARGETS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
    "", "", "", "",
]
_LABEL_ALIASES = {"Pleural Thickening": "Pleural_Thickening"}


class ChestXrayModel(nn.Module):
    """DenseNet121 wrapper with interchangeable ImageNet / chest-X-ray initialization."""

    def __init__(self, backbone: str = "imagenet", pretrained: bool = True):
        super().__init__()
        if backbone not in {"imagenet", "xrv-nih"}:
            raise ValueError("backbone must be 'imagenet' or 'xrv-nih'")
        self.backbone_type = backbone

        if backbone == "imagenet":
            weights = DenseNet121_Weights.DEFAULT if pretrained else None
            self.base = densenet121(weights=weights)
            self.base.classifier = nn.Linear(self.base.classifier.in_features, len(LABELS))
            self.register_buffer("label_indices", torch.arange(len(LABELS), dtype=torch.long))
        else:
            try:
                import torchxrayvision as xrv
            except ImportError as exc:
                raise RuntimeError(
                    "The xrv-nih backbone needs TorchXRayVision. Run: pip install torchxrayvision"
                ) from exc
            if pretrained:
                # Domain-pretrained weights. These are downloadable by TorchXRayVision on first use.
                self.base = xrv.models.DenseNet(
                    weights="densenet121-res224-nih", apply_sigmoid=False
                )
            else:
                self.base = xrv.models.DenseNet(
                    weights=None, num_classes=len(_XRV_TARGETS), apply_sigmoid=False
                )
            # For training we need raw logits, not TorchXRayVision's calibrated operating-point scores.
            if hasattr(self.base, "op_threshs"):
                self.base.op_threshs = None
            actual_targets = list(getattr(self.base, "pathologies", _XRV_TARGETS))
            normalised = {name: i for i, name in enumerate(actual_targets) if name}
            indices = []
            for label in LABELS:
                source_label = _LABEL_ALIASES.get(label, label)
                if source_label not in normalised:
                    raise RuntimeError(f"Pretrained checkpoint does not contain target {label!r}")
                indices.append(normalised[source_label])
            self.register_buffer("label_indices", torch.tensor(indices, dtype=torch.long))

    @property
    def features(self):
        # Used by Grad-CAM.
        return self.base.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base(x)
        if self.backbone_type == "xrv-nih":
            logits = logits.index_select(1, self.label_indices)
        return logits


def build_model(pretrained: bool = True, backbone: str = "imagenet") -> ChestXrayModel:
    return ChestXrayModel(backbone=backbone, pretrained=pretrained)


def load_checkpoint(path: str, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a project checkpoint dictionary with model_state_dict metadata")
    state = checkpoint.get("model_state_dict", checkpoint)
    backbone = checkpoint.get("backbone", "imagenet")
    model = build_model(pretrained=False, backbone=backbone)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint
