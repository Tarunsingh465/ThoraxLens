from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .data import make_transform
from .labels import LABELS
from .model import load_checkpoint


def prepare_image(image: Image.Image, image_size: int = 224, normalization: str = "imagenet"):
    return make_transform(image_size, train=False, normalization=normalization)(image.convert("L")).unsqueeze(0)


def predict(image: Image.Image, checkpoint: str, image_size: int | None = None, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(checkpoint, device)
    image_size = image_size or int(meta.get("image_size", 224))
    normalization = meta.get("input_normalization", "imagenet")
    x = prepare_image(image, image_size, normalization).to(device)
    with torch.inference_mode():
        scores = torch.sigmoid(model(x).float())[0].cpu().numpy()
    return sorted(zip(LABELS, map(float, scores)), key=lambda p: p[1], reverse=True)


def gradcam(image: Image.Image, checkpoint: str, target_label: str, image_size: int | None = None, device: str | None = None):
    """Return (scores, 0..1 CAM). The heatmap is exploratory, not a clinically validated explanation."""
    if target_label not in LABELS:
        raise ValueError(f"target_label must be one of {LABELS}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_checkpoint(checkpoint, device)
    image_size = image_size or int(meta.get("image_size", 224))
    normalization = meta.get("input_normalization", "imagenet")
    activations, gradients = {}, {}
    target_layer = model.features.denseblock4
    def forward_hook(_module, _inputs, output): activations["value"] = output
    def backward_hook(_module, _grad_input, grad_output): gradients["value"] = grad_output[0]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    x = prepare_image(image, image_size, normalization).to(device)
    model.zero_grad(set_to_none=True)
    try:
        logits = model(x)
        logits[0, LABELS.index(target_label)].backward()
        act, grad = activations["value"], gradients["value"]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy(); cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        scores = torch.sigmoid(logits.detach().float())[0].cpu().numpy()
        return dict(zip(LABELS, map(float, scores))), cam
    finally:
        h1.remove(); h2.remove()
