import torch
from src.model import build_model


def test_imagenet_densenet_forward_shape():
    model = build_model(pretrained=False, backbone="imagenet").eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 14)
