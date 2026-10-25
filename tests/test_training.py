import torch
from src.train import reference_weighted_bce


def test_reference_weighted_bce_is_finite_and_differentiable():
    logits = torch.zeros((4, 14), requires_grad=True)
    targets = torch.zeros((4, 14))
    targets[0, 1] = 1
    pos_w = torch.full((14,), 0.9)
    neg_w = torch.full((14,), 0.1)
    loss = reference_weighted_bce(logits, targets, pos_w, neg_w)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
