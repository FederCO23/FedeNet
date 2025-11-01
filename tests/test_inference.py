import torch

from fedenet import build_model


def test_forward_cpu():
    model = build_model().eval()
    x = torch.randn(1, 7, 540, 960)  # dummy input
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1)
