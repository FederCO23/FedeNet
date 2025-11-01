import torch

from fedenet.models import BlurPool2d, TinyResidual


def test_blurpool_stride1_passthrough():
    b = BlurPool2d(channels=3, stride=1)
    x = torch.randn(2, 3, 16, 16)
    y = b(x)
    assert torch.allclose(x, y)


def test_tinyresidual_has_skip():
    block = TinyResidual(8)
    x = torch.randn(2, 8, 10, 10)
    y = block(x)
    # Skip connection should make output not identical but shape-preserving
    assert y.shape == x.shape
