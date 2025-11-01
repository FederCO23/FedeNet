import torch

import fedenet.hubconf as hub


def test_hubconf_builds_model():
    m = hub.fedenet_tiny(pretrained=False, in_ch=7)
    x = torch.randn(1, 7, 540, 960)
    y = m(x)
    assert y.shape == (1, 1)
