import torch

from fedenet.models import FedeNetTiny
from fedenet.weights import load_rgb_pretrained_into_fedenet


def test_load_rgb_pretrained_no_crash(monkeypatch):
    # Mock torchvision to avoid internet use and speed up
    import types

    class FakeM:
        pass

    fake = types.SimpleNamespace()
    conv = torch.nn.Conv2d(3, 32, 3, padding=1, bias=False)
    fake.state_dict = lambda: {"features.0.weight": conv.weight.detach().clone()}

    def fake_efficient(**kw):
        return fake

    import torchvision.models as tvm

    monkeypatch.setattr(tvm, "efficientnet_b0", fake_efficient, raising=True)

    model = FedeNetTiny(in_ch=7)
    load_rgb_pretrained_into_fedenet(model, backbone="efficientnet_b0", device="cpu")
    # If it didn’t raise, we’re good. Do a tiny forward to be safe:
    x = torch.randn(1, 7, 540, 960)
    y = model(x)
    assert y.shape == (1, 1)
