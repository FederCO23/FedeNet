import torch

from fedenet import build_model  # or your constructor
from fedenet.weights import load_rgb_pretrained_into_fedenet


def test_loader_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device).eval()
    conv = model.stem.conv_rgb
    before = conv.weight.detach().clone()
    load_rgb_pretrained_into_fedenet(model, device=device)
    after = conv.weight.detach()
    assert (after - before).abs().sum().item() > 0
    x = torch.randn(1, 7, 540, 960, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1)
