import numpy as np
import torch
from PIL import Image

from fedenet.transforms import AppendFrequencyMaps, NormalizeRGBOnly, ResizePad


def test_resizepad_tensor_to_pil_path():
    # CHW uint8 in 0..255 should go through tensor->PIL branch
    x = torch.zeros(3, 50, 100, dtype=torch.uint8)
    out = ResizePad(540, 960)(x)
    assert isinstance(out, Image.Image)


def test_append_maps_selection():
    img = Image.fromarray(np.zeros((108, 192, 3), dtype=np.uint8))
    x = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    x = x  # (3,H,W)
    x = torch.stack([x])[0]  # just keep as CHW
    # Run as module on BCHW and CHW
    tfm = AppendFrequencyMaps(maps=("sobel", "laplacian"))
    y1 = tfm(x)  # CHW -> CHW+2
    assert y1.shape[0] == 5
    y2 = tfm(x.unsqueeze(0))  # BCHW
    assert y2.shape[1] == 5


def test_normalize_rgb_only_shapes():
    x = torch.rand(7, 12, 20)
    y = NormalizeRGBOnly()(x)
    assert y.shape == x.shape
