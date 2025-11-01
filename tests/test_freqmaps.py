import numpy as np
from PIL import Image

from fedenet import inference_540x960


def test_append_freq_maps_channels():
    tfm = inference_540x960()
    # 1080x1920 black image
    img = Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8))
    x = tfm(img)  # (C,H,W)
    assert x.shape[0] == 7, f"Expected 7 channels (RGB+4), got {x.shape[0]}"
    assert x.shape[1:] == (540, 960)
