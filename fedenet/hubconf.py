# fedenet/hubconf.py
from __future__ import annotations

from typing import Any, Union

from torch import device as torch_device

from fedenet import build_model
from fedenet.weights import load_rgb_pretrained_into_fedenet


def fedenet_tiny(
    pretrained: bool = False,
    in_ch: int = 7,
    map_location: Union[str, torch_device] = "cpu",
    **kwargs: Any,
):
    """
    Torch Hub entrypoint for FedeNetTiny.

    Args:
        pretrained: If True, loads RGB ImageNet weights into the stem (when available).
        in_ch:      Number of input channels (default 7 = RGB + 4 frequency maps).
        map_location: Device or device string to place the model on.
        **kwargs:   Forwarded to build_model / FedeNetTiny (e.g., stem widths, etc.)
    """
    model = build_model(in_ch=in_ch, **kwargs)
    model = model.to(map_location)

    if pretrained:
        # Your existing loader already accepts (model, device=...)
        load_rgb_pretrained_into_fedenet(model, device=map_location)

    model.eval()
    return model
