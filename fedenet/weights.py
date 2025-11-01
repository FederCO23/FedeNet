from __future__ import annotations

import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


@torch.no_grad()
def load_rgb_pretrained_into_fedenet(
    model: torch.nn.Module,
    backbone: str = "efficientnet_b0",
    device: str | torch.device | None = None,
) -> None:
    """
    Load pretrained RGB conv weights from EfficientNet-B0 into model.stem.conv_rgb (3x3).
    """
    backbone = backbone.lower()
    if backbone not in {"efficientnet_b0", "effnet_b0"}:
        raise ValueError(f"Unsupported backbone '{backbone}'. Only 'efficientnet_b0' is supported.")

    m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    w_rgb = m.features[0][0].weight  # (32,3,3,3)

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    stem = getattr(model, "stem", None)
    if stem is None or not hasattr(stem, "conv_rgb"):
        raise AttributeError("Expected model.stem.conv_rgb to exist.")

    conv = stem.conv_rgb
    if conv.in_channels != 3 or conv.kernel_size != (3, 3):
        raise ValueError(f"Expected Conv2d(in_ch=3, k=3). Got {conv}.")

    src_out, dst_out = w_rgb.shape[0], conv.weight.shape[0]
    if src_out > dst_out:
        w_rgb = w_rgb[:dst_out]
    elif src_out < dst_out:
        reps = (dst_out + src_out - 1) // src_out
        w_rgb = w_rgb.repeat(reps, 1, 1, 1)[:dst_out]

    conv.weight.copy_(w_rgb.to(device=device, dtype=conv.weight.dtype))
    if conv.bias is not None:
        conv.bias.zero_()
