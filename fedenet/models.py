from __future__ import annotations

from typing import Optional, cast

import torch
from torch import Tensor, nn


# --------------------------------------------------------------------------------------
# Anti-aliased downsampling
# --------------------------------------------------------------------------------------
class BlurPool2d(nn.Module):
    """
    Channel-wise low-pass (binomial) filter + strided conv for anti-aliased downsampling.
    """

    def __init__(
        self, 
        channels: int,
        filt_size: int = 3,
        stride: int = 2,
        pad_mode: str = "reflect"
        ):
        
        super().__init__()
        if filt_size not in (3, 5):
            raise ValueError("filt_size must be 3 or 5")
        self.stride = int(stride)
        self.pad_mode = pad_mode

        if filt_size == 3:
            k1d = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float32)
        else:  # 5
            k1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)

        k2d = torch.outer(k1d, k1d)
        k2d = (k2d / k2d.sum()).to(dtype=torch.float32)
        kernel = k2d.view(1, 1, k2d.shape[0], k2d.shape[1]).repeat(channels, 1, 1, 1)

        self.kernel: Tensor
        self.register_buffer("kernel", kernel)
        self.groups = channels
        self.pad = k2d.shape[0] // 2

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            return x
        x = nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=self.pad_mode)
        return nn.functional.conv2d(x, self.kernel, stride=self.stride, groups=self.groups)


# --------------------------------------------------------------------------------------
# Stem (anti-aliased), with an explicit conv over RGB for weight-loading compatibility
# --------------------------------------------------------------------------------------
class StemAA(nn.Module):
    """
    Conv on RGB (conv_rgb, 3->out_ch) + Conv on extra K channels (conv_extra, K->out_ch), summed,
    then BN -> SiLU -> BlurPool2d(stride=2).
    Keeps `conv_rgb` attribute so the loader can inject ImageNet RGB weights.
    """

    def __init__(self, in_ch: int, out_ch: int = 32, filt_size: int = 3):
        super().__init__()
        if in_ch < 3:
            raise ValueError("in_ch must be >= 3 (expects RGB + optional frequency maps).")

        self.out_ch = out_ch
        self.has_extra = in_ch > 3
        k_extra = in_ch - 3

        # RGB path (this is the one the weights loader targets)
        self.conv_rgb = nn.Conv2d(3, out_ch, kernel_size=3, stride=1, padding=1, bias=False)

        # Extra-channels path (optional)
        self.conv_extra : Optional[nn.Conv2d] = None
        if self.has_extra:
            self.conv_extra = nn.Conv2d(
                k_extra, 
                out_ch, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.blur = BlurPool2d(out_ch, filt_size=filt_size, stride=2)

        # Kaiming init
        nn.init.kaiming_normal_(self.conv_rgb.weight, nonlinearity="relu")
        if self.conv_extra is not None:
            nn.init.kaiming_normal_(self.conv_extra.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        # Split into RGB and (optional) extra channels
        x_rgb = x[:, :3]
        y = self.conv_rgb(x_rgb)
        if self.has_extra and self.conv_extra is not None:
            x_extra = x[:, 3:]
            y = y + self.conv_extra(x_extra)  # sum the two paths

        y = self.bn(y)
        y = self.act(y)
        y = self.blur(y)
        return y


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------
class DWConvBlock(nn.Module):
    """Depthwise 3x3 + Pointwise 1x1 + BN + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TinyResidual(nn.Module):
    """Lightweight residual: DW 3x3 -> PW 1x1 -> BN -> SiLU + skip."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )
        conv0 = cast(nn.Conv2d, self.block[0])
        conv1 = cast(nn.Conv2d, self.block[1])
        nn.init.kaiming_normal_(conv0.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(conv1.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


# --------------------------------------------------------------------------------------
# Streams + Head (Dual-Stream Architecture)
# --------------------------------------------------------------------------------------
class SpatialStream(nn.Module):
    """Spatial stream with a downsample in the middle."""

    def __init__(self, in_ch: int = 32):
        super().__init__()
        self.stage = nn.Sequential(
            DWConvBlock(in_ch, 48, stride=1),
            TinyResidual(48),
            DWConvBlock(48, 64, stride=2),  # downsample
            TinyResidual(64),
            DWConvBlock(64, 64, stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.stage(x)


class FrequencyStream(nn.Module):
    """Frequency stream: squeeze then downsample."""

    def __init__(self, in_ch: int = 32, out_ch: int = 8):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, 16, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.SiLU(inplace=True)
        self.dw = DWConvBlock(16, out_ch, stride=2)  # downsample
        nn.init.kaiming_normal_(self.squeeze.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw(x)
        return x


class FedeHead(nn.Module):
    """Concat streams → GAP → MLP → 1 logit."""

    def __init__(self, in_ch: int = 72, hidden: int = 32, p_drop: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )
        lin0 = cast(nn.Linear, self.mlp[0])
        lin1 = cast(nn.Linear, self.mlp[3])
        nn.init.kaiming_normal_(lin0.weight, nonlinearity="relu")
        nn.init.zeros_(lin0.bias)
        nn.init.normal_(lin1.weight, std=1e-3)
        nn.init.zeros_(lin1.bias)

    def forward(self, xs: Tensor, xf: Tensor) -> Tensor:
        x = torch.cat([xs, xf], dim=1)        # (B, in_ch, H, W)
        x = self.pool(x).flatten(1)           # (B, in_ch)
        x = self.mlp(x)                       # (B, 1)
        return x


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class FedeNetTiny(nn.Module):
    """
    Dual-stream tiny network with anti-aliased stem.

    Input:  (B, C, H, W); C = 3 + K (RGB + frequency maps)
    Output: (B, 1) logits
    """

    def __init__(self, in_ch: int = 7):
        super().__init__()
        stem_out = 32
        self.stem = StemAA(in_ch=in_ch, out_ch=stem_out, filt_size=3)
        self.spatial = SpatialStream(in_ch=stem_out)
        self.freq = FrequencyStream(in_ch=stem_out, out_ch=8)   # 64 + 8 = 72
        self.head = FedeHead(in_ch=72, hidden=32, p_drop=0.1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)               # (B,32,H/2,W/2)
        xs = self.spatial(x)           # (B,64, ...)
        xf = self.freq(x)              # (B, 8, ...)
        return self.head(xs, xf)       # (B,1)


# --------------------------------------------------------------------------------------
# Factory kept for backward compatibility with tests
# --------------------------------------------------------------------------------------
def build_model(in_ch: int = 7) -> FedeNetTiny:
    return FedeNetTiny(in_ch=in_ch)
