from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import Tensor, nn
from torchvision import transforms

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
TARGET_H, TARGET_W = 540, 960
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------------------
# ResizePad
# --------------------------------------------------------------------------------------
class ResizePad:
    """Resize preserving aspect ratio, then pad to (out_h, out_w). Accepts PIL or CHW Tensor."""

    def __init__(self, out_h: int = TARGET_H, out_w: int = TARGET_W, fill: int = 0):
        self.out_h, self.out_w, self.fill = out_h, out_w, fill

    def __call__(self, img):
        # Accept PIL or Tensor image
        if isinstance(img, torch.Tensor):
            if img.ndim == 3:
                img = TF.to_pil_image(img if img.max() > 1 else (img * 255).byte())
            else:
                raise ValueError("Expected CHW tensor.")

        w, h = img.size
        scale = min(self.out_w / w, self.out_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = TF.resize(img, [new_h, new_w], antialias=True)

        pad_w = self.out_w - new_w
        pad_h = self.out_h - new_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)
        return img


# --------------------------------------------------------------------------------------
# AppendFrequencyMaps (configurable)
# --------------------------------------------------------------------------------------
class AppendFrequencyMaps(nn.Module):
    """
    Append 4 frequency/structure channels to RGB:
      1) Sobel magnitude
      2) Laplacian (3x3)
      3) High-pass (3x3)
      4) Local variance (sqrt) with 5x5 window

    Input:  (C,H,W) or (B,C,H,W) tensor in [0,1] (RGB must be first 3 channels)
    Output: same batch shape, with +4 channels appended (total C+4).
    """

    def __init__(
        self,
        maps: tuple[str, ...] = ("sobel", "laplacian", "highpass", "localvar"),
        localvar_window: int = 7,
        robust_pct: float = 0.99,
    ):
        super().__init__()
        # hyper-params
        self.maps = tuple(maps)
        self.win = int(localvar_window)
        self.robust_pct = float(robust_pct)

        if not (0.0 < self.robust_pct < 1.0):
            raise ValueError("robust_pct must be in (0,1).")

        # typed buffers
        self.box: Tensor
        self.kx: Tensor
        self.ky: Tensor
        self.lap: Tensor
        self.hpf: Tensor
        self.eps: Tensor

        # Box window for local mean/variance
        self.register_buffer(
            "box",
            torch.ones(1, 1, self.win, self.win, dtype=torch.float32) / float(self.win * self.win),
        )
        # Sobel kernels
        self.register_buffer(
            "kx",
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        self.register_buffer(
            "ky",
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        # Laplacian & high-pass kernels
        self.register_buffer(
            "lap",
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "hpf",
            torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(
                1, 1, 3, 3
            ),
        )
        # Epsilon for numerical stability
        self.register_buffer("eps", torch.tensor(1e-12, dtype=torch.float32))

    def _gray(self, x: Tensor) -> Tensor:
        # x: (B,3,H,W) or (3,H,W) in [0,1]
        if x.ndim == 3:
            r, g, b = x[0:1], x[1:2], x[2:3]
        else:
            r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        # ITU-R BT.601 luminance
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _conv3(self, x: Tensor, k: Tensor, groups: int = 1) -> Tensor:
        return F.conv2d(x, k, padding=1, groups=groups)

    def _local_variance(self, xg: Tensor) -> Tensor:
        # xg: (B,1,H,W) or (1,H,W) → make batch
        if xg.ndim == 3:
            xg = xg.unsqueeze(0)
        mean = F.conv2d(xg, self.box, padding=self.win // 2)
        mean2 = F.conv2d(xg * xg, self.box, padding=self.win // 2)
        var = (mean2 - mean * mean).clamp_min(0.0).sqrt()
        return var

    def _robust01(self, t: Tensor) -> Tensor:
        """Scale per-sample to [0,1] using symmetric robust quantiles around the center."""
        # t: (B,1,H,W)
        B, C, H, W = t.shape
        flat = t.reshape(B, -1)
        # symmetric trails: e.g., robust_pct=0.99 -> lo=0.005, hi=0.995
        q_lo = (1.0 - self.robust_pct) * 0.5
        q_hi = 1.0 - q_lo
        lo = torch.quantile(flat, q_lo, dim=1, keepdim=True)
        hi = torch.quantile(flat, q_hi, dim=1, keepdim=True)
        den = (hi - lo).clamp_min(1e-6)
        scaled = (flat - lo) / den
        return scaled.reshape(B, C, H, W).clamp_(0.0, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        single = False
        if x.ndim == 3:
            x = x.unsqueeze(0)  # -> (1,C,H,W)
            single = True
        assert x.shape[1] >= 3, "Expected at least 3 channels (RGB) first."

        x_rgb = x[:, :3]
        xg = self._gray(x_rgb)  # (B,1,H,W)

        # Sobel magnitude
        gx = self._conv3(xg, self.kx)
        gy = self._conv3(xg, self.ky)
        sobel_mag = torch.sqrt(gx * gx + gy * gy + self.eps)

        # Laplacian
        lap = self._conv3(xg, self.lap)

        # High-pass
        hpf = self._conv3(xg, self.hpf)

        # Local variance (sqrt)
        lvar = self._local_variance(xg)

        # Assemble selected maps with robust normalization
        maps_out: list[Tensor] = []
        if "sobel" in self.maps:
            maps_out.append(self._robust01(sobel_mag))
        if "laplacian" in self.maps:
            maps_out.append(self._robust01(lap))
        if "highpass" in self.maps:
            maps_out.append(self._robust01(hpf))
        if "localvar" in self.maps:
            maps_out.append(self._robust01(lvar))

        if maps_out:
            freq = torch.cat(maps_out, dim=1)  # (B, K, H, W)
            base = x[:, :3]
            out = torch.cat([base, freq], dim=1)  # (B, C+K, H, W)
        else:
            out = x  # nothing selected -> passthrough

        if single:
            out = out.squeeze(0)
        return out


# --------------------------------------------------------------------------------------
# Normalizer kept here for completeness with factories below
# --------------------------------------------------------------------------------------
class NormalizeRGBOnly(nn.Module):
    """Normalize only the first 3 channels with ImageNet stats."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        mean_t: Tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t: Tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)

    def forward(self, x: Tensor) -> Tensor:
        from typing import cast

        mean = cast(Tensor, self.mean)
        std = cast(Tensor, self.std)
        x = x.clone()
        x[:3] = (x[:3] - mean) / std
        return x


# --------------------------------------------------------------------------------------
# Factories with knobs exposed
# --------------------------------------------------------------------------------------
def inference_540x960(
    maps: tuple[str, ...] = ("sobel", "laplacian", "highpass", "localvar"),
    localvar_window: int = 7,
    robust_pct: float = 0.99,
):
    return transforms.Compose(
        [
            ResizePad(TARGET_H, TARGET_W, fill=0),
            transforms.ToTensor(),  # [0,1]
            AppendFrequencyMaps(),  # TODO: add real frequency maps later
            AppendFrequencyMaps(
                maps=maps,
                localvar_window=localvar_window,
                robust_pct=robust_pct,
            ),
            NormalizeRGBOnly(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_540x960(
    maps: tuple[str, ...] = ("sobel", "laplacian", "highpass", "localvar"),
    localvar_window: int = 7,
    robust_pct: float = 0.99,
):
    return transforms.Compose(
        [
            ResizePad(TARGET_H, TARGET_W, fill=0),
            transforms.RandomRotation(degrees=5, fill=0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            AppendFrequencyMaps(maps=maps, localvar_window=localvar_window, robust_pct=robust_pct),
            NormalizeRGBOnly(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
