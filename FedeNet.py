"""
FedeNet — A lightweight CNN for binary classification (flip vs notflip)
integrating fixed frequency maps and blur sensitivity at the input level.

Author: Federico Bessi (federico@bessi.dev)
"""
__version__ = "0.1.0"


__all__ = [
    "TARGET_H", "TARGET_W",
    "ResizePad", "AppendFrequencyMaps", "NormalizeRGBOnly",
    "BlurPool2d", "StemAA", "DWConvBlock", "TinyResidual",
    "SpatialStream", "FrequencyStream", "FedeHead", "FedeNetTiny",
    "load_rgb_pretrained_into_fedenet",
]


# ================================
# Imports and Configuration
# ================================

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torchvision.transforms.functional as TF
from torchvision import models

from typing import Tuple, Union, Sequence

# ================================
# Global Constants
# ================================
# Target size that respects the 16:9 frames but is lighter than 1080x1920
# Downscaling x2
TARGET_H, TARGET_W = 540, 960   # keep multiples of 32


# ================================
# Transform Utilities
# ================================
class ResizePad:
    """Resize preserving aspect ratio, then pad to (out_h, out_w).
    Accepts PIL.Image or CHW Tensor in [0,1] or [0,255]."""
    def __init__(self, out_h=TARGET_H, out_w=TARGET_W, fill=0):
        self.out_h = int(out_h)
        self.out_w = int(out_w)
        self.fill = int(fill)
                
    def __call__(self, img: Image.Image):
        # to PIL for consistent resize+pad behavior
        if torch.is_tensor(img):
            if img.ndim == 3 and img.shape[0] in (1,3):  # CHW -> HWC
                img = TF.to_pil_image(img.clamp(0, 1) if img.max() <= 1.0 else img.byte())
            else:
                raise TypeError("ResizePad expects PIL.Image or CHW tensor with C=1 or 3")
            
        w, h = img.size
        scale = min(self.out_w / w, self.out_h / h)
        new_w, new_h = int(w * scale), int(h * scale)  
        img = TF.resize(img, [new_h, new_w], antialias=True)
        
        pad_w = self.out_w - new_w  # considering that all the dataset images are 1080x1920, no need of padding!
        pad_h = self.out_h - new_h  # we will keep these lines of code for more general applications 
        if pad_w < 0 or pad_h < 0:
            raise ValueError("Negative padding computed. Check target size and input.")
        
        # Pad evenly (left, top, right, bottom)
        pad_left  = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top   = pad_h // 2
        pad_bottom= pad_h - pad_top
        img = TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)
        return img


class AppendFrequencyMaps(torch.nn.Module):
    """
    Compute and append fixed frequency maps to an RGB image.
    Input : PIL.Image or torch.Tensor (C,H,W) in [0,1] or [0,255]
    Output: torch.Tensor (3 + K, H, W) in [0,1] per channel (robust-normalized)
    """
    def __init__(self, maps=("sobel", "laplacian", "highpass", "localvar"),
                 localvar_window=7, robust_pct=0.99):
        super().__init__()
        self.maps = tuple(maps)
        self.k = localvar_window      # the local-variance window size 
        self.robust_pct = robust_pct  # upper quantile used for robust normalization, 99th percentile → reduces outlier influence

        # Fixed 3x3 kernels (registered as buffers for device/precision moves)
        sobel_x = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32)
        laplace = torch.tensor([[[0,1,0],[1,-4,1],[0,1,0]]], dtype=torch.float32)  # 4-neigh
        highpas = torch.tensor([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]], dtype=torch.float32)

        # shape (out_channels=1, in_channels=1, kH, kW)
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0))
        self.register_buffer("laplace", laplace.unsqueeze(0))
        self.register_buffer("highpas", highpas.unsqueeze(0))

    @staticmethod
    def _to_gray_tensor(img):
        # Accept PIL or Tensor. Output float tensor (1,H,W) in [0,1]
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)  # (C,H,W) in [0,1]
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.shape[0] == 3:
            img = TF.rgb_to_grayscale(img)
        elif img.shape[0] == 1:
            pass
        else:
            # Fallback: first channel
            img = img[:1]
        return img.clamp(0, 1)

    @staticmethod
    def _robust_norm01(x, pct=0.99, eps=1e-6):
        # Per-image robust scaling to [0,1] using (0, pct) quantiles to avoid outliers
        q_hi = torch.quantile(x.view(1, -1), torch.tensor(pct, device=x.device))
        q_lo = torch.tensor(0.0, device=x.device)  # lower clamp at 0 since magnitudes are >=0
        x = (x - q_lo).clamp(min=0.0) / (q_hi - q_lo + eps)
        return x.clamp(0, 1)

    def _local_variance(self, g1):
        # g1: (1,1,H,W) in [0,1]; window k (odd)
        k = int(self.k)
        pad = k // 2
        w = torch.ones((1,1,k,k), device=g1.device, dtype=g1.dtype) / (k*k)
        mu  = nnF.conv2d(g1, w, padding=pad)
        mu2 = nnF.conv2d(g1*g1, w, padding=pad)
        var = (mu2 - mu*mu).clamp_min(0.0)  # numeric safety
        return var

    def forward(self, img):
        # Keep original RGB (in [0,1])
        if not torch.is_tensor(img):
            rgb = TF.to_tensor(img)  # (3,H,W) [0,1]
        else:
            rgb = img
            if rgb.dtype != torch.float32:
                rgb = rgb.float()
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            if rgb.ndim == 2:
                rgb = rgb.unsqueeze(0).repeat(3,1,1)
        rgb = rgb.clamp(0,1)

        # Gray tensor for frequency ops: (1,1,H,W)
        g = self._to_gray_tensor(rgb).unsqueeze(0)

        H, W = g.shape[-2:]
        maps_out = []

        if "sobel" in self.maps:
            gx = nnF.conv2d(g, self.sobel_x, padding=1)
            gy = nnF.conv2d(g, self.sobel_y, padding=1)
            mag = torch.sqrt(gx*gx + gy*gy)
            maps_out.append(self._robust_norm01(mag.squeeze(0), self.robust_pct))

        if "laplacian" in self.maps:
            lap = nnF.conv2d(g, self.laplace, padding=1).abs()
            maps_out.append(self._robust_norm01(lap.squeeze(0), self.robust_pct))

        if "highpass" in self.maps:
            hp = nnF.conv2d(g, self.highpas, padding=1).abs()
            maps_out.append(self._robust_norm01(hp.squeeze(0), self.robust_pct))

        if "localvar" in self.maps:
            lv = self._local_variance(g).sqrt()  # std ~ more interpretable
            maps_out.append(self._robust_norm01(lv.squeeze(0), self.robust_pct))

        if not maps_out:
            return rgb  # no-op

        freq = torch.cat(maps_out, dim=0)           # (K,H,W)
        out  = torch.cat([rgb, freq], dim=0)        # (3+K,H,W)
        return out
    

# ================================
# Model Components
# ================================
def _binomial_kernel2d(size: int = 3, device=None, dtype=None):
    """2D separable binomial filter (e.g., [1,2,1] ⊗ [1,2,1])."""
    if size == 1:
        k = torch.tensor([1.0], device=device, dtype=dtype)
    elif size == 3:
        k = torch.tensor([1.0, 2.0, 1.0], device=device, dtype=dtype)
    elif size == 5:
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    else:
        # Fallback: triangular weights
        base = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
        k = (1 - (base.abs() / ((size + 1) / 2))).clamp_min(0)
    k = k / k.sum()
    k2 = torch.outer(k, k)
    return k2

class BlurPool2d(nn.Module):
    """
    Depthwise low-pass filter + strided subsample (anti-aliased downsampling).
    Typically used *after* a conv with stride=1 to replace stride=2.
    """
    def __init__(self, channels: int, filt_size: int = 3, stride: int = 2, pad_mode: str = "reflect"):
        super().__init__()
        self.stride = stride
        self.pad_mode = pad_mode
        k = _binomial_kernel2d(filt_size, dtype=torch.float32)
        k = k[None, None, ...]                       # (1,1,H,W)
        k = k.repeat(channels, 1, 1, 1)              # (C,1,H,W)
        self.register_buffer("kernel", k)
        self.groups = channels
        self.pad = filt_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            return x
        x = nnF.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=self.pad_mode)
        return nnF.conv2d(x, self.kernel, stride=self.stride, groups=self.groups)

class StemAA(nn.Module):
    """
    Compact anti-aliased stem
      in:  (B, C_in, H, W)           (e.g., C_in = 3 + K = 7)
      out: (B, 32,  H/2, W/2)        (Conv3x3 + BN + SiLU, then BlurPool stride=2)
    """
    def __init__(self, in_ch: int, out_ch: int = 32, filt_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)
        self.blur = BlurPool2d(out_ch, filt_size=filt_size, stride=2)

        # Kaiming init for conv
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.blur(x)   # anti-aliased downsample (×1/2)
        return x
        


class DWConvBlock(nn.Module):
    """
    Depthwise 3x3 + Pointwise 1x1 + BN + SiLU
    Optionally downsample with stride=2 on the depthwise.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        nn.init.kaiming_normal_(self.dw.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pw.weight, nonlinearity="relu")

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class TinyResidual(nn.Module):
    """Optional residual if shapes match; otherwise acts as identity."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.block[0].weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.block[1].weight, nonlinearity="relu")

    def forward(self, x):
        return x + self.block(x)


class SpatialStream(nn.Module):
    """
    Very small stack; only 2 downsamples total (including stem’s /2).
    Stem out: (B,32,H/2,W/2)
    """
    def __init__(self, in_ch=32):
        super().__init__()
        self.stage = nn.Sequential(
            # keep resolution (H/2,W/2)
            DWConvBlock(in_ch, 48, stride=1),
            TinyResidual(48),

            # downsample once -> (H/4,W/4)
            DWConvBlock(48, 64, stride=2),
            TinyResidual(64),

            # keep resolution (H/4,W/4)
            DWConvBlock(64, 64, stride=1),
        )

    def forward(self, x):
        return self.stage(x)  # (B,64,H/4,W/4)


class FrequencyStream(nn.Module):
    """
    Input must be the *same* tensor from stem (mixed RGB+freq already).
    We keep it tiny: squeeze → 1 DW block → keep channels tiny.
    """
    def __init__(self, in_ch=32, out_ch=8):
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, 16, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.SiLU(inplace=True)

        # downsample once to match spatial stream scale if needed
        self.dw = DWConvBlock(16, out_ch, stride=2)  # -> (H/4,W/4) if stem was /2

    def forward(self, x):
        x = self.squeeze(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dw(x)
        return x  # (B,8,H/4,W/4)


class FedeHead(nn.Module):
    """
    Late concat (spatial 64 + freq 8 = 72) -> GAP -> small MLP -> 1 logit
    """
    def __init__(self, in_ch=72, hidden=32, p_drop=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1, bias=True),
        )
        nn.init.kaiming_normal_(self.mlp[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[3].weight, std=1e-3)
        nn.init.zeros_(self.mlp[3].bias)

    def forward(self, x_spatial, x_freq):
        x = torch.cat([x_spatial, x_freq], dim=1)  # (B,72,H/4,W/4)
        x = self.pool(x).flatten(1)               # (B,72)
        return self.mlp(x)                        # (B,1)


class FedeNetTiny(nn.Module):
    """End-to-end: StemAA -> SpatialStream + FrequencyStream -> FedeHead.

    Args:
        in_ch: Number of input channels (3 + K frequency maps).

    Shape:
        Input: (B, in_ch, H, W)
        Output: (B, 1) logits
    """
    def __init__(self, in_ch):
        super().__init__()
        self.stem = StemAA(in_ch=in_ch, out_ch=32, filt_size=3)
        self.spatial = SpatialStream(in_ch=32)
        self.freq = FrequencyStream(in_ch=32, out_ch=8)
        self.head = FedeHead(in_ch=72, hidden=32, p_drop=0.1)

    def forward(self, x):
        x = self.stem(x)                # (B,32,H/2,W/2)
        xs = self.spatial(x)            # (B,64,H/4,W/4)
        xf = self.freq(x)               # (B, 8,H/4,W/4)
        logit = self.head(xs, xf)       # (B,1)
        return logit


class NormalizeRGBOnly(nn.Module):
    """Normalize only the first 3 channels (RGB); leave extra channels unchanged."""
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        std  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)
        # buffers make it device-safe and picklable
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("NormalizeRGBOnly expects CHW tensor (per-image).")
        rgb = (x[:3] - self.mean) / self.std
        return torch.cat([rgb, x[3:]], dim=0) if x.shape[0] > 3 else rgb


def create_fedenet_tiny(freq_maps=("sobel","laplacian","highpass","localvar")):
    """Convenience: returns a FedeNetTiny model configured for the given frequency maps."""
    in_ch = 3 + len(freq_maps)
    return FedeNetTiny(in_ch=in_ch)


# ================================================================================================
# Hybrid init: copy RGB stem weights from a pretrained backbone into FedeNetTiny
# ================================================================================================

def load_rgb_pretrained_into_fedenet(model: torch.nn.Module,
                                     backbone: str = "efficientnet_b0",
                                     device: torch.device = torch.device("cpu")):
    """
    Copies the pretrained 3->C_out stem conv weights (RGB only) from a torchvision backbone
    into model.stem.conv (which expects 7 input channels).
    The extra 4 channels are random-initialized (small std).
    """
    model.to(device)
    if not hasattr(model, "stem") or not hasattr(model.stem, "conv"):
        raise AttributeError("Model is missing 'stem.conv' to receive pretrained weights.")

    # Get a pretrained backbone and locate its first conv
    if backbone == "efficientnet_b0":
        try:
            # Newer torchvision
            b = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            # Older API fallback
            b = models.efficientnet_b0(pretrained=True)
        # EfficientNet-B0 first conv: features[0][0]
        conv_rgb = b.features[0][0]  # Conv2d(3, 32, k=3, s=2, p=1)
    elif backbone == "resnet18":
        try:
            b = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            b = models.resnet18(pretrained=True)
        # ResNet18 first conv: conv1 (3,64,7x7). We’ll adapt if out_ch mismatches.
        conv_rgb = b.conv1
    else:
        raise ValueError("backbone must be 'efficientnet_b0' or 'resnet18'")

    w_rgb = conv_rgb.weight.data.to(device)   # shape: (C_out_pre, 3, k, k)

    # Prepare FedeNet stem conv weight (C_out_fede, 7, k, k)
    stem = model.stem
    w_fede = stem.conv.weight.data            # (C_out_fede, 7, k, k)
    C_out_fede, C_in_fede, kH, kW = w_fede.shape
    assert C_in_fede == 7, f"Expected 7 input channels, got {C_in_fede}"

    # If backbone out channels differ, project or slice to match FedeNet
    #    For EfficientNet-B0: C_out_pre = 32 matches our 32 → perfect.
    #    For ResNet18: C_out_pre = 64 → we take the first 32 filters (simple, effective).
    if w_rgb.shape[0] != C_out_fede:
        if w_rgb.shape[0] > C_out_fede:
            w_rgb = w_rgb[:C_out_fede]
        else:
            # If backbone has fewer out channels (unlikely), pad with random filters
            extra = torch.randn(C_out_fede - w_rgb.shape[0], 3, kH, kW, device=device) * 0.01
            w_rgb = torch.cat([w_rgb, extra], dim=0)

    # Build new FedeNet weight: copy RGB into [:, :3], random small init for the 4 extra chans
    w_new = torch.zeros_like(w_fede)
    w_new[:, :3, :, :] = w_rgb                      # pretrained RGB
    w_new[:, 3:, :, :] = torch.randn(C_out_fede, 4, kH, kW, device=device) * 0.01  # new chans

    # Assign back and keep training all layers (fine-tune)
    with torch.no_grad():
        stem.conv.weight.copy_(w_new)
    
    # Quick sanity prints
    print(f"[HybridInit] Loaded RGB weights from {backbone} → FedeNet stem.")
    print(f"[HybridInit] Stem conv weight shape now: {stem.conv.weight.shape} (expects 7 input chans).")


