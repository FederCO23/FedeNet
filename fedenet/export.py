from __future__ import annotations

from typing import IO, Tuple, Union

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    out_path: Union[str, IO[bytes]] = "fedenet_tiny.onnx",
    input_size: Tuple[int, int, int, int] = (1, 7, 540, 960),
    opset: int = 17,
) -> Union[str, IO[bytes]]:
    """
    Export a model to ONNX. Keeps args simple to avoid torch.export dynamic shape pitfalls.

    Args:
        model: PyTorch module.
        out_path: Destination file path or a binary buffer (BytesIO).
        input_size: (B, C, H, W) dummy input for tracing.
        opset: ONNX opset version (>= 17 recommended by newer PyTorch).

    Returns:
        The same out_path that was provided.
    """
    model = model.eval().cpu()
    x = torch.randn(*input_size)
    args_tuple: tuple[torch.Tensor, ...] = (x,)

    onnx_kwargs = dict(
        opset_version=opset,
        do_constant_folding=True,
        # No input_names/output_names or dynamic_axes/dynamic_shapes
        # to avoid exporter name/key mismatch
    )

    with torch.inference_mode():
        torch.onnx.export(model, args_tuple, out_path, **onnx_kwargs)  # type: ignore[arg-type]

    return out_path
