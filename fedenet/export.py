from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def export_onnx(
    model: nn.Module, 
    path: str = "fedenet_tiny.onnx",
    input_size: Tuple[int, int, int, int] = (1, 7, 540, 960), 
    opset: int = 17
) -> str:
    """
    Export a model to ONNX with dynamic batch axis.

    Args:
        model: PyTorch module (will be eval()'ed and moved to CPU for export).
        path: Output .onnx path.
        input_size: (B, C, H, W) used to create a dummy input for tracing.
        opset: ONNX opset version.

    Returns:
        The output path.
    """
    model = model.eval().cpu()
    x = torch.randn(*input_size)
    
    # NOTE: wrap inputs as a tuple to satisfy mypy: args: Tuple[Any, ...]
    args_tuple: tuple[torch.Tensor, ...] = (x,)
    
    with torch.inference_mode():
        torch.onnx.export(
            model, 
            args_tuple, 
            path,
            input_names=["input"], 
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
        )
    return path