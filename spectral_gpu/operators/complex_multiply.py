from __future__ import annotations

import os

import torch

from spectral_gpu.cuda import load_extension


def _validate(input_ft: torch.Tensor, weight: torch.Tensor) -> None:
    if input_ft.ndim != 4 or weight.ndim != 4:
        raise ValueError("input_ft must be [batch, in_channels, modes1, modes2] and weight [in, out, modes1, modes2]")
    if not input_ft.is_complex() or not weight.is_complex():
        raise TypeError("complex multiplication requires complex tensors")
    if input_ft.shape[1] != weight.shape[0] or input_ft.shape[2:] != weight.shape[2:]:
        raise ValueError("input and weight frequency dimensions are incompatible")


def complex_multiply_reference(input_ft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference contraction for (a+bi)(c+di) over input channels."""

    _validate(input_ft, weight)
    return torch.einsum("bixy,ioxy->boxy", input_ft, weight)


def complex_multiply_optimized(input_ft: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use the custom CUDA kernel when available, otherwise keep the exact PyTorch path."""

    _validate(input_ft, weight)
    if input_ft.is_cuda and input_ft.dtype == torch.complex64:
        extension = load_extension()
        if extension is not None:
            return extension.complex_mul_forward(input_ft.contiguous(), weight.contiguous())
    return complex_multiply_reference(input_ft, weight)


def backend_name(input_ft: torch.Tensor) -> str:
    if not input_ft.is_cuda:
        return "torch_improved_cpu"
    if os.environ.get("SPECTRAL_GPU_DISABLE_EXTENSION") == "1":
        return "torch_improved_cuda"
    return "custom_complex_cuda" if load_extension() is not None else "torch_reference_cuda"
