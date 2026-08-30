from __future__ import annotations

import os

import torch
from torch import nn

from spectral_gpu.cuda import load_extension

from .complex_multiply import complex_multiply_optimized
from .spectral_conv_reference import SpectralConv2dReference, _validate_input


class SpectralConv2dOptimized(SpectralConv2dReference):
    """Drop-in layer that swaps the channel contraction for the optional CUDA kernel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_input(x, self.modes1, self.modes2)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        extension = load_extension() if x.is_cuda else None
        fused = os.environ.get("SPECTRAL_GPU_ENABLE_FUSED") == "1"
        if extension is not None and fused and x.dtype == torch.float32:
            output_ft = extension.frequency_mul_forward(
                x_ft.contiguous(), self.weight_positive.contiguous(), self.modes1, self.modes2
            )
            return torch.fft.irfft2(output_ft, s=x.shape[-2:], norm="ortho")
        output_ft = torch.zeros(
            x.shape[0], self.out_channels, x.shape[-2], x_ft.shape[-1], dtype=x_ft.dtype, device=x.device
        )
        positive = x_ft[:, :, : self.modes1, : self.modes2].contiguous()
        negative = x_ft[:, :, -self.modes1 :, : self.modes2].contiguous()
        output_ft[:, :, : self.modes1, : self.modes2] = complex_multiply_optimized(
            positive, self.weight_positive
        )
        output_ft[:, :, -self.modes1 :, : self.modes2] = complex_multiply_optimized(
            negative, self.weight_negative
        )
        return torch.fft.irfft2(output_ft, s=x.shape[-2:], norm="ortho")

