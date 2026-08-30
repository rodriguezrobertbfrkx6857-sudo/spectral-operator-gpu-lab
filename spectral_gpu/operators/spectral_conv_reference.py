from __future__ import annotations

import torch
from torch import nn

from .complex_multiply import complex_multiply_reference


def _complex_parameter(shape: tuple[int, ...], generator: torch.Generator | None = None) -> nn.Parameter:
    scale = 1.0 / max(1, shape[0] * shape[1])
    real = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    imaginary = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return nn.Parameter(torch.complex(real, imaginary))


def _validate_input(x: torch.Tensor, modes1: int, modes2: int) -> None:
    if x.ndim != 4:
        raise ValueError("x must have shape [batch, channels, height, width]")
    height, width = x.shape[-2:]
    if modes1 < 1 or modes2 < 1 or modes1 > height // 2 or modes2 > width // 2 + 1:
        raise ValueError("modes must fit the rFFT frequency grid")


def spectral_conv2d_reference(
    x: torch.Tensor,
    weight_positive: torch.Tensor,
    weight_negative: torch.Tensor,
    modes1: int,
    modes2: int,
) -> torch.Tensor:
    _validate_input(x, modes1, modes2)
    x_ft = torch.fft.rfft2(x, norm="ortho")
    output_ft = torch.zeros(
        x.shape[0], weight_positive.shape[1], x.shape[-2], x_ft.shape[-1],
        dtype=x_ft.dtype, device=x.device
    )
    positive = x_ft[:, :, :modes1, :modes2]
    negative = x_ft[:, :, -modes1:, :modes2]
    output_ft[:, :, :modes1, :modes2] = complex_multiply_reference(positive, weight_positive)
    output_ft[:, :, -modes1:, :modes2] = complex_multiply_reference(negative, weight_negative)
    return torch.fft.irfft2(output_ft, s=x.shape[-2:], norm="ortho")


class SpectralConv2dReference(nn.Module):
    """Fourier Neural Operator layer using torch.fft and an einsum ground truth."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, seed: int = 0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.weight_positive = _complex_parameter((in_channels, out_channels, modes1, modes2), generator)
        self.weight_negative = _complex_parameter((in_channels, out_channels, modes1, modes2), generator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return spectral_conv2d_reference(
            x, self.weight_positive, self.weight_negative, self.modes1, self.modes2
        )

