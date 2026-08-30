from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from spectral_gpu.operators.spectral_conv_optimized import SpectralConv2dOptimized
from spectral_gpu.operators.spectral_conv_reference import SpectralConv2dReference


class SpectralBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int, optimized: bool, seed: int) -> None:
        super().__init__()
        layer = SpectralConv2dOptimized if optimized else SpectralConv2dReference
        self.spectral = layer(width, width, modes1, modes2, seed=seed)
        self.local = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.local(x))


class MiniFNO(nn.Module):
    """Small deterministic FNO-like model for operator-level inference experiments."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 16,
        modes1: int = 8,
        modes2: int = 8,
        optimized: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                SpectralBlock(width, modes1, modes2, optimized, seed + 1),
                SpectralBlock(width, modes1, modes2, optimized, seed + 2),
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.projection(x)

