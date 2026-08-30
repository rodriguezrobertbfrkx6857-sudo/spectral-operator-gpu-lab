"""Spectral operator reference and optional CUDA extension."""

from .operators.spectral_conv_reference import SpectralConv2dReference
from .operators.spectral_conv_optimized import SpectralConv2dOptimized

__all__ = ["SpectralConv2dReference", "SpectralConv2dOptimized"]

