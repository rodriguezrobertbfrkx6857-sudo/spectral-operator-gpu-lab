import pytest
import torch

from spectral_gpu.operators.spectral_conv_optimized import SpectralConv2dOptimized
from spectral_gpu.operators.spectral_conv_reference import SpectralConv2dReference


@pytest.mark.parametrize(
    "batch,channels,height,width,modes1,modes2",
    [
        (1, 16, 64, 64, 8, 8),
        (4, 16, 64, 128, 8, 16),
        (1, 32, 128, 128, 16, 16),
        (4, 32, 128, 256, 16, 32),
        (1, 16, 256, 256, 32, 32),
    ],
)
def test_optimized_matches_reference(batch, channels, height, width, modes1, modes2):
    torch.manual_seed(11)
    reference = SpectralConv2dReference(channels, channels, modes1, modes2, seed=13)
    optimized = SpectralConv2dOptimized(channels, channels, modes1, modes2, seed=13)
    optimized.load_state_dict(reference.state_dict())
    x = torch.randn(batch, channels, height, width)
    expected = reference(x)
    actual = optimized(x)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_reference_rejects_invalid_modes():
    layer = SpectralConv2dReference(2, 3, modes1=8, modes2=8)
    with pytest.raises(ValueError):
        layer(torch.randn(1, 2, 8, 16))

