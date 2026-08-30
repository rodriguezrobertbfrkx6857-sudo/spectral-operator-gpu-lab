import pytest
import torch

from spectral_gpu.cuda import extension_status
from spectral_gpu.operators.complex_multiply import (
    complex_multiply_optimized,
    complex_multiply_reference,
)


@pytest.mark.parametrize("batch,in_channels,out_channels,modes1,modes2", [(1, 3, 4, 5, 7), (4, 8, 6, 8, 9)])
def test_complex_multiply_matches_reference(batch, in_channels, out_channels, modes1, modes2):
    torch.manual_seed(7)
    input_ft = torch.randn(batch, in_channels, modes1, modes2, dtype=torch.complex64)
    weight = torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.complex64)
    expected = complex_multiply_reference(input_ft, weight)
    actual = complex_multiply_optimized(input_ft, weight)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_complex_multiply_rejects_real_inputs():
    with pytest.raises(TypeError):
        complex_multiply_reference(torch.ones(1, 2, 3, 4), torch.ones(2, 3, 3, 4, dtype=torch.complex64))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-capable PyTorch is unavailable")
def test_custom_cuda_complex_multiply_executes(monkeypatch):
    monkeypatch.setenv("SPECTRAL_GPU_REQUIRE_EXTENSION", "1")
    torch.manual_seed(41)
    input_ft = torch.randn(1, 16, 8, 9, dtype=torch.complex64, device="cuda")
    weight = torch.randn(16, 32, 8, 9, dtype=torch.complex64, device="cuda")
    expected = complex_multiply_reference(input_ft, weight)
    actual = complex_multiply_optimized(input_ft, weight)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    status = extension_status()
    assert status["available"] is True
    assert status["compiled"] is True
    assert status["backend"] == "custom_cuda"
