# SpectralConv2d

For an input `[B, C_in, H, W]`, `torch.fft.rfft2` produces `[B, C_in, H, W/2+1]` complex coefficients. The layer keeps two low-frequency row bands: the first `modes1` rows and the last `modes1` rows. It contracts each band with a complex weight `[C_in, C_out, modes1, modes2]`, writes the selected coefficients into an output spectrum, and applies `torch.fft.irfft2` to return to the spatial domain.

The reference implementation uses `torch.einsum` as the correctness ground truth. The optimized implementation preserves the FFT library calls and replaces only the channel contraction when the optional complex64 CUDA extension is available. The positive and negative row bands have separate weights, matching the usual 2D FNO construction.

