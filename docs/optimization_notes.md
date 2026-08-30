# Optimization notes

The measured optimization boundary is deliberate:

1. Reference: `torch.fft.rfft2`, frequency slicing, `torch.einsum`, zero-fill, and `torch.fft.irfft2`.
2. Improved PyTorch path: contiguous frequency slices and one explicit output spectrum.
3. Custom CUDA path: one thread computes one output frequency/channel and accumulates complex products in registers.
4. Optional fused path: a CUDA kernel combines frequency selection and complex multiplication into one output-spectrum write. It is opt-in with `SPECTRAL_GPU_ENABLE_FUSED=1` and is never assumed to be faster without a benchmark.

The custom extension does not reimplement FFT. That keeps the experiment focused on the frequency-domain operator while relying on the mature FFT backend for transforms. Correctness is checked against the reference for every benchmark case before timing.

