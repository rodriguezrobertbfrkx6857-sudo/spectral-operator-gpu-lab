"""Lazy loading for the optional PyTorch CUDA extension."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_extension: Any | None = None
_attempted = False
_load_error: str | None = None


def load_extension() -> Any | None:
    global _extension, _attempted, _load_error
    if _attempted:
        return _extension
    _attempted = True
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load

        if not torch.cuda.is_available() or CUDA_HOME is None:
            _load_error = "CUDA runtime or CUDA_HOME is unavailable"
            return None
        if os.environ.get("SPECTRAL_GPU_DISABLE_EXTENSION") == "1":
            _load_error = "disabled by SPECTRAL_GPU_DISABLE_EXTENSION"
            return None
        root = Path(__file__).resolve()
        sources = [str(root / "bindings.cpp"), str(root / "complex_mul_kernel.cu"), str(root / "frequency_kernel.cu")]
        host_flags = ["/O2"] if os.name == "nt" else ["-O3"]
        _extension = load(
            name="spectral_gpu_cuda_ext",
            sources=sources,
            extra_cflags=host_flags,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on an external CUDA toolchain
        _load_error = repr(exc)
        _extension = None
    return _extension


def extension_status() -> dict[str, str | bool | None]:
    extension = load_extension()
    return {"available": extension is not None, "load_error": _load_error}


__all__ = ["load_extension", "extension_status"]
