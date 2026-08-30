"""Lazy loading and status reporting for the optional PyTorch CUDA extension."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

EXTENSION_SOURCE_VERSION = "v2-positive-negative-fusion"

_extension: Any | None = None
_attempted = False
_status: dict[str, Any] = {
    "available": False,
    "compiled": False,
    "backend": "torch_fallback",
    "build_error": None,
    "source_version": EXTENSION_SOURCE_VERSION,
}


def _require_extension() -> bool:
    return os.environ.get("SPECTRAL_GPU_REQUIRE_EXTENSION") == "1"


def _raise_required_failure() -> None:
    if _require_extension() and _extension is None:
        detail = _status.get("build_error") or "CUDA runtime/toolkit is unavailable"
        raise RuntimeError(f"CUDA extension is required but unavailable: {detail}")


def load_extension() -> Any | None:
    global _extension, _attempted
    if os.environ.get("SPECTRAL_GPU_DISABLE_EXTENSION") == "1":
        if _require_extension():
            raise RuntimeError("CUDA extension is disabled by SPECTRAL_GPU_DISABLE_EXTENSION")
        return None
    if _attempted:
        _raise_required_failure()
        return _extension
    _attempted = True
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, load

        if not torch.cuda.is_available() or CUDA_HOME is None:
            _status.update(
                backend="torch_fallback",
                build_error="CUDA runtime or CUDA_HOME is unavailable",
            )
            _raise_required_failure()
            return None
        if os.environ.get("SPECTRAL_GPU_DISABLE_EXTENSION") == "1":
            _status.update(backend="torch_fallback", build_error="disabled by environment")
            _raise_required_failure()
            return None

        root = Path(__file__).resolve()
        sources = [
            str(root / "bindings.cpp"),
            str(root / "complex_mul_kernel.cu"),
            str(root / "frequency_kernel.cu"),
        ]
        host_flags = ["/O2"] if os.name == "nt" else ["-O3"]
        cuda_flags = ["-O3"]
        if os.environ.get("SPECTRAL_GPU_USE_FAST_MATH") == "1":
            cuda_flags.append("--use_fast_math")
        _extension = load(
            name=f"spectral_gpu_cuda_ext_{EXTENSION_SOURCE_VERSION.replace('-', '_')}",
            sources=sources,
            extra_cflags=host_flags,
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
        _status.update(available=True, compiled=True, backend="custom_cuda", build_error=None)
    except Exception as exc:  # pragma: no cover - depends on an external CUDA toolchain
        _status.update(
            available=False,
            compiled=False,
            backend="torch_fallback",
            build_error=repr(exc),
        )
        _extension = None
        _raise_required_failure()
    return _extension


def extension_status() -> dict[str, Any]:
    try:
        load_extension()
    except RuntimeError:
        # Status inspection remains structured even when strict validation is enabled.
        pass
    return dict(_status)


__all__ = ["EXTENSION_SOURCE_VERSION", "load_extension", "extension_status"]
