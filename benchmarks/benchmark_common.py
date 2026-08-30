from __future__ import annotations

import time
import platform
from typing import Callable

import numpy as np


def measure(
    function: Callable[[], object], device: str, warmup: int, iterations: int
) -> tuple[dict[str, float], object]:
    """Measure inference work with synchronized CUDA Event or wall-clock boundaries."""
    import torch

    result: object = None
    with torch.inference_mode():
        for _ in range(warmup):
            result = function()
        if device == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
        samples: list[float] = []
        for _ in range(iterations):
            if device == "cuda":
                start_event.record()
                result = function()
                stop_event.record()
                stop_event.synchronize()
                samples.append(float(start_event.elapsed_time(stop_event)))
            else:
                start = time.perf_counter_ns()
                result = function()
                samples.append((time.perf_counter_ns() - start) / 1.0e6)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "mean_ms": float(np.mean(values)),
        "min_ms": float(np.min(values)),
        "std_ms": float(np.std(values)),
    }, result


def protocol(device: str, quick: bool) -> tuple[int, int]:
    if device == "cuda":
        return (20, 50 if quick else 100)
    return (2, 4 if quick else 8)


def device_name(device: str) -> str:
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name(torch.cuda.current_device())
    return platform.processor() or platform.machine() or "CPU reference"


def status_for(device: str) -> str:
    return "BENCHMARKED_CUDA" if device == "cuda" else "BENCHMARKED_CPU_ONLY"


def custom_cuda_backend(backend: str) -> bool:
    return backend in {"custom_complex_cuda", "fused_frequency_cuda"}
