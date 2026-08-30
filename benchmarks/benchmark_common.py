from __future__ import annotations

import time
from typing import Callable

import numpy as np


def measure(function: Callable[[], object], device: str, warmup: int, iterations: int) -> tuple[dict[str, float], object]:
    result: object = None
    for _ in range(warmup):
        result = function()
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        if device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
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

