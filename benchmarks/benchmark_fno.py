from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark_common import custom_cuda_backend, device_name, measure, protocol, status_for
from spectral_gpu.models.mini_fno import MiniFNO
from spectral_gpu.models.synthetic_data import make_smooth_fields
from spectral_gpu.operators.spectral_conv_optimized import backend_name


def run(quick: bool = False) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    cases = [(1, 32, 8, 8)] if quick else [(1, 64, 8, 8), (1, 128, 16, 16), (1, 256, 32, 32)]
    warmup, iterations = protocol(device, quick)
    records: list[dict] = []
    reference_backend = "torch_reference_cuda" if device == "cuda" else "torch_reference_cpu"
    for batch, side, modes1, modes2 in cases:
        reference = MiniFNO(width=8, modes1=modes1, modes2=modes2, optimized=False, seed=109).to(device)
        optimized = MiniFNO(width=8, modes1=modes1, modes2=modes2, optimized=True, seed=109).to(device)
        optimized.load_state_dict(reference.state_dict())
        x = make_smooth_fields(batch, side, side, seed=113, device=device)
        with torch.inference_mode():
            expected = reference(x)
            actual = optimized(x)
            if device == "cuda":
                torch.cuda.synchronize()
        error = (actual - expected).abs()
        max_error = float(error.max().item())
        mean_error = float(error.mean().item())
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
        reference_stats, _ = measure(lambda: reference(x), device, warmup, iterations)
        optimized_stats, _ = measure(lambda: optimized(x), device, warmup, iterations)
        optimized_backend = backend_name(x)
        speedup = (
            reference_stats["median_ms"] / optimized_stats["median_ms"]
            if custom_cuda_backend(optimized_backend) and optimized_stats["median_ms"] > 0
            else None
        )
        records.append(
            {
                "benchmark": "mini_fno_inference",
                "shape": [batch, 1, side, side],
                "width": 8,
                "modes": [modes1, modes2],
                "parameter_count": sum(parameter.numel() for parameter in reference.parameters()),
                "dtype": "float32",
                "device": device_name(device),
                "hardware_mode": device,
                "warmup": warmup,
                "iterations": iterations,
                "reference": {
                    **reference_stats,
                    "variant": "pytorch_reference",
                    "backend": reference_backend,
                    "status": status_for(device),
                },
                "optimized": {
                    **optimized_stats,
                    "variant": "optimized_operator",
                    "backend": optimized_backend,
                    "status": status_for(device),
                },
                "speedup": speedup,
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
                "note": "Inference only; no dataset download or training claim.",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"schema_version": 2, "benchmark": "mini_fno", "records": run(args.quick)}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
