from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .benchmark_common import measure, protocol
from spectral_gpu.models.mini_fno import MiniFNO
from spectral_gpu.models.synthetic_data import make_smooth_fields


def run(quick: bool = False) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    cases = [(1, 32, 8, 8)] if quick else [(1, 64, 8, 8), (1, 128, 16, 16)]
    warmup, iterations = protocol(device, quick)
    records: list[dict] = []
    for batch, side, modes1, modes2 in cases:
        reference = MiniFNO(width=8, modes1=modes1, modes2=modes2, optimized=False, seed=109).to(device)
        optimized = MiniFNO(width=8, modes1=modes1, modes2=modes2, optimized=True, seed=109).to(device)
        optimized.load_state_dict(reference.state_dict())
        x = make_smooth_fields(batch, side, side, seed=113, device=device)
        with torch.no_grad():
            expected = reference(x)
            actual = optimized(x)
            if device == "cuda":
                torch.cuda.synchronize()
            max_error = float((actual - expected).abs().max().item())
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)
        reference_stats, _ = measure(lambda: reference(x), device, warmup, iterations)
        optimized_stats, _ = measure(lambda: optimized(x), device, warmup, iterations)
        records.append(
            {
                "benchmark": "mini_fno_inference",
                "shape": [batch, 1, side, side],
                "modes": [modes1, modes2],
                "hardware_mode": device,
                "reference": reference_stats,
                "optimized": optimized_stats,
                "speedup": reference_stats["median_ms"] / optimized_stats["median_ms"],
                "max_abs_error": max_error,
                "note": "Inference only; no dataset download or training claim.",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"benchmark": "mini_fno", "records": run(args.quick)}
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
