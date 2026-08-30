#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    data = json.loads((args.results_dir / "operator_results.json").read_text(encoding="utf-8"))
    environment = json.loads((args.results_dir / "environment.json").read_text(encoding="utf-8"))
    lines = [
        "# Operator Benchmark Results",
        "",
        "Generated from JSON emitted by the benchmark programs.",
        "",
        f"- Hardware mode: `{environment['hardware_mode']}`",
        "- Correctness is checked before every measured pair.",
        "- CUDA timing uses CUDA Events; CPU fallback timing uses `perf_counter_ns`.",
        "",
    ]
    for payload in data["benchmarks"]:
        lines.append(f"## {payload['benchmark']}")
        lines.append("")
        for record in payload["records"]:
            if payload["benchmark"] == "operator":
                lines.append(
                    f"- `{record['variant']}` shape `{record['shape']}`, modes `{record['modes']}`: "
                    f"median `{record['median_ms']:.6f} ms`, speedup `{record['speedup']:.3f}x`, "
                    f"max error `{record['max_abs_error_vs_reference']:.3e}`."
                )
            elif payload["benchmark"] == "complex_multiply":
                lines.append(
                    f"- shape `{record['shape']}`: reference median `{record['reference']['median_ms']:.6f} ms`, "
                    f"optimized median `{record['optimized']['median_ms']:.6f} ms`, "
                    f"speedup `{record['speedup']:.3f}x`, max error `{record['max_abs_error']:.3e}`."
                )
            else:
                lines.append(
                    f"- shape `{record['shape']}`, modes `{record['modes']}`: reference median "
                    f"`{record['reference']['median_ms']:.6f} ms`, optimized median "
                    f"`{record['optimized']['median_ms']:.6f} ms`, speedup `{record['speedup']:.3f}x`, "
                    f"max error `{record['max_abs_error']:.3e}`."
                )
        lines.append("")
    if environment["hardware_mode"] != "cuda":
        lines.extend([
            "CUDA status: `NOT BENCHMARKED ON CURRENT HARDWARE`.",
            "All numbers in this report are CPU measurements from the PyTorch fallback.",
            "",
        ])
    report = "\n".join(lines)
    (args.results_dir / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

