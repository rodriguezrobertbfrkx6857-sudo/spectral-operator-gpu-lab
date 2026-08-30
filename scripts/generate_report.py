#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _number(value: object, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}{suffix}"


def _speedup(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}x"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    data = json.loads((args.results_dir / "operator_results.json").read_text(encoding="utf-8"))
    environment = json.loads((args.results_dir / "environment.json").read_text(encoding="utf-8"))
    torch_info = environment.get("python_packages", {}).get("torch", {})
    toolchain = environment.get("toolchain", {})
    lines = [
        "# Operator Benchmark Results",
        "",
        "Generated from JSON emitted by the benchmark programs; timing values are not maintained by hand.",
        "",
        f"- Hardware mode: `{data['hardware_mode']}`",
        "- Correctness is checked before every measured pair.",
        "- CUDA timing uses CUDA Events; CPU fallback timing uses `perf_counter_ns`.",
        "- A custom-CUDA speedup is emitted only when the recorded optimized backend is custom CUDA.",
        "",
    ]
    for payload in data["benchmarks"]:
        lines.append(f"## {payload['benchmark']}")
        lines.append("")
        for record in payload["records"]:
            if payload["benchmark"] == "operator":
                lines.append(
                    f"- `{record['variant']}` shape `{record['shape']}`, modes `{record['modes']}`, "
                    f"backend `{record['backend']}`, status `{record['status']}`: "
                    f"median `{_number(record['median_ms'], ' ms')}`, speedup `{_speedup(record['speedup'])}`, "
                    f"max/mean error `{_number(record['max_abs_error_vs_reference'])}`/"
                    f"`{_number(record['mean_abs_error_vs_reference'])}`."
                )
            elif payload["benchmark"] == "complex_multiply":
                reference = record["reference"]
                optimized = record["optimized"]
                lines.append(
                    f"- shape `{record['shape']}`, optimized backend `{optimized['backend']}`, "
                    f"status `{optimized['status']}`: reference median `{_number(reference['median_ms'], ' ms')}`, "
                    f"optimized median `{_number(optimized['median_ms'], ' ms')}`, "
                    f"speedup `{_speedup(record['speedup'])}`, max/mean error "
                    f"`{_number(record['max_abs_error'])}`/`{_number(record['mean_abs_error'])}`."
                )
            else:
                reference = record["reference"]
                optimized = record["optimized"]
                lines.append(
                    f"- shape `{record['shape']}`, modes `{record['modes']}`, optimized backend "
                    f"`{optimized['backend']}`, status `{optimized['status']}`: reference median "
                    f"`{_number(reference['median_ms'], ' ms')}`, optimized median "
                    f"`{_number(optimized['median_ms'], ' ms')}`, speedup "
                    f"`{_speedup(record['speedup'])}`, max/mean error "
                    f"`{_number(record['max_abs_error'])}`/`{_number(record['mean_abs_error'])}`."
                )
        lines.append("")
    lines.extend(
        [
            "## Environment",
            "",
            f"- OS: {environment['operating_system']['system']} {environment['operating_system']['release']}",
            f"- Python: `{toolchain.get('python', 'unknown')}`",
            f"- PyTorch: `{torch_info.get('version', 'not installed')}`",
            f"- NVIDIA devices: `{len(environment['nvidia']['devices'])}`",
            "",
        ]
    )
    if environment["hardware_mode"] != "cuda":
        lines.extend(
            [
                "CUDA status: `NOT BENCHMARKED ON CURRENT HARDWARE`.",
                "All measured timings in this report are CPU measurements from the PyTorch fallback.",
                "The fused CUDA path is listed as skipped rather than represented by a CPU substitute.",
                "",
            ]
        )
    report = "\n".join(lines)
    (args.results_dir / "results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
