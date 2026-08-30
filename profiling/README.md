# Profiling

Run `python profiling/profile_operator.py` to exercise a deterministic Mini-FNO inference path. On CUDA hardware, wrap the command with Nsight Systems or Nsight Compute and record the exact tool version, GPU, command line, and exported report. The repository does not infer profiler counters from wall-clock timing.

The current environment has no NVIDIA driver, CUDA Toolkit, `ncu`, or `nsys`, so profiling is not available for this checkout.

