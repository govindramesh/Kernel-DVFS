# KernelDVFS MVP

KernelDVFS is a deterministic, hardware-aware DVFS prototype for LLM workloads. The paper-recreation path now follows the local kernel-level waste-reduction flow from the January 13, 2026 paper "Reducing Compute Waste in LLMs through Kernel-Level DVFS":

1. `example_graph.py --scenario paper_iteration` emits a GPT-style training iteration trace with 46 distinct kernel invocations and 24 repeated transformer layers.
2. `profiler.py` profiles each distinct kernel against an `auto` baseline and selects the lowest-energy clock within a tolerated slowdown percentage.
3. `partitioner.py` and `daemon.py` remain available for later productionization work.
4. `runtime_compare.py` runs the paper-style workload sequentially and compares `auto` clocks against per-kernel profiled clocks.

The project is designed to run in two modes:

- `mock`: Uses a deterministic synthetic hardware model and works on any machine.
- `auto`: Tries to use real NVML-backed clock control if `pynvml` and a supported GPU are available, then falls back to mock mode.

## Files

- `profiler.py`: Paper-style offline clock sweeper and profile generator.
- `partitioner.py`: Greedy super-block partitioner and scheduler.
- `daemon.py`: Concurrent runtime actuation simulator.
- `example_graph.py`: Emits either the paper-style GPT iteration trace or the older synthetic block trace.
- `kerneldvfs/kernels.py`: Actual Triton kernels and PyTorch launch wrappers for GEMM, RMSNorm, softmax, and SiLU gating.
- `kerneldvfs/models.py`: Shared dataclasses and JSON serialization.
- `kerneldvfs/nvml_controller.py`: Real and mock NVML control paths.
- `kerneldvfs/paper_recreation.py`: Paper kernel catalog, paper clock domains, and benchmark families.
- `runtime_compare.py`: Sequential workload runner for `auto` vs profiled-clock comparison, including measured clock transition latency.
- `data/execution_trace.json`: Sample transformer-layer execution trace.

## Quickstart

```bash
python3 example_graph.py --scenario paper_iteration --trace-output data/example_execution_trace.json
python3 profiler.py --backend mock --measurement-mode mock --tolerated-slowdown-pct 0.0 --output data/profiles.json
python3 partitioner.py --profiles data/profiles.json --trace data/example_execution_trace.json --output data/superblock_schedule.json
python3 daemon.py --schedule data/superblock_schedule.json --backend mock
```

## Real Runtime Comparison

Once you have a real profile file, you can compare the sequential workload under `auto` clocks and per-kernel profiled clocks:

```bash
python3 profiler.py --backend real --measurement-mode real --device-index 0 --tolerated-slowdown-pct 0.0 --output data/profiles.json
python3 runtime_compare.py --backend real --device-index 0 --device cuda:0 --profiles data/profiles.json --mode both --output data/runtime_comparison.json
```

`runtime_compare.py`:

- runs the paper-style kernel sequence sequentially
- uses `auto` clocks for one run
- uses the profiled target clocks before every kernel event for the second run
- records total workload wall time, summed kernel runtime, estimated energy, and observed clock-transition latency

## Dependency Notes

Optional GPU-backed execution uses:

- `nvidia-ml-py`
- `torch`
- `triton`
- `kernel_tuner`

The profiler ships with a mock paper-recreation workload model so the pipeline remains testable without those dependencies. If the real stack is present, the code will use genuine NVML clock queries and locking and benchmark representative CUDA kernels for each paper kernel family.

## Kernel Implementations

Actual Triton kernels now live in `kerneldvfs/kernels.py`:

- `matmul_kernel`: Used for `fused_qkv_matmul`, `attention_scores_matmul`, `attention_value_matmul`, `mlp_up_proj_matmul`, and `mlp_down_proj_matmul`.
- `rmsnorm_kernel`: Row-wise RMSNorm over the hidden dimension.
- `softmax_kernel`: Row-wise masked-softmax core without the masking prepass.
- `silu_gate_kernel`: Fused SiLU-and-gate elementwise kernel for the MLP path.

PyTorch launch helpers are exposed as `launch_matmul`, `launch_rmsnorm`, `launch_softmax`, and `launch_silu_gate`. The `example_inputs()` helper builds representative LLM-shaped CUDA tensors for each kernel when `torch` and `triton` are available.

## Runnable Example Graph

`example_graph.py` executes this kernel sequence with artificial tensors:

1. `fused_qkv_matmul`
2. `rmsnorm`
3. `attention_scores_matmul`
4. `masked_softmax`
5. `attention_value_matmul`
6. `mlp_up_proj_matmul`
7. `silu_gate`
8. `mlp_down_proj_matmul`

It writes:

- `data/example_graph.json`: Node-level graph description with inputs, outputs, and shapes.
- `data/example_execution_trace.json`: Measured durations in the exact format expected by `partitioner.py`.

CPU-safe example:

```bash
python3 example_graph.py --backend torch --device cpu --dtype float32
python3 partitioner.py --profiles data/profiles.json --trace data/example_execution_trace.json --output data/superblock_schedule.json
python3 daemon.py --schedule data/superblock_schedule.json --backend mock
```

If you have CUDA plus Triton available, you can switch to:

```bash
python3 example_graph.py --backend triton --device cuda --dtype float16
```

## Real Profiling

`profiler.py` now supports two profiling modes:

- `--measurement-mode mock`: Uses the synthetic runtime and energy model.
- `--measurement-mode real`: Locks each candidate clock pair, warms up the actual Triton kernels, measures them with CUDA events, and reads energy from NVML when available.

Example GPU run:

```bash
python3 profiler.py --backend real --measurement-mode real --device-index 0 --warmup-runs 5 --benchmark-runs 25 --output data/profiles.json
```

If the environment cannot satisfy real profiling requirements, `--measurement-mode auto` will fall back to mock profiling while keeping the output format unchanged.

If NVML clock-lock APIs return `Not Supported`, the controller now falls back to `nvidia-smi`. To force that path to run through non-interactive `sudo`, set:

```bash
export KERNELDVFS_NVIDIA_SMI_SUDO=1
```

You can also point to a specific binary with:

```bash
export KERNELDVFS_NVIDIA_SMI=/usr/bin/nvidia-smi
```

You can also pass those directly as CLI flags instead:

```bash
python3 profiler.py --backend real --measurement-mode real --nvidia-smi-sudo --nvidia-smi-path /usr/bin/nvidia-smi --device-index 0 --warmup-runs 5 --benchmark-runs 25 --output data/profiles.json
```
