# KernelDVFS MVP

KernelDVFS is a deterministic, hardware-aware DVFS prototype for LLM workloads. This MVP implements the three-stage pipeline described in the architecture spec:

1. `profiler.py` discovers per-kernel "performance-locked energy floor" clock targets.
2. `partitioner.py` groups trace events into super-blocks and computes a look-ahead actuation schedule.
3. `daemon.py` replays the schedule with a concurrent power-actuation thread.

The project is designed to run in two modes:

- `mock`: Uses a deterministic synthetic hardware model and works on any machine.
- `auto`: Tries to use real NVML-backed clock control if `pynvml` and a supported GPU are available, then falls back to mock mode.

## Files

- `profiler.py`: Offline clock sweeper and profile generator.
- `partitioner.py`: Greedy super-block partitioner and scheduler.
- `daemon.py`: Concurrent runtime actuation simulator.
- `example_graph.py`: Runs a synthetic transformer-style kernel graph with artificial data and emits a partitioner-ready trace.
- `kerneldvfs/kernels.py`: Actual Triton kernels and PyTorch launch wrappers for GEMM, RMSNorm, softmax, and SiLU gating.
- `kerneldvfs/models.py`: Shared dataclasses and JSON serialization.
- `kerneldvfs/nvml_controller.py`: Real and mock NVML control paths.
- `data/execution_trace.json`: Sample transformer-layer execution trace.

## Quickstart

```bash
python3 example_graph.py --backend torch --device cpu --trace-output data/example_execution_trace.json
python3 profiler.py --backend mock --output data/profiles.json
python3 partitioner.py --profiles data/profiles.json --trace data/example_execution_trace.json --output data/superblock_schedule.json
python3 daemon.py --schedule data/superblock_schedule.json --backend mock
```

## Dependency Notes

Optional GPU-backed execution uses:

- `nvidia-ml-py`
- `torch`
- `triton`
- `kernel_tuner`

The profiler ships with a mock Triton-style workload model so the pipeline remains testable without those dependencies. If the real stack is present, the code will use genuine NVML clock queries and locking; timing still falls back to the synthetic benchmark model unless a custom benchmark hook is provided.

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
