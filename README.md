# KernelDVFS

KernelDVFS is a research-to-production prototype based on the paper *Reducing Compute Waste in LLMs through Kernel-Level DVFS*. The repo has two complementary paths:

- a paper-recreation path for profiling and aggregating per-kernel DVFS decisions
- a real CUDA demo path for running a small custom workflow through bundled CUDA kernels and visualizing the results

## What’s in the repo

- `profiler.py`: profiles kernels against `auto` clocks and selects the lowest-energy clock within a slowdown bound
- `runtime_compare.py`: aggregates isolated kernel measurements over a workflow, paper-style
- `dashboard.py`: generates a standalone HTML dashboard
- `demo_web.py`: local web workbench for running the custom workflow path
- `demo_pipeline.py`: CLI wrapper for the same custom workflow path
- `kerneldvfs/paper_recreation.py`: forward-only transformer-style kernel catalog used for the paper recreation flow
- `kerneldvfs/custom_cuda_kernels.py`: registry and extension loader for the bundled CUDA demo kernels
- `kerneldvfs/cuda_kernels/`: CUDA sources used by the custom demo path

## Repository Layout

```text
.
├── kerneldvfs/
│   ├── cuda_kernels/          # Bundled CUDA kernels for the custom demo
│   ├── custom_cuda_kernels.py # Kernel registry + torch extension loader
│   ├── nvml_controller.py     # Real/mock clock control
│   ├── paper_recreation.py    # Built-in paper-style workload definitions
│   └── workload_loader.py     # Custom workflow and kernel-definition loading
├── data/
│   ├── demo_kernels.json      # Sample custom kernel list
│   └── demo_workflow.json     # Sample workflow definition
├── profiler.py
├── runtime_compare.py
├── dashboard.py
├── demo_pipeline.py
└── demo_web.py
```

## Installation

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For real GPU runs you also need:

- an NVIDIA GPU with supported clock control
- a working driver and `nvidia-smi`
- the CUDA toolkit, including `nvcc`
- `CUDA_HOME` set, or a standard CUDA install path such as `/usr/local/cuda`

## Paper Recreation Flow

The built-in paper path uses a forward-only transformer-style kernel inventory and supports both mock and real profiling.

Generate profiles:

```bash
python3 profiler.py \
  --backend real \
  --measurement-mode real \
  --device-index 0 \
  --nvidia-smi-sudo \
  --output data/profiles.json
```

Aggregate workload totals:

```bash
python3 runtime_compare.py \
  --profiles data/profiles.json \
  --output data/runtime_comparison.json
```

Build the dashboard:

```bash
python3 dashboard.py \
  --profiles data/profiles.json \
  --runtime-compare data/runtime_comparison.json \
  --output data/dashboard.html
```

This comparison is intentionally **offline aggregation**, matching the paper’s main methodology:

- each unique kernel is profiled in isolation
- `runtime_compare.py` sums the isolated measurements over the workflow order
- the repo does not replay live clock changes per kernel for the paper comparison

## Custom CUDA Demo

The custom demo path is **real-run only**. Instead of accepting arbitrary Python kernel bodies, it accepts a kernel list whose names map to bundled CUDA kernels in `kerneldvfs/cuda_kernels/`.

Sample kernel list:

```json
{
  "kernels": [
    { "kernel_name": "rmsnorm" },
    { "kernel_name": "softmax" },
    { "kernel_name": "silu" }
  ]
}
```

Sample workflow:

```json
{
  "prefix": [],
  "layer_kernel_order": ["rmsnorm", "softmax", "silu"],
  "num_layers": 4,
  "suffix": []
}
```

Run it from the CLI:

```bash
python3 demo_pipeline.py \
  --kernel-defs data/demo_kernels.json \
  --workflow data/demo_workflow.json
```

Or launch the web workbench:

```bash
python3 demo_web.py --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The web workbench:

- accepts a kernel list and workflow JSON
- runs the real profiler path
- streams subprocess output into the page
- shows the current pipeline step
- generates the dashboard and JSON artifacts

## Bundled CUDA Kernels

The custom demo currently ships with three small CUDA operators:

- `rmsnorm`
- `softmax`
- `silu`

These are compiled at runtime through `torch.utils.cpp_extension.load(...)`.

The operator set and naming were inspired by [FlashInfer](https://github.com/flashinfer-ai/flashinfer), but the CUDA code in this repo is local and simplified for the demo path.

## Notes

- Real clock changes may require `sudo nvidia-smi`, so the demo pipeline enables `--nvidia-smi-sudo` by default.
- The web workbench stores generated outputs under `data/web_runs/`.
- Large generated artifacts, runtime outputs, and local extension builds are gitignored.
