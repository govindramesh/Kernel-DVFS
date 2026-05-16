from __future__ import annotations

from dataclasses import replace
from typing import Any

from .custom_cuda_kernels import custom_cuda_category, get_custom_cuda_kernel, has_custom_cuda_kernel
from .models import read_json
from .transformer_workload import KernelSpec


def load_kernel_specs_file(path: str) -> list[KernelSpec]:
    payload = read_json(path)
    specs: list[KernelSpec] = []
    for item in payload["kernels"]:
        if "kernel_name" not in item:
            raise ValueError("Each custom kernel must include 'kernel_name'")
        kernel_name = str(item["kernel_name"])
        if not has_custom_cuda_kernel(kernel_name):
            raise ValueError(f"Unknown custom CUDA kernel '{kernel_name}'")
        kernel = get_custom_cuda_kernel(kernel_name)
        specs.append(
            KernelSpec(
                kernel_name=kernel_name,
                family=custom_cuda_category(kernel_name),
                phase=item.get("phase", "custom"),
                baseline_ms=0.0,
                optimal_core_mhz=0,
                optimal_mem_mhz=0,
                static_power_watts=0.0,
                dynamic_power_watts=0.0,
                repeat_count=int(item.get("repeat_count", 1)),
                description=item.get("description", kernel.description),
            )
        )
    return specs


def load_workflow_file(path: str) -> dict[str, Any]:
    return read_json(path)


def expanded_trace_from_workflow(specs: list[KernelSpec], workflow: dict[str, Any]) -> list[KernelSpec]:
    spec_by_name = {spec.kernel_name: spec for spec in specs}
    num_layers = int(workflow.get("num_layers", 1))
    trace: list[KernelSpec] = []
    for name in workflow.get("prefix", []):
        trace.append(replace(spec_by_name[name], repeat_count=1))
    layer_order = workflow.get("layer_kernel_order", [])
    for _layer_index in range(num_layers):
        for name in layer_order:
            trace.append(replace(spec_by_name[name], repeat_count=1))
    for name in workflow.get("suffix", []):
        trace.append(replace(spec_by_name[name], repeat_count=1))
    if not trace and workflow.get("events"):
        for name in workflow["events"]:
            trace.append(replace(spec_by_name[name], repeat_count=1))
    return trace


def execution_graph_from_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    if workflow.get("events"):
        return {
            "prefix": workflow["events"],
            "layer_kernel_order": [],
            "suffix": [],
            "num_layers": 1,
        }
    return {
        "prefix": workflow.get("prefix", []),
        "layer_kernel_order": workflow.get("layer_kernel_order", []),
        "suffix": workflow.get("suffix", []),
        "num_layers": int(workflow.get("num_layers", 1)),
    }
