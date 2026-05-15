from __future__ import annotations

from dataclasses import replace
from typing import Any

from .models import read_json
from .paper_recreation import PaperKernelSpec


def load_kernel_specs_file(path: str) -> list[PaperKernelSpec]:
    payload = read_json(path)
    specs: list[PaperKernelSpec] = []
    for item in payload["kernels"]:
        specs.append(
            PaperKernelSpec(
                kernel_name=item["kernel_name"],
                family=item["family"],
                phase=item.get("phase", "custom"),
                baseline_ms=float(item["baseline_ms"]),
                optimal_core_mhz=int(item["optimal_core_mhz"]),
                optimal_mem_mhz=int(item["optimal_mem_mhz"]),
                static_power_watts=float(item["static_power_watts"]),
                dynamic_power_watts=float(item["dynamic_power_watts"]),
                repeat_count=int(item.get("repeat_count", 1)),
                m=int(item.get("m", 0)),
                n=int(item.get("n", 0)),
                k=int(item.get("k", 0)),
                rows=int(item.get("rows", 0)),
                cols=int(item.get("cols", 0)),
                elements=int(item.get("elements", 0)),
                heads=int(item.get("heads", 0)),
                description=item.get("description", item["kernel_name"]),
            )
        )
    return specs


def load_workflow_file(path: str) -> dict[str, Any]:
    return read_json(path)


def expanded_trace_from_workflow(specs: list[PaperKernelSpec], workflow: dict[str, Any]) -> list[PaperKernelSpec]:
    spec_by_name = {spec.kernel_name: spec for spec in specs}
    num_layers = int(workflow.get("num_layers", 1))
    trace: list[PaperKernelSpec] = []
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
