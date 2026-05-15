from __future__ import annotations

import argparse
from typing import Any

from kerneldvfs.models import ProfileResult, read_json, write_json
from kerneldvfs.paper_recreation import PaperKernelSpec, expanded_trace_specs, paper_kernel_specs
from kerneldvfs.workload_loader import (
    execution_graph_from_workflow,
    expanded_trace_from_workflow,
    load_kernel_specs_file,
    load_workflow_file,
)


def load_profiles(path: str) -> dict[str, ProfileResult]:
    payload = read_json(path)
    return {name: ProfileResult.from_dict(item) for name, item in payload["profiles"].items()}


def identify_trace_regions(specs: list[PaperKernelSpec], num_layers: int) -> tuple[list[PaperKernelSpec], list[PaperKernelSpec], list[PaperKernelSpec]]:
    repeated_start = next((index for index, spec in enumerate(specs) if spec.repeat_count == num_layers), len(specs))
    repeated_end = repeated_start
    while repeated_end < len(specs) and specs[repeated_end].repeat_count == num_layers:
        repeated_end += 1
    return specs[:repeated_start], specs[repeated_start:repeated_end], specs[repeated_end:]


def aggregate_events(
    trace_specs: list[PaperKernelSpec],
    profiles: dict[str, ProfileResult],
    num_layers: int,
    iterations: int,
    execution_graph: dict[str, Any],
) -> dict[str, Any]:
    unique_specs: list[PaperKernelSpec] = []
    prefix_names = set(execution_graph.get("prefix", []))
    repeated_order = list(execution_graph.get("layer_kernel_order", []))
    repeated_names = set(repeated_order)
    suffix_names = set(execution_graph.get("suffix", []))

    events: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = [
        {
            "layer_index": layer_index,
            "events": [],
            "auto_time_ms": 0.0,
            "profiled_time_ms": 0.0,
            "auto_energy_mj": 0.0,
            "profiled_energy_mj": 0.0,
        }
        for layer_index in range(num_layers)
    ]

    auto_total_time_ms = 0.0
    profiled_total_time_ms = 0.0
    auto_total_energy_mj = 0.0
    profiled_total_energy_mj = 0.0
    energy_supported = True
    for iteration in range(iterations):
        repeated_cursor = 0
        for event_index, spec in enumerate(trace_specs):
            profile = profiles[spec.kernel_name]
            layer_index: int | None = None
            region = "prefix"
            if spec.kernel_name in repeated_names:
                layer_index = repeated_cursor // max(len(repeated_order), 1)
                repeated_cursor += 1
                region = "layer"
            elif spec.kernel_name in suffix_names:
                region = "suffix"
            elif spec.kernel_name in prefix_names:
                region = "prefix"

            auto_time_ms = float(profile.baseline_ms)
            profiled_time_ms = float(profile.selected_runtime_ms)
            auto_energy_mj = profile.baseline_energy_mj
            profiled_energy_mj = float(profile.estimated_energy_mj)
            if auto_energy_mj is None:
                energy_supported = False

            event_payload = {
                "iteration": iteration,
                "event_index": event_index,
                "kernel_name": spec.kernel_name,
                "phase": spec.phase,
                "family": spec.family,
                "layer_index": layer_index,
                "region": region,
                "auto_time_ms": round(auto_time_ms, 6),
                "profiled_time_ms": round(profiled_time_ms, 6),
                "time_delta_ms": round(profiled_time_ms - auto_time_ms, 6),
                "auto_energy_mj": None if auto_energy_mj is None else round(auto_energy_mj, 6),
                "profiled_energy_mj": round(profiled_energy_mj, 6),
                "energy_delta_mj": None if auto_energy_mj is None else round(profiled_energy_mj - auto_energy_mj, 6),
                "target_clock": profile.target_clock.to_dict(),
            }
            events.append(event_payload)

            auto_total_time_ms += auto_time_ms
            profiled_total_time_ms += profiled_time_ms
            if auto_energy_mj is not None:
                auto_total_energy_mj += auto_energy_mj
            profiled_total_energy_mj += profiled_energy_mj

            if layer_index is not None:
                layer_summary = layers[layer_index]
                layer_summary["events"].append(event_payload)
                layer_summary["auto_time_ms"] += auto_time_ms
                layer_summary["profiled_time_ms"] += profiled_time_ms
                if auto_energy_mj is not None:
                    layer_summary["auto_energy_mj"] += auto_energy_mj
                layer_summary["profiled_energy_mj"] += profiled_energy_mj

    for layer_summary in layers:
        layer_summary["auto_time_ms"] = round(layer_summary["auto_time_ms"], 6)
        layer_summary["profiled_time_ms"] = round(layer_summary["profiled_time_ms"], 6)
        layer_summary["time_delta_ms"] = round(layer_summary["profiled_time_ms"] - layer_summary["auto_time_ms"], 6)
        layer_summary["auto_energy_mj"] = None if not energy_supported else round(layer_summary["auto_energy_mj"], 6)
        layer_summary["profiled_energy_mj"] = round(layer_summary["profiled_energy_mj"], 6)
        layer_summary["energy_delta_mj"] = (
            None if not energy_supported else round(layer_summary["profiled_energy_mj"] - layer_summary["auto_energy_mj"], 6)
        )

    return {
        "execution_graph": execution_graph,
        "events": events,
        "layers": layers,
        "auto": {
            "time_to_complete_ms": round(auto_total_time_ms, 6),
            "total_energy_mj": None if not energy_supported else round(auto_total_energy_mj, 6),
        },
        "profiled": {
            "time_to_complete_ms": round(profiled_total_time_ms, 6),
            "total_energy_mj": round(profiled_total_energy_mj, 6),
        },
        "comparison": {
            "wall_runtime_delta_ms": round(profiled_total_time_ms - auto_total_time_ms, 6),
            "wall_runtime_delta_pct": round(((profiled_total_time_ms / auto_total_time_ms) - 1.0) * 100.0, 6)
            if auto_total_time_ms
            else None,
            "energy_delta_mj": None if not energy_supported else round(profiled_total_energy_mj - auto_total_energy_mj, 6),
            "energy_delta_pct": round(((profiled_total_energy_mj / auto_total_energy_mj) - 1.0) * 100.0, 6)
            if energy_supported and auto_total_energy_mj
            else None,
        },
        "energy_supported": energy_supported,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate isolated kernel measurements into a paper-style workload comparison")
    parser.add_argument("--profiles", default="data/profiles.json")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--kernel-defs", default=None, help="Optional JSON file describing custom kernel definitions")
    parser.add_argument("--workflow", default=None, help="Optional JSON file describing workflow order")
    parser.add_argument("--output", default="data/runtime_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.profiles)
    if args.kernel_defs and args.workflow:
        specs = load_kernel_specs_file(args.kernel_defs)
        workflow = load_workflow_file(args.workflow)
        trace_specs = expanded_trace_from_workflow(specs=specs, workflow=workflow)
        execution_graph = execution_graph_from_workflow(workflow)
        num_layers = int(execution_graph.get("num_layers", 1))
    else:
        trace_specs = expanded_trace_specs(num_layers=args.num_layers)
        execution_graph = {
            "prefix": ["k00_tokpos_embedding_add"],
            "layer_kernel_order": [
                "k01_block_pre_attn_layernorm",
                "k02_block_qkv_projection",
                "k03_block_qkv_permute",
                "k04_block_attn_scores",
                "k05_block_attn_softmax",
                "k06_block_attn_context",
                "k07_block_attn_output_projection",
                "k08_block_attn_residual_add",
                "k09_block_pre_mlp_layernorm",
                "k10_block_mlp_expand",
                "k11_block_mlp_gelu",
                "k12_block_mlp_project",
                "k13_block_mlp_residual_add",
            ],
            "suffix": ["k14_final_layernorm", "k15_logits_projection"],
            "num_layers": args.num_layers,
        }
        num_layers = args.num_layers
    payload = aggregate_events(
        trace_specs=trace_specs,
        profiles=profiles,
        num_layers=num_layers,
        iterations=args.iterations,
        execution_graph=execution_graph,
    )
    payload["metadata"] = {
        "profiles_path": args.profiles,
        "num_layers": num_layers,
        "iterations": args.iterations,
        "kernel_defs_path": args.kernel_defs,
        "workflow_path": args.workflow,
        "comparison_style": "paper_offline_aggregation",
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
