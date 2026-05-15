from __future__ import annotations

import argparse
from typing import Any

from kerneldvfs.models import ProfileResult, read_json, write_json
from kerneldvfs.paper_recreation import PaperKernelSpec, expanded_trace_specs, paper_kernel_specs


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
) -> dict[str, Any]:
    unique_specs = paper_kernel_specs(num_layers=num_layers)
    prefix, repeated, suffix = identify_trace_regions(unique_specs, num_layers=num_layers)
    prefix_names = {spec.kernel_name for spec in prefix}
    repeated_names = {spec.kernel_name for spec in repeated}
    suffix_names = {spec.kernel_name for spec in suffix}

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
                layer_index = repeated_cursor // len(repeated)
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
        "execution_graph": {
            "prefix": [spec.kernel_name for spec in prefix],
            "layer_kernel_order": [spec.kernel_name for spec in repeated],
            "suffix": [spec.kernel_name for spec in suffix],
            "num_layers": num_layers,
        },
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
    parser.add_argument("--output", default="data/runtime_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.profiles)
    trace_specs = expanded_trace_specs(num_layers=args.num_layers)
    payload = aggregate_events(
        trace_specs=trace_specs,
        profiles=profiles,
        num_layers=args.num_layers,
        iterations=args.iterations,
    )
    payload["metadata"] = {
        "profiles_path": args.profiles,
        "num_layers": args.num_layers,
        "iterations": args.iterations,
        "comparison_style": "paper_offline_aggregation",
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
