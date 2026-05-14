from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

from kerneldvfs.models import ProfileResult, read_json, write_json
from kerneldvfs.nvml_controller import BaseClockController, create_clock_controller
from kerneldvfs.paper_recreation import (
    PaperKernelSpec,
    build_family_inputs,
    expanded_trace_specs,
    paper_kernel_specs,
    run_family_kernel,
)

LOGGER = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


@dataclass(frozen=True)
class RunConfig:
    mode: str
    iterations: int
    num_layers: int
    device: str
    dtype_name: str
    poll_interval_us: int
    transition_timeout_ms: float


def resolve_dtype(name: str) -> Any:
    if torch is None:
        raise RuntimeError("runtime_compare.py requires torch")
    lookup = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in lookup:
        raise ValueError(f"Unsupported dtype '{name}'")
    return lookup[name]


def load_profiles(path: str) -> dict[str, ProfileResult]:
    payload = read_json(path)
    return {name: ProfileResult.from_dict(item) for name, item in payload["profiles"].items()}


def prepare_inputs(specs: list[PaperKernelSpec], device: str, dtype_name: str) -> dict[str, tuple[Any, ...]]:
    if torch is None:
        raise RuntimeError("runtime_compare.py requires torch")
    dtype = resolve_dtype(dtype_name)
    return {spec.kernel_name: build_family_inputs(spec, torch=torch, device=device, dtype=dtype) for spec in specs}


def measure_kernel_runtime_ms(spec: PaperKernelSpec, args: tuple[Any, ...], device: str) -> float:
    if torch is None:
        raise RuntimeError("runtime_compare.py requires torch")
    if device.startswith("cuda"):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        run_family_kernel(spec, torch=torch, args=args)
        end_event.record()
        torch.cuda.synchronize()
        return float(start_event.elapsed_time(end_event))
    start_ns = time.perf_counter_ns()
    run_family_kernel(spec, torch=torch, args=args)
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def sample_energy_delta_mj(
    controller: BaseClockController,
    start_energy_mj: int | None,
    start_power_mw: int | None,
    elapsed_ms: float,
) -> tuple[float | None, int | None]:
    end_energy_mj = controller.get_total_energy_consumption_mj()
    if start_energy_mj is not None and end_energy_mj is not None and end_energy_mj >= start_energy_mj:
        return float(end_energy_mj - start_energy_mj), end_energy_mj
    end_power_mw = controller.get_power_usage_mw()
    if start_power_mw is not None and end_power_mw is not None:
        avg_power_mw = (start_power_mw + end_power_mw) / 2.0
        return avg_power_mw * elapsed_ms / 1000.0, end_energy_mj
    return None, end_energy_mj


def wait_for_clock_target(
    controller: BaseClockController,
    target_clock: Any,
    poll_interval_us: int,
    timeout_ms: float,
) -> tuple[float | None, Any]:
    if target_clock is None:
        return None, controller.get_current_clock_setting()
    start_ns = time.perf_counter_ns()
    timeout_ns = int(timeout_ms * 1_000_000)
    poll_ns = max(1, poll_interval_us) * 1_000
    while True:
        current = controller.get_current_clock_setting()
        if current == target_clock:
            return (time.perf_counter_ns() - start_ns) / 1_000_000.0, current
        now_ns = time.perf_counter_ns()
        if now_ns - start_ns >= timeout_ns:
            return None, current
        remaining_ns = timeout_ns - (now_ns - start_ns)
        time.sleep(min(poll_ns, remaining_ns) / 1_000_000_000.0)


def run_workload(
    run_config: RunConfig,
    controller: BaseClockController,
    profiles: dict[str, ProfileResult],
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("runtime_compare.py requires torch")
    unique_specs = paper_kernel_specs(num_layers=run_config.num_layers)
    spec_by_name = {spec.kernel_name: spec for spec in unique_specs}
    trace_specs = expanded_trace_specs(num_layers=run_config.num_layers)
    inputs = prepare_inputs(unique_specs, device=run_config.device, dtype_name=run_config.dtype_name)

    if run_config.device.startswith("cuda"):
        torch.cuda.set_device(run_config.device)
        torch.cuda.synchronize()

    per_event: list[dict[str, Any]] = []
    total_kernel_ms = 0.0
    total_transition_ms = 0.0
    transition_observed_ms: list[float] = []
    missed_transitions = 0
    kernel_energy_mj_total = 0.0
    transition_energy_mj_total = 0.0
    energy_samples = 0
    workload_start_ns = time.perf_counter_ns()

    if run_config.mode == "auto":
        controller.reset_locked_clocks()

    for iteration in range(run_config.iterations):
        for event_index, event_spec in enumerate(trace_specs):
            profile = profiles[event_spec.kernel_name]
            requested_clock = profile.target_clock if run_config.mode == "profiled" else None
            transition_command_ms: float | None = None
            observed_transition_ms: float | None = None
            observed_clock = controller.get_current_clock_setting()

            if run_config.mode == "profiled":
                should_actuate = True
                if should_actuate:
                    transition_start_energy = controller.get_total_energy_consumption_mj()
                    transition_start_power = controller.get_power_usage_mw()
                    command_start_ns = time.perf_counter_ns()
                    controller.set_locked_clocks(requested_clock)
                    transition_command_ms = (time.perf_counter_ns() - command_start_ns) / 1_000_000.0
                    observed_transition_ms, observed_clock = wait_for_clock_target(
                        controller=controller,
                        target_clock=requested_clock,
                        poll_interval_us=run_config.poll_interval_us,
                        timeout_ms=run_config.transition_timeout_ms,
                    )
                    total_transition_ms += transition_command_ms
                    if observed_transition_ms is not None:
                        transition_observed_ms.append(observed_transition_ms)
                    else:
                        missed_transitions += 1
                    transition_energy_mj, _ = sample_energy_delta_mj(
                        controller=controller,
                        start_energy_mj=transition_start_energy,
                        start_power_mw=transition_start_power,
                        elapsed_ms=observed_transition_ms if observed_transition_ms is not None else transition_command_ms,
                    )
                    if transition_energy_mj is not None:
                        transition_energy_mj_total += transition_energy_mj
                        energy_samples += 1

            kernel_start_energy = controller.get_total_energy_consumption_mj()
            kernel_start_power = controller.get_power_usage_mw()
            kernel_runtime_ms = measure_kernel_runtime_ms(
                spec=spec_by_name[event_spec.kernel_name],
                args=inputs[event_spec.kernel_name],
                device=run_config.device,
            )
            kernel_energy_mj, _ = sample_energy_delta_mj(
                controller=controller,
                start_energy_mj=kernel_start_energy,
                start_power_mw=kernel_start_power,
                elapsed_ms=kernel_runtime_ms,
            )
            total_kernel_ms += kernel_runtime_ms
            if kernel_energy_mj is not None:
                kernel_energy_mj_total += kernel_energy_mj
                energy_samples += 1

            per_event.append(
                {
                    "iteration": iteration,
                    "event_index": event_index,
                    "kernel_name": event_spec.kernel_name,
                    "mode": run_config.mode,
                    "requested_clock": requested_clock.to_dict() if requested_clock is not None else None,
                    "observed_clock": observed_clock.to_dict() if observed_clock is not None else None,
                    "transition_command_ms": None if transition_command_ms is None else round(transition_command_ms, 6),
                    "transition_observed_ms": None if observed_transition_ms is None else round(observed_transition_ms, 6),
                    "kernel_runtime_ms": round(kernel_runtime_ms, 6),
                    "kernel_energy_mj": None if kernel_energy_mj is None else round(kernel_energy_mj, 6),
                }
            )

    if run_config.device.startswith("cuda"):
        torch.cuda.synchronize()
    workload_wall_ms = (time.perf_counter_ns() - workload_start_ns) / 1_000_000.0

    transition_summary = {
        "count": len(transition_observed_ms),
        "missed": missed_transitions,
        "avg_ms": round(mean(transition_observed_ms), 6) if transition_observed_ms else None,
        "max_ms": round(max(transition_observed_ms), 6) if transition_observed_ms else None,
        "min_ms": round(min(transition_observed_ms), 6) if transition_observed_ms else None,
    }
    total_energy_mj = kernel_energy_mj_total + transition_energy_mj_total if energy_samples > 0 else None
    return {
        "mode": run_config.mode,
        "iterations": run_config.iterations,
        "num_layers": run_config.num_layers,
        "events_per_iteration": len(trace_specs),
        "total_events": len(per_event),
        "time_to_complete_ms": round(workload_wall_ms, 6),
        "workload_wall_ms": round(workload_wall_ms, 6),
        "kernel_runtime_ms": round(total_kernel_ms, 6),
        "clock_transition_command_ms": round(total_transition_ms, 6),
        "kernel_energy_mj": round(kernel_energy_mj_total, 6) if energy_samples > 0 else None,
        "clock_transition_energy_mj": round(transition_energy_mj_total, 6) if energy_samples > 0 else None,
        "total_energy_mj": round(total_energy_mj, 6) if total_energy_mj is not None else None,
        "energy_supported": energy_samples > 0,
        "transition_latency": transition_summary,
        "events": per_event,
    }


def build_output(auto_result: dict[str, Any] | None, profiled_result: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    if auto_result is not None and profiled_result is not None:
        auto_wall = float(auto_result["time_to_complete_ms"])
        profiled_wall = float(profiled_result["time_to_complete_ms"])
        auto_energy = auto_result["total_energy_mj"]
        profiled_energy = profiled_result["total_energy_mj"]
        comparison = {
            "wall_runtime_delta_ms": round(profiled_wall - auto_wall, 6),
            "wall_runtime_delta_pct": round(((profiled_wall / auto_wall) - 1.0) * 100.0, 6) if auto_wall else None,
            "energy_delta_mj": round(profiled_energy - auto_energy, 6) if auto_energy is not None and profiled_energy is not None else None,
            "energy_delta_pct": round(((profiled_energy / auto_energy) - 1.0) * 100.0, 6) if auto_energy not in (None, 0) and profiled_energy is not None else None,
        }
    return {
        "metadata": {
            "profiles_path": args.profiles,
            "backend": args.backend,
            "device_index": args.device_index,
            "device": args.device,
            "dtype": args.dtype,
            "mode": args.mode,
            "iterations": args.iterations,
            "num_layers": args.num_layers,
            "profiled_per_kernel_actuation": True,
            "poll_interval_us": args.poll_interval_us,
            "transition_timeout_ms": args.transition_timeout_ms,
        },
        "auto": auto_result,
        "profiled": profiled_result,
        "comparison": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper workload sequentially and compare auto vs profiled clocks")
    parser.add_argument("--profiles", default="data/profiles.json")
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--mode", choices=["auto", "profiled", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--poll-interval-us", type=int, default=100)
    parser.add_argument("--transition-timeout-ms", type=float, default=50.0)
    parser.add_argument("--switching-latency-ms", type=float, default=2.5)
    parser.add_argument("--nvidia-smi-path", default=None)
    parser.add_argument("--nvidia-smi-sudo", action="store_true")
    parser.add_argument("--output", default="data/runtime_comparison.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    profiles = load_profiles(args.profiles)
    controller = create_clock_controller(
        backend=args.backend,
        device_index=args.device_index,
        switching_latency_ms=args.switching_latency_ms,
        nvidia_smi_path=args.nvidia_smi_path,
        nvidia_smi_sudo=args.nvidia_smi_sudo,
    )
    auto_result: dict[str, Any] | None = None
    profiled_result: dict[str, Any] | None = None

    try:
        if torch is None:
            raise RuntimeError("runtime_compare.py requires torch")
        if controller.mode == "real" and not args.device.startswith("cuda"):
            raise RuntimeError("real backend requires a CUDA device like cuda:0")

        if args.mode in {"auto", "both"}:
            LOGGER.info("Running workload with auto clocks")
            auto_result = run_workload(
                RunConfig(
                    mode="auto",
                    iterations=args.iterations,
                    num_layers=args.num_layers,
                    device=args.device,
                    dtype_name=args.dtype,
                    poll_interval_us=args.poll_interval_us,
                    transition_timeout_ms=args.transition_timeout_ms,
                ),
                controller=controller,
                profiles=profiles,
            )
            controller.reset_locked_clocks()

        if args.mode in {"profiled", "both"}:
            LOGGER.info("Running workload with profiled clocks")
            profiled_result = run_workload(
                RunConfig(
                    mode="profiled",
                    iterations=args.iterations,
                    num_layers=args.num_layers,
                    device=args.device,
                    dtype_name=args.dtype,
                    poll_interval_us=args.poll_interval_us,
                    transition_timeout_ms=args.transition_timeout_ms,
                ),
                controller=controller,
                profiles=profiles,
            )

        write_json(args.output, build_output(auto_result, profiled_result, args))
        LOGGER.info("Wrote runtime comparison to %s", args.output)
    finally:
        try:
            controller.reset_locked_clocks()
        except Exception:
            LOGGER.exception("Failed to reset locked clocks during teardown")
        controller.close()


if __name__ == "__main__":
    main()
