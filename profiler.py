from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from typing import Any

from kerneldvfs.kernels import benchmarkable_kernels, kernel_entrypoints
from kerneldvfs.models import ClockSetting, ProfileResult, write_json
from kerneldvfs.nvml_controller import BaseClockController, create_clock_controller

LOGGER = logging.getLogger(__name__)

try:
    import kernel_tuner  # type: ignore  # noqa: F401
except ImportError:
    kernel_tuner = None

try:
    import torch  # type: ignore  # noqa: F401
except ImportError:
    torch = None

try:
    import triton  # type: ignore  # noqa: F401
except ImportError:
    triton = None


@dataclass(frozen=True)
class KernelWorkload:
    kernel_name: str
    category: str
    baseline_ms: float
    optimal_core_mhz: int
    optimal_mem_mhz: int
    static_power_watts: float
    dynamic_power_watts: float


def default_workloads() -> list[KernelWorkload]:
    return [
        KernelWorkload("fused_qkv_matmul", "compute", 1.80, 1230, 1215, 62.0, 126.0),
        KernelWorkload("rmsnorm", "memory", 0.42, 1080, 1215, 48.0, 52.0),
        KernelWorkload("attention_scores_matmul", "compute", 1.55, 1230, 1215, 60.0, 118.0),
        KernelWorkload("masked_softmax", "memory", 0.38, 1080, 1593, 46.0, 54.0),
        KernelWorkload("attention_value_matmul", "compute", 1.62, 1230, 1215, 60.0, 120.0),
        KernelWorkload("mlp_up_proj_matmul", "compute", 1.90, 1410, 1215, 66.0, 140.0),
        KernelWorkload("silu_gate", "memory", 0.31, 1080, 1215, 45.0, 42.0),
        KernelWorkload("mlp_down_proj_matmul", "compute", 1.70, 1230, 1215, 62.0, 128.0),
    ]


class BenchmarkHarness:
    def __init__(self, controller: BaseClockController, margin_ms: float) -> None:
        self.controller = controller
        self.margin_ms = margin_ms

    def profile(self, workloads: list[KernelWorkload]) -> list[ProfileResult]:
        supported_pairs = self.controller.get_supported_clock_pairs()
        max_pair = max(supported_pairs, key=lambda pair: pair.score())
        ascending_pairs = sorted(supported_pairs, key=lambda pair: pair.score())
        results: list[ProfileResult] = []

        for workload in workloads:
            baseline_ms, baseline_energy = self._measure(workload, max_pair)
            selected_clock = max_pair
            selected_runtime_ms = baseline_ms
            selected_energy = baseline_energy

            for candidate in ascending_pairs:
                runtime_ms, energy_mj = self._measure(workload, candidate)
                if runtime_ms <= baseline_ms + self.margin_ms:
                    selected_clock = candidate
                    selected_runtime_ms = runtime_ms
                    selected_energy = energy_mj
                    break

            result = ProfileResult(
                kernel_name=workload.kernel_name,
                target_clock=selected_clock,
                baseline_ms=baseline_ms,
                selected_runtime_ms=selected_runtime_ms,
                estimated_energy_mj=selected_energy,
                backend=self.controller.mode,
                metadata={
                    "category": workload.category,
                    "margin_ms": self.margin_ms,
                    "supported_pairs": len(supported_pairs),
                    "kernel_tuner_available": kernel_tuner is not None,
                    "torch_available": torch is not None,
                    "triton_available": triton is not None,
                },
            )
            LOGGER.info(
                "Profiled %s -> core=%s mem=%s baseline=%.4fms selected=%.4fms energy=%.4fmJ",
                workload.kernel_name,
                result.target_clock.core_mhz,
                result.target_clock.mem_mhz,
                result.baseline_ms,
                result.selected_runtime_ms,
                result.estimated_energy_mj,
            )
            results.append(result)
        return results

    def _measure(self, workload: KernelWorkload, setting: ClockSetting) -> tuple[float, float]:
        core_penalty = max(0.0, workload.optimal_core_mhz - setting.core_mhz) / max(workload.optimal_core_mhz, 1)
        mem_penalty = max(0.0, workload.optimal_mem_mhz - setting.mem_mhz) / max(workload.optimal_mem_mhz, 1)

        if workload.category == "compute":
            slowdown_factor = 1.0 + (1.15 * core_penalty) + (0.25 * mem_penalty)
        else:
            slowdown_factor = 1.0 + (0.25 * core_penalty) + (1.10 * mem_penalty)

        runtime_ms = workload.baseline_ms * slowdown_factor
        power_watts = workload.static_power_watts + workload.dynamic_power_watts * (
            0.55 * (setting.core_mhz / max(workload.optimal_core_mhz, setting.core_mhz))
            + 0.45 * (setting.mem_mhz / max(workload.optimal_mem_mhz, setting.mem_mhz))
        )
        energy_mj = power_watts * (runtime_ms / 1000.0) * 1000.0
        return runtime_ms, energy_mj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KernelDVFS offline profiler")
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--margin-ms", type=float, default=0.02)
    parser.add_argument("--switching-latency-ms", type=float, default=2.5)
    parser.add_argument("--output", default="data/profiles.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_output(results: list[ProfileResult], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "metadata": {
            "backend": args.backend,
            "device_index": args.device_index,
            "margin_ms": args.margin_ms,
            "switching_latency_ms": args.switching_latency_ms,
            "workload_count": len(results),
            "kernel_module": "kerneldvfs/kernels.py",
            "kernel_entrypoints": kernel_entrypoints(),
            "benchmarkable_kernels": benchmarkable_kernels(),
        },
        "profiles": {result.kernel_name: result.to_dict() for result in results},
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    controller = create_clock_controller(
        backend=args.backend,
        device_index=args.device_index,
        switching_latency_ms=args.switching_latency_ms,
    )
    try:
        harness = BenchmarkHarness(controller=controller, margin_ms=args.margin_ms)
        results = harness.profile(default_workloads())
        payload = build_output(results, args)
        write_json(args.output, payload)
        LOGGER.info("Wrote profiles to %s", args.output)
    finally:
        try:
            controller.reset_locked_clocks()
        except Exception:
            LOGGER.debug("Clock reset skipped or failed during profiler teardown", exc_info=True)
        controller.close()


if __name__ == "__main__":
    main()
