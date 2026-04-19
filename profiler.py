from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import logging
import time
from typing import Any

from kerneldvfs.models import ClockSetting, ProfileResult, write_json
from kerneldvfs.nvml_controller import BaseClockController, create_clock_controller

LOGGER = logging.getLogger(__name__)

KERNEL_ENTRYPOINTS = {
    "fused_qkv_matmul": "launch_matmul",
    "attention_scores_matmul": "launch_matmul",
    "attention_value_matmul": "launch_matmul",
    "mlp_up_proj_matmul": "launch_matmul",
    "mlp_down_proj_matmul": "launch_matmul",
    "rmsnorm": "launch_rmsnorm",
    "masked_softmax": "launch_softmax",
    "silu_gate": "launch_silu_gate",
}

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


@dataclass(frozen=True)
class MeasurementConfig:
    mode: str
    warmup_runs: int
    benchmark_runs: int
    device: str
    settle_time_ms: float


@dataclass
class RealMeasurementContext:
    torch: Any
    kernels_module: Any
    inputs: dict[str, Any]
    launchers: dict[str, Any]
    energy_source: str


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
    def __init__(self, controller: BaseClockController, margin_ms: float, measurement: MeasurementConfig) -> None:
        self.controller = controller
        self.margin_ms = margin_ms
        self.measurement = measurement
        self._resolved_measurement_mode: str | None = None
        self._real_context: RealMeasurementContext | None = None

    def _resolve_measurement_mode(self) -> str:
        if self._resolved_measurement_mode is not None:
            return self._resolved_measurement_mode
        requested = self.measurement.mode
        if requested == "mock":
            self._resolved_measurement_mode = "mock"
            return self._resolved_measurement_mode
        try:
            self._real_context = self._build_real_context()
            self._resolved_measurement_mode = "real"
        except Exception as exc:
            if requested == "real":
                raise RuntimeError(f"Real profiling requested, but GPU kernel benchmarking is unavailable: {exc}") from exc
            LOGGER.warning("Falling back to mock profiler measurements: %s", exc)
            self._resolved_measurement_mode = "mock"
        return self._resolved_measurement_mode

    def _build_real_context(self) -> RealMeasurementContext:
        if self.controller.mode != "real":
            raise RuntimeError(f"real measurement requires a real NVML controller, got '{self.controller.mode}'")
        if torch is None:
            raise RuntimeError("torch is not installed")
        if not self.measurement.device.startswith("cuda"):
            raise RuntimeError(f"real measurement requires a CUDA device, got '{self.measurement.device}'")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")

        kernels_module = importlib.import_module("kerneldvfs.kernels")
        if not getattr(kernels_module, "TRITON_AVAILABLE", False):
            raise RuntimeError("kerneldvfs.kernels could not initialize Triton")

        torch.cuda.set_device(self.measurement.device)
        inputs = kernels_module.example_inputs(device=self.measurement.device)
        launchers = {
            "fused_qkv_matmul": kernels_module.launch_matmul,
            "attention_scores_matmul": kernels_module.launch_matmul,
            "attention_value_matmul": kernels_module.launch_matmul,
            "mlp_up_proj_matmul": kernels_module.launch_matmul,
            "mlp_down_proj_matmul": kernels_module.launch_matmul,
            "rmsnorm": kernels_module.launch_rmsnorm,
            "masked_softmax": kernels_module.launch_softmax,
            "silu_gate": kernels_module.launch_silu_gate,
        }
        energy_source = "nvml_total_energy" if self.controller.get_total_energy_consumption_mj() is not None else "nvml_power_samples"
        return RealMeasurementContext(
            torch=torch,
            kernels_module=kernels_module,
            inputs=inputs,
            launchers=launchers,
            energy_source=energy_source,
        )

    def profile(self, workloads: list[KernelWorkload]) -> list[ProfileResult]:
        measurement_mode = self._resolve_measurement_mode()
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
                    "measurement_mode": measurement_mode,
                    "warmup_runs": self.measurement.warmup_runs,
                    "benchmark_runs": self.measurement.benchmark_runs,
                    "energy_source": self._energy_source(measurement_mode),
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
        if self._resolve_measurement_mode() == "real":
            return self._measure_real(workload, setting)
        return self._measure_mock(workload, setting)

    def _measure_real(self, workload: KernelWorkload, setting: ClockSetting) -> tuple[float, float]:
        assert self._real_context is not None
        context = self._real_context
        launch = context.launchers[workload.kernel_name]
        args = context.inputs[workload.kernel_name]
        if not isinstance(args, tuple):
            args = (args,)

        self.controller.set_locked_clocks(setting)
        time.sleep(max(self.measurement.settle_time_ms, self.controller.switching_latency_ms) / 1000.0)

        for _ in range(self.measurement.warmup_runs):
            launch(*args)
        context.torch.cuda.synchronize()

        start_energy_mj = self.controller.get_total_energy_consumption_mj()
        timings_ms: list[float] = []
        sampled_energy_mj = 0.0

        for _ in range(self.measurement.benchmark_runs):
            start_power_mw = self.controller.get_power_usage_mw()
            start_event = context.torch.cuda.Event(enable_timing=True)
            end_event = context.torch.cuda.Event(enable_timing=True)
            start_event.record()
            launch(*args)
            end_event.record()
            context.torch.cuda.synchronize()
            elapsed_ms = float(start_event.elapsed_time(end_event))
            timings_ms.append(elapsed_ms)
            end_power_mw = self.controller.get_power_usage_mw()
            if start_energy_mj is None and start_power_mw is not None and end_power_mw is not None:
                sampled_energy_mj += ((start_power_mw + end_power_mw) / 2.0) * elapsed_ms / 1000.0

        end_energy_mj = self.controller.get_total_energy_consumption_mj()
        runtime_ms = sum(timings_ms) / len(timings_ms)
        if start_energy_mj is not None and end_energy_mj is not None and end_energy_mj >= start_energy_mj:
            energy_mj = (end_energy_mj - start_energy_mj) / max(self.measurement.benchmark_runs, 1)
        else:
            energy_mj = sampled_energy_mj / max(self.measurement.benchmark_runs, 1)
        return runtime_ms, energy_mj

    def _measure_mock(self, workload: KernelWorkload, setting: ClockSetting) -> tuple[float, float]:
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

    def _energy_source(self, measurement_mode: str) -> str:
        if measurement_mode == "mock":
            return "synthetic_model"
        assert self._real_context is not None
        return self._real_context.energy_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KernelDVFS offline profiler")
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--measurement-mode", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--margin-ms", type=float, default=0.02)
    parser.add_argument("--switching-latency-ms", type=float, default=2.5)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=25)
    parser.add_argument("--settle-time-ms", type=float, default=10.0)
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
            "measurement_mode": args.measurement_mode,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
            "settle_time_ms": args.settle_time_ms,
            "workload_count": len(results),
            "kernel_module": "kerneldvfs/kernels.py",
            "kernel_entrypoints": KERNEL_ENTRYPOINTS,
            "benchmarkable_kernels": list(KERNEL_ENTRYPOINTS.keys()),
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
        measurement = MeasurementConfig(
            mode=args.measurement_mode,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            device=f"cuda:{args.device_index}",
            settle_time_ms=args.settle_time_ms,
        )
        harness = BenchmarkHarness(controller=controller, margin_ms=args.margin_ms, measurement=measurement)
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
