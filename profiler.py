from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import time
from typing import Any

from kerneldvfs.models import ClockSetting, ProfileResult, write_json
from kerneldvfs.nvml_controller import BaseClockController, create_clock_controller
from kerneldvfs.paper_recreation import (
    PaperKernelSpec,
    build_family_inputs,
    family_category,
    paper_kernel_specs,
    run_family_kernel,
    spec_shapes,
)

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
    inputs: dict[str, Any]
    energy_source: str


def default_workloads(num_layers: int) -> list[KernelWorkload]:
    specs = paper_kernel_specs(num_layers=num_layers)
    return [
        KernelWorkload(
            kernel_name=spec.kernel_name,
            category=family_category(spec.family),
            baseline_ms=spec.baseline_ms,
            optimal_core_mhz=spec.optimal_core_mhz,
            optimal_mem_mhz=spec.optimal_mem_mhz,
            static_power_watts=spec.static_power_watts,
            dynamic_power_watts=spec.dynamic_power_watts,
        )
        for spec in specs
    ]


class BenchmarkHarness:
    def __init__(
        self,
        controller: BaseClockController,
        tolerated_slowdown_pct: float,
        measurement: MeasurementConfig,
        specs: list[PaperKernelSpec],
    ) -> None:
        self.controller = controller
        self.tolerated_slowdown_pct = tolerated_slowdown_pct
        self.measurement = measurement
        self.specs = {spec.kernel_name: spec for spec in specs}
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

        torch.cuda.set_device(self.measurement.device)
        dtype = torch.float16
        inputs = {
            spec.kernel_name: build_family_inputs(spec, torch=torch, device=self.measurement.device, dtype=dtype)
            for spec in self.specs.values()
        }
        energy_source = "nvml_total_energy" if self.controller.get_total_energy_consumption_mj() is not None else "nvml_power_samples"
        return RealMeasurementContext(
            torch=torch,
            inputs=inputs,
            energy_source=energy_source,
        )

    def profile(self, workloads: list[KernelWorkload]) -> list[ProfileResult]:
        measurement_mode = self._resolve_measurement_mode()
        candidate_pairs = self.controller.get_supported_clock_pairs()
        results: list[ProfileResult] = []

        for workload in workloads:
            baseline_ms, baseline_energy = self._measure_auto(workload)
            selected_clock = max(candidate_pairs, key=lambda pair: pair.score())
            selected_runtime_ms = baseline_ms
            selected_energy = baseline_energy

            for candidate in candidate_pairs:
                runtime_ms, energy_mj = self._measure(workload, candidate)
                allowed_runtime_ms = baseline_ms * (1.0 + self.tolerated_slowdown_pct / 100.0)
                if runtime_ms <= allowed_runtime_ms and energy_mj <= selected_energy:
                    selected_clock = candidate
                    selected_runtime_ms = runtime_ms
                    selected_energy = energy_mj

            spec = self.specs[workload.kernel_name]
            result = ProfileResult(
                kernel_name=workload.kernel_name,
                target_clock=selected_clock,
                baseline_ms=baseline_ms,
                selected_runtime_ms=selected_runtime_ms,
                estimated_energy_mj=selected_energy,
                backend=self.controller.mode,
                metadata={
                    "category": workload.category,
                    "family": spec.family,
                    "phase": spec.phase,
                    "paper_kernel_description": spec.description,
                    "repeat_count": spec.repeat_count,
                    "shapes": spec_shapes(spec),
                    "tolerated_slowdown_pct": self.tolerated_slowdown_pct,
                    "supported_pairs": len(candidate_pairs),
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

    def _measure_auto(self, workload: KernelWorkload) -> tuple[float, float]:
        if self._resolve_measurement_mode() == "real":
            return self._measure_real_auto(workload)
        return self._measure_mock_auto(workload)

    def _measure(self, workload: KernelWorkload, setting: ClockSetting) -> tuple[float, float]:
        if self._resolve_measurement_mode() == "real":
            return self._measure_real(workload, setting)
        return self._measure_mock(workload, setting)

    def _measure_real_auto(self, workload: KernelWorkload) -> tuple[float, float]:
        try:
            self.controller.reset_locked_clocks()
        except Exception:
            LOGGER.debug("Failed to reset clocks before auto baseline measurement", exc_info=True)
        time.sleep(max(self.measurement.settle_time_ms, self.controller.switching_latency_ms) / 1000.0)
        return self._measure_real_impl(workload)

    def _measure_real(self, workload: KernelWorkload, setting: ClockSetting) -> tuple[float, float]:
        self.controller.set_locked_clocks(setting)
        time.sleep(max(self.measurement.settle_time_ms, self.controller.switching_latency_ms) / 1000.0)
        return self._measure_real_impl(workload)

    def _measure_real_impl(self, workload: KernelWorkload) -> tuple[float, float]:
        assert self._real_context is not None
        context = self._real_context
        spec = self.specs[workload.kernel_name]
        args = context.inputs[workload.kernel_name]
        if not isinstance(args, tuple):
            args = (args,)

        for _ in range(self.measurement.warmup_runs):
            run_family_kernel(spec, torch=context.torch, args=args)
        context.torch.cuda.synchronize()

        start_energy_mj = self.controller.get_total_energy_consumption_mj()
        timings_ms: list[float] = []
        sampled_energy_mj = 0.0

        for _ in range(self.measurement.benchmark_runs):
            start_power_mw = self.controller.get_power_usage_mw()
            start_event = context.torch.cuda.Event(enable_timing=True)
            end_event = context.torch.cuda.Event(enable_timing=True)
            start_event.record()
            run_family_kernel(spec, torch=context.torch, args=args)
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

    def _measure_mock_auto(self, workload: KernelWorkload) -> tuple[float, float]:
        runtime_ms = workload.baseline_ms
        auto_core_mhz = max(workload.optimal_core_mhz, 1680)
        auto_mem_mhz = max(workload.optimal_mem_mhz, 9501)
        power_watts = workload.static_power_watts + workload.dynamic_power_watts
        energy_mj = power_watts * (runtime_ms / 1000.0) * 1000.0
        LOGGER.debug(
            "Mock auto baseline for %s assumed at core=%s mem=%s",
            workload.kernel_name,
            auto_core_mhz,
            auto_mem_mhz,
        )
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
            0.55 * (setting.core_mhz / max(workload.optimal_core_mhz, 1))
            + 0.45 * (setting.mem_mhz / max(workload.optimal_mem_mhz, 1))
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
    parser.add_argument("--nvidia-smi-path", default=None)
    parser.add_argument("--nvidia-smi-sudo", action="store_true")
    parser.add_argument("--tolerated-slowdown-pct", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=24)
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
            "num_layers": args.num_layers,
            "tolerated_slowdown_pct": args.tolerated_slowdown_pct,
            "switching_latency_ms": args.switching_latency_ms,
            "measurement_mode": args.measurement_mode,
            "nvidia_smi_path": args.nvidia_smi_path,
            "nvidia_smi_sudo": args.nvidia_smi_sudo,
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
            "settle_time_ms": args.settle_time_ms,
            "workload_count": len(results),
            "profile_style": "paper_local_waste",
            "baseline_clock_mode": "auto",
            "benchmarkable_kernels": [result.kernel_name for result in results],
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
        nvidia_smi_path=args.nvidia_smi_path,
        nvidia_smi_sudo=args.nvidia_smi_sudo,
    )
    try:
        measurement = MeasurementConfig(
            mode=args.measurement_mode,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
            device=f"cuda:{args.device_index}",
            settle_time_ms=args.settle_time_ms,
        )
        specs = paper_kernel_specs(num_layers=args.num_layers)
        harness = BenchmarkHarness(
            controller=controller,
            tolerated_slowdown_pct=args.tolerated_slowdown_pct,
            measurement=measurement,
            specs=specs,
        )
        results = harness.profile(default_workloads(num_layers=args.num_layers))
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
