from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Iterable

from .models import ClockSetting

LOGGER = logging.getLogger(__name__)

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None


class NVMLControllerError(RuntimeError):
    """Raised when clock control fails."""


@dataclass
class ControllerCapabilities:
    mode: str
    switching_latency_ms: float


class BaseClockController:
    def __init__(self, device_index: int = 0, switching_latency_ms: float = 2.5) -> None:
        self.device_index = device_index
        self.switching_latency_ms = switching_latency_ms

    @property
    def capabilities(self) -> ControllerCapabilities:
        return ControllerCapabilities(mode=self.mode, switching_latency_ms=self.switching_latency_ms)

    @property
    def mode(self) -> str:
        raise NotImplementedError

    def get_supported_clock_pairs(self) -> list[ClockSetting]:
        raise NotImplementedError

    def set_locked_clocks(self, setting: ClockSetting) -> None:
        raise NotImplementedError

    def reset_locked_clocks(self) -> None:
        raise NotImplementedError

    def get_power_usage_mw(self) -> int | None:
        return None

    def get_total_energy_consumption_mj(self) -> int | None:
        return None

    def close(self) -> None:
        return None

    @staticmethod
    def _deduplicate(settings: Iterable[ClockSetting]) -> list[ClockSetting]:
        deduped = sorted(set(settings), key=lambda item: item.score())
        return deduped


class MockClockController(BaseClockController):
    def __init__(self, device_index: int = 0, switching_latency_ms: float = 2.5) -> None:
        super().__init__(device_index=device_index, switching_latency_ms=switching_latency_ms)
        self._current: ClockSetting | None = None
        self._history: list[dict[str, float | int]] = []

    @property
    def mode(self) -> str:
        return "mock"

    def get_supported_clock_pairs(self) -> list[ClockSetting]:
        core_clocks = [900, 1080, 1230, 1410]
        mem_clocks = [877, 1215, 1593]
        return self._deduplicate(ClockSetting(core_mhz=core, mem_mhz=mem) for core in core_clocks for mem in mem_clocks)

    def set_locked_clocks(self, setting: ClockSetting) -> None:
        start = time.perf_counter()
        time.sleep(self.switching_latency_ms / 1000.0)
        self._current = setting
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._history.append(
            {
                "ts_ms": time.perf_counter() * 1000.0,
                "core_mhz": setting.core_mhz,
                "mem_mhz": setting.mem_mhz,
                "elapsed_ms": elapsed_ms,
            }
        )
        LOGGER.info("Mock clock transition applied: core=%s mem=%s elapsed=%.3fms", setting.core_mhz, setting.mem_mhz, elapsed_ms)

    def reset_locked_clocks(self) -> None:
        self._current = None
        LOGGER.info("Mock clock transition reset")


class PynvmlClockController(BaseClockController):
    def __init__(self, device_index: int = 0, switching_latency_ms: float = 2.5) -> None:
        if pynvml is None:
            raise NVMLControllerError("pynvml is not installed")
        super().__init__(device_index=device_index, switching_latency_ms=switching_latency_ms)
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:
            raise NVMLControllerError(f"Unable to initialize NVML device {device_index}: {exc}") from exc

    @property
    def mode(self) -> str:
        return "real"

    def get_supported_clock_pairs(self) -> list[ClockSetting]:
        pairs: list[ClockSetting] = []
        try:
            memory_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(self._handle)
            for mem_clock in memory_clocks:
                graphics_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(self._handle, mem_clock)
                pairs.extend(ClockSetting(core_mhz=int(core_clock), mem_mhz=int(mem_clock)) for core_clock in graphics_clocks)
        except Exception as exc:
            raise NVMLControllerError(f"Failed to query supported clock pairs: {exc}") from exc
        deduped = self._deduplicate(pairs)
        if not deduped:
            raise NVMLControllerError("No supported clock pairs returned by NVML")
        return deduped

    def set_locked_clocks(self, setting: ClockSetting) -> None:
        try:
            pynvml.nvmlDeviceSetMemoryLockedClocks(self._handle, setting.mem_mhz, setting.mem_mhz)
            pynvml.nvmlDeviceSetGpuLockedClocks(self._handle, setting.core_mhz, setting.core_mhz)
        except Exception as exc:
            raise NVMLControllerError(
                f"Failed to set locked clocks core={setting.core_mhz} mem={setting.mem_mhz}: {exc}"
            ) from exc
        LOGGER.info("Locked real GPU clocks to core=%s mem=%s", setting.core_mhz, setting.mem_mhz)

    def reset_locked_clocks(self) -> None:
        errors: list[str] = []
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
        except Exception as exc:
            errors.append(f"gpu reset failed: {exc}")
        try:
            pynvml.nvmlDeviceResetMemoryLockedClocks(self._handle)
        except Exception as exc:
            errors.append(f"memory reset failed: {exc}")
        if errors:
            raise NVMLControllerError("; ".join(errors))
        LOGGER.info("Reset real GPU clock locks")

    def get_power_usage_mw(self) -> int | None:
        try:
            return int(pynvml.nvmlDeviceGetPowerUsage(self._handle))
        except Exception:
            LOGGER.debug("Failed to query NVML power usage", exc_info=True)
            return None

    def get_total_energy_consumption_mj(self) -> int | None:
        try:
            return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            LOGGER.debug("Failed to query NVML total energy consumption", exc_info=True)
            return None

    def close(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            LOGGER.debug("nvmlShutdown() failed during controller close", exc_info=True)


def create_clock_controller(
    backend: str = "auto",
    device_index: int = 0,
    switching_latency_ms: float = 2.5,
) -> BaseClockController:
    if backend == "mock":
        return MockClockController(device_index=device_index, switching_latency_ms=switching_latency_ms)
    if backend == "real":
        return PynvmlClockController(device_index=device_index, switching_latency_ms=switching_latency_ms)
    if backend != "auto":
        raise ValueError(f"Unsupported backend '{backend}'")
    try:
        return PynvmlClockController(device_index=device_index, switching_latency_ms=switching_latency_ms)
    except Exception as exc:
        LOGGER.warning("Falling back to mock clock controller: %s", exc)
        return MockClockController(device_index=device_index, switching_latency_ms=switching_latency_ms)
