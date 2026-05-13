from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import shutil
import subprocess
import time
from typing import Iterable

from .models import ClockSetting
from .paper_recreation import PAPER_CORE_CLOCKS, PAPER_MEMORY_CLOCKS

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

    def get_current_clock_setting(self) -> ClockSetting | None:
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
        core_clocks = PAPER_CORE_CLOCKS
        mem_clocks = PAPER_MEMORY_CLOCKS
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

    def get_current_clock_setting(self) -> ClockSetting | None:
        return self._current


class PynvmlClockController(BaseClockController):
    def __init__(
        self,
        device_index: int = 0,
        switching_latency_ms: float = 2.5,
        nvidia_smi_path: str | None = None,
        nvidia_smi_sudo: bool = False,
    ) -> None:
        if pynvml is None:
            raise NVMLControllerError("pynvml is not installed")
        super().__init__(device_index=device_index, switching_latency_ms=switching_latency_ms)
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._clock_api: str | None = None
            resolved_path = nvidia_smi_path or os.environ.get("KERNELDVFS_NVIDIA_SMI", "nvidia-smi")
            self._nvidia_smi = shutil.which(resolved_path) if os.path.basename(resolved_path) == resolved_path else resolved_path
            self._use_sudo_for_nvidia_smi = nvidia_smi_sudo or os.environ.get("KERNELDVFS_NVIDIA_SMI_SUDO", "0") == "1"
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
            self._clock_api = "locked_clocks"
            LOGGER.info("Locked real GPU clocks via locked-clocks API to core=%s mem=%s", setting.core_mhz, setting.mem_mhz)
            return
        except Exception as exc:
            LOGGER.warning(
                "Locked-clocks API unavailable for core=%s mem=%s, trying applications clocks: %s",
                setting.core_mhz,
                setting.mem_mhz,
                exc,
            )

        try:
            pynvml.nvmlDeviceSetApplicationsClocks(self._handle, setting.mem_mhz, setting.core_mhz)
            self._clock_api = "applications_clocks"
            LOGGER.info(
                "Locked real GPU clocks via applications-clocks API to core=%s mem=%s",
                setting.core_mhz,
                setting.mem_mhz,
            )
            return
        except Exception as exc:
            LOGGER.warning(
                "Applications-clocks API unavailable for core=%s mem=%s, trying nvidia-smi: %s",
                setting.core_mhz,
                setting.mem_mhz,
                exc,
            )

        self._set_clocks_via_nvidia_smi(setting)

    def reset_locked_clocks(self) -> None:
        errors: list[str] = []
        if self._clock_api in (None, "locked_clocks"):
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
            except Exception as exc:
                if self._clock_api == "locked_clocks":
                    errors.append(f"gpu reset failed: {exc}")
            try:
                pynvml.nvmlDeviceResetMemoryLockedClocks(self._handle)
            except Exception as exc:
                if self._clock_api == "locked_clocks":
                    errors.append(f"memory reset failed: {exc}")
        if self._clock_api == "applications_clocks":
            try:
                pynvml.nvmlDeviceResetApplicationsClocks(self._handle)
            except Exception as exc:
                errors.append(f"applications reset failed: {exc}")
        if self._clock_api == "nvidia_smi_locked_clocks":
            try:
                self._reset_clocks_via_nvidia_smi()
            except Exception as exc:
                errors.append(str(exc))
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

    def get_current_clock_setting(self) -> ClockSetting | None:
        try:
            core_mhz = int(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_GRAPHICS))
            mem_mhz = int(pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM))
            return ClockSetting(core_mhz=core_mhz, mem_mhz=mem_mhz)
        except Exception:
            LOGGER.debug("Failed to query NVML current clock info", exc_info=True)
            return None

    def close(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            LOGGER.debug("nvmlShutdown() failed during controller close", exc_info=True)

    def _nvidia_smi_prefix(self) -> list[str]:
        if self._nvidia_smi is None:
            raise NVMLControllerError("nvidia-smi not found on PATH")
        if self._use_sudo_for_nvidia_smi:
            return ["sudo", "-n", self._nvidia_smi]
        return [self._nvidia_smi]

    def _run_nvidia_smi(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = self._nvidia_smi_prefix() + ["-i", str(self.device_index), *args]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or f"exit status {completed.returncode}"
            raise NVMLControllerError(f"nvidia-smi {' '.join(args)} failed: {stderr}")
        return completed

    def _set_clocks_via_nvidia_smi(self, setting: ClockSetting) -> None:
        errors: list[str] = []

        try:
            self._run_nvidia_smi("-pm", "1")
        except Exception as exc:
            LOGGER.debug("Unable to enable persistence mode before clock lock", exc_info=True)
            errors.append(str(exc))

        try:
            self._run_nvidia_smi("-lgc", f"{setting.core_mhz},{setting.core_mhz}")
            try:
                self._run_nvidia_smi("-lmc", f"{setting.mem_mhz},{setting.mem_mhz}")
            except Exception as exc:
                LOGGER.warning("nvidia-smi memory lock failed for mem=%s: %s", setting.mem_mhz, exc)
            self._clock_api = "nvidia_smi_locked_clocks"
            LOGGER.info(
                "Locked real GPU clocks via nvidia-smi locked-clocks API to core=%s mem=%s",
                setting.core_mhz,
                setting.mem_mhz,
            )
            return
        except Exception as exc:
            errors.append(str(exc))

        try:
            self._run_nvidia_smi("-ac", f"{setting.mem_mhz},{setting.core_mhz}")
            self._clock_api = "applications_clocks"
            LOGGER.info(
                "Locked real GPU clocks via nvidia-smi applications-clocks API to core=%s mem=%s",
                setting.core_mhz,
                setting.mem_mhz,
            )
            return
        except Exception as exc:
            errors.append(str(exc))

        raise NVMLControllerError(
            f"Failed to set clocks core={setting.core_mhz} mem={setting.mem_mhz} using NVML and nvidia-smi fallbacks: {'; '.join(errors)}"
        )

    def _reset_clocks_via_nvidia_smi(self) -> None:
        errors: list[str] = []
        try:
            self._run_nvidia_smi("-rgc")
        except Exception as exc:
            errors.append(str(exc))
        try:
            self._run_nvidia_smi("-rmc")
        except Exception:
            LOGGER.debug("nvidia-smi does not support -rmc reset on this GPU", exc_info=True)
        try:
            self._run_nvidia_smi("-rac")
        except Exception:
            LOGGER.debug("nvidia-smi applications-clocks reset unavailable on this GPU", exc_info=True)
        if errors:
            raise NVMLControllerError("; ".join(errors))


def create_clock_controller(
    backend: str = "auto",
    device_index: int = 0,
    switching_latency_ms: float = 2.5,
    nvidia_smi_path: str | None = None,
    nvidia_smi_sudo: bool = False,
) -> BaseClockController:
    if backend == "mock":
        return MockClockController(device_index=device_index, switching_latency_ms=switching_latency_ms)
    if backend == "real":
        return PynvmlClockController(
            device_index=device_index,
            switching_latency_ms=switching_latency_ms,
            nvidia_smi_path=nvidia_smi_path,
            nvidia_smi_sudo=nvidia_smi_sudo,
        )
    if backend != "auto":
        raise ValueError(f"Unsupported backend '{backend}'")
    try:
        return PynvmlClockController(
            device_index=device_index,
            switching_latency_ms=switching_latency_ms,
            nvidia_smi_path=nvidia_smi_path,
            nvidia_smi_sudo=nvidia_smi_sudo,
        )
    except Exception as exc:
        LOGGER.warning("Falling back to mock clock controller: %s", exc)
        return MockClockController(device_index=device_index, switching_latency_ms=switching_latency_ms)
