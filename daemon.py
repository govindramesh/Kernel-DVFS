from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any

from kerneldvfs.models import ClockSetting, SuperBlock, read_json
from kerneldvfs.nvml_controller import BaseClockController, create_clock_controller

LOGGER = logging.getLogger(__name__)


def load_schedule(path: str) -> tuple[dict[str, Any], list[SuperBlock]]:
    payload = read_json(path)
    return payload["metadata"], [SuperBlock.from_dict(item) for item in payload["blocks"]]


def sleep_until_ns(target_ns: int) -> None:
    while True:
        now_ns = time.perf_counter_ns()
        remaining_ns = target_ns - now_ns
        if remaining_ns <= 0:
            return
        if remaining_ns > 1_000_000:
            time.sleep((remaining_ns - 250_000) / 1_000_000_000)
            continue
        if remaining_ns > 100_000:
            time.sleep(0)
            continue


class PowerActuationThread(threading.Thread):
    def __init__(
        self,
        controller: BaseClockController,
        schedule: list[SuperBlock],
        start_ns: int,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="power-daemon", daemon=True)
        self.controller = controller
        self.schedule = schedule
        self.start_ns = start_ns
        self.stop_event = stop_event
        self.last_setting: ClockSetting | None = None
        self.errors: list[BaseException] = []

    def run(self) -> None:
        try:
            for block in self.schedule:
                if self.stop_event.is_set():
                    return
                if block.trigger_time_ms <= 0.0:
                    trigger_ns = time.perf_counter_ns()
                else:
                    trigger_ns = self.start_ns + int(block.trigger_time_ms * 1_000_000)
                    sleep_until_ns(trigger_ns)
                if self.stop_event.is_set():
                    return
                if self.last_setting == block.target_clock:
                    LOGGER.info("Skipping redundant clock actuation for %s", block.block_id)
                    continue
                actuation_start_ns = time.perf_counter_ns()
                self.controller.set_locked_clocks(block.target_clock)
                actuation_end_ns = time.perf_counter_ns()
                LOGGER.info(
                    "Actuated %s at %+0.3fms from trigger, duration=%.3fms target=(core=%s mem=%s)",
                    block.block_id,
                    (actuation_start_ns - trigger_ns) / 1_000_000.0,
                    (actuation_end_ns - actuation_start_ns) / 1_000_000.0,
                    block.target_clock.core_mhz,
                    block.target_clock.mem_mhz,
                )
                self.last_setting = block.target_clock
        except BaseException as exc:
            self.errors.append(exc)
            self.stop_event.set()


def replay_main_thread(schedule: list[SuperBlock], start_ns: int, stop_event: threading.Event) -> None:
    for block in schedule:
        if stop_event.is_set():
            return
        block_start_ns = start_ns + int(block.start_time_ms * 1_000_000)
        block_end_ns = start_ns + int(block.end_time_ms * 1_000_000)
        sleep_until_ns(block_start_ns)
        LOGGER.info(
            "Main thread entering %s at t=%.3fms with %d kernels",
            block.block_id,
            (time.perf_counter_ns() - start_ns) / 1_000_000.0,
            len(block.kernels),
        )
        for kernel in block.kernels:
            if stop_event.is_set():
                return
            LOGGER.info("  kernel=%s duration=%.3fms", kernel.kernel_name, kernel.duration_ms)
            time.sleep(kernel.duration_ms / 1000.0)
        sleep_until_ns(block_end_ns)
        LOGGER.info("Main thread completed %s at t=%.3fms", block.block_id, (time.perf_counter_ns() - start_ns) / 1_000_000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KernelDVFS runtime actuation simulator")
    parser.add_argument("--schedule", default="data/superblock_schedule.json")
    parser.add_argument("--backend", choices=["auto", "mock", "real"], default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--nvidia-smi-path", default=None)
    parser.add_argument("--nvidia-smi-sudo", action="store_true")
    parser.add_argument("--switching-latency-ms", type=float, default=2.5)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
    )
    metadata, schedule = load_schedule(args.schedule)
    if not schedule:
        raise SystemExit("schedule is empty")

    switching_latency_ms = float(metadata.get("switching_latency_ms", args.switching_latency_ms))
    controller = create_clock_controller(
        backend=args.backend,
        device_index=args.device_index,
        switching_latency_ms=switching_latency_ms,
        nvidia_smi_path=args.nvidia_smi_path,
        nvidia_smi_sudo=args.nvidia_smi_sudo,
    )
    stop_event = threading.Event()
    start_ns = time.perf_counter_ns() + max(5_000_000, int((switching_latency_ms + 0.5) * 1_000_000))
    daemon_thread = PowerActuationThread(controller=controller, schedule=schedule, start_ns=start_ns, stop_event=stop_event)

    try:
        LOGGER.info(
            "Starting runtime replay with %d super-blocks, backend=%s, switching_latency=%.3fms",
            len(schedule),
            controller.mode,
            switching_latency_ms,
        )
        daemon_thread.start()
        replay_main_thread(schedule=schedule, start_ns=start_ns, stop_event=stop_event)
        daemon_thread.join(timeout=5.0)
        if daemon_thread.errors:
            raise daemon_thread.errors[0]
    finally:
        stop_event.set()
        try:
            controller.reset_locked_clocks()
        except Exception:
            LOGGER.exception("Failed to reset locked clocks during teardown")
        controller.close()
        LOGGER.info("Runtime teardown complete")


if __name__ == "__main__":
    main()
