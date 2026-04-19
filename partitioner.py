from __future__ import annotations

import argparse
import logging
from typing import Any

from kerneldvfs.models import ClockSetting, ProfileResult, SuperBlock, TraceEvent, read_json, write_json

LOGGER = logging.getLogger(__name__)


def load_profiles(path: str) -> dict[str, ProfileResult]:
    payload = read_json(path)
    profiles_payload = payload["profiles"]
    return {name: ProfileResult.from_dict(item) for name, item in profiles_payload.items()}


def load_trace(path: str) -> list[TraceEvent]:
    payload = read_json(path)
    return [TraceEvent.from_dict(item) for item in payload["trace"]]


def similar_frequencies(left: ClockSetting, right: ClockSetting, core_tol_mhz: int, mem_tol_mhz: int) -> bool:
    return abs(left.core_mhz - right.core_mhz) <= core_tol_mhz and abs(left.mem_mhz - right.mem_mhz) <= mem_tol_mhz


def block_clock(events: list[TraceEvent], profiles: dict[str, ProfileResult]) -> ClockSetting:
    return ClockSetting(
        core_mhz=max(profiles[event.kernel_name].target_clock.core_mhz for event in events),
        mem_mhz=max(profiles[event.kernel_name].target_clock.mem_mhz for event in events),
    )


def block_duration(events: list[TraceEvent]) -> float:
    return sum(event.duration_ms for event in events)


def greedy_group(
    trace: list[TraceEvent],
    profiles: dict[str, ProfileResult],
    core_tol_mhz: int,
    mem_tol_mhz: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: list[TraceEvent] = []

    for event in trace:
        if not current:
            current = [event]
            continue
        current_clock = block_clock(current, profiles)
        next_clock = profiles[event.kernel_name].target_clock
        if similar_frequencies(current_clock, next_clock, core_tol_mhz=core_tol_mhz, mem_tol_mhz=mem_tol_mhz):
            current.append(event)
        else:
            blocks.append({"kernels": current, "merged_for_latency": False})
            current = [event]

    if current:
        blocks.append({"kernels": current, "merged_for_latency": False})
    return blocks


def merge_short_blocks(
    blocks: list[dict[str, Any]],
    profiles: dict[str, ProfileResult],
    min_block_ms: float,
) -> list[dict[str, Any]]:
    if len(blocks) <= 1:
        return blocks

    index = 0
    while index < len(blocks):
        current_duration = block_duration(blocks[index]["kernels"])
        if current_duration >= min_block_ms or len(blocks) == 1:
            index += 1
            continue

        left_index = index - 1 if index > 0 else None
        right_index = index + 1 if index < len(blocks) - 1 else None

        if left_index is None and right_index is None:
            index += 1
            continue

        def requirement_score(block_index: int) -> tuple[int, int, float]:
            setting = block_clock(blocks[block_index]["kernels"], profiles)
            return (setting.core_mhz, setting.mem_mhz, block_duration(blocks[block_index]["kernels"]))

        if left_index is None:
            target_index = right_index
        elif right_index is None:
            target_index = left_index
        else:
            left_score = requirement_score(left_index)
            right_score = requirement_score(right_index)
            target_index = left_index if left_score >= right_score else right_index

        target_index = int(target_index)
        source_events = blocks[index]["kernels"]

        if target_index < index:
            blocks[target_index]["kernels"].extend(source_events)
            blocks[target_index]["merged_for_latency"] = True
            del blocks[index]
            index = max(target_index, 0)
        else:
            blocks[target_index]["kernels"] = source_events + blocks[target_index]["kernels"]
            blocks[target_index]["merged_for_latency"] = True
            del blocks[index]
    return blocks


def materialize_schedule(
    blocks: list[dict[str, Any]],
    profiles: dict[str, ProfileResult],
    switching_latency_ms: float,
) -> list[SuperBlock]:
    schedule: list[SuperBlock] = []
    cursor_ms = 0.0

    for block_index, block in enumerate(blocks):
        kernels = block["kernels"]
        start_time_ms = cursor_ms
        duration_ms = block_duration(kernels)
        end_time_ms = start_time_ms + duration_ms
        schedule.append(
            SuperBlock(
                block_id=f"sb{block_index}",
                kernels=kernels,
                target_clock=block_clock(kernels, profiles),
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                trigger_time_ms=max(0.0, start_time_ms - switching_latency_ms),
                merged_for_latency=bool(block["merged_for_latency"]),
            )
        )
        cursor_ms = end_time_ms
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KernelDVFS super-block partitioner")
    parser.add_argument("--profiles", default="data/profiles.json")
    parser.add_argument("--trace", default="data/execution_trace.json")
    parser.add_argument("--output", default="data/superblock_schedule.json")
    parser.add_argument("--switching-latency-ms", type=float, default=2.5)
    parser.add_argument("--core-tol-mhz", type=int, default=180)
    parser.add_argument("--mem-tol-mhz", type=int, default=400)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def build_output(schedule: list[SuperBlock], args: argparse.Namespace) -> dict[str, Any]:
    iteration_duration_ms = schedule[-1].end_time_ms if schedule else 0.0
    return {
        "metadata": {
            "profiles_path": args.profiles,
            "trace_path": args.trace,
            "switching_latency_ms": args.switching_latency_ms,
            "core_tolerance_mhz": args.core_tol_mhz,
            "mem_tolerance_mhz": args.mem_tol_mhz,
            "block_count": len(schedule),
            "iteration_duration_ms": round(iteration_duration_ms, 6),
        },
        "blocks": [block.to_dict() for block in schedule],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    profiles = load_profiles(args.profiles)
    trace = load_trace(args.trace)
    initial_blocks = greedy_group(trace, profiles, core_tol_mhz=args.core_tol_mhz, mem_tol_mhz=args.mem_tol_mhz)
    merged_blocks = merge_short_blocks(initial_blocks, profiles, min_block_ms=args.switching_latency_ms)
    schedule = materialize_schedule(merged_blocks, profiles, switching_latency_ms=args.switching_latency_ms)
    write_json(args.output, build_output(schedule, args))
    LOGGER.info("Wrote %s super-blocks to %s", len(schedule), args.output)


if __name__ == "__main__":
    main()
