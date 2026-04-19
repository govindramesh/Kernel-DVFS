from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True)
class ClockSetting:
    core_mhz: int
    mem_mhz: int

    def score(self) -> tuple[int, int]:
        return (self.core_mhz, self.mem_mhz)

    def to_dict(self) -> dict[str, int]:
        return {"core_mhz": self.core_mhz, "mem_mhz": self.mem_mhz}


@dataclass
class ProfileResult:
    kernel_name: str
    target_clock: ClockSetting
    baseline_ms: float
    selected_runtime_ms: float
    estimated_energy_mj: float
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_clock"] = self.target_clock.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProfileResult":
        return cls(
            kernel_name=payload["kernel_name"],
            target_clock=ClockSetting(**payload["target_clock"]),
            baseline_ms=float(payload["baseline_ms"]),
            selected_runtime_ms=float(payload["selected_runtime_ms"]),
            estimated_energy_mj=float(payload["estimated_energy_mj"]),
            backend=str(payload["backend"]),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class TraceEvent:
    kernel_name: str
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {"kernel_name": self.kernel_name, "duration_ms": self.duration_ms}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEvent":
        return cls(kernel_name=payload["kernel_name"], duration_ms=float(payload["duration_ms"]))


@dataclass
class SuperBlock:
    block_id: str
    kernels: list[TraceEvent]
    target_clock: ClockSetting
    start_time_ms: float
    end_time_ms: float
    trigger_time_ms: float
    merged_for_latency: bool = False

    @property
    def duration_ms(self) -> float:
        return self.end_time_ms - self.start_time_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "target_clock": self.target_clock.to_dict(),
            "start_time_ms": round(self.start_time_ms, 6),
            "end_time_ms": round(self.end_time_ms, 6),
            "duration_ms": round(self.duration_ms, 6),
            "trigger_time_ms": round(self.trigger_time_ms, 6),
            "merged_for_latency": self.merged_for_latency,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SuperBlock":
        return cls(
            block_id=payload["block_id"],
            kernels=[TraceEvent.from_dict(item) for item in payload["kernels"]],
            target_clock=ClockSetting(**payload["target_clock"]),
            start_time_ms=float(payload["start_time_ms"]),
            end_time_ms=float(payload["end_time_ms"]),
            trigger_time_ms=float(payload["trigger_time_ms"]),
            merged_for_latency=bool(payload.get("merged_for_latency", False)),
        )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
