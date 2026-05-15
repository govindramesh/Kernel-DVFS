from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def ask_for_path(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or (default or "")


def run_step(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def build_pipeline_commands(
    *,
    kernel_defs: str,
    workflow: str,
    profiles_output: str,
    runtime_output: str,
    dashboard_output: str,
    num_layers: int | None,
    device_index: int,
    nvidia_smi_sudo: bool,
    tolerated_slowdown_pct: float,
) -> dict[str, list[str]]:
    profiler_cmd = [
        sys.executable,
        "profiler.py",
        "--backend",
        "real",
        "--measurement-mode",
        "real",
        "--kernel-defs",
        kernel_defs,
        "--tolerated-slowdown-pct",
        str(tolerated_slowdown_pct),
        "--output",
        profiles_output,
        "--device-index",
        str(device_index),
    ]
    runtime_cmd = [
        sys.executable,
        "runtime_compare.py",
        "--profiles",
        profiles_output,
        "--kernel-defs",
        kernel_defs,
        "--workflow",
        workflow,
        "--output",
        runtime_output,
    ]
    dashboard_cmd = [
        sys.executable,
        "dashboard.py",
        "--profiles",
        profiles_output,
        "--runtime-compare",
        runtime_output,
        "--output",
        dashboard_output,
    ]

    if num_layers is not None:
        runtime_cmd.extend(["--num-layers", str(num_layers)])
    if nvidia_smi_sudo:
        profiler_cmd.append("--nvidia-smi-sudo")
    return {"profiler": profiler_cmd, "runtime": runtime_cmd, "dashboard": dashboard_cmd}


def run_pipeline(**kwargs: Any) -> dict[str, str]:
    commands = build_pipeline_commands(**kwargs)
    print("Profiling kernels...")
    run_step(commands["profiler"])
    print("Aggregating workflow...")
    run_step(commands["runtime"])
    print("Building dashboard...")
    run_step(commands["dashboard"])
    return {
        "profiles_output": kwargs["profiles_output"],
        "runtime_output": kwargs["runtime_output"],
        "dashboard_output": kwargs["dashboard_output"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the custom KernelDVFS pipeline from kernel and workflow JSON files")
    parser.add_argument("--kernel-defs", default=None)
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--profiles-output", default="data/demo_profiles.json")
    parser.add_argument("--runtime-output", default="data/demo_runtime.json")
    parser.add_argument("--dashboard-output", default="data/demo_dashboard.html")
    parser.add_argument("--num-layers", type=int, default=None, help="Override workflow num_layers if desired")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--nvidia-smi-sudo", action="store_true", default=True)
    parser.add_argument("--no-nvidia-smi-sudo", dest="nvidia_smi_sudo", action="store_false")
    parser.add_argument("--tolerated-slowdown-pct", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_kernel_defs = str(ROOT / "data" / "demo_kernels.json")
    default_workflow = str(ROOT / "data" / "demo_workflow.json")

    kernel_defs = args.kernel_defs or ask_for_path("Kernel definitions file", default_kernel_defs)
    workflow = args.workflow or ask_for_path("Workflow file", default_workflow)

    outputs = run_pipeline(
        kernel_defs=kernel_defs,
        workflow=workflow,
        profiles_output=args.profiles_output,
        runtime_output=args.runtime_output,
        dashboard_output=args.dashboard_output,
        num_layers=args.num_layers,
        device_index=args.device_index,
        nvidia_smi_sudo=args.nvidia_smi_sudo,
        tolerated_slowdown_pct=args.tolerated_slowdown_pct,
    )
    print(f"Dashboard written to {Path(outputs['dashboard_output']).resolve()}")


if __name__ == "__main__":
    main()
