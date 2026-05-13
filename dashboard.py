from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percent_delta(new_value: float | None, old_value: float | None) -> str:
    if new_value is None or old_value in (None, 0):
        return "n/a"
    return f"{((new_value / old_value) - 1.0) * 100.0:+.2f}%"


def format_number(value: float | None, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def build_kernel_rows(profiles_payload: dict[str, Any]) -> str:
    rows: list[str] = []
    profiles = profiles_payload["profiles"]
    ordered = sorted(
        profiles.values(),
        key=lambda item: (
            item.get("metadata", {}).get("phase", ""),
            item.get("metadata", {}).get("family", ""),
            item["kernel_name"],
        ),
    )
    for item in ordered:
        metadata = item.get("metadata", {})
        rows.append(
            """
            <tr>
              <td>{kernel}</td>
              <td>{phase}</td>
              <td>{family}</td>
              <td>{auto_runtime}</td>
              <td>{ideal_runtime}</td>
              <td>{runtime_delta}</td>
              <td>{auto_energy}</td>
              <td>{ideal_energy}</td>
              <td>{energy_delta}</td>
              <td>{clock}</td>
            </tr>
            """.format(
                kernel=item["kernel_name"],
                phase=metadata.get("phase", "n/a"),
                family=metadata.get("family", "n/a"),
                auto_runtime=format_number(item.get("baseline_ms"), digits=4, suffix=" ms"),
                ideal_runtime=format_number(item.get("selected_runtime_ms"), digits=4, suffix=" ms"),
                runtime_delta=percent_delta(item.get("selected_runtime_ms"), item.get("baseline_ms")),
                auto_energy=format_number(item.get("baseline_energy_mj"), digits=3, suffix=" mJ"),
                ideal_energy=format_number(item.get("estimated_energy_mj"), digits=3, suffix=" mJ"),
                energy_delta=percent_delta(item.get("estimated_energy_mj"), item.get("baseline_energy_mj")),
                clock=f"{item['target_clock']['core_mhz']} / {item['target_clock']['mem_mhz']} MHz",
            )
        )
    return "\n".join(rows)


def build_runtime_summary(runtime_payload: dict[str, Any]) -> str:
    auto = runtime_payload.get("auto")
    profiled = runtime_payload.get("profiled")
    comparison = runtime_payload.get("comparison", {})

    def block(title: str, payload: dict[str, Any] | None) -> str:
        if payload is None:
            return f"""
            <section class="summary-card">
              <h3>{title}</h3>
              <p>No run data.</p>
            </section>
            """
        transition = payload.get("transition_latency", {})
        return f"""
        <section class="summary-card">
          <h3>{title}</h3>
          <dl>
            <div><dt>Time to Complete</dt><dd>{format_number(payload.get('time_to_complete_ms'), suffix=' ms')}</dd></div>
            <div><dt>Kernel Runtime</dt><dd>{format_number(payload.get('kernel_runtime_ms'), suffix=' ms')}</dd></div>
            <div><dt>Transition Command Time</dt><dd>{format_number(payload.get('clock_transition_command_ms'), suffix=' ms')}</dd></div>
            <div><dt>Total Energy</dt><dd>{format_number(payload.get('total_energy_mj'), suffix=' mJ')}</dd></div>
            <div><dt>Transition Avg</dt><dd>{format_number(transition.get('avg_ms'), suffix=' ms')}</dd></div>
            <div><dt>Transition Missed</dt><dd>{transition.get('missed', 0)}</dd></div>
          </dl>
        </section>
        """

    comparison_block = f"""
    <section class="summary-card accent">
      <h3>Comparison</h3>
      <dl>
        <div><dt>Runtime Delta</dt><dd>{format_number(comparison.get('wall_runtime_delta_ms'), suffix=' ms')}</dd></div>
        <div><dt>Runtime Delta %</dt><dd>{comparison.get('wall_runtime_delta_pct', 'n/a')}</dd></div>
        <div><dt>Energy Delta</dt><dd>{format_number(comparison.get('energy_delta_mj'), suffix=' mJ')}</dd></div>
        <div><dt>Energy Delta %</dt><dd>{comparison.get('energy_delta_pct', 'n/a')}</dd></div>
      </dl>
    </section>
    """
    return block("Auto Run", auto) + block("Profiled Run", profiled) + comparison_block


def build_runtime_events_table(runtime_payload: dict[str, Any]) -> str:
    profiled = runtime_payload.get("profiled")
    if not profiled:
        return "<p>No profiled runtime events available.</p>"
    rows: list[str] = []
    for event in profiled.get("events", [])[:200]:
        rows.append(
            """
            <tr>
              <td>{iteration}</td>
              <td>{event_index}</td>
              <td>{kernel}</td>
              <td>{clock}</td>
              <td>{transition_cmd}</td>
              <td>{transition_seen}</td>
              <td>{runtime}</td>
              <td>{energy}</td>
            </tr>
            """.format(
                iteration=event["iteration"],
                event_index=event["event_index"],
                kernel=event["kernel_name"],
                clock=(
                    "auto"
                    if event.get("requested_clock") is None
                    else f"{event['requested_clock']['core_mhz']} / {event['requested_clock']['mem_mhz']} MHz"
                ),
                transition_cmd=format_number(event.get("transition_command_ms"), suffix=" ms"),
                transition_seen=format_number(event.get("transition_observed_ms"), suffix=" ms"),
                runtime=format_number(event.get("kernel_runtime_ms"), suffix=" ms"),
                energy=format_number(event.get("kernel_energy_mj"), suffix=" mJ"),
            )
        )
    return "\n".join(rows)


def build_dashboard_html(profiles_payload: dict[str, Any], runtime_payload: dict[str, Any], source_paths: dict[str, str]) -> str:
    kernel_rows = build_kernel_rows(profiles_payload)
    runtime_summary = build_runtime_summary(runtime_payload)
    runtime_event_rows = build_runtime_events_table(runtime_payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KernelDVFS Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255,255,255,0.78);
      --ink: #1d2a2f;
      --muted: #5b6a70;
      --line: rgba(29,42,47,0.14);
      --accent: #b85042;
      --accent-soft: rgba(184,80,66,0.12);
      --good: #256f3a;
      --bad: #a23535;
      --shadow: 0 24px 60px rgba(52, 44, 30, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184,80,66,0.15), transparent 28%),
        radial-gradient(circle at top right, rgba(36,111,58,0.14), transparent 24%),
        linear-gradient(180deg, #f7f2ea, var(--bg));
    }}
    .shell {{
      width: min(1480px, calc(100vw - 32px));
      margin: 20px auto 40px;
    }}
    .hero {{
      padding: 28px 30px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,248,240,0.72));
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.7rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 940px;
      font-size: 1rem;
    }}
    .source-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .source-pill {{
      padding: 12px 14px;
      border-radius: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      font-size: 0.92rem;
    }}
    .section {{
      margin-top: 20px;
      padding: 22px;
      border-radius: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 1.45rem;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .summary-card {{
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}
    .summary-card.accent {{
      background: linear-gradient(135deg, rgba(184,80,66,0.12), rgba(255,255,255,0.78));
    }}
    .summary-card h3 {{
      margin: 0 0 10px;
      font-size: 1.05rem;
    }}
    dl {{
      margin: 0;
      display: grid;
      gap: 8px;
    }}
    dl div {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border-top: 1px dashed var(--line);
      padding-top: 8px;
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; font-weight: 700; }}
    .table-wrap {{
      overflow: auto;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.78);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1120px;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f6eee2;
      z-index: 1;
    }}
    tbody tr:nth-child(odd) {{
      background: rgba(248,242,234,0.55);
    }}
    .footnote {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>KernelDVFS Dashboard</h1>
      <p class="subtitle">Per-kernel auto-versus-ideal profiling results on top, whole-workload runtime comparison on the bottom. This page is generated from the current profiling and runtime comparison JSON outputs.</p>
      <div class="source-strip">
        <div class="source-pill"><strong>Profiles</strong><br>{source_paths['profiles']}</div>
        <div class="source-pill"><strong>Runtime Compare</strong><br>{source_paths['runtime_compare']}</div>
        <div class="source-pill"><strong>Profile Style</strong><br>{profiles_payload.get('metadata', {}).get('profile_style', 'n/a')}</div>
      </div>
    </section>

    <section class="section">
      <h2>Kernel Table</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Kernel</th>
              <th>Phase</th>
              <th>Family</th>
              <th>Auto Runtime</th>
              <th>Ideal Runtime</th>
              <th>Runtime Delta</th>
              <th>Auto Energy</th>
              <th>Ideal Energy</th>
              <th>Energy Delta</th>
              <th>Ideal Clock</th>
            </tr>
          </thead>
          <tbody>
            {kernel_rows}
          </tbody>
        </table>
      </div>
      <p class="footnote">Runtime delta compares profiled-clock runtime against auto-baseline runtime. Energy delta compares profiled-clock energy against auto-baseline energy.</p>
    </section>

    <section class="section">
      <h2>Runtime Compare</h2>
      <div class="summary-grid">
        {runtime_summary}
      </div>
    </section>

    <section class="section">
      <h2>Profiled Run Events</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Iter</th>
              <th>Event</th>
              <th>Kernel</th>
              <th>Requested Clock</th>
              <th>Command Time</th>
              <th>Observed Transition</th>
              <th>Kernel Runtime</th>
              <th>Kernel Energy</th>
            </tr>
          </thead>
          <tbody>
            {runtime_event_rows}
          </tbody>
        </table>
      </div>
      <p class="footnote">The event table is capped at the first 200 profiled events to keep the page usable.</p>
    </section>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a standalone HTML dashboard for KernelDVFS results")
    parser.add_argument("--profiles", default="data/profiles.json")
    parser.add_argument("--runtime-compare", default="data/runtime_comparison.json")
    parser.add_argument("--output", default="data/dashboard.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles_payload = load_json(args.profiles)
    runtime_payload = load_json(args.runtime_compare)
    html = build_dashboard_html(
        profiles_payload=profiles_payload,
        runtime_payload=runtime_payload,
        source_paths={"profiles": args.profiles, "runtime_compare": args.runtime_compare},
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
