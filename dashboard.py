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
        key=lambda item: item["kernel_name"],
    )
    for item in ordered:
        rows.append(
            f"""
            <tr>
              <td>{item["kernel_name"]}</td>
              <td>{format_number(item.get("baseline_ms"), digits=4, suffix=" ms")}</td>
              <td>{format_number(item.get("selected_runtime_ms"), digits=4, suffix=" ms")}</td>
              <td>{percent_delta(item.get("selected_runtime_ms"), item.get("baseline_ms"))}</td>
              <td>{format_number(item.get("baseline_energy_mj"), digits=3, suffix=" mJ")}</td>
              <td>{format_number(item.get("estimated_energy_mj"), digits=3, suffix=" mJ")}</td>
              <td>{percent_delta(item.get("estimated_energy_mj"), item.get("baseline_energy_mj"))}</td>
              <td>{item["target_clock"]["core_mhz"]} / {item["target_clock"]["mem_mhz"]} MHz</td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_workload_summary(runtime_payload: dict[str, Any]) -> str:
    auto = runtime_payload.get("auto", {})
    profiled = runtime_payload.get("profiled", {})
    comparison = runtime_payload.get("comparison", {})
    return f"""
    <section class="summary-card">
      <h3>Auto Workload</h3>
      <dl>
        <div><dt>Total Time</dt><dd>{format_number(auto.get("time_to_complete_ms"), suffix=" ms")}</dd></div>
        <div><dt>Total Energy</dt><dd>{format_number(auto.get("total_energy_mj"), suffix=" mJ")}</dd></div>
      </dl>
    </section>
    <section class="summary-card">
      <h3>Profiled Workload</h3>
      <dl>
        <div><dt>Total Time</dt><dd>{format_number(profiled.get("time_to_complete_ms"), suffix=" ms")}</dd></div>
        <div><dt>Total Energy</dt><dd>{format_number(profiled.get("total_energy_mj"), suffix=" mJ")}</dd></div>
      </dl>
    </section>
    <section class="summary-card accent">
      <h3>Delta</h3>
      <dl>
        <div><dt>Runtime Delta</dt><dd>{format_number(comparison.get("wall_runtime_delta_ms"), suffix=" ms")}</dd></div>
        <div><dt>Runtime Delta %</dt><dd>{comparison.get("wall_runtime_delta_pct", "n/a")}</dd></div>
        <div><dt>Energy Delta</dt><dd>{format_number(comparison.get("energy_delta_mj"), suffix=" mJ")}</dd></div>
        <div><dt>Energy Delta %</dt><dd>{comparison.get("energy_delta_pct", "n/a")}</dd></div>
      </dl>
    </section>
    """


def build_layer_rows(runtime_payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for layer in runtime_payload.get("layers", []):
        rows.append(
            f"""
            <tr>
              <td>Layer {layer["layer_index"] + 1}</td>
              <td>{len(layer.get("events", []))}</td>
              <td>{format_number(layer.get("auto_time_ms"), suffix=" ms")}</td>
              <td>{format_number(layer.get("profiled_time_ms"), suffix=" ms")}</td>
              <td>{percent_delta(layer.get("profiled_time_ms"), layer.get("auto_time_ms"))}</td>
              <td>{format_number(layer.get("auto_energy_mj"), suffix=" mJ")}</td>
              <td>{format_number(layer.get("profiled_energy_mj"), suffix=" mJ")}</td>
              <td>{percent_delta(layer.get("profiled_energy_mj"), layer.get("auto_energy_mj"))}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_event_rows(runtime_payload: dict[str, Any]) -> str:
    rows: list[str] = []
    for event in runtime_payload.get("events", [])[:240]:
        layer_label = "n/a" if event.get("layer_index") is None else str(event["layer_index"] + 1)
        rows.append(
            f"""
            <tr>
              <td>{event["event_index"]}</td>
              <td>{event["kernel_name"]}</td>
              <td>{layer_label}</td>
              <td>{format_number(event.get("auto_time_ms"), suffix=" ms")}</td>
              <td>{format_number(event.get("profiled_time_ms"), suffix=" ms")}</td>
              <td>{format_number(event.get("auto_energy_mj"), suffix=" mJ")}</td>
              <td>{format_number(event.get("profiled_energy_mj"), suffix=" mJ")}</td>
              <td>{event["target_clock"]["core_mhz"]} / {event["target_clock"]["mem_mhz"]} MHz</td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_execution_graph(runtime_payload: dict[str, Any]) -> str:
    graph = runtime_payload.get("execution_graph", {})
    prefix = graph.get("prefix", [])
    repeated = graph.get("layer_kernel_order", [])
    suffix = graph.get("suffix", [])
    num_layers = int(graph.get("num_layers", 0))

    prefix_html = "".join(f'<span class="graph-node prelude">{name}</span>' for name in prefix)
    repeated_html = "".join(f'<span class="graph-node repeated">{name}</span>' for name in repeated)
    suffix_html = "".join(f'<span class="graph-node output">{name}</span>' for name in suffix)

    return f"""
    <div class="graph-grid">
      <div class="graph-lane">
        <h3>Prelude</h3>
        <div class="graph-strip">{prefix_html}</div>
      </div>
      <div class="graph-lane">
        <h3>Transformer Layer x {num_layers}</h3>
        <div class="graph-strip">{repeated_html}</div>
      </div>
      <div class="graph-lane">
        <h3>Output</h3>
        <div class="graph-strip">{suffix_html}</div>
      </div>
    </div>
    """


def build_dashboard_html(profiles_payload: dict[str, Any], runtime_payload: dict[str, Any], source_paths: dict[str, str]) -> str:
    kernel_rows = build_kernel_rows(profiles_payload)
    workload_summary = build_workload_summary(runtime_payload)
    layer_rows = build_layer_rows(runtime_payload)
    event_rows = build_event_rows(runtime_payload)
    execution_graph = build_execution_graph(runtime_payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KernelDVFS Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255,255,255,0.82);
      --ink: #1d2a2f;
      --muted: #5b6a70;
      --line: rgba(29,42,47,0.14);
      --accent: #b85042;
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
    .shell {{ width: min(1500px, calc(100vw - 32px)); margin: 20px auto 40px; }}
    .hero, .section {{
      border: 1px solid var(--line);
      border-radius: 26px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px 30px; }}
    .section {{ margin-top: 20px; padding: 22px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.7rem); line-height: 0.96; letter-spacing: -0.04em; }}
    .subtitle, .footnote {{ color: var(--muted); }}
    .source-strip, .summary-grid {{ display: grid; gap: 12px; }}
    .source-strip {{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); margin-top: 20px; }}
    .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .source-pill, .summary-card {{
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}
    .summary-card.accent {{ background: linear-gradient(135deg, rgba(184,80,66,0.12), rgba(255,255,255,0.78)); }}
    h2 {{ margin: 0 0 12px; font-size: 1.45rem; }}
    h3 {{ margin: 0 0 10px; font-size: 1.02rem; }}
    dl {{ margin: 0; display: grid; gap: 8px; }}
    dl div {{ display: flex; justify-content: space-between; gap: 12px; border-top: 1px dashed var(--line); padding-top: 8px; }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; font-weight: 700; }}
    .table-wrap {{ overflow: auto; border-radius: 18px; border: 1px solid var(--line); background: rgba(255,255,255,0.78); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1080px; font-size: 0.92rem; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    thead th {{ position: sticky; top: 0; background: #f6eee2; z-index: 1; }}
    tbody tr:nth-child(odd) {{ background: rgba(248,242,234,0.55); }}
    .graph-grid {{ display: grid; gap: 16px; }}
    .graph-lane {{ padding: 16px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.68); }}
    .graph-strip {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
    .graph-node {{
      display: inline-flex; align-items: center; padding: 10px 12px; border-radius: 999px;
      border: 1px solid var(--line); background: #fff8f0; font-size: 0.88rem; white-space: nowrap;
    }}
    .graph-node.repeated {{ background: #f1f7f1; }}
    .graph-node.output {{ background: #fdf1ef; }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>KernelDVFS Dashboard</h1>
      <p class="subtitle">This dashboard follows the paper-style comparison path: isolated kernel measurements are aggregated over the execution trace instead of replaying live clock changes per kernel.</p>
      <div class="source-strip">
        <div class="source-pill"><strong>Profiles</strong><br>{source_paths['profiles']}</div>
        <div class="source-pill"><strong>Runtime Compare</strong><br>{source_paths['runtime_compare']}</div>
        <div class="source-pill"><strong>Comparison Style</strong><br>{runtime_payload.get('metadata', {}).get('comparison_style', 'n/a')}</div>
      </div>
    </section>

    <section class="section">
      <h2>Kernel Table</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Kernel</th>
              <th>Auto Runtime</th>
              <th>Ideal Runtime</th>
              <th>Runtime Delta</th>
              <th>Auto Energy</th>
              <th>Ideal Energy</th>
              <th>Energy Delta</th>
              <th>Ideal Clock</th>
            </tr>
          </thead>
          <tbody>{kernel_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>Execution Graph</h2>
      {execution_graph}
      <p class="footnote">The middle lane is repeated once per transformer layer in the aggregated forward workload.</p>
    </section>

    <section class="section">
      <h2>Whole Workload</h2>
      <div class="summary-grid">{workload_summary}</div>
    </section>

    <section class="section">
      <h2>Per-Layer Totals</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Layer</th>
              <th>Events</th>
              <th>Auto Time</th>
              <th>Profiled Time</th>
              <th>Time Delta</th>
              <th>Auto Energy</th>
              <th>Profiled Energy</th>
              <th>Energy Delta</th>
            </tr>
          </thead>
          <tbody>{layer_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <h2>Execution Events</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Event</th>
              <th>Kernel</th>
              <th>Layer</th>
              <th>Auto Time</th>
              <th>Profiled Time</th>
              <th>Auto Energy</th>
              <th>Profiled Energy</th>
              <th>Ideal Clock</th>
            </tr>
          </thead>
          <tbody>{event_rows}</tbody>
        </table>
      </div>
      <p class="footnote">The event table is capped at the first 240 events to keep the page responsive.</p>
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
