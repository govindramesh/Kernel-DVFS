from __future__ import annotations

import argparse
import cgi
import html
import json
import shutil
import subprocess
import traceback
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from demo_pipeline import ROOT, build_pipeline_commands


RUNS_DIR = ROOT / "data" / "web_runs"
DEFAULT_KERNELS = ROOT / "data" / "demo_kernels.json"
DEFAULT_WORKFLOW = ROOT / "data" / "demo_workflow.json"


def page_template(body: str) -> bytes:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KernelDVFS Workbench</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: rgba(255,255,255,0.86);
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
    .shell {{ width: min(1180px, calc(100vw - 32px)); margin: 20px auto 40px; }}
    .hero, .panel {{
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 28px 30px; }}
    .panel {{ margin-top: 20px; padding: 22px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 0.96; letter-spacing: -0.04em; }}
    h2 {{ margin: 0 0 14px; font-size: 1.35rem; }}
    p, li, label {{ color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; }}
    .stack {{ display: grid; gap: 14px; }}
    .field {{ display: grid; gap: 8px; }}
    textarea {{
      min-height: 260px;
      width: 100%;
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.8);
      font: 0.92rem/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--ink);
      resize: vertical;
    }}
    input[type="file"], select, input[type="number"] {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.8);
      color: var(--ink);
    }}
    .controls {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .submit {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 14px 18px;
      background: linear-gradient(135deg, #b85042, #d87046);
      color: white;
      font: inherit;
      cursor: pointer;
      width: fit-content;
    }}
    .hint {{ font-size: 0.92rem; }}
    .result-card {{
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }}
    .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 6px;
    }}
    .metric-value {{
      display: block;
      font-size: 1.35rem;
      font-weight: 700;
      color: var(--ink);
    }}
    a {{ color: var(--accent); }}
    pre {{
      padding: 14px 16px;
      border-radius: 16px;
      background: #fff8f0;
      border: 1px solid var(--line);
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--ink);
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    {body}
  </main>
</body>
</html>
""".encode("utf-8")


def index_body() -> str:
    kernels_text = DEFAULT_KERNELS.read_text(encoding="utf-8")
    workflow_text = DEFAULT_WORKFLOW.read_text(encoding="utf-8")
    return f"""
    <section class="hero">
      <h1>KernelDVFS Workbench</h1>
      <p>Upload or edit kernel definitions and workflow JSON, then run profiling, aggregation, and dashboard generation from one page.</p>
    </section>
    <section class="panel">
      <h2>Run Pipeline</h2>
      <form method="post" action="/run" enctype="multipart/form-data" class="stack">
        <div class="controls">
          <div class="field">
            <label for="backend">Backend</label>
            <select id="backend" name="backend">
              <option value="mock" selected>mock</option>
              <option value="real">real</option>
              <option value="auto">auto</option>
            </select>
          </div>
          <div class="field">
            <label for="measurement_mode">Measurement Mode</label>
            <select id="measurement_mode" name="measurement_mode">
              <option value="mock" selected>mock</option>
              <option value="real">real</option>
              <option value="auto">auto</option>
            </select>
          </div>
          <div class="field">
            <label for="num_layers">Optional Layer Override</label>
            <input id="num_layers" name="num_layers" type="number" min="1" step="1" placeholder="Leave blank to use workflow file">
          </div>
          <div class="field">
            <label for="tolerated_slowdown_pct">Tolerated Slowdown %</label>
            <input id="tolerated_slowdown_pct" name="tolerated_slowdown_pct" type="number" min="0" step="0.01" value="0.0">
          </div>
        </div>

        <div class="grid">
          <div class="stack">
            <div class="field">
              <label for="kernel_defs_text">Kernel Definitions JSON</label>
              <textarea id="kernel_defs_text" name="kernel_defs_text">{html.escape(kernels_text)}</textarea>
            </div>
            <div class="field">
              <label for="kernel_defs_file">Or upload kernel definitions file</label>
              <input id="kernel_defs_file" name="kernel_defs_file" type="file" accept=".json,application/json">
            </div>
          </div>
          <div class="stack">
            <div class="field">
              <label for="workflow_text">Workflow JSON</label>
              <textarea id="workflow_text" name="workflow_text">{html.escape(workflow_text)}</textarea>
            </div>
            <div class="field">
              <label for="workflow_file">Or upload workflow file</label>
              <input id="workflow_file" name="workflow_file" type="file" accept=".json,application/json">
            </div>
          </div>
        </div>

        <p class="hint">If you upload files, they override the text areas. Outputs are stored under <code>data/web_runs/...</code>.</p>
        <button class="submit" type="submit">Run</button>
      </form>
    </section>
    """


def result_body(
    run_id: str,
    dashboard_href: str,
    profiles_href: str,
    runtime_href: str,
    metrics: dict[str, str],
    logs: str,
) -> str:
    return f"""
    <section class="hero">
      <h1>Run Complete</h1>
      <p>The profiling, aggregation, and dashboard generation steps finished for run <code>{html.escape(run_id)}</code>.</p>
    </section>
    <section class="panel">
      <h2>Results</h2>
      <div class="metrics">
        <div class="metric"><span class="metric-label">Profiled Kernels</span><span class="metric-value">{metrics["profiled_kernels"]}</span></div>
        <div class="metric"><span class="metric-label">Workflow Events</span><span class="metric-value">{metrics["workflow_events"]}</span></div>
        <div class="metric"><span class="metric-label">Auto Time</span><span class="metric-value">{metrics["auto_time"]}</span></div>
        <div class="metric"><span class="metric-label">Profiled Time</span><span class="metric-value">{metrics["profiled_time"]}</span></div>
        <div class="metric"><span class="metric-label">Auto Energy</span><span class="metric-value">{metrics["auto_energy"]}</span></div>
        <div class="metric"><span class="metric-label">Profiled Energy</span><span class="metric-value">{metrics["profiled_energy"]}</span></div>
      </div>
    </section>
    <section class="panel">
      <h2>Artifacts</h2>
      <div class="stack">
        <div class="result-card">
          <strong>Dashboard</strong><br>
          <a href="{dashboard_href}" target="_blank" rel="noopener">Open dashboard</a>
        </div>
        <div class="result-card">
          <strong>Profiles JSON</strong><br>
          <a href="{profiles_href}" target="_blank" rel="noopener">Open profiles</a>
        </div>
        <div class="result-card">
          <strong>Runtime Compare JSON</strong><br>
          <a href="{runtime_href}" target="_blank" rel="noopener">Open aggregated runtime results</a>
        </div>
        <div class="result-card">
          <a href="/">Start another run</a>
        </div>
      </div>
    </section>
    <section class="panel">
      <h2>Logs</h2>
      <pre>{html.escape(logs)}</pre>
    </section>
    """


def error_body(message: str, details: str) -> str:
    return f"""
    <section class="hero">
      <h1>Pipeline Failed</h1>
      <p>{html.escape(message)}</p>
    </section>
    <section class="panel">
      <h2>Details</h2>
      <pre>{html.escape(details)}</pre>
      <p><a href="/">Back to workbench</a></p>
    </section>
    """


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "KernelDVFSDemo/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(page_template(index_body()))
            return
        if parsed.path.startswith("/artifacts/"):
            relative = parsed.path.removeprefix("/artifacts/")
            target = RUNS_DIR / relative
            if target.is_file() and target.resolve().is_relative_to(RUNS_DIR.resolve()):
                self._send_file(target)
                return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                },
            )
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=False)

            kernel_defs_path = run_dir / "kernels.json"
            workflow_path = run_dir / "workflow.json"
            self._write_uploaded_or_text(form, "kernel_defs_file", "kernel_defs_text", kernel_defs_path)
            self._write_uploaded_or_text(form, "workflow_file", "workflow_text", workflow_path)

            profiles_output = str(run_dir / "profiles.json")
            runtime_output = str(run_dir / "runtime.json")
            dashboard_output = str(run_dir / "dashboard.html")

            backend = self._field_value(form, "backend", "mock")
            measurement_mode = self._field_value(form, "measurement_mode", "mock")
            num_layers_raw = self._field_value(form, "num_layers", "").strip()
            num_layers = int(num_layers_raw) if num_layers_raw else None
            tolerated_slowdown_pct = float(self._field_value(form, "tolerated_slowdown_pct", "0.0"))

            commands = build_pipeline_commands(
                kernel_defs=str(kernel_defs_path),
                workflow=str(workflow_path),
                backend=backend,
                measurement_mode=measurement_mode,
                profiles_output=profiles_output,
                runtime_output=runtime_output,
                dashboard_output=dashboard_output,
                num_layers=num_layers,
                device_index=0,
                nvidia_smi_sudo=True,
                tolerated_slowdown_pct=tolerated_slowdown_pct,
            )
            logs = self._run_commands_with_logs(commands)

            profiles_payload = json.loads(Path(profiles_output).read_text(encoding="utf-8"))
            runtime_payload = json.loads(Path(runtime_output).read_text(encoding="utf-8"))
            metrics = {
                "profiled_kernels": str(len(profiles_payload.get("profiles", {}))),
                "workflow_events": str(len(runtime_payload.get("events", []))),
                "auto_time": self._format_metric(runtime_payload.get("auto", {}).get("time_to_complete_ms"), " ms"),
                "profiled_time": self._format_metric(runtime_payload.get("profiled", {}).get("time_to_complete_ms"), " ms"),
                "auto_energy": self._format_metric(runtime_payload.get("auto", {}).get("total_energy_mj"), " mJ"),
                "profiled_energy": self._format_metric(runtime_payload.get("profiled", {}).get("total_energy_mj"), " mJ"),
            }

            dashboard_href = f"/artifacts/{run_id}/dashboard.html"
            profiles_href = f"/artifacts/{run_id}/profiles.json"
            runtime_href = f"/artifacts/{run_id}/runtime.json"
            self._send_html(
                page_template(result_body(run_id, dashboard_href, profiles_href, runtime_href, metrics, logs))
            )
        except Exception as exc:
            details = "".join(traceback.format_exception(exc))
            self._send_html(page_template(error_body(str(exc), details)), status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _write_uploaded_or_text(self, form: cgi.FieldStorage, file_field: str, text_field: str, output_path: Path) -> None:
        upload = form[file_field] if file_field in form else None
        if upload is not None and getattr(upload, "file", None) is not None and upload.filename:
            with output_path.open("wb") as target:
                shutil.copyfileobj(upload.file, target)
            return
        text = self._field_value(form, text_field, "").strip()
        if not text:
            raise ValueError(f"Missing required input for {text_field}")
        parsed = json.loads(text)
        output_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")

    def _field_value(self, form: cgi.FieldStorage, name: str, default: str) -> str:
        if name not in form:
            return default
        value = form.getvalue(name)
        if isinstance(value, list):
            return str(value[0])
        return default if value is None else str(value)

    def _run_commands_with_logs(self, commands: dict[str, list[str]]) -> str:
        log_chunks: list[str] = []
        for step_name in ("profiler", "runtime", "dashboard"):
            cmd = commands[step_name]
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            log_chunks.append(f"$ {' '.join(cmd)}")
            if completed.stdout.strip():
                log_chunks.append(completed.stdout.strip())
            if completed.stderr.strip():
                log_chunks.append(completed.stderr.strip())
            if completed.returncode != 0:
                raise RuntimeError("\n\n".join(log_chunks))
        return "\n\n".join(log_chunks)

    def _format_metric(self, value: object, suffix: str) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.3f}{suffix}"
        except Exception:
            return f"{value}{suffix}"

    def _send_html(self, body: bytes, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        data = path.read_bytes()
        content_type = "application/json" if path.suffix == ".json" else "text/html; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KernelDVFS demo as a local webpage")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"KernelDVFS web demo running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
