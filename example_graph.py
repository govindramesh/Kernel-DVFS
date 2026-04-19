from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable

from kerneldvfs.models import write_json

LOGGER = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

from kerneldvfs import kernels


@dataclass(frozen=True)
class GraphConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int
    mlp_ratio: int
    dtype: str
    device: str

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        return self.hidden_size * self.mlp_ratio


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("example_graph.py requires torch to be installed")


def resolve_dtype(name: str) -> Any:
    require_torch()
    lookup = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in lookup:
        raise ValueError(f"Unsupported dtype '{name}'")
    return lookup[name]


def select_runtime(backend: str, device: str) -> str:
    require_torch()
    if backend == "torch":
        return "torch"
    if backend == "triton":
        if not kernels.TRITON_AVAILABLE or device != "cuda":
            raise RuntimeError("Triton backend requires torch+triton with CUDA tensors")
        return "triton"
    if backend != "auto":
        raise ValueError(f"Unsupported backend '{backend}'")
    if kernels.TRITON_AVAILABLE and device == "cuda":
        return "triton"
    return "torch"


def make_inputs(config: GraphConfig) -> dict[str, Any]:
    require_torch()
    if config.hidden_size % config.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")

    dtype = resolve_dtype(config.dtype)
    tokens = config.batch_size * config.seq_len
    hidden = config.hidden_size
    mlp_hidden = config.mlp_hidden_size

    torch.manual_seed(7)

    return {
        "x": torch.randn((tokens, hidden), device=config.device, dtype=dtype),
        "qkv_weight": torch.randn((hidden, hidden * 3), device=config.device, dtype=dtype),
        "rms_weight": torch.randn((hidden,), device=config.device, dtype=dtype),
        "out_proj_weight": torch.randn((hidden, hidden), device=config.device, dtype=dtype),
        "up_gate_weight": torch.randn((hidden, mlp_hidden * 2), device=config.device, dtype=dtype),
        "down_proj_weight": torch.randn((mlp_hidden, hidden), device=config.device, dtype=dtype),
    }


def torch_rmsnorm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return normalized * weight


def apply_causal_mask(scores: Any, seq_len: int) -> Any:
    require_torch()
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=scores.device, dtype=torch.float32),
        diagonal=1,
    )
    return scores.float() + mask.unsqueeze(0)


def attention_scores_op(q: Any, k: Any, runtime: str) -> Any:
    if runtime == "triton":
        outputs = [kernels.launch_matmul(q[index], k[index].transpose(0, 1)) for index in range(q.shape[0])]
        return torch.stack(outputs, dim=0)
    return torch.matmul(q, k.transpose(-1, -2))


def attention_values_op(attn_probs: Any, v: Any, runtime: str) -> Any:
    if runtime == "triton":
        outputs = [kernels.launch_matmul(attn_probs[index], v[index]) for index in range(attn_probs.shape[0])]
        return torch.stack(outputs, dim=0)
    return torch.matmul(attn_probs, v)


def run_kernel(name: str, fn: Callable[[], Any]) -> tuple[Any, float]:
    require_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    output = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    LOGGER.info("Executed %s in %.4f ms", name, duration_ms)
    return output, duration_ms


def build_graph(config: GraphConfig, runtime: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require_torch()
    tensors = make_inputs(config)
    seq_len = config.seq_len
    batch_size = config.batch_size
    num_heads = config.num_heads
    head_dim = config.head_dim
    hidden = config.hidden_size
    mlp_hidden = config.mlp_hidden_size

    trace: list[dict[str, Any]] = []
    graph_nodes: list[dict[str, Any]] = []

    x = tensors["x"]
    qkv_weight = tensors["qkv_weight"]
    rms_weight = tensors["rms_weight"]
    out_proj_weight = tensors["out_proj_weight"]
    up_gate_weight = tensors["up_gate_weight"]
    down_proj_weight = tensors["down_proj_weight"]

    if runtime == "triton":
        matmul = kernels.launch_matmul
        rmsnorm = kernels.launch_rmsnorm
        softmax = kernels.launch_softmax
        silu_gate = kernels.launch_silu_gate
    else:
        matmul = lambda a, b: torch.matmul(a, b)
        rmsnorm = torch_rmsnorm
        softmax = lambda x: torch.softmax(x, dim=-1)
        silu_gate = lambda gate, up: torch.nn.functional.silu(gate) * up

    qkv, duration = run_kernel("fused_qkv_matmul", lambda: matmul(x, qkv_weight))
    trace.append({"kernel_name": "fused_qkv_matmul", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "fused_qkv_matmul",
            "inputs": ["x", "qkv_weight"],
            "outputs": ["qkv"],
            "shape": [x.shape[0], qkv_weight.shape[1]],
        }
    )

    normalized, duration = run_kernel("rmsnorm", lambda: rmsnorm(x, rms_weight))
    trace.append({"kernel_name": "rmsnorm", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "rmsnorm",
            "inputs": ["x", "rms_weight"],
            "outputs": ["normalized"],
            "shape": list(normalized.shape),
        }
    )

    q, k, v = torch.chunk(qkv, 3, dim=-1)
    q = q.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)
    attn_scores, duration = run_kernel("attention_scores_matmul", lambda: attention_scores_op(q, k, runtime))
    trace.append({"kernel_name": "attention_scores_matmul", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "attention_scores_matmul",
            "inputs": ["q", "k"],
            "outputs": ["attention_scores"],
            "shape": list(attn_scores.shape),
        }
    )

    masked_scores = apply_causal_mask(attn_scores, seq_len)
    if runtime == "triton":
        attn_probs, duration = run_kernel(
            "masked_softmax",
            lambda: torch.stack([softmax(masked_scores[index].to(qkv.dtype)) for index in range(masked_scores.shape[0])], dim=0),
        )
    else:
        attn_probs, duration = run_kernel("masked_softmax", lambda: softmax(masked_scores.to(qkv.dtype)))
    trace.append({"kernel_name": "masked_softmax", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "masked_softmax",
            "inputs": ["attention_scores"],
            "outputs": ["attention_probs"],
            "shape": list(attn_probs.shape),
        }
    )

    v = v.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)
    context, duration = run_kernel("attention_value_matmul", lambda: attention_values_op(attn_probs, v, runtime))
    trace.append({"kernel_name": "attention_value_matmul", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "attention_value_matmul",
            "inputs": ["attention_probs", "v"],
            "outputs": ["context"],
            "shape": list(context.shape),
        }
    )

    context = context.reshape(batch_size, num_heads, seq_len, head_dim).permute(0, 2, 1, 3).reshape(batch_size * seq_len, hidden)
    attention_out = matmul(context, out_proj_weight)

    up_gate, duration = run_kernel("mlp_up_proj_matmul", lambda: matmul(normalized + attention_out, up_gate_weight))
    trace.append({"kernel_name": "mlp_up_proj_matmul", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "mlp_up_proj_matmul",
            "inputs": ["normalized", "attention_out", "up_gate_weight"],
            "outputs": ["up_gate"],
            "shape": list(up_gate.shape),
        }
    )

    gate, up = torch.chunk(up_gate, 2, dim=-1)
    activated, duration = run_kernel("silu_gate", lambda: silu_gate(gate, up))
    trace.append({"kernel_name": "silu_gate", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "silu_gate",
            "inputs": ["gate", "up"],
            "outputs": ["activated"],
            "shape": list(activated.shape),
        }
    )

    output, duration = run_kernel("mlp_down_proj_matmul", lambda: matmul(activated.reshape(batch_size * seq_len, mlp_hidden), down_proj_weight))
    trace.append({"kernel_name": "mlp_down_proj_matmul", "duration_ms": round(duration, 6)})
    graph_nodes.append(
        {
            "kernel_name": "mlp_down_proj_matmul",
            "inputs": ["activated", "down_proj_weight"],
            "outputs": ["output"],
            "shape": list(output.shape),
        }
    )

    graph = {
        "metadata": {
            "graph_name": "synthetic_transformer_block",
            "runtime": runtime,
            "device": config.device,
            "dtype": config.dtype,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
            "final_output_shape": list(output.shape),
        },
        "nodes": graph_nodes,
    }
    return graph, trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic transformer-style kernel graph and emit a trace")
    parser.add_argument("--backend", choices=["auto", "torch", "triton"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--graph-output", default="data/example_graph.json")
    parser.add_argument("--trace-output", default="data/example_execution_trace.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    config = GraphConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dtype=args.dtype,
        device=args.device,
    )
    runtime = select_runtime(args.backend, args.device)
    graph, trace = build_graph(config=config, runtime=runtime)

    write_json(args.graph_output, graph)
    write_json(
        args.trace_output,
        {
            "metadata": {
                **graph["metadata"],
                "description": "Measured execution trace for the synthetic transformer block",
            },
            "trace": trace,
        },
    )
    LOGGER.info("Wrote runnable graph to %s", args.graph_output)
    LOGGER.info("Wrote execution trace to %s", args.trace_output)


if __name__ == "__main__":
    main()
