from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaperKernelSpec:
    kernel_name: str
    family: str
    phase: str
    baseline_ms: float
    optimal_core_mhz: int
    optimal_mem_mhz: int
    static_power_watts: float
    dynamic_power_watts: float
    repeat_count: int = 1
    m: int = 0
    n: int = 0
    k: int = 0
    rows: int = 0
    cols: int = 0
    elements: int = 0
    heads: int = 0
    description: str = ""


PAPER_CORE_CLOCKS = [630, 840, 1050, 1260, 1470, 1680]
PAPER_MEMORY_CLOCKS = [5001, 9251, 9501]


def _spec(
    idx: int,
    name: str,
    family: str,
    phase: str,
    baseline_ms: float,
    optimal_core_mhz: int,
    optimal_mem_mhz: int,
    static_power_watts: float,
    dynamic_power_watts: float,
    *,
    repeat_count: int = 1,
    m: int = 0,
    n: int = 0,
    k: int = 0,
    rows: int = 0,
    cols: int = 0,
    elements: int = 0,
    heads: int = 0,
) -> PaperKernelSpec:
    return PaperKernelSpec(
        kernel_name=f"k{idx:02d}_{name}",
        family=family,
        phase=phase,
        baseline_ms=baseline_ms,
        optimal_core_mhz=optimal_core_mhz,
        optimal_mem_mhz=optimal_mem_mhz,
        static_power_watts=static_power_watts,
        dynamic_power_watts=dynamic_power_watts,
        repeat_count=repeat_count,
        m=m,
        n=n,
        k=k,
        rows=rows,
        cols=cols,
        elements=elements,
        heads=heads,
        description=name,
    )


def paper_kernel_specs(num_layers: int = 12) -> list[PaperKernelSpec]:
    # Forward-only GPT-2 style kernel inventory aligned to the public llm.c gpt2_forward structure.
    # Representative dimensions are based on GPT-2-small style widths and head count while keeping
    # sequence length practical for repeated benchmarking.
    tokens = 256
    hidden = 768
    mlp_hidden = hidden * 4
    vocab = 50257
    heads = 12
    head_dim = hidden // heads
    attn_rows = tokens * heads

    return [
        _spec(0, "tokpos_embedding_add", "embedding_add", "embedding", 0.17, 840, 9501, 44.0, 22.0, rows=tokens, cols=hidden, elements=tokens * hidden),
        _spec(1, "block_pre_attn_layernorm", "layernorm", "forward", 0.28, 1260, 9501, 49.0, 28.0, repeat_count=num_layers, rows=tokens, cols=hidden),
        _spec(2, "block_qkv_projection", "gemm", "forward", 0.96, 3090, 7001, 60.0, 118.0, repeat_count=num_layers, m=tokens, n=hidden * 3, k=hidden),
        _spec(3, "block_qkv_permute", "permute", "forward", 0.11, 2400, 14001, 42.0, 18.0, repeat_count=num_layers, rows=tokens, cols=hidden * 3, elements=tokens * hidden * 3, heads=heads),
        _spec(4, "block_attn_scores", "gemm", "forward", 0.51, 3090, 13365, 56.0, 92.0, repeat_count=num_layers, m=attn_rows, n=tokens, k=head_dim),
        _spec(5, "block_attn_softmax", "softmax", "forward", 0.19, 2400, 14001, 46.0, 24.0, repeat_count=num_layers, rows=attn_rows, cols=tokens),
        _spec(6, "block_attn_context", "gemm", "forward", 0.49, 3000, 13365, 55.0, 88.0, repeat_count=num_layers, m=attn_rows, n=head_dim, k=tokens),
        _spec(7, "block_attn_output_projection", "gemm", "forward", 0.73, 3090, 7001, 58.0, 104.0, repeat_count=num_layers, m=tokens, n=hidden, k=hidden),
        _spec(8, "block_attn_residual_add", "residual_add", "forward", 0.10, 2100, 14001, 43.0, 16.0, repeat_count=num_layers, rows=tokens, cols=hidden, elements=tokens * hidden),
        _spec(9, "block_pre_mlp_layernorm", "layernorm", "forward", 0.27, 1260, 9501, 49.0, 27.0, repeat_count=num_layers, rows=tokens, cols=hidden),
        _spec(10, "block_mlp_expand", "gemm", "forward", 1.22, 3090, 7001, 61.0, 132.0, repeat_count=num_layers, m=tokens, n=mlp_hidden, k=hidden),
        _spec(11, "block_mlp_gelu", "gelu", "forward", 0.16, 2100, 14001, 45.0, 20.0, repeat_count=num_layers, rows=tokens, cols=mlp_hidden, elements=tokens * mlp_hidden),
        _spec(12, "block_mlp_project", "gemm", "forward", 1.08, 3090, 7001, 60.0, 124.0, repeat_count=num_layers, m=tokens, n=hidden, k=mlp_hidden),
        _spec(13, "block_mlp_residual_add", "residual_add", "forward", 0.10, 2100, 14001, 43.0, 16.0, repeat_count=num_layers, rows=tokens, cols=hidden, elements=tokens * hidden),
        _spec(14, "final_layernorm", "layernorm", "output", 0.29, 1260, 9501, 49.0, 28.0, rows=tokens, cols=hidden),
        _spec(15, "logits_projection", "gemm", "output", 3.48, 3090, 7001, 63.0, 150.0, m=tokens, n=vocab, k=hidden),
    ]


def expanded_trace_specs(num_layers: int = 12) -> list[PaperKernelSpec]:
    expanded: list[PaperKernelSpec] = []
    for spec in paper_kernel_specs(num_layers=num_layers):
        expanded.extend([spec] * spec.repeat_count)
    return expanded


def family_category(family: str) -> str:
    if family == "gemm":
        return "compute"
    return "memory"


def spec_shapes(spec: PaperKernelSpec) -> dict[str, int]:
    return {
        "m": spec.m,
        "n": spec.n,
        "k": spec.k,
        "rows": spec.rows,
        "cols": spec.cols,
        "elements": spec.elements,
        "heads": spec.heads,
        "repeat_count": spec.repeat_count,
    }


def build_family_inputs(spec: PaperKernelSpec, torch: Any, device: str, dtype: Any) -> tuple[Any, ...]:
    if spec.family == "gemm":
        return (
            torch.randn((spec.m, spec.k), device=device, dtype=dtype),
            torch.randn((spec.k, spec.n), device=device, dtype=dtype),
        )
    if spec.family == "softmax":
        return (torch.randn((spec.rows, spec.cols), device=device, dtype=dtype),)
    if spec.family == "layernorm":
        return (
            torch.randn((spec.rows, spec.cols), device=device, dtype=dtype),
            torch.randn((spec.cols,), device=device, dtype=dtype),
        )
    if spec.family in {"gelu", "permute", "embedding_backward"}:
        width = spec.cols or spec.elements or 1
        return (torch.randn((spec.rows, width), device=device, dtype=dtype),)
    if spec.family in {"residual_add", "bias_add", "embedding_add"}:
        return (
            torch.randn((spec.rows, spec.cols), device=device, dtype=dtype),
            torch.randn((spec.rows, spec.cols), device=device, dtype=dtype),
        )
    if spec.family == "bias_reduce":
        return (torch.randn((spec.rows, spec.cols), device=device, dtype=dtype),)
    raise ValueError(f"Unsupported paper kernel family '{spec.family}'")


def run_family_kernel(spec: PaperKernelSpec, torch: Any, args: tuple[Any, ...]) -> Any:
    if spec.family == "gemm":
        left, right = args
        return torch.matmul(left, right)
    if spec.family == "softmax":
        (x,) = args
        return torch.softmax(x, dim=-1)
    if spec.family == "layernorm":
        x, weight = args
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight)
    if spec.family == "gelu":
        (x,) = args
        return torch.nn.functional.gelu(x)
    if spec.family == "permute":
        (x,) = args
        return x.reshape(x.shape[0], -1, max(1, x.shape[1] // max(1, spec.heads or 1))).transpose(0, 1).contiguous()
    if spec.family in {"residual_add", "bias_add", "embedding_add"}:
        left, right = args
        return left + right
    if spec.family == "bias_reduce":
        (x,) = args
        return x.sum(dim=0)
    if spec.family == "embedding_backward":
        (x,) = args
        return x.sum(dim=0)
    raise ValueError(f"Unsupported paper kernel family '{spec.family}'")
