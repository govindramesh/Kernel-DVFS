from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE = triton is not None and tl is not None and torch is not None


def require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton kernels require torch and triton to be installed in the active environment")


def next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def kernel_entrypoints() -> dict[str, str]:
    return {
        "fused_qkv_matmul": "launch_matmul",
        "attention_scores_matmul": "launch_matmul",
        "attention_value_matmul": "launch_matmul",
        "mlp_up_proj_matmul": "launch_matmul",
        "mlp_down_proj_matmul": "launch_matmul",
        "rmsnorm": "launch_rmsnorm",
        "masked_softmax": "launch_softmax",
        "silu_gate": "launch_silu_gate",
    }


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            a_mask = (offs_m[:, None] < M) & ((k_start + offs_k[None, :]) < K)
            b_mask = ((k_start + offs_k[:, None]) < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(tl.float16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


    @triton.jit
    def rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        y_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / N
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * weight
        y_ptrs = y_ptr + row * stride_ym + cols * stride_yn
        tl.store(y_ptrs, y.to(tl.float16), mask=mask)


    @triton.jit
    def softmax_kernel(
        x_ptr,
        y_ptr,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_ptrs = x_ptr + row * stride_xm + cols * stride_xn
        row_values = tl.load(x_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        row_values = row_values - tl.max(row_values, axis=0)
        numerator = tl.exp(row_values)
        denominator = tl.sum(numerator, axis=0)
        output = numerator / denominator
        y_ptrs = y_ptr + row * stride_ym + cols * stride_yn
        tl.store(y_ptrs, output.to(tl.float16), mask=mask)


    @triton.jit
    def silu_gate_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = gate * tl.sigmoid(gate) * up
        tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


def launch_matmul(a: Any, b: Any) -> Any:
    require_triton()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("launch_matmul expects rank-2 tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible matmul shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("launch_matmul expects CUDA tensors")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def launch_rmsnorm(x: Any, weight: Any, eps: float = 1e-6) -> Any:
    require_triton()
    if x.ndim != 2:
        raise ValueError("launch_rmsnorm expects a rank-2 input tensor")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("launch_rmsnorm expects a 1D weight tensor sized to the hidden dimension")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("launch_rmsnorm expects CUDA tensors")

    rows, cols = x.shape
    y = torch.empty_like(x)
    block_size = min(max(128, next_power_of_2(cols)), 4096)
    rmsnorm_kernel[(rows,)](
        x,
        weight,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        cols,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=min(max(block_size // 256, 1), 8),
    )
    return y


def launch_softmax(x: Any) -> Any:
    require_triton()
    if x.ndim != 2:
        raise ValueError("launch_softmax expects a rank-2 tensor")
    if not x.is_cuda:
        raise ValueError("launch_softmax expects CUDA tensors")

    rows, cols = x.shape
    y = torch.empty_like(x)
    block_size = min(max(128, next_power_of_2(cols)), 4096)
    softmax_kernel[(rows,)](
        x,
        y,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        cols,
        BLOCK_SIZE=block_size,
        num_warps=min(max(block_size // 256, 1), 8),
    )
    return y


def launch_silu_gate(gate: Any, up: Any) -> Any:
    require_triton()
    if gate.shape != up.shape:
        raise ValueError(f"launch_silu_gate expects matching input shapes, got {tuple(gate.shape)} and {tuple(up.shape)}")
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("launch_silu_gate expects CUDA tensors")

    output = torch.empty_like(gate)
    n_elements = gate.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    silu_gate_kernel[grid](gate, up, output, n_elements, BLOCK_SIZE=block_size, num_warps=4)
    return output


def example_inputs(device: str = "cuda") -> dict[str, Any]:
    require_triton()
    hidden = 4096
    tokens = 128
    heads = 32
    head_dim = hidden // heads
    return {
        "fused_qkv_matmul": (
            torch.randn((tokens, hidden), device=device, dtype=torch.float16),
            torch.randn((hidden, hidden * 3), device=device, dtype=torch.float16),
        ),
        "attention_scores_matmul": (
            torch.randn((tokens, head_dim), device=device, dtype=torch.float16),
            torch.randn((head_dim, tokens), device=device, dtype=torch.float16),
        ),
        "attention_value_matmul": (
            torch.randn((tokens, tokens), device=device, dtype=torch.float16),
            torch.randn((tokens, head_dim), device=device, dtype=torch.float16),
        ),
        "mlp_up_proj_matmul": (
            torch.randn((tokens, hidden), device=device, dtype=torch.float16),
            torch.randn((hidden, hidden * 4), device=device, dtype=torch.float16),
        ),
        "mlp_down_proj_matmul": (
            torch.randn((tokens, hidden * 4), device=device, dtype=torch.float16),
            torch.randn((hidden * 4, hidden), device=device, dtype=torch.float16),
        ),
        "rmsnorm": (
            torch.randn((tokens, hidden), device=device, dtype=torch.float16),
            torch.randn((hidden,), device=device, dtype=torch.float16),
        ),
        "masked_softmax": (
            torch.randn((tokens * heads, tokens), device=device, dtype=torch.float16),
        ),
        "silu_gate": (
            torch.randn((tokens, hidden * 4), device=device, dtype=torch.float16),
            torch.randn((tokens, hidden * 4), device=device, dtype=torch.float16),
        ),
    }


def benchmarkable_kernels() -> list[str]:
    return list(kernel_entrypoints().keys())
