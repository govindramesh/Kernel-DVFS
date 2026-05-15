from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR = ROOT / "kerneldvfs" / "cuda_kernels"


@dataclass(frozen=True)
class CustomCudaKernel:
    kernel_name: str
    category: str
    description: str
    source_files: tuple[str, ...]
    build_inputs: Callable[[Any, str, Any], tuple[Any, ...]]
    run: Callable[[Any, tuple[Any, ...]], Any]


def _build_rmsnorm_inputs(torch: Any, device: str, dtype: Any) -> tuple[Any, ...]:
    return (
        torch.randn((128, 512), device=device, dtype=dtype),
        torch.randn((512,), device=device, dtype=dtype),
    )


def _build_silu_inputs(torch: Any, device: str, dtype: Any) -> tuple[Any, ...]:
    return (torch.randn((128, 2048), device=device, dtype=dtype),)


def _build_softmax_inputs(torch: Any, device: str, dtype: Any) -> tuple[Any, ...]:
    return (torch.randn((1024, 128), device=device, dtype=dtype),)


def _run_rmsnorm(ext: Any, args: tuple[Any, ...]) -> Any:
    x, weight = args
    return ext.rmsnorm_forward(x, weight, 1e-5)


def _run_silu(ext: Any, args: tuple[Any, ...]) -> Any:
    (x,) = args
    return ext.silu_forward(x)


def _run_softmax(ext: Any, args: tuple[Any, ...]) -> Any:
    (x,) = args
    return ext.row_softmax_forward(x)


CUSTOM_CUDA_KERNELS: dict[str, CustomCudaKernel] = {
    "rmsnorm": CustomCudaKernel(
        kernel_name="rmsnorm",
        category="memory",
        description="CUDA RMSNorm kernel inspired by FlashInfer normalization operators",
        source_files=("binding.cpp", "rmsnorm_kernel.cu", "silu_kernel.cu", "softmax_kernel.cu"),
        build_inputs=_build_rmsnorm_inputs,
        run=_run_rmsnorm,
    ),
    "silu": CustomCudaKernel(
        kernel_name="silu",
        category="compute",
        description="CUDA SiLU activation kernel inspired by FlashInfer activation operators",
        source_files=("binding.cpp", "rmsnorm_kernel.cu", "silu_kernel.cu", "softmax_kernel.cu"),
        build_inputs=_build_silu_inputs,
        run=_run_silu,
    ),
    "softmax": CustomCudaKernel(
        kernel_name="softmax",
        category="memory",
        description="CUDA row softmax kernel for attention-style score normalization",
        source_files=("binding.cpp", "rmsnorm_kernel.cu", "silu_kernel.cu", "softmax_kernel.cu"),
        build_inputs=_build_softmax_inputs,
        run=_run_softmax,
    ),
}


def has_custom_cuda_kernel(kernel_name: str) -> bool:
    return kernel_name in CUSTOM_CUDA_KERNELS


def get_custom_cuda_kernel(kernel_name: str) -> CustomCudaKernel:
    if kernel_name not in CUSTOM_CUDA_KERNELS:
        raise KeyError(f"Unknown custom CUDA kernel '{kernel_name}'")
    return CUSTOM_CUDA_KERNELS[kernel_name]


def custom_cuda_category(kernel_name: str) -> str:
    return get_custom_cuda_kernel(kernel_name).category


@lru_cache(maxsize=1)
def load_custom_cuda_extension(torch: Any) -> Any:
    from torch.utils.cpp_extension import load

    sources = [str(CUDA_DIR / filename) for filename in ("binding.cpp", "rmsnorm_kernel.cu", "silu_kernel.cu", "softmax_kernel.cu")]
    build_directory = ROOT / "data" / "torch_extensions" / "kerneldvfs_custom_cuda"
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="kerneldvfs_custom_cuda",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_directory),
        verbose=False,
    )
