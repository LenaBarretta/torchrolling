"""Argument checks, dtype rules and the choice of backend, shared by rolling() and ewm()."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from torchrolling import _triton
except ImportError:  # pragma: no cover - Triton is only installed on Linux
    _triton = None  # type: ignore[assignment]


def check_int(name: str, value: object, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return value


def as_float(x: object, name: str = "x") -> Tensor:
    """``x`` as a floating tensor: integers become the default float dtype."""
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.is_complex():
        raise TypeError("complex tensors are not supported")
    if x.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    return x if x.is_floating_point() else x.to(torch.get_default_dtype())


def as_other(other: object, shape: torch.Size) -> Tensor:
    """The second series of a pairwise statistic, which must be shaped like ``x``."""
    other = as_float(other, "other")
    if other.shape != shape:
        raise ValueError(
            f"other must have the same shape as x {tuple(shape)}, got {tuple(other.shape)}"
        )
    return other


def accumulator(dtype: torch.dtype, acc_dtype: torch.dtype | None) -> torch.dtype:
    """dtype for sums and moments: the input's, but at least float32."""
    if acc_dtype is None:
        return torch.float64 if dtype == torch.float64 else torch.float32
    if not isinstance(acc_dtype, torch.dtype) or not acc_dtype.is_floating_point:
        raise TypeError(f"acc_dtype must be a floating torch.dtype, got {acc_dtype!r}")
    return acc_dtype


def use_triton(*tensors: Tensor, window: int = 0, limit: str = "") -> bool:
    """Whether the Triton kernels can compute this: CUDA (or the interpreter, in tests), no
    gradients needed, and ``window`` within the kernel's limit (``_triton.<limit>``)."""
    x = tensors[0]
    return (
        _triton is not None
        and x.numel() > 0
        and (not limit or window <= getattr(_triton, limit))
        and (x.is_cuda or _triton.INTERPRET)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))
    )


def kernel_dtype(x: Tensor) -> Tensor:
    """The kernels read float32 or float64."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def dtype_name(dtype: torch.dtype) -> str:
    return "float64" if dtype == torch.float64 else "float32"  # pragma: no cover - needs Triton
