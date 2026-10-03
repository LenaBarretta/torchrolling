# mypy: disable-error-code="no-untyped-def"
"""Triton kernels. Importing this module fails where Triton is not installed (macOS, Windows).

Rolling quantile: each program takes ``BLOCK_T`` consecutive outputs of one series, loads
their windows as a ``[BLOCK_T, BLOCK_W]`` tile, pushes missing values to the end with +inf,
sorts every row in registers and reads the answer straight out of the sorted tile.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor

MAX_WINDOW = 1024
# CPU tensors only work in Triton's interpreter, which the test suite uses without a GPU.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
MODES = {"linear": 0, "lower": 1, "higher": 2, "midpoint": 3, "nearest": 4}


@triton.jit
def _quantile_kernel(
    x_ptr,
    q_ptr,
    out_ptr,
    length,
    window,
    shift,
    min_periods,
    MODE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
    j = tl.arange(0, BLOCK_W)
    pos = (t + shift - window + 1)[:, None] + j[None, :]
    inside = (j[None, :] < window) & (pos >= 0) & (pos < length) & (t[:, None] < length)
    v = tl.load(x_ptr + row * length + pos, mask=inside, other=float("nan"))
    valid = inside & (v == v) & (tl.abs(v) != float("inf"))
    n = tl.sum(valid.to(tl.int32), axis=1)
    s = tl.sort(tl.where(valid, v, float("inf")), dim=1)
    s = tl.where(j[None, :] < n[:, None], s, 0.0)  # drop the +inf padding again

    # Same arithmetic as pandas: the rank is q * (n - 1) in float64.
    rank = tl.load(q_ptr) * (n - 1).to(tl.float64)
    lo = tl.floor(rank).to(tl.int32)
    hi = tl.ceil(rank).to(tl.int32)
    frac = (rank - lo.to(tl.float64)).to(s.dtype)
    a = tl.sum(tl.where(j[None, :] == lo[:, None], s, 0.0), axis=1)
    b = tl.sum(tl.where(j[None, :] == hi[:, None], s, 0.0), axis=1)
    if MODE == 0:
        res = a + (b - a) * frac
    elif MODE == 1:
        res = a
    elif MODE == 2:
        res = b
    elif MODE == 3:
        res = (a + b) * 0.5
    else:  # nearest, ties to even like pandas
        res = tl.where((frac > 0.5) | ((frac == 0.5) & (lo % 2 == 1)), b, a)

    ok = (n >= min_periods) & (n > 0)
    tl.store(out_ptr + row * length + t, tl.where(ok, res, float("nan")), mask=t < length)


def _quantile(
    x: Tensor, window: int, shift: int, min_periods: int, q: float, interpolation: str
) -> Tensor:
    """Rolling quantile over the last dim of ``x``. Missing values are NaN or +-inf."""
    length = x.shape[-1]
    flat = x.reshape(-1, length).contiguous()
    out = torch.empty_like(flat)
    q_tensor = torch.tensor([q], dtype=torch.float64, device=x.device)
    block_w = triton.next_power_of_2(window)
    block_t = max(1, min(64, 4096 // block_w))
    grid = (flat.shape[0], triton.cdiv(length, block_t))
    _quantile_kernel[grid](
        flat,
        q_tensor,
        out,
        length,
        window,
        shift,
        min_periods,
        MODE=MODES[interpolation],
        BLOCK_T=block_t,
        BLOCK_W=block_w,
    )
    return out.reshape(x.shape)


quantile: Callable[[Tensor, int, int, int, float, str], Tensor] = _quantile

if hasattr(torch.library, "custom_op"):
    # torch >= 2.4: as a registered op, torch.compile keeps the kernel call inside the graph
    # (it only needs the output's shape, from the fake implementation) instead of breaking it.
    _op = torch.library.custom_op(
        "torchrolling::quantile",
        _quantile,
        mutates_args=(),
        schema="(Tensor x, int window, int shift, int min_periods, float q, str interpolation)"
        " -> Tensor",
    )

    @_op.register_fake
    def _(x: Tensor, window: int, shift: int, min_periods: int, q: float, interpolation: str):
        return torch.empty_like(x)

    quantile = _op
