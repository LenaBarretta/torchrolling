# mypy: disable-error-code="no-untyped-def"
"""Triton kernels. Importing this module fails where Triton is not installed (macOS, Windows).

Rolling statistics (``_rolling_kernel``): the same van Herk / Gil-Werman split as the torch
code, fused into one pass. A program owns ``ROWS`` consecutive blocks of ``window`` elements
of one series. It loads them twice as ``[ROWS, BLOCK_W]`` tiles: once as the blocks
themselves (tails, scanned backwards) and once shifted by one block and one element (heads,
scanned forwards), so that row ``b``, column ``o`` of both tiles belongs to the window that
starts at offset ``o`` of block ``b``. Every window then combines two numbers in the same
tile position: no data moves between registers, and each element is read about twice, both
times from cache.

Rolling quantile, small windows (``_quantile_sort_kernel``): every window is loaded as a row
of a tile and sorted in registers. Larger windows (``_quantile_select_kernel``): a program
sorts the whole segment its outputs need once, with each value's position packed into the
sort key, then walks the sorted segment once, counting for every output how many of its own
values it has passed: O(window) per output instead of O(window log^2 window).

Exponentially weighted statistics (``_ewm_kernel``): one program per series walks it in
chunks; inside a chunk the affine recurrences are scanned in parallel, and the state at the
chunk's end is carried into the next one.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from torch import Tensor

if not hasattr(tl, "gather"):  # pragma: no cover - torch < 2.6 ships an older Triton
    raise ImportError("torchrolling's kernels need Triton 3.2 or newer")

# CPU tensors only work in Triton's interpreter, which the test suite uses without a GPU.
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
INTERPOLATIONS = {"linear": 0, "lower": 1, "higher": 2, "midpoint": 3, "nearest": 4}
STATS = {
    "count": 0,
    "sum": 1,
    "mean": 2,
    "var": 3,
    "std": 4,
    "skew": 5,
    "kurt": 6,
    "min": 7,
    "max": 8,
    "cov": 9,
    "corr": 10,
}
EWM_MODES = {"mean": 0, "var": 1, "cov": 2, "corr": 3}

# Largest window the rolling and quantile kernels take; beyond it the torch code is already
# efficient.
MAX_WINDOW = 4096
QUANTILE_WINDOW = 4096

# Launch configurations, measured on a T4 with bench/tune.py. Rolling statistics come in
# groups by how many tensors they keep alive.
GROUPS = {0: "light", 1: "light", 2: "light", 7: "light", 8: "light"}
GROUPS |= {3: "moments", 4: "moments", 9: "moments", 5: "heavy", 6: "heavy", 10: "heavy"}
# Windows up to DIRECT_WINDOW[group] read every window straight from cache (_rolling_direct_
# kernel); larger ones use the van Herk split (_rolling_kernel).
DIRECT_WINDOW = {"light": 32, "moments": 32, "heavy": 32}
# {group: {BLOCK_W: (tile elements, num_warps)}}; sizes in between use the closest one below.
DIRECT_CONFIGS: dict[str, dict[int, tuple[int, int]]] = {"light": {}, "moments": {}, "heavy": {}}
DIRECT_DEFAULT = {"light": (2048, 4), "moments": (2048, 4), "heavy": (1024, 4)}
ROLLING_CONFIGS: dict[str, dict[int, tuple[int, int]]] = {
    "light": {
        16: (512, 2), 32: (512, 4), 64: (512, 2), 128: (512, 4), 256: (1024, 4),
        512: (1024, 2), 1024: (1024, 2), 2048: (2048, 4), 4096: (4096, 4),
    },
    "moments": {
        16: (512, 4), 32: (512, 8), 64: (512, 4), 128: (512, 4), 256: (512, 2),
        512: (512, 2), 1024: (1024, 4), 2048: (2048, 4), 4096: (4096, 8),
    },
    "heavy": {
        16: (512, 4), 32: (512, 4), 64: (512, 2), 128: (512, 4), 256: (512, 4),
        512: (512, 2), 1024: (1024, 4), 2048: (2048, 4), 4096: (4096, 8),
    },
}  # fmt: skip
ROLLING_DEFAULT = {"light": (1024, 4), "moments": (1024, 4), "heavy": (1024, 4)}
# Quantiles, by next_power_of_2(window): ("sort", 0, num_warps) sorts every window,
# ("select", BLOCK_T, num_warps) uses the select kernel.
QUANTILE_CONFIGS: dict[int, tuple[str, int, int]] = {
    4: ("sort", 0, 2), 8: ("sort", 0, 2), 16: ("sort", 0, 2), 32: ("select", 32, 2),
    64: ("select", 64, 2), 128: ("select", 256, 4), 256: ("select", 256, 4),
    1024: ("select", 512, 2),
}  # fmt: skip
SORT_WINDOW = 16  # without an entry: sort up to here, select beyond
# EWM: (time steps per chunk, num_warps).
EWM_CONFIG = (1024, 4)


def _nearest(table: dict[int, tuple], size: int) -> tuple | None:  # type: ignore[type-arg]
    """The entry for ``size``, or for the closest measured size below it (above if none)."""
    if not table:
        return None
    below = [k for k in table if k <= size]
    return table[max(below) if below else min(table)]


COUNT = tl.constexpr(0)
SUM = tl.constexpr(1)
MEAN = tl.constexpr(2)
VAR = tl.constexpr(3)
STD = tl.constexpr(4)
SKEW = tl.constexpr(5)
KURT = tl.constexpr(6)
MIN = tl.constexpr(7)
MAX = tl.constexpr(8)
COV = tl.constexpr(9)
CORR = tl.constexpr(10)
TINY_VARIANCE = tl.constexpr(1e-14)


@triton.jit
def _finite(v):
    return (v == v) & (tl.abs(v) != float("inf"))


@triton.jit
def _minimum(a, b):
    return tl.minimum(a, b)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _affine1(a1, b1, a2, b2):
    return a1 * a2, a2 * b1 + b2


@triton.jit
def _affine2(a1, b1, c1, a2, b2, c2):
    return a1 * a2, a2 * b1 + b2, a2 * c1 + c2


@triton.jit
def _affine3(a1, b1, c1, d1, a2, b2, c2, d2):
    return a1 * a2, a2 * b1 + b2, a2 * c1 + c2, a2 * d1 + d2


@triton.jit
def _anchor(x, valid, o, LAST: tl.constexpr, BLOCK_W: tl.constexpr):
    """Per tile row: its last (tails) or first (heads) valid value, as a column."""
    if LAST:
        at = tl.max(tl.where(valid, o, -1), axis=1)
    else:
        at = tl.min(tl.where(valid, o, BLOCK_W), axis=1)
    return tl.sum(tl.where(o == at[:, None], x, 0.0), axis=1)[:, None]


@triton.jit
def _rolling_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    length,
    window,
    shift,
    min_periods,
    ddof,
    tiles_per_row,
    STAT: tl.constexpr,
    PAIR: tl.constexpr,
    ACC: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    base = (pid // tiles_per_row).to(tl.int64) * length
    b = (pid % tiles_per_row) * ROWS + tl.arange(0, ROWS)[:, None]
    o = tl.arange(0, BLOCK_W)[None, :]
    in_block = o < window
    # Tail of the window starting at (b, o): block b from offset o on. Head: block b + 1 up
    # to offset o - 1, kept in column o of the shifted tile (column 0 is an empty head).
    p_tail = b * window + o - window + 1
    p_head = b * window + o
    tail_in = in_block & (p_tail >= 0) & (p_tail < length)
    head_in = in_block & (o >= 1) & (p_head < length)
    xt = tl.load(x_ptr + base + p_tail, mask=tail_in, other=0.0)
    xh = tl.load(x_ptr + base + p_head, mask=head_in, other=0.0)
    t = b * window + o - shift
    store = in_block & (t >= 0) & (t < length)

    if STAT == COUNT:
        n = tl.cumsum((tail_in & (xt == xt)).to(tl.int32), axis=1, reverse=True)
        n += tl.cumsum((head_in & (xh == xh)).to(tl.int32), axis=1)
        end = t + shift
        inside = tl.minimum(end, length - 1) - tl.maximum(end - window + 1, 0) + 1
        res = n.to(ACC)
        ok = inside >= min_periods
    elif STAT == MIN or STAT == MAX:
        vt = tail_in & _finite(xt)
        vh = head_in & _finite(xh)
        n = tl.cumsum(vt.to(tl.int32), axis=1, reverse=True) + tl.cumsum(vh.to(tl.int32), axis=1)
        if STAT == MIN:
            a = tl.associative_scan(tl.where(vt, xt, float("inf")), 1, _minimum, reverse=True)
            res = tl.minimum(a, tl.associative_scan(tl.where(vh, xh, float("inf")), 1, _minimum))
        else:
            a = tl.associative_scan(tl.where(vt, xt, -float("inf")), 1, _maximum, reverse=True)
            res = tl.maximum(a, tl.associative_scan(tl.where(vh, xh, -float("inf")), 1, _maximum))
        ok = (n >= min_periods) & (n > 0)
    else:
        vt = tail_in & _finite(xt)
        vh = head_in & _finite(xh)
        if PAIR:
            yt = tl.load(y_ptr + base + p_tail, mask=tail_in, other=0.0).to(ACC)
            yh = tl.load(y_ptr + base + p_head, mask=head_in, other=0.0).to(ACC)
            vt = vt & _finite(yt)
            vh = vh & _finite(yh)
        xt = xt.to(ACC)
        xh = xh.to(ACC)
        # Shift each side by a value from that same side (see _rolling.py).
        cxt = _anchor(xt, vt, o, True, BLOCK_W)
        cxh = _anchor(xh, vh, o, False, BLOCK_W)
        zxt = tl.where(vt, xt - cxt, 0.0)
        zxh = tl.where(vh, xh - cxh, 0.0)
        na = tl.cumsum(vt.to(ACC), axis=1, reverse=True)
        nb = tl.cumsum(vh.to(ACC), axis=1)
        n = na + nb
        sxa = tl.cumsum(zxt, axis=1, reverse=True)
        sxb = tl.cumsum(zxh, axis=1)
        enough = (n >= min_periods) & (n > 0)
        both = (na > 0) & (nb > 0)
        ra = 1.0 / tl.maximum(na, 1.0)
        rb = 1.0 / tl.maximum(nb, 1.0)
        dxa = sxa * ra
        dxb = sxb * rb
        dx = tl.where(both, (cxh + dxb) - (cxt + dxa), 0.0)
        if STAT == SUM:
            res = sxa + na * cxt + sxb + nb * cxh
            ok = n >= min_periods
        elif STAT == MEAN:
            res = (sxa + na * cxt + sxb + nb * cxh) / tl.maximum(n, 1.0)
            ok = enough
        elif STAT == COV or STAT == CORR:
            cyt = _anchor(yt, vt, o, True, BLOCK_W)
            cyh = _anchor(yh, vh, o, False, BLOCK_W)
            zyt = tl.where(vt, yt - cyt, 0.0)
            zyh = tl.where(vh, yh - cyh, 0.0)
            sya = tl.cumsum(zyt, axis=1, reverse=True)
            syb = tl.cumsum(zyh, axis=1)
            dy = tl.where(both, (cyh + syb * rb) - (cyt + sya * ra), 0.0)
            weight = na * nb / tl.maximum(n, 1.0)
            ca = tl.cumsum(zxt * zyt, axis=1, reverse=True) - sxa * sya * ra
            cb = tl.cumsum(zxh * zyh, axis=1) - sxb * syb * rb
            c = ca + cb + dx * dy * weight
            if STAT == COV:
                res = c / tl.maximum(n - ddof, 1.0)
                ok = enough & (n > ddof)
            else:
                m2x = tl.cumsum(zxt * zxt, axis=1, reverse=True) - sxa * dxa
                m2x += tl.cumsum(zxh * zxh, axis=1) - sxb * dxb + dx * dx * weight
                m2y = tl.cumsum(zyt * zyt, axis=1, reverse=True) - sya * sya * ra
                m2y += tl.cumsum(zyh * zyh, axis=1) - syb * syb * rb + dy * dy * weight
                denom = tl.maximum(m2x, 0.0) * tl.maximum(m2y, 0.0)
                spread = denom > 0
                res = c / tl.sqrt(tl.where(spread, denom, 1.0))
                res = tl.minimum(tl.maximum(res, -1.0), 1.0)
                ok = enough & (n > ddof) & spread
        else:
            # Central moments of each side from the shifted power sums, then merged.
            s2a = tl.cumsum(zxt * zxt, axis=1, reverse=True)
            s2b = tl.cumsum(zxh * zxh, axis=1)
            m2a = tl.maximum(s2a - sxa * dxa, 0.0)
            m2b = tl.maximum(s2b - sxb * dxb, 0.0)
            dn = dx / tl.maximum(n, 1.0)
            m2 = m2a + m2b + dx * dn * na * nb
            if STAT == VAR or STAT == STD:
                res = m2 / tl.maximum(n - ddof, 1.0)
                if STAT == STD:
                    res = tl.sqrt(res)
                ok = enough & (n > ddof)
            else:
                s3a = tl.cumsum(zxt * zxt * zxt, axis=1, reverse=True)
                s3b = tl.cumsum(zxh * zxh * zxh, axis=1)
                m3a = s3a - 3 * dxa * s2a + 2 * dxa * dxa * sxa
                m3b = s3b - 3 * dxb * s2b + 2 * dxb * dxb * sxb
                m3 = m3a + m3b + dx * dn * dn * na * nb * (na - nb) + 3 * dn * (na * m2b - nb * m2a)
                nza = tl.cumsum((zxt != 0).to(tl.int32), axis=1, reverse=True)
                nzb = tl.cumsum((zxh != 0).to(tl.int32), axis=1)
                constant = (nza + nzb == 0) & ((na == 0) | (nb == 0) | (cxt == cxh))
                bvar = m2 / tl.maximum(n, 1.0)
                flat = bvar <= TINY_VARIANCE
                bvar = tl.where(flat, 1.0, bvar)
                if STAT == SKEW:
                    res = tl.sqrt(tl.maximum(n * (n - 1), 0.0)) * (m3 / tl.maximum(n, 1.0))
                    res = res / (tl.maximum(n - 2, 1.0) * bvar * tl.sqrt(bvar))
                    res = tl.where(constant, 0.0, tl.where(flat, float("nan"), res))
                    ok = enough & (n >= 3)
                else:
                    s4a = tl.cumsum(zxt * zxt * zxt * zxt, axis=1, reverse=True)
                    s4b = tl.cumsum(zxh * zxh * zxh * zxh, axis=1)
                    m4a = s4a - 4 * dxa * s3a + 6 * dxa * dxa * s2a - 3 * dxa * dxa * dxa * sxa
                    m4b = s4b - 4 * dxb * s3b + 6 * dxb * dxb * s2b - 3 * dxb * dxb * dxb * sxb
                    m4 = (
                        m4a
                        + m4b
                        + dx * dn * dn * dn * na * nb * (na * na - na * nb + nb * nb)
                        + 6 * dn * dn * (na * na * m2b + nb * nb * m2a)
                        + 4 * dn * (na * m3b - nb * m3a)
                    )
                    d = m4 / tl.maximum(n, 1.0)
                    res = (n * n - 1) * d / (bvar * bvar) - 3 * (n - 1) * (n - 1)
                    res = res / tl.maximum((n - 2) * (n - 3), 1.0)
                    res = tl.where(constant, -3.0, tl.where(flat, float("nan"), res))
                    ok = enough & (n >= 4)
    tl.store(out_ptr + base + t, tl.where(ok, res, float("nan")), mask=store)


@triton.jit
def _rolling_direct_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    length,
    window,
    shift,
    min_periods,
    ddof,
    tiles_per_row,
    STAT: tl.constexpr,
    PAIR: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Small windows: every output reads its whole window (from cache) and reduces it. Moments
    are two-pass (mean first, then deviations), and a window whose valid values are all equal
    gets exactly zero spread, as in the van Herk kernel and pandas."""
    pid = tl.program_id(0)
    base = (pid // tiles_per_row).to(tl.int64) * length
    t = (pid % tiles_per_row) * BLOCK_T + tl.arange(0, BLOCK_T)
    j = tl.arange(0, BLOCK_W)[None, :]
    pos = (t + shift - window + 1)[:, None] + j
    inside = (j < window) & (pos >= 0) & (pos < length) & (t[:, None] < length)
    x = tl.load(x_ptr + base + pos, mask=inside, other=0.0)
    store = t < length

    if STAT == COUNT:
        n = tl.sum((inside & (x == x)).to(tl.int32), axis=1)
        end = t + shift
        positions = tl.minimum(end, length - 1) - tl.maximum(end - window + 1, 0) + 1
        res = n.to(ACC)
        ok = positions >= min_periods
    elif STAT == MIN or STAT == MAX:
        valid = inside & _finite(x)
        n = tl.sum(valid.to(tl.int32), axis=1)
        if STAT == MIN:
            res = tl.min(tl.where(valid, x, float("inf")), axis=1)
        else:
            res = tl.max(tl.where(valid, x, -float("inf")), axis=1)
        ok = (n >= min_periods) & (n > 0)
    else:
        valid = inside & _finite(x)
        if PAIR:
            y = tl.load(y_ptr + base + pos, mask=inside, other=0.0)
            valid = valid & _finite(y)
            y = tl.where(valid, y.to(ACC), 0.0)
        x = tl.where(valid, x.to(ACC), 0.0)
        n = tl.sum(valid.to(ACC), axis=1)
        sx = tl.sum(x, axis=1)
        enough = (n >= min_periods) & (n > 0)
        if STAT == SUM:
            res = sx
            ok = n >= min_periods
        elif STAT == MEAN:
            res = sx / tl.maximum(n, 1.0)
            ok = enough
        else:
            dx = tl.where(valid, x - (sx / tl.maximum(n, 1.0))[:, None], 0.0)
            lo = tl.min(tl.where(valid, x, float("inf")), axis=1)
            hi = tl.max(tl.where(valid, x, -float("inf")), axis=1)
            flat_x = lo == hi
            m2 = tl.where(flat_x, 0.0, tl.sum(dx * dx, axis=1))
            if STAT == COV or STAT == CORR:
                dy = tl.where(valid, y - (tl.sum(y, axis=1) / tl.maximum(n, 1.0))[:, None], 0.0)
                c = tl.sum(dx * dy, axis=1)
                if STAT == COV:
                    res = c / tl.maximum(n - ddof, 1.0)
                    ok = enough & (n > ddof)
                else:
                    ylo = tl.min(tl.where(valid, y, float("inf")), axis=1)
                    yhi = tl.max(tl.where(valid, y, -float("inf")), axis=1)
                    m2y = tl.where(ylo == yhi, 0.0, tl.sum(dy * dy, axis=1))
                    denom = m2 * m2y
                    spread = denom > 0
                    res = c / tl.sqrt(tl.where(spread, denom, 1.0))
                    res = tl.minimum(tl.maximum(res, -1.0), 1.0)
                    ok = enough & (n > ddof) & spread
            elif STAT == VAR or STAT == STD:
                res = m2 / tl.maximum(n - ddof, 1.0)
                if STAT == STD:
                    res = tl.sqrt(res)
                ok = enough & (n > ddof)
            else:
                bvar = m2 / tl.maximum(n, 1.0)
                tiny = bvar <= TINY_VARIANCE
                bvar = tl.where(tiny, 1.0, bvar)
                m3 = tl.sum(dx * dx * dx, axis=1)
                if STAT == SKEW:
                    res = tl.sqrt(tl.maximum(n * (n - 1), 0.0)) * (m3 / tl.maximum(n, 1.0))
                    res = res / (tl.maximum(n - 2, 1.0) * bvar * tl.sqrt(bvar))
                    res = tl.where(flat_x, 0.0, tl.where(tiny, float("nan"), res))
                    ok = enough & (n >= 3)
                else:
                    d = tl.sum(dx * dx * dx * dx, axis=1) / tl.maximum(n, 1.0)
                    res = (n * n - 1) * d / (bvar * bvar) - 3 * (n - 1) * (n - 1)
                    res = res / tl.maximum((n - 2) * (n - 3), 1.0)
                    res = tl.where(flat_x, -3.0, tl.where(tiny, float("nan"), res))
                    ok = enough & (n >= 4)
    tl.store(out_ptr + base + t, tl.where(ok, res, float("nan")), mask=store)


@triton.jit
def _interpolate(a, b, frac, lo, MODE: tl.constexpr):
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
    return res


@triton.jit
def _quantile_sort_kernel(
    x_ptr,
    q_ptr,
    out_ptr,
    length,
    window,
    shift,
    min_periods,
    tiles_per_row,
    MODE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    base = (pid // tiles_per_row).to(tl.int64) * length
    t = (pid % tiles_per_row) * BLOCK_T + tl.arange(0, BLOCK_T)
    j = tl.arange(0, BLOCK_W)
    pos = (t + shift - window + 1)[:, None] + j[None, :]
    inside = (j[None, :] < window) & (pos >= 0) & (pos < length) & (t[:, None] < length)
    v = tl.load(x_ptr + base + pos, mask=inside, other=float("nan"))
    valid = inside & _finite(v)
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
    res = _interpolate(a, b, frac, lo, MODE)
    ok = (n >= min_periods) & (n > 0)
    tl.store(out_ptr + base + t, tl.where(ok, res, float("nan")), mask=t < length)


@triton.jit
def _quantile_select_kernel(
    x_ptr,
    q_ptr,
    out_ptr,
    scratch_ptr,
    length,
    window,
    shift,
    min_periods,
    tiles_per_row,
    num_tiles,
    MODE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SEG: tl.constexpr,
    IDX_BITS: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    i = tl.arange(0, SEG)
    j = tl.arange(0, BLOCK_T)
    scratch = scratch_ptr + pid.to(tl.int64) * SEG
    q = tl.load(q_ptr)
    for tile in range(pid, num_tiles, tl.num_programs(0)):
        base = (tile // tiles_per_row).to(tl.int64) * length
        t = (tile % tiles_per_row) * BLOCK_T + j
        start = (tile % tiles_per_row) * BLOCK_T + shift - window + 1  # segment's first position
        pos = start + i
        inside = (i < BLOCK_T + window - 1) & (pos >= 0) & (pos < length)
        v = tl.load(x_ptr + base + pos, mask=inside, other=float("nan"))
        valid = inside & _finite(v)
        # Order-preserving integer keys with the position in the low bits: float32 keeps
        # every bit; float64 gives up its lowest IDX_BITS (ties within ~1e-12 relative).
        if WIDE:
            bits = v.to(tl.int64, bitcast=True)
            bits = tl.where(bits < 0, bits ^ 0x7FFFFFFFFFFFFFFF, bits)
            key = ((bits >> IDX_BITS) << IDX_BITS) | i.to(tl.int64)
        else:
            bits = v.to(tl.int32, bitcast=True)
            bits = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
            key = (bits.to(tl.int64) << 32) | i.to(tl.int64)
        key = tl.sort(tl.where(valid, key, 0x7FFFFFFFFFFFFFFF))
        tl.store(scratch + i, key)
        total = tl.sum(valid.to(tl.int32))

        # Valid values per window, from two shifted loads (no moves between registers).
        first = tl.load(x_ptr + base + start + j, mask=(start + j >= 0) & (start + j < length))
        last_pos = start + j + window - 1
        last = tl.load(x_ptr + base + last_pos, mask=(last_pos >= 0) & (last_pos < length))
        a_in = ((start + j >= 0) & (start + j < length) & _finite(first)).to(tl.int32)
        b_in = ((last_pos >= 0) & (last_pos < length) & _finite(last)).to(tl.int32)
        head = tl.sum(tl.where(i < window - 1, valid, False).to(tl.int32))
        n = head + tl.cumsum(b_in, axis=0) - (tl.cumsum(a_in, axis=0) - a_in)
        rank = q * (n - 1).to(tl.float64)
        lo = tl.floor(rank).to(tl.int32)
        hi = tl.ceil(rank).to(tl.int32)
        tl.debug_barrier()

        seen = tl.zeros([BLOCK_T], dtype=tl.int32)
        a = tl.zeros([BLOCK_T], dtype=v.dtype)
        b = tl.zeros([BLOCK_T], dtype=v.dtype)
        for k in range(0, total):
            idx = (tl.load(scratch + k) & ((1 << IDX_BITS) - 1)).to(tl.int32)
            value = tl.load(x_ptr + base + start + idx)
            member = (idx >= j) & (idx < j + window)
            seen += member.to(tl.int32)
            a = tl.where(member & (seen == lo + 1), value, a)
            b = tl.where(member & (seen == hi + 1), value, b)
        tl.debug_barrier()

        frac = (rank - lo.to(tl.float64)).to(v.dtype)
        res = _interpolate(a, b, frac, lo, MODE)
        ok = (n >= min_periods) & (n > 0)
        tl.store(out_ptr + base + t, tl.where(ok, res, float("nan")), mask=t < length)


@triton.jit
def _ewm_kernel(
    x_ptr,
    y_ptr,
    params_ptr,
    out_ptr,
    length,
    min_periods,
    MODE: tl.constexpr,
    PAIR: tl.constexpr,
    BIAS: tl.constexpr,
    SQRT: tl.constexpr,
    ADJUST: tl.constexpr,
    IGNORE_NA: tl.constexpr,
    ACC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    base = tl.program_id(0).to(tl.int64) * length
    alpha = tl.load(params_ptr).to(ACC)
    f = tl.load(params_ptr + 1).to(ACC)  # 1 - alpha, as pandas computes it
    log_f = tl.load(params_ptr + 2).to(ACC)
    i = tl.arange(0, BLOCK)
    zero = tl.sum(tl.zeros([BLOCK], dtype=ACC))
    # State carried from one chunk to the next.
    before = zero  # adjust=True: total weight before the newest value
    last_obs = zero.to(tl.int32) - 1  # adjust=False: position of the latest observation
    nobs = zero
    found = zero.to(tl.int32)
    anchor_x = zero
    anchor_y = zero
    mean_x = zero
    mean_y = zero
    cxy = zero
    cxx = zero
    cyy = zero
    q = zero
    for t0 in range(0, length, BLOCK):
        t = t0 + i
        inside = t < length
        prev_in = inside & (t >= 1)
        x = tl.load(x_ptr + base + t, mask=inside, other=float("nan"))
        xp = tl.load(x_ptr + base + t - 1, mask=prev_in, other=float("nan"))
        obs = inside & _finite(x)
        obs_prev = prev_in & _finite(xp)
        if PAIR:
            y = tl.load(y_ptr + base + t, mask=inside, other=float("nan"))
            yp = tl.load(y_ptr + base + t - 1, mask=prev_in, other=float("nan"))
            obs = obs & _finite(y)
            obs_prev = obs_prev & _finite(yp)
        o = obs.to(ACC)

        # The share s of the newest value and the share kept of the old mean, computed
        # directly (1 - s would lose all precision when s is close to 1).
        if ADJUST:
            if IGNORE_NA:
                decay = tl.where(obs, f, 1.0)
            else:
                decay = tl.zeros([BLOCK], dtype=ACC) + f
            # before_t = decay_t * (before_{t-1} + obs_{t-1})
            ca, cb = tl.associative_scan((decay, decay * obs_prev.to(ACC)), 0, _affine1)
            u = ca * before + cb
            before = tl.sum(tl.where(i == BLOCK - 1, u, 0.0))
            s = tl.where(obs, 1.0 / (u + 1.0), 0.0)
            kept = tl.where(obs, u / (u + 1.0), 1.0)
        else:
            latest = tl.associative_scan(tl.where(obs_prev, t - 1, -1), 0, _maximum)
            latest = tl.maximum(latest, last_obs)
            last_obs = tl.max(tl.where(obs, t, latest))
            if IGNORE_NA:
                gap = tl.zeros([BLOCK], dtype=ACC) + 1.0
            else:
                gap = (t - latest).to(ACC)
            old = tl.where(latest < 0, 0.0, tl.exp(gap * log_f))
            s = tl.where(obs, alpha / (old + alpha), 0.0)
            kept = tl.where(obs, old / (old + alpha), 1.0)
        n = nobs + tl.cumsum(o, axis=0)
        nobs = tl.sum(tl.where(i == BLOCK - 1, n, 0.0))

        # Shift by the series' first observation, so constant stretches stay exactly 0.
        here = tl.min(tl.where(obs, i, BLOCK))
        anchor_x = tl.where(found > 0, anchor_x, tl.sum(tl.where(i == here, x, 0.0)).to(ACC))
        zx = tl.where(obs, x.to(ACC) - anchor_x, 0.0)
        if PAIR:
            anchor_y = tl.where(found > 0, anchor_y, tl.sum(tl.where(i == here, y, 0.0)).to(ACC))
            zy = tl.where(obs, y.to(ACC) - anchor_y, 0.0)
        found = found | (here < BLOCK).to(tl.int32)

        if MODE == 0:
            ca, cb = tl.associative_scan((kept, s * zx), 0, _affine1)
            m = ca * mean_x + cb
            mean_x = tl.sum(tl.where(i == BLOCK - 1, m, 0.0))
            res = m + anchor_x
            ok = n >= min_periods
        else:
            if PAIR:
                ca, cb, cc = tl.associative_scan((kept, s * zx, s * zy), 0, _affine2)
                mx = ca * mean_x + cb
                my = ca * mean_y + cc
            else:
                ca, cb = tl.associative_scan((kept, s * zx), 0, _affine1)
                mx = ca * mean_x + cb
                my = mx
            jump_x = tl.where(i == 0, mean_x, tl.gather(mx, tl.maximum(i - 1, 0), 0)) - mx
            mean_x = tl.sum(tl.where(i == BLOCK - 1, mx, 0.0))
            if PAIR:
                jump_y = tl.where(i == 0, mean_y, tl.gather(my, tl.maximum(i - 1, 0), 0)) - my
                mean_y = tl.sum(tl.where(i == BLOCK - 1, my, 0.0))
            else:
                jump_y = jump_x
                zy = zx
            dev_x = zx - mx
            dev_y = zy - my
            ok = n >= min_periods
            if MODE == 3:
                ca, cb, cc, cd = tl.associative_scan(
                    (
                        kept,
                        kept * jump_x * jump_y + s * dev_x * dev_y,
                        kept * jump_x * jump_x + s * dev_x * dev_x,
                        kept * jump_y * jump_y + s * dev_y * dev_y,
                    ),
                    0,
                    _affine3,
                )
                c = ca * cxy + cb
                vx = ca * cxx + cc
                vy = ca * cyy + cd
                cxy = tl.sum(tl.where(i == BLOCK - 1, c, 0.0))
                cxx = tl.sum(tl.where(i == BLOCK - 1, vx, 0.0))
                cyy = tl.sum(tl.where(i == BLOCK - 1, vy, 0.0))
                denom = vx * vy
                spread = denom > 0
                res = c / tl.sqrt(tl.where(spread, denom, 1.0))
                res = tl.minimum(tl.maximum(res, -1.0), 1.0)
                ok = ok & spread
            else:
                ca, cb = tl.associative_scan(
                    (kept, kept * jump_x * jump_y + s * dev_x * dev_y), 0, _affine1
                )
                res = ca * cxy + cb
                cxy = tl.sum(tl.where(i == BLOCK - 1, res, 0.0))
                if not BIAS:
                    # 1 - (sum of squared weights) / W^2, with no subtraction (see _ewm.py).
                    ca, cb = tl.associative_scan((kept * kept, 2.0 * kept * s), 0, _affine1)
                    qt = ca * q + cb
                    q = tl.sum(tl.where(i == BLOCK - 1, qt, 0.0))
                    ok = ok & (qt > 0)
                    res = res / tl.where(qt > 0, qt, 1.0)
                if SQRT:
                    res = tl.sqrt(res)
        tl.store(out_ptr + base + t, tl.where(ok, res, float("nan")), mask=inside)


_TL = {"float32": tl.float32, "float64": tl.float64}


def _rows(x: Tensor) -> Tensor:
    return x.reshape(-1, x.shape[-1]).contiguous()


def _rolling(
    x: Tensor,
    y: Tensor | None,
    window: int,
    shift: int,
    min_periods: int,
    ddof: int,
    stat: str,
    acc: str,
) -> Tensor:
    """Rolling ``stat`` over the last dim of ``x`` (and ``y`` for cov/corr)."""
    length, code = x.shape[-1], STATS[stat]
    flat = _rows(x)
    other = flat if y is None else _rows(y)
    out_dtype = x.dtype if stat in ("min", "max") else getattr(torch, acc)
    out = torch.empty(flat.shape, dtype=out_dtype, device=x.device)
    block_w = max(16, triton.next_power_of_2(window))
    group = GROUPS[code]
    if window <= DIRECT_WINDOW[group]:
        block_w = max(4, triton.next_power_of_2(window))
        tile, warps = _nearest(DIRECT_CONFIGS[group], block_w) or DIRECT_DEFAULT[group]
        block_t = max(1, tile // block_w)
        tiles = triton.cdiv(length, block_t)
        _rolling_direct_kernel[(flat.shape[0] * tiles,)](
            flat,
            other,
            out,
            length,
            window,
            shift,
            min_periods,
            ddof,
            tiles,
            STAT=code,
            PAIR=y is not None,
            ACC=_TL[acc],
            BLOCK_T=block_t,
            BLOCK_W=block_w,
            num_warps=warps,
        )
        return out.reshape(x.shape)
    tile, warps = _nearest(ROLLING_CONFIGS[group], block_w) or ROLLING_DEFAULT[group]
    rows = max(1, tile // block_w)
    tiles = triton.cdiv((length - 1 + shift) // window + 1, rows)
    _rolling_kernel[(flat.shape[0] * tiles,)](
        flat,
        other,
        out,
        length,
        window,
        shift,
        min_periods,
        ddof,
        tiles,
        STAT=code,
        PAIR=y is not None,
        ACC=_TL[acc],
        ROWS=rows,
        BLOCK_W=block_w,
        num_warps=warps,
    )
    return out.reshape(x.shape)


def _quantile(
    x: Tensor, window: int, shift: int, min_periods: int, q: float, interpolation: str
) -> Tensor:
    """Rolling quantile over the last dim of ``x``. Missing values are NaN or +-inf."""
    length = x.shape[-1]
    flat = _rows(x)
    out = torch.empty_like(flat)
    q_tensor = torch.tensor([q], dtype=torch.float64, device=x.device)
    mode = INTERPOLATIONS[interpolation]
    if window <= SORT_WINDOW:
        default = ("sort", 0, 4)
    else:
        default = ("select", min(512, max(128, triton.next_power_of_2(window))), 4)
    kind, block_t, warps = _nearest(QUANTILE_CONFIGS, triton.next_power_of_2(window)) or default
    if kind == "sort" and window > 1024:  # the sort kernel holds whole windows in registers
        kind, block_t = default[0], default[1]
    if kind == "sort":
        block_w = triton.next_power_of_2(window)
        block_t = max(1, min(64, 4096 // block_w))
        tiles = triton.cdiv(length, block_t)
        _quantile_sort_kernel[(flat.shape[0] * tiles,)](
            flat,
            q_tensor,
            out,
            length,
            window,
            shift,
            min_periods,
            tiles,
            MODE=mode,
            BLOCK_T=block_t,
            BLOCK_W=block_w,
            num_warps=warps,
        )
        return out.reshape(x.shape)
    seg = triton.next_power_of_2(block_t + window - 1)
    tiles = triton.cdiv(length, block_t)
    num_tiles = flat.shape[0] * tiles
    if x.is_cuda:  # pragma: no cover - needs a GPU
        programs = torch.cuda.get_device_properties(x.device).multi_processor_count * 8
    else:
        programs = 4
    programs = min(num_tiles, programs)
    scratch = torch.empty(programs * seg, dtype=torch.int64, device=x.device)
    _quantile_select_kernel[(programs,)](
        flat,
        q_tensor,
        out,
        scratch,
        length,
        window,
        shift,
        min_periods,
        tiles,
        num_tiles,
        MODE=mode,
        BLOCK_T=block_t,
        SEG=seg,
        IDX_BITS=max(1, (seg - 1).bit_length()),
        WIDE=x.dtype == torch.float64,
        num_warps=warps,
    )
    return out.reshape(x.shape)


def _ewm(
    x: Tensor,
    y: Tensor | None,
    alpha: float,
    min_periods: int,
    mode: str,
    bias: bool,
    sqrt: bool,
    adjust: bool,
    ignore_na: bool,
    acc: str,
) -> Tensor:
    """Exponentially weighted ``mode`` over the last dim of ``x`` (and ``y``)."""
    length = x.shape[-1]
    flat = _rows(x)
    other = flat if y is None else _rows(y)
    out = torch.empty(flat.shape, dtype=getattr(torch, acc), device=x.device)
    f = 1 - alpha
    params = torch.tensor(
        [alpha, f, math.log(f) if f > 0 else -math.inf], dtype=torch.float64, device=x.device
    )
    _ewm_kernel[(flat.shape[0],)](
        flat,
        other,
        params,
        out,
        length,
        min_periods,
        MODE=EWM_MODES[mode],
        PAIR=y is not None,
        BIAS=bias,
        SQRT=sqrt,
        ADJUST=adjust,
        IGNORE_NA=ignore_na,
        ACC=_TL[acc],
        BLOCK=EWM_CONFIG[0],
        num_warps=EWM_CONFIG[1],
    )
    return out.reshape(x.shape)


rolling: Callable[..., Tensor] = _rolling
quantile: Callable[[Tensor, int, int, int, float, str], Tensor] = _quantile
ewm: Callable[..., Tensor] = _ewm

if hasattr(torch.library, "custom_op"):
    # torch >= 2.4: as registered ops, torch.compile keeps the kernel calls inside the graph
    # (it only needs the outputs' shapes, from the fake implementations) instead of breaking.
    _rolling_op = torch.library.custom_op(
        "torchrolling::rolling",
        _rolling,
        mutates_args=(),
        schema="(Tensor x, Tensor? y, int window, int shift, int min_periods, int ddof,"
        " str stat, str acc) -> Tensor",
    )

    @_rolling_op.register_fake
    def _(x, y, window, shift, min_periods, ddof, stat, acc):
        dtype = x.dtype if stat in ("min", "max") else getattr(torch, acc)
        return torch.empty(x.shape, dtype=dtype, device=x.device)

    _quantile_op = torch.library.custom_op(
        "torchrolling::quantile",
        _quantile,
        mutates_args=(),
        schema="(Tensor x, int window, int shift, int min_periods, float q, str interpolation)"
        " -> Tensor",
    )

    @_quantile_op.register_fake
    def _(x, window, shift, min_periods, q, interpolation):
        return torch.empty_like(x)

    _ewm_op = torch.library.custom_op(
        "torchrolling::ewm",
        _ewm,
        mutates_args=(),
        schema="(Tensor x, Tensor? y, float alpha, int min_periods, str mode, bool bias,"
        " bool sqrt, bool adjust, bool ignore_na, str acc) -> Tensor",
    )

    @_ewm_op.register_fake
    def _(x, y, alpha, min_periods, mode, bias, sqrt, adjust, ignore_na, acc):
        return torch.empty(x.shape, dtype=getattr(torch, acc), device=x.device)

    rolling, quantile, ewm = _rolling_op, _quantile_op, _ewm_op
