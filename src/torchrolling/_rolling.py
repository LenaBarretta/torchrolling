"""Rolling window statistics with pandas semantics.

Every statistic is built from one trick (van Herk / Gil-Werman): cut the padded series into
blocks of exactly ``window`` elements and scan each block forwards (prefix) and backwards
(suffix). Any window then covers at most the tail of one block and the head of the next, so
its value is ``combine(suffix[start], prefix[end])``. That gives O(1) work per element for
any window size, uses O(n) memory, runs on any device, and only sums ``2 * window`` numbers
at a time, which keeps float error small even on very long series.

Moments (variance, skew, kurtosis, covariance) need care, because plain power sums cancel
badly. Each side of a window is shifted by a value that lies inside that same side: a head
always starts at its block's start, so it contains the block's first valid value, and a
tail always contains its block's last one. Power sums of the shifted values are as well
conditioned as the window itself, a constant window gives exact zeros, and the two sides are
merged with the pairwise formulas of Chan et al. and Pébay.

Quantiles cannot be split that way: they sort every window, over chunks of windows so memory
stays bounded.

On CUDA, when no gradients are needed, all of this runs as fused Triton kernels instead
(see ``_triton.py``), with the same algorithms and the same results.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from torchrolling import _common
from torchrolling._common import (
    accumulator,
    as_float,
    as_other,
    check_int,
    dtype_name,
    kernel_dtype,
)

try:
    from torchrolling import _triton
except ImportError:  # pragma: no cover - Triton is only installed on Linux
    _triton = None  # type: ignore[assignment]

Scan = Callable[[Tensor], Tensor]
Combine = Callable[[Tensor, Tensor], Tensor]


INTERPOLATIONS = ("linear", "lower", "higher", "midpoint", "nearest")
# Upper bound on window elements sorted at once by the torch quantile code.
_CHUNK = 1 << 22
# pandas gives NaN skew and kurtosis when the (biased) variance is at most this.
_TINY_VARIANCE = 1e-14


def _cumsum(t: Tensor) -> Tensor:
    return t.cumsum(-1)


def _cummax(t: Tensor) -> Tensor:
    return t.cummax(-1).values


def _cummin(t: Tensor) -> Tensor:
    return t.cummin(-1).values


def _select(windows: Tensor, counts: Tensor, q: float, interpolation: str) -> Tensor:
    """The ``q``-quantile of each window (NaN for missing), with pandas' interpolation."""
    s = windows.sort(-1).values  # NaN sorts last
    exact = torch.float32 if s.device.type == "mps" else torch.float64  # MPS has no float64
    rank = q * (counts.to(exact) - 1)  # pandas computes the rank in float64 too
    lo, hi = rank.floor(), rank.ceil()
    a = s.gather(-1, lo.clamp(min=0).long().unsqueeze(-1)).squeeze(-1)
    b = s.gather(-1, hi.clamp(min=0).long().unsqueeze(-1)).squeeze(-1)
    frac = (rank - lo).to(s.dtype)
    if interpolation == "linear":
        return a + (b - a) * frac
    if interpolation == "lower":
        return a
    if interpolation == "higher":
        return b
    if interpolation == "midpoint":
        return (a + b) / 2
    # nearest, ties to even like pandas
    return torch.where((frac > 0.5) | ((frac == 0.5) & (lo % 2 == 1)), b, a)


def _central(n: Tensor, shift: Tensor, sums: Sequence[Tensor]) -> list[Tensor]:
    """Mean and central moments M2, M3, ... of one side, from sums of (value - shift)^k."""
    d = sums[0] / n.clamp(min=1)
    out = [shift + d]
    if len(sums) > 1:
        out.append((sums[1] - sums[0] * d).clamp(min=0))
    if len(sums) > 2:
        out.append(sums[2] - 3 * d * sums[1] + 2 * d * d * sums[0])
    if len(sums) > 3:
        out.append(sums[3] - 4 * d * sums[2] + 6 * d * d * sums[1] - 3 * d * d * d * sums[0])
    return out


def _merge(na: Tensor, a: Sequence[Tensor], nb: Tensor, b: Sequence[Tensor]) -> list[Tensor]:
    """Central moments [M2, M3, M4][:k] of the union of two sides (Chan et al., Pébay)."""
    n = na + nb
    delta = torch.where((na > 0) & (nb > 0), b[0] - a[0], 0.0)
    dn = delta / n.clamp(min=1)
    out = [a[1] + b[1] + delta * dn * na * nb]
    if len(a) > 2:
        out.append(
            a[2] + b[2] + delta * dn * dn * na * nb * (na - nb) + 3 * dn * (na * b[1] - nb * a[1])
        )
    if len(a) > 3:
        out.append(
            a[3]
            + b[3]
            + delta * dn * dn * dn * na * nb * (na * na - na * nb + nb * nb)
            + 6 * dn * dn * (na * na * b[1] + nb * nb * a[1])
            + 4 * dn * (na * b[2] - nb * a[2])
        )
    return out


def rolling(
    x: Tensor,
    window: int,
    *,
    min_periods: int | None = None,
    center: bool = False,
    dim: int = -1,
    acc_dtype: torch.dtype | None = None,
) -> Rolling:
    """Rolling window over ``dim`` of ``x``, like ``pandas.Series.rolling``.

    Missing values (NaN and +-inf, as in pandas) are skipped. A result is NaN when its window
    holds fewer than ``min_periods`` valid values (default: ``window``). Sums and moments are
    accumulated in ``acc_dtype`` (default float64, float32 on MPS); results keep the dtype
    of ``x``.
    """
    return Rolling(x, window, min_periods=min_periods, center=center, dim=dim, acc_dtype=acc_dtype)


class Rolling:
    """Rolling window statistics. Create it with :func:`rolling`.

    Intermediate results (counts, moments) are cached, so asking one object for several
    statistics is cheaper than building a new one for each.
    """

    def __init__(
        self,
        x: Tensor,
        window: int,
        *,
        min_periods: int | None = None,
        center: bool = False,
        dim: int = -1,
        acc_dtype: torch.dtype | None = None,
    ) -> None:
        x = as_float(x)
        self._window = check_int("window", window, 1)
        if min_periods is None:
            min_periods = window
        self._min_periods = check_int("min_periods", min_periods, 0)
        if min_periods > window:
            raise ValueError(f"min_periods ({min_periods}) must be <= window ({window})")
        self._dtype = x.dtype
        self._dim = dim
        self._shape = x.shape
        # Windows end at their own index; with center=True they end (window - 1) // 2 later.
        self._shift = (window - 1) // 2 if center else 0
        self._acc = accumulator(x.dtype, acc_dtype)
        x = x.movedim(dim, -1)
        self._orig = x
        self._valid = torch.isfinite(x)
        self._x = torch.where(self._valid, x.to(self._acc), 0.0)
        self._n: Tensor | None = None
        self._moments_cache: tuple[int, list[Tensor]] = (0, [])

    def count(self) -> Tensor:
        """Number of non-NaN values in each window.

        Like pandas' ``count``, this counts +-inf, and gives NaN only where the window has
        fewer than ``min_periods`` positions inside the series (at its ends).
        """
        if (out := self._kernel("count")) is not None:  # pragma: no cover - needs Triton
            return out
        notnan = (~torch.isnan(self._orig)).to(self._x.dtype)
        total = self._reduce(notnan, 0.0, _cumsum, torch.add)
        end = torch.arange(total.shape[-1], device=total.device) + self._shift
        inside = end.clamp(max=total.shape[-1] - 1) - (end - self._window + 1).clamp(min=0) + 1
        return self._finish(total, inside >= self._min_periods)

    def sum(self) -> Tensor:
        """Sum of valid values in each window."""
        if (out := self._kernel("sum")) is not None:  # pragma: no cover - needs Triton
            return out
        total = self._reduce(self._x, 0.0, _cumsum, torch.add)
        return self._finish(total, self._count() >= self._min_periods)

    def mean(self) -> Tensor:
        """Mean of valid values in each window."""
        if (out := self._kernel("mean")) is not None:  # pragma: no cover - needs Triton
            return out
        n = self._count()
        total = self._reduce(self._x, 0.0, _cumsum, torch.add)
        return self._finish(total / n.clamp(min=1), self._enough(n))

    def var(self, ddof: int = 1) -> Tensor:
        """Variance of valid values in each window (``ddof=1`` by default, like pandas)."""
        ddof = check_int("ddof", ddof, 0)
        if (out := self._kernel("var", ddof=ddof)) is not None:  # pragma: no cover - needs Triton
            return out
        n, m2 = self._moments(2)[:2]
        return self._finish(m2 / (n - ddof).clamp(min=1), self._enough(n) & (n > ddof))

    def std(self, ddof: int = 1) -> Tensor:
        """Standard deviation of valid values in each window."""
        ddof = check_int("ddof", ddof, 0)
        if (out := self._kernel("std", ddof=ddof)) is not None:  # pragma: no cover - needs Triton
            return out
        return self.var(ddof).sqrt()

    def skew(self) -> Tensor:
        """Unbiased skewness of valid values in each window, like pandas.

        NaN for fewer than 3 values or a (biased) variance at most 1e-14; 0 for a window of
        identical values.
        """
        if (out := self._kernel("skew")) is not None:  # pragma: no cover - needs Triton
            return out
        n, m2, m3, constant = self._moments(3)
        b = m2 / n.clamp(min=1)
        flat = b <= _TINY_VARIANCE
        b = torch.where(flat, 1.0, b)
        skew = (n * (n - 1)).clamp(min=0).sqrt() * (m3 / n.clamp(min=1))
        skew = skew / ((n - 2).clamp(min=1) * b * b.sqrt())
        skew = torch.where(constant, 0.0, torch.where(flat, math.nan, skew))
        return self._finish(skew, self._enough(n) & (n >= 3))

    def kurt(self) -> Tensor:
        """Unbiased excess kurtosis (Fisher) of valid values in each window, like pandas.

        NaN for fewer than 4 values or a (biased) variance at most 1e-14; -3 for a window of
        identical values.
        """
        if (out := self._kernel("kurt")) is not None:  # pragma: no cover - needs Triton
            return out
        n, m2, _, m4, constant = self._moments(4)
        b = m2 / n.clamp(min=1)
        flat = b <= _TINY_VARIANCE
        b = torch.where(flat, 1.0, b)
        d = m4 / n.clamp(min=1)
        k = (n * n - 1) * d / (b * b) - 3 * (n - 1) * (n - 1)
        k = k / ((n - 2) * (n - 3)).clamp(min=1)
        k = torch.where(constant, -3.0, torch.where(flat, math.nan, k))
        return self._finish(k, self._enough(n) & (n >= 4))

    def cov(self, other: Tensor, ddof: int = 1) -> Tensor:
        """Covariance with ``other`` (same shape as ``x``) in each window.

        As in pandas, only positions where both series are valid are used.
        """
        ddof = check_int("ddof", ddof, 0)
        y = self._other(other)
        if (out := self._kernel("cov", y, ddof)) is not None:  # pragma: no cover - needs Triton
            return out
        n, c, _, _, dtype = self._comoments(y, squares=False)
        return self._finish(c / (n - ddof).clamp(min=1), self._enough(n) & (n > ddof), dtype)

    def corr(self, other: Tensor, ddof: int = 1) -> Tensor:
        """Pearson correlation with ``other`` (same shape as ``x``) in each window.

        Only positions where both series are valid are used. NaN where either series is
        constant in the window. ``ddof`` cancels out; it only makes windows with at most
        ``ddof`` values NaN, as in pandas.
        """
        ddof = check_int("ddof", ddof, 0)
        y = self._other(other)
        if (out := self._kernel("corr", y, ddof)) is not None:  # pragma: no cover - needs Triton
            return out
        n, c, m2x, m2y, dtype = self._comoments(y, squares=True)
        denom = m2x * m2y
        spread = denom > 0
        corr = (c / torch.where(spread, denom, 1.0).sqrt()).clamp(-1, 1)
        return self._finish(corr, self._enough(n) & (n > ddof) & spread, dtype)

    def min(self) -> Tensor:
        """Minimum of valid values in each window."""
        if (out := self._kernel("min")) is not None:  # pragma: no cover - needs Triton
            return out
        values = torch.where(self._valid, self._orig, math.inf)
        lowest = self._reduce(values, math.inf, _cummin, torch.minimum)
        return self._finish(lowest, self._enough(self._count()))

    def max(self) -> Tensor:
        """Maximum of valid values in each window."""
        if (out := self._kernel("max")) is not None:  # pragma: no cover - needs Triton
            return out
        values = torch.where(self._valid, self._orig, -math.inf)
        highest = self._reduce(values, -math.inf, _cummax, torch.maximum)
        return self._finish(highest, self._enough(self._count()))

    def median(self) -> Tensor:
        """Median of valid values in each window."""
        return self.quantile(0.5)

    def quantile(self, q: float, interpolation: str = "linear") -> Tensor:
        """``q``-quantile of valid values in each window, with pandas' interpolation modes."""
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not 0 <= q <= 1:
            raise ValueError(f"q must be a number between 0 and 1, got {q!r}")
        if interpolation not in INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {INTERPOLATIONS}, got {interpolation!r}"
            )
        # Sorting needs no extra precision, but float16 is too coarse for the interpolation.
        x = kernel_dtype(self._orig)
        window = self._window
        if _common.use_triton(x, window=window, limit="QUANTILE_WINDOW"):  # pragma: no cover
            out = _triton.quantile(
                x, window, self._shift, self._min_periods, float(q), interpolation
            )
            return out.to(self._dtype).movedim(-1, self._dim)
        return self._quantile_torch(x, float(q), interpolation)

    def _quantile_torch(self, x: Tensor, q: float, interpolation: str) -> Tensor:
        w, length = self._window, x.shape[-1]
        values = torch.where(self._valid, x, math.nan)
        if values.numel() == 0:
            return self._finish(values, self._valid)
        flat = values.reshape(-1, length)
        counts = self._count().reshape(-1, length)
        padded = F.pad(flat, (w - 1, self._shift), value=math.nan)
        windows = padded.unfold(-1, w, 1)[:, self._shift :, :]
        # Chunks of whole rows when they fit, otherwise of time steps within rows.
        steps = min(length, max(1, _CHUNK // w))
        rows = max(1, _CHUNK // (steps * w))
        out = torch.cat(
            [
                torch.cat(
                    [
                        _select(
                            windows[r : r + rows, t : t + steps],
                            counts[r : r + rows, t : t + steps],
                            q,
                            interpolation,
                        )
                        for t in range(0, length, steps)
                    ],
                    dim=-1,
                )
                for r in range(0, flat.shape[0], rows)
            ]
        )
        return self._finish(out.reshape(values.shape), self._enough(self._count()))

    def _kernel(self, stat: str, other: Tensor | None = None, ddof: int = 0) -> Tensor | None:
        """``stat`` from the fused Triton kernel, or None where that does not apply."""
        tensors = [self._orig] if other is None else [self._orig, other]
        if self._acc not in (torch.float32, torch.float64) or not _common.use_triton(
            *tensors, window=self._window, limit="MAX_WINDOW"
        ):
            return None
        out = _triton.rolling(  # pragma: no cover - needs Triton
            kernel_dtype(self._orig),
            None if other is None else kernel_dtype(other),
            self._window,
            self._shift,
            self._min_periods,
            ddof,
            stat,
            dtype_name(self._acc),
        )
        dtype = (
            self._dtype if other is None else torch.promote_types(self._dtype, other.dtype)
        )  # pragma: no cover
        return out.to(dtype).movedim(-1, self._dim)  # pragma: no cover - needs Triton

    def _other(self, other: Tensor) -> Tensor:
        """The second series of cov/corr, with its time dimension last."""
        return as_other(other, self._shape).movedim(self._dim, -1)

    def _count(self) -> Tensor:
        if self._n is None:
            self._n = self._reduce(self._valid.to(self._x.dtype), 0.0, _cumsum, torch.add)
        return self._n

    def _enough(self, n: Tensor) -> Tensor:
        # Statistics of an empty window are NaN even when min_periods is 0.
        return (n >= self._min_periods) & (n > 0)

    def _finish(self, values: Tensor, ok: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        out = torch.where(ok, values, math.nan).to(dtype or self._dtype)
        return out.movedim(-1, self._dim)

    def _moments(self, order: int) -> list[Tensor]:
        """[count, M2, ..., M_order] of every window; for order >= 3 also a constant flag."""
        cached_order, cached = self._moments_cache
        if cached_order < order:
            cached = self._compute_moments(order)
            self._moments_cache = (order, cached)
        return cached[:order] + cached[-1:] if order >= 3 else cached[:order]

    def _compute_moments(self, order: int) -> list[Tensor]:
        valid = self._valid.to(self._x.dtype)
        vb, tail, tail_shift, head, head_shift = self._anchored(self._x, valid)
        n_a, n_b = self._sides(vb, vb)
        shift_a, shift_b = self._sides(tail_shift, head_shift, scan=False)
        sums_a, sums_b = zip(
            *(self._sides(tail**k, head**k) for k in range(1, order + 1)), strict=True
        )
        moments = _merge(n_a, _central(n_a, shift_a, sums_a), n_b, _central(n_b, shift_b, sums_b))
        out = [n_a + n_b, *moments]
        if order >= 3:
            # Shifted values are exactly zero only where they equal their side's anchor.
            nz_a, nz_b = self._sides((tail != 0).to(vb.dtype), (head != 0).to(vb.dtype))
            same = (n_a == 0) | (n_b == 0) | (shift_a == shift_b)
            out.append((nz_a + nz_b == 0) & same)
        return out

    def _comoments(
        self, other: Tensor, squares: bool
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, torch.dtype]:
        """Count, co-moment C and (if ``squares``) both M2 over the jointly valid values."""
        y = other
        dtype = torch.promote_types(self._dtype, y.dtype)
        valid = self._valid & torch.isfinite(y)
        vf = valid.to(self._x.dtype)
        vb, tx, tx_shift, hx, hx_shift = self._anchored(torch.where(valid, self._x, 0.0), vf)
        _, ty, ty_shift, hy, hy_shift = self._anchored(torch.where(valid, y.to(vf.dtype), 0.0), vf)
        n_a, n_b = self._sides(vb, vb)
        n = n_a + n_b
        sx, sy = self._sides(tx, hx), self._sides(ty, hy)
        mx = self._side_means((n_a, n_b), sx, self._sides(tx_shift, hx_shift, scan=False))
        my = self._side_means((n_a, n_b), sy, self._sides(ty_shift, hy_shift, scan=False))
        both = (n_a > 0) & (n_b > 0)
        dx = torch.where(both, mx[1] - mx[0], 0.0)
        dy = torch.where(both, my[1] - my[0], 0.0)
        weight = n_a * n_b / n.clamp(min=1)

        def comoment(
            products: tuple[Tensor, Tensor], p: tuple[Tensor, Tensor], q: tuple[Tensor, Tensor]
        ) -> Tensor:
            side_a = products[0] - p[0] * q[0] / n_a.clamp(min=1)
            side_b = products[1] - p[1] * q[1] / n_b.clamp(min=1)
            return side_a + side_b

        c = comoment(self._sides(tx * ty, hx * hy), sx, sy) + dx * dy * weight
        if not squares:
            return n, c, c, c, dtype
        m2x = comoment(self._sides(tx * tx, hx * hx), sx, sx) + dx * dx * weight
        m2y = comoment(self._sides(ty * ty, hy * hy), sy, sy) + dy * dy * weight
        return n, c, m2x.clamp(min=0), m2y.clamp(min=0), dtype

    def _side_means(
        self, n: tuple[Tensor, Tensor], sums: tuple[Tensor, Tensor], shifts: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        return (
            shifts[0] + sums[0] / n[0].clamp(min=1),
            shifts[1] + sums[1] / n[1].clamp(min=1),
        )

    def _anchored(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Blocks of ``valid``, and blocks of ``x`` minus an anchor from the same window side.

        Returns (valid, tail values, tail anchor, head values, head anchor). Heads are shifted
        by their block's first valid value and tails by its last, so the anchor is always one
        of the values it is subtracted from; missing values become exactly zero.
        """
        xb, vb = self._blocks(x, 0.0), self._blocks(valid, 0.0)
        first = xb.gather(-1, vb.argmax(-1, keepdim=True))
        last = xb.gather(-1, self._window - 1 - vb.flip(-1).argmax(-1, keepdim=True))
        return vb, (xb - last) * vb, last.expand_as(xb), (xb - first) * vb, first.expand_as(xb)

    def _sides(self, tail: Tensor, head: Tensor, scan: bool = True) -> tuple[Tensor, Tensor]:
        """Per window: ``tail`` summed over its part of the earlier block, ``head`` over the
        later one (or, with ``scan=False``, just read at the window's start and end)."""
        length = self._x.shape[-1]
        if not scan:
            return self._halves(tail, head, length)
        a, b = self._halves(_cumsum(tail.flip(-1)).flip(-1), _cumsum(head), length)
        # A window that is exactly one block has no separate tail.
        return torch.where(self._aligned(length, a.device), 0.0, a), b

    def _reduce(self, values: Tensor, identity: float, scan: Scan, combine: Combine) -> Tensor:
        """Apply an associative reduction to every window of ``values`` (last dim)."""
        length = values.shape[-1]
        blocks = self._blocks(values, identity)
        suffix = scan(blocks.flip(-1)).flip(-1)
        tail, head = self._halves(suffix, scan(blocks), length)
        return torch.where(self._aligned(length, values.device), head, combine(tail, head))

    def _blocks(self, values: Tensor, identity: float) -> Tensor:
        """Pad ``values`` (last dim) and cut it into blocks of exactly ``window`` elements."""
        w, length = self._window, values.shape[-1]
        left = w - 1
        right = self._shift + (-(left + length + self._shift)) % w
        padded = F.pad(values, (left, right), value=identity)
        return padded.reshape(*padded.shape[:-1], padded.shape[-1] // w, w)

    def _halves(self, suffix: Tensor, prefix: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """Per window: the backward scan at its start (tail) and forward scan at its end (head)."""
        flat = (*prefix.shape[:-2], prefix.shape[-2] * prefix.shape[-1])
        start = self._shift
        end = start + self._window - 1
        return (
            suffix.reshape(flat)[..., start : start + length],
            prefix.reshape(flat)[..., end : end + length],
        )

    def _aligned(self, length: int, device: torch.device) -> Tensor:
        # A window that starts on a block boundary is exactly one block: its head alone.
        return torch.arange(self._shift, self._shift + length, device=device) % self._window == 0
