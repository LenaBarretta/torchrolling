"""Exponentially weighted statistics with pandas semantics.

pandas updates the weighted mean one observation at a time:
``mean_t = (1 - s_t) * mean_{t-1} + s_t * x_t``, where ``s_t`` is the share of the newest
weight in the total. The shares depend only on where values are missing, never on the
values, so they are computed up front, and the mean, the weighted covariance and the bias
correction all become affine recurrences ``y_t = a_t * y_{t-1} + b_t``. Those are composed
in parallel (``_affine_scan``), which is stable because every ``a_t`` lies in [0, 1].

Values are shifted by each series' first observation before the scan, so a constant
stretch gives exactly zero variance, as in pandas.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from torchrolling._common import accumulator, as_float, as_other, check_int

# Steps composed directly (in log2 rounds) before moving one level up.
_SCAN_BLOCK = 64


def _compose(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Inclusive prefix composition of the maps y -> a * y + b along the last dim."""
    k = 1
    while k < b.shape[-1]:
        b = a * F.pad(b[..., :-k], (k, 0)) + b
        a = a * F.pad(a[..., :-k], (k, 0), value=1.0)
        k *= 2
    return a, b


def _affine_scan(a: Tensor, b: Tensor) -> Tensor:
    """``y_t = a_t * y_{t-1} + b_t`` along the last dim, starting from ``y_{-1} = 0``.

    ``b`` may have extra leading dims (several series sharing the same ``a``). Blocks of
    ``_SCAN_BLOCK`` steps are composed directly; their end values are chained by the same
    scan one level up, so the work is O(n log _SCAN_BLOCK).
    """
    length = b.shape[-1]
    if length <= _SCAN_BLOCK:
        return _compose(a, b)[1]
    pad = -length % _SCAN_BLOCK
    blocks = (length + pad) // _SCAN_BLOCK
    a = F.pad(a, (0, pad), value=1.0).reshape(*a.shape[:-1], blocks, _SCAN_BLOCK)
    b = F.pad(b, (0, pad)).reshape(*b.shape[:-1], blocks, _SCAN_BLOCK)
    a, b = _compose(a, b)
    ends = _affine_scan(a[..., -1], b[..., -1])
    starts = F.pad(ends[..., :-1], (1, 0))
    return (a * starts.unsqueeze(-1) + b).flatten(-2)[..., :length]


def _center_of_mass(
    com: float | None, span: float | None, halflife: float | None, alpha: float | None
) -> float:
    given = {"com": com, "span": span, "halflife": halflife, "alpha": alpha}
    given = {k: v for k, v in given.items() if v is not None}
    if len(given) != 1:
        raise ValueError("pass exactly one of com, span, halflife or alpha")
    ((name, value),) = given.items()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if name == "com":
        if value < 0:
            raise ValueError(f"com must be >= 0, got {value!r}")
        return float(value)
    if name == "span":
        if value < 1:
            raise ValueError(f"span must be >= 1, got {value!r}")
        return (value - 1) / 2
    if name == "halflife":
        if value <= 0:
            raise ValueError(f"halflife must be > 0, got {value!r}")
        return 1 / (1 - math.exp(math.log(0.5) / value)) - 1
    if not 0 < value <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {value!r}")
    return (1 - value) / value


def ewm(
    x: Tensor,
    com: float | None = None,
    span: float | None = None,
    halflife: float | None = None,
    alpha: float | None = None,
    *,
    min_periods: int = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    dim: int = -1,
    acc_dtype: torch.dtype | None = None,
) -> Ewm:
    """Exponentially weighted window over ``dim`` of ``x``, like ``pandas.Series.ewm``.

    Give exactly one of ``com``, ``span``, ``halflife`` or ``alpha``. Missing values (NaN
    and +-inf) are skipped; ``adjust`` and ``ignore_na`` mean what they mean in pandas. A
    result is NaN until ``min_periods`` (at least 1) values have been seen.
    """
    return Ewm(
        x,
        com,
        span,
        halflife,
        alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
        dim=dim,
        acc_dtype=acc_dtype,
    )


class Ewm:
    """Exponentially weighted statistics. Create it with :func:`ewm`."""

    def __init__(
        self,
        x: Tensor,
        com: float | None = None,
        span: float | None = None,
        halflife: float | None = None,
        alpha: float | None = None,
        *,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        dim: int = -1,
        acc_dtype: torch.dtype | None = None,
    ) -> None:
        x = as_float(x)
        self._alpha = 1 / (1 + _center_of_mass(com, span, halflife, alpha))
        self._min_periods = max(check_int("min_periods", min_periods, 0), 1)
        self._adjust = bool(adjust)
        self._ignore_na = bool(ignore_na)
        self._dtype = x.dtype
        self._dim = dim
        self._shape = x.shape
        self._acc = accumulator(x.device, acc_dtype)
        self._orig = x.movedim(dim, -1)

    def mean(self) -> Tensor:
        """Exponentially weighted mean."""
        x, obs = self._prepare(self._orig)
        if x.numel() == 0:
            return self._finish(x, obs)
        s, kept, nobs = self._shares(obs)
        anchor = x.gather(-1, obs.to(torch.uint8).argmax(-1, keepdim=True))
        mean = _affine_scan(kept, s * (x - anchor)) + anchor
        return self._finish(mean, nobs >= self._min_periods)

    def var(self, bias: bool = False) -> Tensor:
        """Exponentially weighted variance (bias-corrected unless ``bias=True``, like pandas)."""
        return self._cov(None, bias=bias)

    def std(self, bias: bool = False) -> Tensor:
        """Exponentially weighted standard deviation."""
        return self.var(bias).sqrt()

    def cov(self, other: Tensor, bias: bool = False) -> Tensor:
        """Exponentially weighted covariance with ``other`` (same shape as ``x``)."""
        return self._cov(as_other(other, self._shape).movedim(self._dim, -1), bias=bias)

    def corr(self, other: Tensor) -> Tensor:
        """Exponentially weighted correlation with ``other`` (same shape as ``x``).

        NaN while either series has been constant so far.
        """
        y = as_other(other, self._shape).movedim(self._dim, -1)
        dtype = torch.promote_types(self._dtype, y.dtype)
        x, obs = self._prepare(self._orig, y)
        if x.numel() == 0:
            return self._finish(x, obs, dtype)
        y = torch.where(obs, y.to(self._acc), 0.0)
        s, kept, nobs = self._shares(obs)
        cxy, cxx, cyy = self._comoments(s, kept, [x, y], [(0, 1), (0, 0), (1, 1)])
        denom = cxx * cyy
        spread = denom > 0
        corr = (cxy / torch.where(spread, denom, 1.0).sqrt()).clamp(-1, 1)
        return self._finish(corr, (nobs >= self._min_periods) & spread, dtype)

    def _cov(self, y: Tensor | None, bias: bool) -> Tensor:
        """Covariance of ``x`` with ``y``, or variance of ``x`` when ``y`` is None."""
        dtype = self._dtype if y is None else torch.promote_types(self._dtype, y.dtype)
        x, obs = self._prepare(self._orig, y)
        if x.numel() == 0:
            return self._finish(x, obs, dtype)
        s, kept, nobs = self._shares(obs)
        if y is None:
            (cov,) = self._comoments(s, kept, [x], [(0, 0)])
        else:
            y = torch.where(obs, y.to(self._acc), 0.0)
            (cov,) = self._comoments(s, kept, [x, y], [(0, 1)])
        ok = nobs >= self._min_periods
        if not bias:
            # pandas multiplies by W^2 / (W^2 - sum of squared weights). With r the ratio
            # (sum of squared weights) / W^2, r_t = kept^2 r_{t-1} + s^2, and since
            # kept + s = 1, q = 1 - r follows q_t = kept^2 q_{t-1} + 2 kept s: no cancellation.
            q = _affine_scan(kept * kept, 2 * kept * s)
            ok = ok & (q > 0)
            cov = cov / torch.where(q > 0, q, 1.0)
        return self._finish(cov, ok, dtype)

    def _comoments(
        self, s: Tensor, kept: Tensor, series: list[Tensor], pairs: list[tuple[int, int]]
    ) -> list[Tensor]:
        """Weighted (biased) covariances of the given pairs of ``series``.

        The series are zero where unobserved. pandas' update is
        ``c_t = kept * (c_{t-1} + dmean_x * dmean_y) + s * (x - mean_x) * (y - mean_y)``.
        """
        v = torch.stack(series)
        first = (s > 0).to(torch.uint8).argmax(-1, keepdim=True)
        v = v - v.gather(-1, first.expand(*v.shape[:-1], 1))
        means = _affine_scan(kept, s * v)
        jump = F.pad(means[..., :-1], (1, 0)) - means
        dev = v - means
        b = torch.stack([kept * jump[i] * jump[j] + s * dev[i] * dev[j] for i, j in pairs])
        return list(_affine_scan(kept, b))

    def _prepare(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``x`` in the accumulator dtype, zero where it (or ``y``) is missing; and the mask."""
        obs = torch.isfinite(x) if y is None else torch.isfinite(x) & torch.isfinite(y)
        return torch.where(obs, x.to(self._acc), 0.0), obs

    def _shares(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Per step, the weights of the new value (s) and of the old mean (kept), and nobs.

        Where nothing is observed s = 0 and kept = 1. Both are computed directly rather than
        one as 1 minus the other, which would lose all precision when alpha is close to 1.
        """
        f = 1 - self._alpha  # as pandas computes it, so alpha == 1 gives exactly 0
        o = obs.to(self._acc)
        if self._adjust:
            # The total weight decays every step (only at observations with ignore_na)
            # and gains 1 per observation.
            decay = o * f + (1 - o) if self._ignore_na else torch.full_like(o, f)
            total = _affine_scan(decay, o)
            before = decay * F.pad(total[..., :-1], (1, 0))
            s = o / total.clamp(min=1)
            kept = torch.where(obs, before / total.clamp(min=1), 1.0)
        else:
            # The previous total is normalised to 1, then decays over the steps since the
            # previous observation; the new value gets weight alpha.
            steps = torch.arange(obs.shape[-1], device=obs.device)
            last = torch.where(obs, steps, -1).cummax(-1).values
            previous = F.pad(last[..., :-1], (1, 0), value=-1)
            gap = torch.ones_like(o) if self._ignore_na else (steps - previous).to(self._acc)
            old = torch.where(previous < 0, 0.0, torch.full_like(o, f).pow(gap))
            s = o * self._alpha / (old + self._alpha)
            kept = torch.where(obs, old / (old + self._alpha), 1.0)
        return s, kept, o.cumsum(-1)

    def _finish(self, values: Tensor, ok: Tensor, dtype: torch.dtype | None = None) -> Tensor:
        out = torch.where(ok, values, math.nan).to(dtype or self._dtype)
        return out.movedim(-1, self._dim)
