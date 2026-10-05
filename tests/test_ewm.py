"""Exponentially weighted statistics, checked against pandas."""

from __future__ import annotations

import math
from typing import Any

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import torchrolling
from torchrolling import Ewm, _ewm

values = st.one_of(
    st.integers(-20, 20).map(float),
    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    st.sampled_from([math.nan, math.inf, -math.inf]),
)
# With alpha within ~1e-2 of 1, pandas' own bias correction cancels (W^2 - sum of squared
# weights, both close to 1) and drifts from the exact answer; torchrolling does not. Exactly
# alpha = 1 is still covered.
decays = st.one_of(
    st.fixed_dictionaries({"com": st.just(0.0) | st.floats(0.01, 20)}),
    st.fixed_dictionaries({"span": st.just(1.0) | st.floats(1.02, 40)}),
    st.fixed_dictionaries({"halflife": st.floats(0.2, 20)}),
    st.fixed_dictionaries({"alpha": st.floats(0.01, 0.99) | st.just(1.0)}),
)


@st.composite
def cases(draw: st.DrawFn) -> dict[str, Any]:
    return {
        **draw(decays),
        "min_periods": draw(st.integers(0, 5)),
        "adjust": draw(st.booleans()),
        "ignore_na": draw(st.booleans()),
    }


def check(got: torch.Tensor, want: pd.Series | np.ndarray[Any, Any]) -> None:
    want = want.to_numpy() if isinstance(want, pd.Series) else want
    # pandas divides by an exactly zero variance in corr (+-inf or NaN); torchrolling: NaN.
    expected = torch.tensor(np.where(np.isinf(want), np.nan, want), dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=1e-8, atol=1e-8, equal_nan=True)


def exact(
    x: list[float], y: list[float], params: dict[str, Any]
) -> tuple[list[float], list[float]]:
    """Mean of ``x`` and unbiased covariance: pandas' ``ewmcov`` loop in 60-digit arithmetic.

    Two reasons not to compare with pandas directly here:

    - pandas computes the bias correction as W^2 - (sum of squared weights), two numbers
      close to 1 once the weights have decayed a lot (alpha near 1, long gaps), so its
      float64 answer can be off in the 8th digit, or 0 instead of NaN after one observation.
    - pandas special-cases alpha=0.5 (com=1) in ``mean()`` with adjust=False and
      ignore_na=False across missing values (``[1, nan, 0]`` gives 0.25, not the documented
      1/3; pandas-dev/pandas#66523). torchrolling follows the documented weights, as pandas
      does for every other alpha.
    """
    mp.mp.dps = 60
    decay = [params.get(k) for k in ("com", "span", "halflife", "alpha")]
    alpha = mp.mpf(1 / (1 + _ewm._center_of_mass(*decay)))  # the float alpha both use
    factor, new = 1 - alpha, mp.mpf(1) if params["adjust"] else alpha
    minp = max(params["min_periods"], 1)
    means: list[float] = []
    out: list[float] = []
    mean_x = mean_y = cov = sum_wt = sum_wt2 = old_wt = mp.mpf(0)
    seen = nobs = 0
    for a, b in zip(x, y, strict=True):
        observed = math.isfinite(a) and math.isfinite(b)
        nobs += observed
        if not seen:
            if observed:
                seen = 1
                mean_x, mean_y, cov, sum_wt, sum_wt2, old_wt = a, b, 0, 1, 1, 1
        elif observed or not params["ignore_na"]:
            sum_wt *= factor
            sum_wt2 *= factor * factor
            old_wt *= factor
            if observed:
                old_x, old_y = mean_x, mean_y
                mean_x = (old_wt * old_x + new * a) / (old_wt + new)
                mean_y = (old_wt * old_y + new * b) / (old_wt + new)
                cov = (
                    old_wt * (cov + (old_x - mean_x) * (old_y - mean_y))
                    + new * (a - mean_x) * (b - mean_y)
                ) / (old_wt + new)
                sum_wt += new
                sum_wt2 += new * new
                old_wt += new
                if not params["adjust"]:
                    sum_wt /= old_wt
                    sum_wt2 /= old_wt * old_wt
                    old_wt = mp.mpf(1)
        enough = seen and nobs >= minp
        means.append(float(mean_x) if enough else math.nan)
        # One observation has W^2 == sum of squared weights exactly; skip the rounding.
        denominator = sum_wt * sum_wt - sum_wt2
        ok = enough and nobs >= 2 and denominator > 0
        out.append(float(sum_wt * sum_wt / denominator * cov) if ok else math.nan)
    return means, out


pairs = st.integers(0, 40).flatmap(
    lambda n: st.tuples(*[st.lists(values, min_size=n, max_size=n)] * 2)
)


@pytest.mark.parametrize(("method", "kw"), [("mean", {}), ("var", {"bias": True})])
@settings(max_examples=300, deadline=None)
@given(data=st.lists(values, max_size=40), params=cases())
def test_matches_pandas(
    backend: str, method: str, kw: dict[str, bool], data: list[float], params: dict[str, Any]
) -> None:
    decay = [params.get(k) for k in ("com", "span", "halflife", "alpha")]
    if method == "mean" and not params["adjust"] and not params["ignore_na"]:
        # pandas' special case for alpha=0.5 (see exact())
        assume(_ewm._center_of_mass(*decay) != 1 or all(map(math.isfinite, data)))
    want = getattr(pd.Series(data, dtype="float64").ewm(**params), method)(**kw)
    got = getattr(torchrolling.ewm(torch.tensor(data, dtype=torch.float64), **params), method)(**kw)
    check(got, want)


@pytest.mark.parametrize(("method", "kw"), [("cov", {"bias": True}), ("corr", {})])
@settings(max_examples=200, deadline=None)
@given(pair=pairs, params=cases())
def test_pairwise_matches_pandas(
    backend: str,
    method: str,
    kw: dict[str, bool],
    pair: tuple[list[float], list[float]],
    params: dict[str, Any],
) -> None:
    x, y = (pd.Series(v, dtype="float64") for v in pair)
    want = getattr(x.ewm(**params), method)(y, **kw)
    tx, ty = (torch.tensor(v, dtype=torch.float64) for v in pair)
    got = getattr(torchrolling.ewm(tx, **params), method)(ty, **kw)
    check(got, want)


@settings(max_examples=300, deadline=None)
@given(pair=pairs, params=cases(), same=st.booleans())
def test_matches_exact_pandas_algorithm(
    backend: str, pair: tuple[list[float], list[float]], params: dict[str, Any], same: bool
) -> None:
    x, y = (pair[0], pair[0]) if same else pair
    tx, ty = (torch.tensor(v, dtype=torch.float64) for v in (x, y))
    e = torchrolling.ewm(tx, **params)
    got = e.var() if same else e.cov(ty)
    cov = torch.tensor(exact(x, y, params)[1], dtype=torch.float64)
    torch.testing.assert_close(got, cov, rtol=1e-9, atol=1e-9, equal_nan=True)
    mean = torch.tensor(exact(x, x, params)[0], dtype=torch.float64)
    torch.testing.assert_close(e.mean(), mean, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("method", ["mean", "var", "std"])
@pytest.mark.parametrize("adjust", [True, False])
def test_long_series(backend: str, method: str, adjust: bool) -> None:
    # Longer than _SCAN_BLOCK ** 2 steps, so the scan recurses twice.
    rng = np.random.default_rng(0)
    data = rng.standard_normal(10_000).cumsum() + 1e4
    data[rng.random(10_000) < 0.1] = np.nan
    want = getattr(pd.Series(data).ewm(span=300, adjust=adjust), method)()
    got = getattr(torchrolling.ewm(torch.tensor(data), span=300, adjust=adjust), method)()
    check(got, want)


def test_constant_series_has_zero_variance(backend: str) -> None:
    x = torch.full((50,), 3.7, dtype=torch.float64)
    x[10] = math.nan
    out = torchrolling.ewm(x, alpha=0.3).var(bias=True)
    assert (out == 0).all()
    assert torchrolling.ewm(x, alpha=0.3).corr(torch.arange(50.0)).isnan().all()


def test_batch_along_any_dim(backend: str) -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.standard_normal((30, 4)))
    frame.iloc[3, 1] = math.nan
    want = frame.ewm(halflife=4, min_periods=2).std().to_numpy()
    x = torch.tensor(frame.to_numpy())
    check(torchrolling.ewm(x, halflife=4, min_periods=2, dim=0).std(), want)
    check(torchrolling.ewm(x.T, halflife=4, min_periods=2).std(), want.T)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("method", ["mean", "var", "std"])
def test_keeps_float_dtype(backend: str, dtype: torch.dtype, method: str) -> None:
    out = getattr(torchrolling.ewm(torch.arange(10, dtype=dtype), alpha=0.5), method)()
    assert out.dtype == dtype
    assert out.shape == (10,)


def test_pairwise_dtype_promotes(backend: str) -> None:
    x = torch.arange(6, dtype=torch.float32)
    out = torchrolling.ewm(x, alpha=0.5).cov(x.double() ** 2)
    assert out.dtype == torch.float64


def test_integers_become_default_float(backend: str) -> None:
    out = torchrolling.ewm(torch.arange(6), alpha=1.0).mean()
    assert out.dtype == torch.get_default_dtype()
    torch.testing.assert_close(out, torch.arange(6.0))


def test_float32_accumulation_is_close(backend: str) -> None:
    x = torch.randn(4, 500, dtype=torch.float64)
    want = torchrolling.ewm(x, span=20).std()
    got = torchrolling.ewm(x.float(), span=20, acc_dtype=torch.float32).std()
    torch.testing.assert_close(got.double(), want, rtol=1e-4, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("method", ["mean", "var", "std", "cov", "corr"])
def test_empty_series(backend: str, method: str) -> None:
    x = torch.empty(3, 0)
    e = torchrolling.ewm(x, alpha=0.5)
    out = getattr(e, method)(x) if method in ("cov", "corr") else getattr(e, method)()
    assert out.shape == (3, 0)


@pytest.mark.parametrize("method", ["mean", "var", "std", "cov", "corr"])
@pytest.mark.parametrize("adjust", [True, False])
def test_gradients(method: str, adjust: bool) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 80, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 80, dtype=torch.float64)

    def f(t: torch.Tensor) -> torch.Tensor:
        e = torchrolling.ewm(t, span=5, adjust=adjust)
        out: torch.Tensor = (
            getattr(e, method)(y) if method in ("cov", "corr") else getattr(e, method)()
        )
        return out[..., 2:]

    assert torch.autograd.gradcheck(f, (x,))


def test_bias_correction_near_alpha_one(backend: str) -> None:
    # Two observations two steps apart: the exact unbiased covariance here is 1/2.
    x = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    y = torch.tensor([0.0, math.nan, 1.0], dtype=torch.float64)
    out = torchrolling.ewm(x, com=1e-5, adjust=False).cov(y)
    torch.testing.assert_close(out[2], torch.tensor(0.5, dtype=torch.float64), rtol=1e-12, atol=0)


def test_scan_matches_loop() -> None:
    torch.manual_seed(1)
    a, b = torch.rand(3, 1000, dtype=torch.float64), torch.randn(3, 1000, dtype=torch.float64)
    want, y = torch.empty_like(b), torch.zeros(3, dtype=torch.float64)
    for t in range(b.shape[-1]):
        y = a[:, t] * y + b[:, t]
        want[:, t] = y
    torch.testing.assert_close(_ewm._affine_scan(a, b), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("method", ["mean", "var", "std", "corr"])
def test_cuda_matches_cpu(method: str) -> None:  # pragma: no cover
    x = torch.randn(64, 3000, dtype=torch.float64)
    x[x > 2] = math.nan
    y = torch.randn(64, 3000, dtype=torch.float64)

    def run(t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        e = torchrolling.ewm(t, halflife=30, min_periods=5)
        out: torch.Tensor = e.corr(u) if method == "corr" else getattr(e, method)()
        return out

    torch.testing.assert_close(run(x.cuda(), y.cuda()).cpu(), run(x, y), equal_nan=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one of com, span, halflife or alpha"),
        ({"com": 1, "alpha": 0.5}, "exactly one of com, span, halflife or alpha"),
        ({"com": -1}, "com must be >= 0"),
        ({"span": 0.5}, "span must be >= 1"),
        ({"halflife": 0}, "halflife must be > 0"),
        ({"alpha": 0}, r"alpha must be in \(0, 1\]"),
        ({"alpha": 1.5}, r"alpha must be in \(0, 1\]"),
        ({"alpha": True}, "alpha must be a number"),
        ({"com": math.nan}, "com must be a number"),
        ({"alpha": 0.5, "min_periods": -1}, "min_periods must be an integer >= 0"),
    ],
)
def test_bad_arguments(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        torchrolling.ewm(torch.zeros(5), **kwargs)


def test_bad_other() -> None:
    e = torchrolling.ewm(torch.zeros(5), alpha=0.5)
    assert isinstance(e, Ewm)
    with pytest.raises(ValueError, match="other must have the same shape as x"):
        e.cov(torch.zeros(4))
    with pytest.raises(TypeError, match=r"other must be a torch\.Tensor"):
        e.corr([0.0] * 5)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["mean", "var", "std"])
def test_torch_compile(backend: str, method: str) -> None:
    torch.compiler.reset()  # each method is a new graph for the same function
    torch.manual_seed(5)
    x = torch.randn(3, 200)
    x[0, 5] = math.nan

    def f(t: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = getattr(torchrolling.ewm(t, span=10), method)()
        return out

    compiled = torch.compile(f, fullgraph=True, backend="aot_eager")
    torch.testing.assert_close(compiled(x), f(x), equal_nan=True)
