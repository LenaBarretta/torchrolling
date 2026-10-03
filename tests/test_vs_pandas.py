"""torchrolling must match pandas exactly (up to float rounding)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import torchrolling

METHODS = ["count", "sum", "mean", "var", "std", "min", "max"]

values = st.one_of(
    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    st.sampled_from([math.nan, math.inf, -math.inf]),
)
# pandas computes skew, kurt, cov and corr from raw power sums, which lose precision on
# nearly constant windows; small integers keep its answers exact enough to compare against.
small = st.one_of(
    st.integers(-20, 20).map(float),
    st.sampled_from([math.nan, math.inf, -math.inf]),
)


@st.composite
def cases(
    draw: st.DrawFn, elements: st.SearchStrategy[float] = values
) -> tuple[list[float], int, int | None, bool]:
    data = draw(st.lists(elements, max_size=40))
    window = draw(st.integers(1, 12))
    min_periods = draw(st.none() | st.integers(0, window))
    center = draw(st.booleans())
    return data, window, min_periods, center


def expected(
    data: list[float], window: int, min_periods: int | None, center: bool, method: str, **kw: int
) -> torch.Tensor:
    r = pd.Series(data, dtype="float64").rolling(window, min_periods=min_periods, center=center)
    return torch.tensor(getattr(r, method)(**kw).to_numpy(), dtype=torch.float64)


def actual(
    data: list[float], window: int, min_periods: int | None, center: bool, method: str, **kw: int
) -> torch.Tensor:
    x = torch.tensor(data, dtype=torch.float64)
    r = torchrolling.rolling(x, window, min_periods=min_periods, center=center)
    out: torch.Tensor = getattr(r, method)(**kw)
    return out


def check(got: torch.Tensor, want: torch.Tensor) -> None:
    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("method", METHODS)
@settings(max_examples=300, deadline=None)
@given(case=cases())
def test_matches_pandas(method: str, case: tuple[list[float], int, int | None, bool]) -> None:
    check(actual(*case, method), expected(*case, method))


@pytest.mark.parametrize("method", ["var", "std"])
@pytest.mark.parametrize("ddof", [0, 2])
@settings(max_examples=100, deadline=None)
@given(case=cases())
def test_ddof_matches_pandas(
    method: str, ddof: int, case: tuple[list[float], int, int | None, bool]
) -> None:
    check(actual(*case, method, ddof=ddof), expected(*case, method, ddof=ddof))


@pytest.mark.parametrize("method", METHODS)
def test_large_offset_variance_is_stable(method: str) -> None:
    rng = np.random.default_rng(0)
    data = (1e6 + rng.standard_normal(5000)).tolist()
    check(actual(data, 50, None, False, method), expected(data, 50, None, False, method))


@pytest.mark.parametrize("method", METHODS)
def test_long_series_large_window(method: str) -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal(20_000).cumsum().tolist()
    check(actual(data, 997, 10, True, method), expected(data, 997, 10, True, method))


@pytest.mark.parametrize("method", METHODS)
def test_batch_along_any_dim(method: str) -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.standard_normal((30, 4)))
    frame.iloc[3, 1] = math.nan
    want = torch.tensor(getattr(frame.rolling(5, min_periods=2), method)().to_numpy())
    x = torch.tensor(frame.to_numpy())
    check(getattr(torchrolling.rolling(x, 5, min_periods=2, dim=0), method)(), want)
    check(getattr(torchrolling.rolling(x.T, 5, min_periods=2), method)(), want.T)


def test_variance_follows_local_spread_not_level() -> None:
    # A drifting series with tiny local noise: centring on one global mean loses all precision.
    rng = np.random.default_rng(6)
    data = 1e4 + rng.standard_normal(4000).cumsum() * 50 + rng.standard_normal(4000) * 1e-3
    windows = np.lib.stride_tricks.sliding_window_view(data, 20)
    want = torch.tensor(np.concatenate([np.full(19, np.nan), windows.std(axis=1, ddof=1)]))
    got = torchrolling.rolling(torch.tensor(data), 20).std()
    torch.testing.assert_close(got, want, rtol=1e-6, atol=0, equal_nan=True)


def expected_per_window(
    data: list[float], window: int, min_periods: int | None, center: bool, method: str
) -> torch.Tensor:
    """pandas on every window separately.

    pandas 3.0's rolling skew and kurt keep state between windows and return NaN for every
    window after two missing values in a row (e.g. ``[0, nan, nan, 0, 0, 1]`` with window 3
    ends in NaN, though ``[0, 0, 1]`` alone has skew 1.73), so the reference starts fresh.
    """
    shift = (window - 1) // 2 if center else 0
    padded = [math.nan] * (window - 1) + data + [math.nan] * shift
    out = []
    for t in range(len(data)):
        piece = pd.Series(padded[t + shift : t + shift + window], dtype="float64")
        out.append(getattr(piece.rolling(window, min_periods=min_periods), method)().iloc[-1])
    return torch.tensor(out, dtype=torch.float64)


@pytest.mark.parametrize("method", ["skew", "kurt"])
@settings(max_examples=300, deadline=None)
@given(case=cases(small))
def test_shape_moments_match_pandas(
    method: str, case: tuple[list[float], int, int | None, bool]
) -> None:
    got, want = actual(*case, method), expected_per_window(*case, method)
    torch.testing.assert_close(got, want, rtol=1e-7, atol=1e-7, equal_nan=True)


@pytest.mark.parametrize("method", ["skew", "kurt"])
def test_shape_moments_on_real_data(method: str) -> None:
    rng = np.random.default_rng(7)
    data = (rng.standard_normal(3000) * 5 + 100).tolist()
    data[17] = math.nan
    check(actual(data, 40, 10, True, method), expected(data, 40, 10, True, method))


@st.composite
def pairs(draw: st.DrawFn) -> tuple[list[float], list[float], int, int | None, bool]:
    data, window, min_periods, center = draw(cases(small))
    other = draw(st.lists(small, min_size=len(data), max_size=len(data)))
    return data, other, window, min_periods, center


def pandas_pairwise(
    method: str, case: tuple[list[float], list[float], int, int | None, bool], ddof: int
) -> torch.Tensor:
    data, other, window, min_periods, center = case
    r = pd.Series(data, dtype="float64").rolling(window, min_periods=min_periods, center=center)
    out = getattr(r, method)(pd.Series(other, dtype="float64"), ddof=ddof).to_numpy()
    # pandas divides by an exactly zero variance there (+-inf or NaN); torchrolling says NaN.
    return torch.tensor(np.where(np.isinf(out), np.nan, out), dtype=torch.float64)


@pytest.mark.parametrize("method", ["cov", "corr"])
@pytest.mark.parametrize("ddof", [0, 1])
@settings(max_examples=200, deadline=None)
@given(case=pairs())
def test_pairwise_matches_pandas(
    method: str, ddof: int, case: tuple[list[float], list[float], int, int | None, bool]
) -> None:
    data, other, window, min_periods, center = case
    r = torchrolling.rolling(
        torch.tensor(data, dtype=torch.float64), window, min_periods=min_periods, center=center
    )
    got = getattr(r, method)(torch.tensor(other, dtype=torch.float64), ddof=ddof)
    check(got, pandas_pairwise(method, case, ddof))


def test_corr_of_correlated_noise() -> None:
    rng = np.random.default_rng(8)
    x = rng.standard_normal(5000)
    y = 0.6 * x + 0.8 * rng.standard_normal(5000) + 1e5
    want = pd.Series(x).rolling(100).corr(pd.Series(y))
    got = torchrolling.rolling(torch.tensor(x), 100).corr(torch.tensor(y))
    check(got, torch.tensor(want.to_numpy()))
