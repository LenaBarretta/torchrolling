"""Rolling median and quantile, on both backends, checked against pandas."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

import torchrolling
from torchrolling import _rolling

BACKENDS = ["auto", "torch"]
INTERPOLATIONS = ["linear", "lower", "higher", "midpoint", "nearest"]

values = st.one_of(
    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    st.sampled_from([math.nan, math.inf, -math.inf, 0.0, 1.0]),
)


@contextmanager
def backend(name: str) -> Iterator[None]:
    """ "auto" uses Triton when available (on CPU via its interpreter); "torch" never does."""
    if name == "torch":
        with mock.patch.object(_rolling, "_use_triton", return_value=False):
            yield
    else:
        yield


@st.composite
def cases(draw: st.DrawFn) -> tuple[list[float], int, int | None, bool]:
    data = draw(st.lists(values, max_size=40))
    window = draw(st.integers(1, 12))
    min_periods = draw(st.none() | st.integers(0, window))
    center = draw(st.booleans())
    return data, window, min_periods, center


def check(got: torch.Tensor, want: pd.Series) -> None:
    expected = torch.tensor(want.to_numpy(), dtype=torch.float64)
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize("name", BACKENDS)
@settings(max_examples=150, deadline=None)
@given(
    case=cases(),
    q=st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 1.0]),
    interpolation=st.sampled_from(INTERPOLATIONS),
)
def test_quantile_matches_pandas(
    name: str, case: tuple[list[float], int, int | None, bool], q: float, interpolation: str
) -> None:
    data, window, min_periods, center = case
    r = pd.Series(data, dtype="float64").rolling(window, min_periods=min_periods, center=center)
    want = r.quantile(q, interpolation=interpolation)  # type: ignore[arg-type]
    x = torch.tensor(data, dtype=torch.float64)
    with backend(name):
        got = torchrolling.rolling(x, window, min_periods=min_periods, center=center).quantile(
            q, interpolation
        )
    check(got, want)


@pytest.mark.parametrize("name", BACKENDS)
@settings(max_examples=100, deadline=None)
@given(case=cases())
def test_median_matches_pandas(name: str, case: tuple[list[float], int, int | None, bool]) -> None:
    data, window, min_periods, center = case
    r = pd.Series(data, dtype="float64").rolling(window, min_periods=min_periods, center=center)
    x = torch.tensor(data, dtype=torch.float64)
    with backend(name):
        got = torchrolling.rolling(x, window, min_periods=min_periods, center=center).median()
    check(got, r.median())


@pytest.mark.parametrize("name", BACKENDS)
def test_batch_along_any_dim(name: str) -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.standard_normal((50, 3)))
    frame.iloc[7, 2] = math.nan
    want = torch.tensor(frame.rolling(9, min_periods=4, center=True).quantile(0.3).to_numpy())
    x = torch.tensor(frame.to_numpy())
    with backend(name):
        got = torchrolling.rolling(x, 9, min_periods=4, center=True, dim=0).quantile(0.3)
    torch.testing.assert_close(got, want, equal_nan=True)


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_low_precision_input(name: str, dtype: torch.dtype) -> None:
    x = torch.arange(20, dtype=dtype).flip(0)
    with backend(name):
        out = torchrolling.rolling(x, 4).median()
    assert out.dtype == dtype
    torch.testing.assert_close(out[3:], torch.arange(17.5, 1, -1, dtype=dtype))


def test_large_window_uses_torch_and_matches_pandas() -> None:
    rng = np.random.default_rng(4)
    data = rng.standard_normal(3000)
    want = pd.Series(data).rolling(1500, min_periods=10).quantile(0.9)
    got = torchrolling.rolling(torch.tensor(data), 1500, min_periods=10).quantile(0.9)
    check(got, want)


def test_chunks_join_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_rolling, "_CHUNK", 64)
    rng = np.random.default_rng(5)
    data = rng.standard_normal((3, 100))
    want = pd.DataFrame(data.T).rolling(7).median().to_numpy().T
    with backend("torch"):
        got = torchrolling.rolling(torch.tensor(data), 7).median()
    torch.testing.assert_close(got, torch.tensor(want), equal_nan=True)


def test_gradients() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 15, dtype=torch.float64, requires_grad=True)

    def f(t: torch.Tensor) -> torch.Tensor:
        return torchrolling.rolling(t, 5).quantile(0.3)[..., 4:]

    assert torch.autograd.gradcheck(f, (x,))


@pytest.mark.parametrize("name", BACKENDS)
def test_empty_series(name: str) -> None:
    with backend(name):
        assert torchrolling.rolling(torch.empty(2, 0), 3).median().shape == (2, 0)


@pytest.mark.parametrize("q", [-0.1, 1.5, True, "0.5", math.nan])
def test_bad_q(q: object) -> None:
    with pytest.raises(ValueError, match="q must be a number between 0 and 1"):
        torchrolling.rolling(torch.zeros(5), 2).quantile(q)  # type: ignore[arg-type]


def test_bad_interpolation() -> None:
    with pytest.raises(ValueError, match="interpolation must be one of"):
        torchrolling.rolling(torch.zeros(5), 2).quantile(0.5, "cubic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_cuda_kernel_matches_cpu(interpolation: str) -> None:  # pragma: no cover
    x = torch.randn(32, 2000, dtype=torch.float64)
    x[x > 2] = math.nan
    want = torchrolling.rolling(x, 101, min_periods=20, center=True).quantile(0.37, interpolation)
    got = torchrolling.rolling(x.cuda(), 101, min_periods=20, center=True).quantile(
        0.37, interpolation
    )
    torch.testing.assert_close(got.cpu(), want, equal_nan=True)
