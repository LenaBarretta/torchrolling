from __future__ import annotations

import math

import pytest
import torch

import torchrolling
from torchrolling import Rolling

METHODS = ["count", "sum", "mean", "var", "std", "skew", "kurt", "min", "max"]
PAIRWISE = ["cov", "corr"]


def test_version() -> None:
    assert torchrolling.__version__ == "0.1.0"


def test_example(backend: str) -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    r = torchrolling.rolling(x, 3)
    assert isinstance(r, Rolling)
    torch.testing.assert_close(
        r.mean(), torch.tensor([math.nan, math.nan, 2.0, 3.0, 4.0]), equal_nan=True
    )
    torch.testing.assert_close(
        r.max(), torch.tensor([math.nan, math.nan, 3.0, 4.0, 5.0]), equal_nan=True
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("method", METHODS)
def test_keeps_float_dtype(backend: str, dtype: torch.dtype, method: str) -> None:
    out = getattr(torchrolling.rolling(torch.arange(10, dtype=dtype), 3), method)()
    assert out.dtype == dtype
    assert out.shape == (10,)


def test_integers_become_default_float(backend: str) -> None:
    out = torchrolling.rolling(torch.arange(6), 2).sum()
    assert out.dtype == torch.get_default_dtype()
    torch.testing.assert_close(out[1:], torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0]))


@pytest.mark.parametrize("method", METHODS + PAIRWISE + ["median"])
@pytest.mark.parametrize("shape", [(3, 0), (0, 5), (0, 0)])
def test_empty(backend: str, method: str, shape: tuple[int, int]) -> None:
    x = torch.empty(shape)
    r = torchrolling.rolling(x, 4)
    out = getattr(r, method)(x) if method in PAIRWISE else getattr(r, method)()
    assert out.shape == shape


@pytest.mark.parametrize("method", PAIRWISE)
def test_pairwise_dtype_promotes(backend: str, method: str) -> None:
    x = torch.arange(6, dtype=torch.float32)
    out = getattr(torchrolling.rolling(x, 3), method)(x.double() ** 2)
    assert out.dtype == torch.float64


def test_corr_is_nan_where_a_series_is_constant(backend: str) -> None:
    x = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0])
    out = torchrolling.rolling(x, 3).corr(torch.arange(5.0))
    torch.testing.assert_close(
        out, torch.tensor([math.nan, math.nan, math.nan, math.sqrt(0.75), 1.0]), equal_nan=True
    )


def test_constant_windows(backend: str) -> None:
    x = torch.tensor([2.5] * 6 + [1.0], dtype=torch.float64)
    r = torchrolling.rolling(x, 4)
    assert (r.var()[3:6] == 0).all()
    assert (r.skew()[3:6] == 0).all()
    assert (r.kurt()[3:6] == -3).all()
    assert r.skew()[6] != 0


def test_statistics_share_one_cache() -> None:
    torch.manual_seed(2)
    x = torch.randn(3, 50, dtype=torch.float64)
    shared = torchrolling.rolling(x, 7)
    # Ask in an order that grows, shrinks and grows the cached moments again.
    for method in ["var", "kurt", "skew", "std", "kurt", "var"]:
        fresh = getattr(torchrolling.rolling(x, 7), method)()
        torch.testing.assert_close(getattr(shared, method)(), fresh, equal_nan=True)


@pytest.mark.parametrize("method", METHODS + PAIRWISE)
def test_float32_accumulation_is_close(backend: str, method: str) -> None:
    torch.manual_seed(3)
    x, y = torch.randn(4, 300, dtype=torch.float64), torch.randn(4, 300, dtype=torch.float64)

    def run(t: torch.Tensor, u: torch.Tensor, acc: torch.dtype | None) -> torch.Tensor:
        r = torchrolling.rolling(t, 30, min_periods=5, acc_dtype=acc)
        out: torch.Tensor = getattr(r, method)(u) if method in PAIRWISE else getattr(r, method)()
        return out

    got = run(x.float(), y.float(), torch.float32).double()
    # Fourth powers in float32: kurtosis near 0 is only good to ~1e-5 absolute.
    atol = 1e-4 if method in ("skew", "kurt") else 1e-5
    torch.testing.assert_close(got, run(x, y, None), rtol=1e-4, atol=atol, equal_nan=True)


def test_bad_acc_dtype() -> None:
    with pytest.raises(TypeError, match=r"acc_dtype must be a floating torch\.dtype"):
        torchrolling.rolling(torch.zeros(5), 2, acc_dtype=torch.int64)


@pytest.mark.parametrize("method", PAIRWISE)
def test_bad_other(method: str) -> None:
    r = torchrolling.rolling(torch.zeros(2, 5), 2)
    with pytest.raises(ValueError, match=r"other must have the same shape as x \(2, 5\)"):
        getattr(r, method)(torch.zeros(5))
    with pytest.raises(ValueError, match="ddof must be an integer >= 0"):
        getattr(r, method)(torch.zeros(2, 5), ddof=-1)


@pytest.mark.parametrize("method", [*METHODS, "median"])
def test_torch_compile(backend: str, method: str) -> None:
    torch.compiler.reset()  # each method is a new graph for the same function
    torch.manual_seed(4)
    x = torch.randn(3, 40)
    x[0, 5] = math.nan

    def f(t: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = getattr(torchrolling.rolling(t, 6, min_periods=2), method)()
        return out

    compiled = torch.compile(f, fullgraph=True, backend="aot_eager")
    torch.testing.assert_close(compiled(x), f(x), equal_nan=True)


def test_window_of_one_is_identity_with_nan_for_missing(backend: str) -> None:
    x = torch.tensor([1.0, math.nan, math.inf, 4.0])
    torch.testing.assert_close(
        torchrolling.rolling(x, 1).max(),
        torch.tensor([1.0, math.nan, math.nan, 4.0]),
        equal_nan=True,
    )


def test_sum_of_empty_window_is_zero_when_min_periods_is_zero(backend: str) -> None:
    x = torch.tensor([math.nan, math.nan, 1.0])
    torch.testing.assert_close(
        torchrolling.rolling(x, 2, min_periods=0).sum(), torch.tensor([0.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        torchrolling.rolling(x, 2, min_periods=0).mean(),
        torch.tensor([math.nan, math.nan, 1.0]),
        equal_nan=True,
    )


@pytest.mark.parametrize("method", [m for m in METHODS if m != "count"] + PAIRWISE)
def test_gradients(method: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 12, dtype=torch.float64, requires_grad=True)
    y = torch.randn(2, 12, dtype=torch.float64)

    def f(t: torch.Tensor) -> torch.Tensor:
        r = torchrolling.rolling(t, 4)
        out: torch.Tensor = getattr(r, method)(y) if method in PAIRWISE else getattr(r, method)()
        return out[..., 3:]

    assert torch.autograd.gradcheck(f, (x,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("method", METHODS + PAIRWISE)
def test_cuda_matches_cpu(method: str) -> None:  # pragma: no cover
    x = torch.randn(64, 1000, dtype=torch.float64)
    x[x > 2] = math.nan
    y = torch.randn(64, 1000, dtype=torch.float64)

    def run(t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r = torchrolling.rolling(t, 37, min_periods=5, center=True)
        out: torch.Tensor = getattr(r, method)(u) if method in PAIRWISE else getattr(r, method)()
        return out

    torch.testing.assert_close(run(x.cuda(), y.cuda()).cpu(), run(x, y), equal_nan=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window": 0}, "window must be an integer >= 1"),
        ({"window": 2.5}, "window must be an integer >= 1"),
        ({"window": True}, "window must be an integer >= 1"),
        ({"window": 3, "min_periods": -1}, "min_periods must be an integer >= 0"),
        ({"window": 3, "min_periods": 4}, r"min_periods \(4\) must be <= window \(3\)"),
    ],
)
def test_bad_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        torchrolling.rolling(torch.zeros(5), **kwargs)  # type: ignore[arg-type]


def test_bad_ddof() -> None:
    with pytest.raises(ValueError, match="ddof must be an integer >= 0"):
        torchrolling.rolling(torch.zeros(5), 2).var(ddof=-1)


def test_bad_inputs() -> None:
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor, got list"):
        torchrolling.rolling([1.0, 2.0], 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="complex"):
        torchrolling.rolling(torch.zeros(3, dtype=torch.complex64), 2)
    with pytest.raises(ValueError, match="at least one dimension"):
        torchrolling.rolling(torch.tensor(1.0), 2)
