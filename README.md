# torchrolling

[![PyPI](https://img.shields.io/pypi/v/torchrolling)](https://pypi.org/project/torchrolling/)
[![Python](https://img.shields.io/pypi/pyversions/torchrolling)](https://pypi.org/project/torchrolling/)
[![CI](https://github.com/LenaBarretta/torchrolling/actions/workflows/ci.yml/badge.svg)](https://github.com/LenaBarretta/torchrolling/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Pandas-style rolling and exponentially weighted statistics for PyTorch tensors, fast on
GPU.** Only depends on torch.

```python
import torch
import torchrolling

x = torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0])
torchrolling.rolling(x, 3, min_periods=2).mean()
# tensor([nan, 1.5000, 1.5000, 3.0000, 4.5000])

prices = torch.randn(512, 10_000, device="cuda").cumsum(-1)  # [series, time]
returns = prices.diff(dim=-1, prepend=prices[..., :1])
r = torchrolling.rolling(returns, 60)
features = torch.stack([r.mean(), r.std(), r.skew(), r.median(), r.corr(returns.roll(1, -1))])
trend = torchrolling.ewm(prices, span=20).mean()
```

## Why

`pandas.Series.rolling` works on one column on the CPU. When your series already live in a
`[batch, time]` tensor on the GPU (features computed on the fly, inside a model, after
augmentation), going through pandas means a round trip per batch. The usual torch workaround,
`x.unfold(-1, w, 1).mean(-1)`, does `w` times more work than needed, and for a median or
quantile it copies a tensor `w` times bigger than `x`. torch has no rolling median and no
exponential moving average at all.

torchrolling computes rolling sums, means, counts, variances, skew, kurtosis, covariances,
correlations, minima and maxima in O(1) work per element and O(n) memory for any window
size; medians and quantiles with a Triton kernel on the GPU; and exponentially weighted
means, variances and correlations with a parallel scan. It works on any device, supports
autograd and `torch.compile`, and gives the same numbers as pandas.

## Install

```bash
pip install torchrolling
```

Python 3.10+, torch 2.0+ (2.4+ for `torch.compile` of the Triton quantile kernel).

## Usage

### Rolling windows

`torchrolling.rolling(x, window, *, min_periods=None, center=False, dim=-1, acc_dtype=None)`
returns an object with these methods. Each returns a tensor shaped like `x`.

| Method | pandas equivalent |
| --- | --- |
| `.count()` | `.rolling(...).count()` |
| `.sum()` | `.rolling(...).sum()` |
| `.mean()` | `.rolling(...).mean()` |
| `.var(ddof=1)` / `.std(ddof=1)` | `.rolling(...).var(ddof=1)` / `.std(ddof=1)` |
| `.skew()` / `.kurt()` | `.rolling(...).skew()` / `.kurt()` |
| `.min()` / `.max()` | `.rolling(...).min()` / `.max()` |
| `.median()` | `.rolling(...).median()` |
| `.quantile(q, interpolation="linear")` | `.rolling(...).quantile(q, interpolation="linear")` |
| `.cov(other, ddof=1)` / `.corr(other)` | `.rolling(...).cov(other)` / `.corr(other)` |

Call several methods on one object: counts and moments are computed once and shared.

### Exponentially weighted windows

`torchrolling.ewm(x, com=None, span=None, halflife=None, alpha=None, *, min_periods=0,
adjust=True, ignore_na=False, dim=-1, acc_dtype=None)` takes exactly one of `com`, `span`,
`halflife`, `alpha`, as in pandas.

| Method | pandas equivalent |
| --- | --- |
| `.mean()` | `.ewm(...).mean()` |
| `.var(bias=False)` / `.std(bias=False)` | `.ewm(...).var(bias=False)` / `.std(bias=False)` |
| `.cov(other, bias=False)` / `.corr(other)` | `.ewm(...).cov(other)` / `.corr(other)` |

### Semantics

The test suite checks everything against pandas with property-based tests (hypothesis):

- the output has the same length; windows that are not full yet give NaN unless
  `min_periods` allows them;
- NaN and ±inf are treated as missing and skipped (pandas does the same). `count()` is the
  exception, as in pandas: it counts ±inf;
- `center=True` centres the window the way pandas does;
- pairwise statistics (`cov`, `corr`) use only the positions where both series are valid;
- integer input becomes the default float dtype; float input keeps its dtype;
- `interpolation` is one of `linear`, `lower`, `higher`, `midpoint`, `nearest`, as in pandas.

Where pandas itself is wrong, torchrolling gives the correct answer, and the tests compare
against pandas' algorithm run in exact arithmetic instead:

- pandas 3.0's rolling `skew`/`kurt` return NaN for every window after two missing values
  in a row;
- pandas' bias correction in `ewm(...).var()`/`.cov()` loses digits when the weights have
  decayed a lot (alpha close to 1, long gaps), and sometimes gives 0 instead of NaN after a
  single observation;
- pandas 3.0 changed `ewm(adjust=False).mean()` across missing values, away from its
  documented weights and from its own `var`/`cov`. torchrolling uses the documented
  weights (the pandas 2 result) for every method;
- `corr` is NaN, not ±inf, where one series is constant in the window.

### Precision and speed: `acc_dtype`

Sums and moments are accumulated in float64 by default (float32 on Apple MPS, which has no
float64), so results match pandas to 1e-9 or better, whatever the input dtype. Minima, maxima
and quantiles are exact in the input dtype and never use the accumulator. Consumer GPUs run
float64 much slower than float32; when float32 accuracy (~1e-6 relative) is enough, pass
`acc_dtype=torch.float32`.

## How it works

The series is padded and cut into blocks of exactly `window` elements. Each block is scanned
forwards (prefix) and backwards (suffix) with `cumsum`, `cummax` or `cummin`. Any window then
covers the tail of one block and the head of the next, so its value is
`combine(suffix[start], prefix[end])` (the van Herk / Gil-Werman algorithm). For sums this
also means only `2 * window` numbers are ever added together, so error does not grow with
series length.

Moments (variance, skew, kurtosis, covariance) come from power sums, which cancel badly when
taken raw. Each side of a window is therefore shifted by a value that lies inside that same
side: a head starts at its block's start, so it contains the block's first valid value, and
a tail contains its block's last one. The shifted sums are as well conditioned as the window
itself, a constant window gives exact zeros (so its variance is exactly 0, as in pandas), and
the two sides are merged with the pairwise formulas of Chan, Golub and LeVeque and of Pébay.

Quantiles cannot be split into halves. On CUDA a Triton kernel loads a tile of windows,
sorts each one in registers and reads the answer from the sorted row (windows up to 1024).
Elsewhere, for larger windows, or when gradients are needed, torchrolling falls back to
`torch.nanquantile` over chunks of windows, so memory stays bounded.

Exponentially weighted statistics follow pandas' update rule
`mean_t = (1 - s_t) * mean_{t-1} + s_t * x_t`. The weights `s_t` depend only on where values
are missing, so they are computed up front, and the mean, the weighted covariance and the
bias correction become affine recurrences `y_t = a_t * y_{t-1} + b_t`. These are composed in
parallel in blocks of 64 steps (Hillis-Steele), which is stable because every `a_t` lies in
[0, 1].

## Benchmarks

Numbers come from [`bench/bench.py`](bench/bench.py), run on Kaggle with
[`notebooks/kaggle_gpu.ipynb`](notebooks/kaggle_gpu.ipynb). Raw results are in
[`bench/results/`](bench/results/). Neither folder is part of the installed package.

_GPU results will be added after the first Kaggle run._

On the CPU, use pandas or polars: they are faster there. torchrolling is for data that is
already on the GPU.

## License

MIT
