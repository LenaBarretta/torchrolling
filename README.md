<p align="center">
  <img src="https://raw.githubusercontent.com/LenaBarretta/torchrolling/main/docs/logo.png" alt="torchrolling logo" width="160">
</p>

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
size; medians and quantiles; and exponentially weighted means, variances and correlations
with a parallel scan. On CUDA every statistic is a single fused Triton kernel. It works on
any device, supports autograd and `torch.compile`, and gives the same numbers as pandas.

## Install

```bash
pip install torchrolling
```

Python 3.10+, torch 2.0+. The CUDA kernels need Triton 3.2+, which comes with torch 2.6+;
with an older torch, the same statistics run as plain torch operations.

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

### Precision: `acc_dtype`

Sums and moments are accumulated in the input's precision, but at least float32: float64
input is accumulated in float64 and matches pandas to 1e-9 or better; float32, float16 and
bfloat16 input is accumulated in float32, which is accurate to about 1e-6 relative. Minima,
maxima and quantiles are exact in the input dtype. To accumulate float32 data in float64,
pass `acc_dtype=torch.float64`. On consumer and inference GPUs (T4, RTX), float64 is much
slower than float32.

### Gradients

Every statistic supports autograd. On CUDA, the fused Triton kernels compute the forward
pass when no gradient is needed (inference, feature pipelines, `torch.no_grad()`); when one
is, torchrolling runs the same algorithms as plain torch operations instead, which autograd
differentiates. Both give the same results.

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

Quantiles cannot be split into halves; the torch code sorts every window, in chunks so that
memory stays bounded.

On CUDA, each of these is a single fused Triton kernel. For rolling statistics, a program
loads a run of whole blocks twice, once as is and once shifted by one block and one element,
so that the tail and the head of every window sit at the same position of two register
tiles; both scans and the merge happen in registers, in one pass over memory. For quantiles
with windows above 64, a program sorts the segment its outputs need once, with each value's
position packed into the sort key, then walks the sorted segment once, counting for every
window how many of its own values it has passed: O(window) work per output instead of
O(window log² window). Smaller windows sort each window directly.

Exponentially weighted statistics follow pandas' update rule
`mean_t = (1 - s_t) * mean_{t-1} + s_t * x_t`. The weights `s_t` depend only on where values
are missing, so they are computed up front, and the mean, the weighted covariance and the
bias correction become affine recurrences `y_t = a_t * y_{t-1} + b_t`. These are composed in
parallel in blocks of 64 steps (Hillis-Steele), which is stable because every `a_t` lies in
[0, 1]. On CUDA, one program per series scans it chunk by chunk and carries the state across
chunks.

## Benchmarks

Milliseconds, best of 3 runs, lower is better; the fastest in each row is bold. Measured on
a Tesla T4 (Kaggle) with float32 data, torch 2.11 and Triton 3.6, by
[`bench/bench.py`](bench/bench.py) via [`notebooks/kaggle_gpu.ipynb`](notebooks/kaggle_gpu.ipynb).
Each library gets its data where it wants it (tensors on the GPU, a DataFrame for pandas and
polars), and only the computation is timed. For EWM, `window` is the span. All raw numbers,
including the CPU runs and float64 accumulation, are in [`bench/results/`](bench/results/).

**1,000 series × 10,000 points**

| | window | torchrolling | torch `unfold` | cuDF | pandas | polars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean | 20 | 4.2 | **0.9** | 164 | 247 | 68 |
|  | 200 | **4.0** | 6.4 | 265 | 262 | 68 |
|  | 1000 | **3.1** | 38 | 350 | 238 | 62 |
| std | 20 | 5.1 | **1.2** | 291 | 330 | 98 |
|  | 200 | **4.4** | 15 | 486 | 356 | 101 |
|  | 1000 | **2.3** | 80 | 1,487 | 316 | 85 |
| max | 20 | 3.4 | **1.5** | 154 | 415 | 104 |
|  | 200 | **2.0** | 3.2 | 170 | 424 | 110 |
|  | 1000 | **1.9** | 30 | 207 | 386 | 97 |
| median | 20 | **18** | 142 | — | 4,721 | 480 |
|  | 200 | **44** | out of memory | — | 5,998 | 425 |
|  | 1000 | **81** | out of memory | — | 8,293 | 423 |
| corr | 20 | **8.3** | 64 | — | 1,594 | — |
|  | 200 | **8.4** | out of memory | — | 1,462 | — |
|  | 1000 | **7.3** | out of memory | — | 1,416 | — |
| ewm mean | 20 | **1.1** | — | 948 | 147 | 54 |
|  | 200 | **1.0** | — | 902 | 137 | 54 |
|  | 1000 | **1.0** | — | 876 | 137 | 53 |

**10,000 series × 10,000 points** (GPU libraries only)

| | window | torchrolling | torch `unfold` | cuDF |
| --- | ---: | ---: | ---: | ---: |
| mean | 20 | 42 | **5.7** | 2,230 |
|  | 200 | **40** | 50 | 3,200 |
|  | 1000 | **30** | 262 | 4,055 |
| std | 20 | 50 | **14** | 3,574 |
|  | 200 | **40** | 136 | 5,566 |
|  | 1000 | **27** | 804 | 15,515 |
| max | 20 | 31 | **9.7** | 2,042 |
|  | 200 | **22** | 57 | 2,303 |
|  | 1000 | **19** | 294 | 2,573 |
| median | 20 | **106** | out of memory | — |
|  | 200 | **221** | out of memory | — |
|  | 1000 | **830** | out of memory | — |
| corr | 20 | **58** | out of memory | — |
|  | 200 | **50** | out of memory | — |
|  | 1000 | **48** | out of memory | — |
| ewm mean | 20 | **5.3** | — | 10,136 |
|  | 200 | **4.7** | — | 10,117 |
|  | 1000 | **4.5** | — | 9,648 |

`torch unfold` is the usual plain-torch workaround, for example `x.unfold(-1, w, 1).mean(-1)`:
it has no missing-value handling and no `min_periods`, and for median and corr it copies a
tensor `window` times the size of the input. On small windows without missing values it is
the fastest way to get mean, std or max; everywhere else torchrolling is. "—" means the
library has no such statistic.

**Reading these numbers honestly.** Part of the gap to pandas and polars is simply a GPU
against a modest CPU: Kaggle gives them 2–4 cores, and on a large many-core server polars
would close much of it. That is the situation torchrolling is for (data already on the GPU),
but on the CPU, use pandas or polars: they are faster there. cuDF is measured at its weakest
shape, thousands of short columns, which it processes one by one; on a few long series it
would be much closer. Transfers between CPU and GPU are not timed, and neither is the first
call of each statistic, which compiles its Triton kernel (a few seconds, then cached on disk).

The kernels' launch configurations were tuned on a T4. Results are the same on every GPU;
only speed can differ. On a GPU or system where a kernel cannot run, torchrolling warns once
and computes the same statistics with plain torch operations.

## License

MIT
