# Changelog

## 0.1.0

First release.

- `rolling(...)`: `count`, `sum`, `mean`, `var`, `std`, `skew`, `kurt`, `min`, `max`,
  `median`, `quantile`, and pairwise `cov` and `corr`, with pandas semantics. O(1) work per
  element for everything except quantiles.
- `ewm(...)`: `mean`, `var`, `std`, `cov`, `corr` with pandas' `com`/`span`/`halflife`/
  `alpha`, `adjust`, `ignore_na` and `min_periods`.
- On CUDA, every statistic runs as one fused Triton kernel (Triton 3.2+, torch 2.6+).
- Autograd everywhere, `torch.compile(fullgraph=True)` support. Sums and moments accumulate
  in the input's precision (at least float32); `acc_dtype` overrides it.
