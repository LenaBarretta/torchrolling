# Changelog

## 0.1.0

First release.

- `rolling(...)`: `count`, `sum`, `mean`, `var`, `std`, `skew`, `kurt`, `min`, `max`,
  `median`, `quantile`, and pairwise `cov` and `corr`, with pandas semantics. O(1) work per
  element for everything except quantiles, which use a Triton kernel on CUDA.
- `ewm(...)`: `mean`, `var`, `std`, `cov`, `corr` with pandas' `com`/`span`/`halflife`/
  `alpha`, `adjust`, `ignore_na` and `min_periods`.
- Autograd everywhere, `torch.compile(fullgraph=True)` support, and `acc_dtype` to trade
  float64 accumulation for float32 speed.
