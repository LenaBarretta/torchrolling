Device: Apple M3 Max (CPU, 12 performance + 4 efficiency cores). Best of several runs, in milliseconds (lower is better).

### mean

| series x length | window | pandas | polars |
| --- | --- | --- | --- |
| 100 x 10,000 | 20 | 7.2 | 1.2 |
| 100 x 10,000 | 200 | 7.1 | 1.2 |
| 100 x 10,000 | 1000 | 7.1 | 1.1 |
| 1,000 x 10,000 | 20 | 70.3 | 10.1 |
| 1,000 x 10,000 | 200 | 70.5 | 9.6 |
| 1,000 x 10,000 | 1000 | 70.8 | 9.3 |

### std

| series x length | window | pandas | polars |
| --- | --- | --- | --- |
| 100 x 10,000 | 20 | 13.9 | 1.5 |
| 100 x 10,000 | 200 | 13.9 | 1.7 |
| 100 x 10,000 | 1000 | 13.3 | 1.8 |
| 1,000 x 10,000 | 20 | 137.5 | 13.3 |
| 1,000 x 10,000 | 200 | 137.5 | 14.7 |
| 1,000 x 10,000 | 1000 | 132.6 | 14.4 |

### max

| series x length | window | pandas | polars |
| --- | --- | --- | --- |
| 100 x 10,000 | 20 | 14.9 | 1.6 |
| 100 x 10,000 | 200 | 15.0 | 1.5 |
| 100 x 10,000 | 1000 | 14.6 | 1.6 |
| 1,000 x 10,000 | 20 | 149.0 | 13.0 |
| 1,000 x 10,000 | 200 | 148.0 | 13.0 |
| 1,000 x 10,000 | 1000 | 148.2 | 12.4 |

### median

| series x length | window | pandas | polars |
| --- | --- | --- | --- |
| 100 x 10,000 | 20 | 227.5 | 4.3 |
| 100 x 10,000 | 200 | 295.2 | 4.2 |
| 100 x 10,000 | 1000 | 307.5 | 4.2 |
| 1,000 x 10,000 | 20 | 2239.4 | 39.9 |
| 1,000 x 10,000 | 200 | 2908.6 | 39.2 |
| 1,000 x 10,000 | 1000 | 3080.5 | 36.7 |

### corr

| series x length | window | pandas | polars |
| --- | --- | --- | --- |
| 100 x 10,000 | 20 | 54.2 | not supported (polars has no column-wise rolling corr) |
| 100 x 10,000 | 200 | 53.1 | not supported (polars has no column-wise rolling corr) |
| 100 x 10,000 | 1000 | 51.0 | not supported (polars has no column-wise rolling corr) |
| 1,000 x 10,000 | 20 | 526.0 | not supported (polars has no column-wise rolling corr) |
| 1,000 x 10,000 | 200 | 526.7 | not supported (polars has no column-wise rolling corr) |
| 1,000 x 10,000 | 1000 | 511.8 | not supported (polars has no column-wise rolling corr) |

### ewm_mean

| series x length | window | pandas | polars |
| --- | --- | --- | --- |
| 100 x 10,000 | 20 | 4.8 | 1.6 |
| 100 x 10,000 | 200 | 5.0 | 1.3 |
| 100 x 10,000 | 1000 | 4.8 | 1.2 |
| 1,000 x 10,000 | 20 | 49.0 | 10.0 |
| 1,000 x 10,000 | 200 | 48.6 | 9.9 |
| 1,000 x 10,000 | 1000 | 48.5 | 11.1 |

