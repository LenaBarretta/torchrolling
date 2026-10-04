Device: Tesla T4. Best of several runs, in milliseconds (lower is better).

### mean

| series x length | window | torchrolling (cuda) | torchrolling fp64 acc (cuda) | torch unfold (cuda) | torchrolling (cpu) | torch unfold (cpu) | pandas | polars | cuDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 x 10,000 | 20 | 0.7 | 2.6 | 0.2 | 25.4 | 6.0 | 31.4 | 6.8 | 14.7 |
| 100 x 10,000 | 200 | 0.6 | 2.6 | 0.7 | 13.2 | 9.2 | 23.3 | 7.9 | 25.3 |
| 100 x 10,000 | 1000 | 0.5 | 1.9 | 4.5 | 13.5 | 26.0 | 23.8 | 7.2 | 54.5 |
| 1,000 x 10,000 | 20 | 4.2 | 24.1 | 0.9 | 410.4 | 76.2 | 247.5 | 68.4 | 163.9 |
| 1,000 x 10,000 | 200 | 4.0 | 24.1 | 6.4 | 432.3 | skipped, needs 7 GB | 262.0 | 67.9 | 265.2 |
| 1,000 x 10,000 | 1000 | 3.1 | 16.2 | 38.4 | 429.0 | skipped, needs 37 GB | 237.7 | 62.2 | 349.7 |
| 10,000 x 10,000 | 20 | 41.5 | 114.9 | 5.7 | - | - | - | - | 2230.2 |
| 10,000 x 10,000 | 200 | 39.5 | 116.1 | 50.1 | - | - | - | - | 3199.8 |
| 10,000 x 10,000 | 1000 | 29.9 | 85.6 | 261.7 | - | - | - | - | 4055.0 |

### std

| series x length | window | torchrolling (cuda) | torchrolling fp64 acc (cuda) | torch unfold (cuda) | torchrolling (cpu) | torch unfold (cpu) | pandas | polars | cuDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 x 10,000 | 20 | 0.7 | 5.6 | 0.3 | 71.8 | 219.2 | 32.3 | 9.9 | 27.7 |
| 100 x 10,000 | 200 | 0.6 | 4.4 | 2.2 | 62.8 | 872.4 | 30.6 | 10.0 | 61.2 |
| 100 x 10,000 | 1000 | 0.6 | 4.0 | 12.4 | 67.3 | 3505.5 | 32.5 | 9.9 | 148.0 |
| 1,000 x 10,000 | 20 | 5.1 | 39.8 | 1.2 | 1145.5 | 2252.4 | 329.6 | 98.1 | 291.3 |
| 1,000 x 10,000 | 200 | 4.4 | 30.3 | 14.7 | 1216.3 | skipped, needs 7 GB | 356.4 | 101.5 | 485.8 |
| 1,000 x 10,000 | 1000 | 2.3 | 16.2 | 80.0 | 1215.0 | skipped, needs 37 GB | 315.7 | 84.6 | 1486.8 |
| 10,000 x 10,000 | 20 | 50.4 | 228.8 | 14.2 | - | - | - | - | 3574.1 |
| 10,000 x 10,000 | 200 | 40.1 | 188.3 | 136.3 | - | - | - | - | 5566.0 |
| 10,000 x 10,000 | 1000 | 26.6 | 165.7 | 803.5 | - | - | - | - | 15514.5 |

### max

| series x length | window | torchrolling (cuda) | torchrolling fp64 acc (cuda) | torch unfold (cuda) | torchrolling (cpu) | torch unfold (cpu) | pandas | polars | cuDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 x 10,000 | 20 | 0.5 | 0.5 | 0.2 | 32.5 | 19.4 | 46.8 | 11.3 | 13.9 |
| 100 x 10,000 | 200 | 0.4 | 0.5 | 0.5 | 17.2 | 21.2 | 38.3 | 10.7 | 13.6 |
| 100 x 10,000 | 1000 | 0.4 | 0.4 | 2.0 | 16.8 | 53.6 | 39.2 | 10.0 | 20.1 |
| 1,000 x 10,000 | 20 | 3.4 | 4.1 | 1.5 | 652.1 | 198.3 | 415.3 | 103.9 | 154.5 |
| 1,000 x 10,000 | 200 | 2.0 | 3.0 | 3.2 | 618.5 | skipped, needs 7 GB | 424.3 | 110.1 | 170.4 |
| 1,000 x 10,000 | 1000 | 1.9 | 2.8 | 30.3 | 602.5 | skipped, needs 37 GB | 385.8 | 97.1 | 206.7 |
| 10,000 x 10,000 | 20 | 31.1 | 32.5 | 9.7 | - | - | - | - | 2041.9 |
| 10,000 x 10,000 | 200 | 22.1 | 32.4 | 56.9 | - | - | - | - | 2303.2 |
| 10,000 x 10,000 | 1000 | 19.4 | 29.4 | 294.3 | - | - | - | - | 2572.8 |

### median

| series x length | window | torchrolling (cuda) | torchrolling fp64 acc (cuda) | torch unfold (cuda) | torchrolling (cpu) | torch unfold (cpu) | pandas | polars | cuDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 x 10,000 | 20 | 2.0 | 2.0 | 23.4 | 251.5 | 406.2 | 409.3 | 44.7 | not supported (Rolling.median() is not yet implemented) |
| 100 x 10,000 | 200 | 4.7 | 4.7 | 217.5 | 4334.4 | 5744.5 | 652.4 | 44.5 | not supported (Rolling.median() is not yet implemented) |
| 100 x 10,000 | 1000 | 19.5 | 19.4 | out of memory | 27010.4 | 31810.6 | 788.5 | 43.4 | not supported (Rolling.median() is not yet implemented) |
| 1,000 x 10,000 | 20 | 18.1 | 16.1 | 141.8 | 2789.7 | 4553.0 | 4721.3 | 480.0 | not supported (Rolling.median() is not yet implemented) |
| 1,000 x 10,000 | 200 | 43.8 | 23.1 | out of memory | 42616.7 | skipped, needs 7 GB | 5998.4 | 425.2 | not supported (Rolling.median() is not yet implemented) |
| 1,000 x 10,000 | 1000 | 81.1 | 83.5 | out of memory | 270691.9 | skipped, needs 37 GB | 8293.2 | 423.3 | not supported (Rolling.median() is not yet implemented) |
| 10,000 x 10,000 | 20 | 106.1 | 114.3 | out of memory | - | - | - | - | not supported (Rolling.median() is not yet implemented) |
| 10,000 x 10,000 | 200 | 221.1 | 230.7 | out of memory | - | - | - | - | not supported (Rolling.median() is not yet implemented) |
| 10,000 x 10,000 | 1000 | 829.9 | 844.8 | out of memory | - | - | - | - | not supported (Rolling.median() is not yet implemented) |

### corr

| series x length | window | torchrolling (cuda) | torchrolling fp64 acc (cuda) | torch unfold (cuda) | torchrolling (cpu) | torch unfold (cpu) | pandas | polars | cuDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 x 10,000 | 20 | 1.0 | 8.4 | 8.1 | 119.2 | 213.0 | 143.7 | not supported (polars has no column-wise rolling corr) | not supported (AttributeError) |
| 100 x 10,000 | 200 | 1.1 | 7.8 | 44.0 | 114.0 | 1796.1 | 144.7 | not supported (polars has no column-wise rolling corr) | not supported (AttributeError) |
| 100 x 10,000 | 1000 | 0.9 | 6.3 | 193.2 | 122.8 | 8376.8 | 133.9 | not supported (polars has no column-wise rolling corr) | not supported (AttributeError) |
| 1,000 x 10,000 | 20 | 8.3 | 59.9 | 63.8 | 2646.2 | 2344.9 | 1594.0 | not supported (polars has no column-wise rolling corr) | not supported (AttributeError) |
| 1,000 x 10,000 | 200 | 8.4 | 36.1 | out of memory | 2446.5 | skipped, needs 7 GB | 1462.5 | not supported (polars has no column-wise rolling corr) | not supported (AttributeError) |
| 1,000 x 10,000 | 1000 | 7.3 | 35.7 | out of memory | 2535.1 | skipped, needs 37 GB | 1416.2 | not supported (polars has no column-wise rolling corr) | not supported (AttributeError) |
| 10,000 x 10,000 | 20 | 58.0 | 333.9 | out of memory | - | - | - | - | not supported (AttributeError) |
| 10,000 x 10,000 | 200 | 50.1 | 304.8 | out of memory | - | - | - | - | not supported (AttributeError) |
| 10,000 x 10,000 | 1000 | 47.7 | 268.2 | out of memory | - | - | - | - | not supported (AttributeError) |

### ewm_mean

| series x length | window | torchrolling (cuda) | torchrolling fp64 acc (cuda) | torch unfold (cuda) | torchrolling (cpu) | torch unfold (cpu) | pandas | polars | cuDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 x 10,000 | 20 | 0.4 | 1.7 | not supported (torch has no exponential moving average) | 60.1 | not supported (torch has no exponential moving average) | 13.1 | 5.5 | 84.6 |
| 100 x 10,000 | 200 | 0.4 | 1.6 | not supported (torch has no exponential moving average) | 35.7 | not supported (torch has no exponential moving average) | 11.4 | 5.7 | 89.8 |
| 100 x 10,000 | 1000 | 0.4 | 1.6 | not supported (torch has no exponential moving average) | 35.3 | not supported (torch has no exponential moving average) | 12.4 | 5.7 | 83.9 |
| 1,000 x 10,000 | 20 | 1.1 | 12.1 | not supported (torch has no exponential moving average) | 1699.1 | not supported (torch has no exponential moving average) | 147.1 | 54.3 | 947.8 |
| 1,000 x 10,000 | 200 | 1.0 | 12.1 | not supported (torch has no exponential moving average) | 1627.3 | not supported (torch has no exponential moving average) | 137.0 | 54.2 | 901.8 |
| 1,000 x 10,000 | 1000 | 1.0 | 12.1 | not supported (torch has no exponential moving average) | 1616.8 | not supported (torch has no exponential moving average) | 136.9 | 52.9 | 875.6 |
| 10,000 x 10,000 | 20 | 5.3 | 51.2 | not supported (torch has no exponential moving average) | - | - | - | - | 10135.7 |
| 10,000 x 10,000 | 200 | 4.7 | 47.6 | not supported (torch has no exponential moving average) | - | - | - | - | 10116.9 |
| 10,000 x 10,000 | 1000 | 4.5 | 49.3 | not supported (torch has no exponential moving average) | - | - | - | - | 9647.5 |

