# Model 1 benchmark

Test set: MM-Fit `test_unseen` (people not in training).

| Architecture | Inputs | Macro-F1 unseen | Accuracy unseen | Macro-F1 seen | Params | Train minutes |
|---|---|---:|---:|---:|---:|---:|
| cnn | pose | 0.7847 | 0.8992 | 0.9246 | 153291 | 0.36 |
| cnn | imu | 0.8423 | 0.9358 | 0.9658 | 143691 | 0.74 |
| cnn | both | 0.8478 | 0.9384 | 0.9743 | 295563 | 1.75 |
| lstm | pose | 0.7566 | 0.9051 | 0.9522 | 169483 | 4.19 |
| lstm | imu | 0.7458 | 0.9091 | 0.9609 | 154123 | 6.71 |
| lstm | both | 0.8212 | 0.8999 | 0.9351 | 322187 | 4.21 |
| transformer | pose | 0.8355 | 0.8942 | 0.9421 | 84811 | 8.5 |
| transformer | imu | 0.7455 | 0.889 | 0.9403 | 82891 | 8.26 |
| transformer | both | 0.8798 | 0.9495 | 0.975 | 166283 | 24.72 |

Best unseen macro-F1: **0.8798** with `transformer` / `both` (`20261022_m1_transformer_both`).
