# Week 2 validation package · 7 Nov 2026

## Data-source change

One teammate was injured. We have **no own recordings** and did not invent any. MM-Fit RGB (Zenodo 7672767) + watch IMU/HR replaces that role. Personal baselines are **within-session** (earlier sets of the same exercise). RPE is dropped. Live Galaxy Watch link remains **untested**.

## Model 2 (EC3D, step 2.2)

Honest conclusion (also in `m2_benchmark.md`): ST-GCN squat 55.6 ± 12.4% vs 2D RF 53.8 ± 6.1% — the squat lead is within fold noise. RF wins on lunge and plank. Do not treat ST-GCN as the overall winner. Authors' 90.9% is a different setup (context only).

## Kafka (step 2.3, replay)

Broker that worked: **Docker `apache/kafka`**. KRaft binaries blocked (`wmic` missing).

| Topic | Median msg/s | Median latency |
|---|---:|---:|
| `pose.keypoints` (w20 RGB) | 29.8 | 48 ms |
| `watch.hr` (w20 `sw_l_hr`) | 0.79 | 19 ms |

Sink → `data/interim/replay/` only. Metrics: `experiments/20261107_kafka_replay/metrics.json`.

## RQ4 (step 2.4, w20 only)

| Pose source | M1 macro-F1 pose / both | Rep MAE | Sync ≤3 fr. | M2 P(correct) |
|---|---:|---:|---:|---:|
| Dataset pose | 0.85 / 0.66 | 0.93 | 8/10 | 0.08 |
| MediaPipe from RGB | 0.80 / 0.48 | 1.56 | (same map) | 0.14 |

No form labels on MM-Fit. Extra videos (w00/w12/w13) were downloading but not finished — shrunk to w20.

## Spark rep table (step 2.5)

- **5,918** rep rows, 616 sets, 21 sessions
- Within-session baselines; `has_baseline` rate **0.67**
- HR mask forced 0 on w05/w10/w18
