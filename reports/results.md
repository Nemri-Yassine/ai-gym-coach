# Results · RQ1–RQ4 + Big Data

Numbers below are copied from `experiments/*/metrics.json` (and the week reports).

## RQ1 · Recognition on unseen people

- Best Model 1: transformer pose+IMU, macro-F1 **0.880** on `test_unseen` (`reports/m1_benchmark.md`).
- Rep counting macro MAE **0.49** on 616 MM-Fit sets (`reports/rep_counting.md`).

## RQ2 · Heart rate vs movement (B4 vs B3)

- B4 (personal+HR) risk AUC **0.8033740546829552**, lead **0.8545454545454545**, FA **2.0**
- B3 (personal, no HR) risk AUC **0.8140356444021365**, lead **0.8545454545454545**, FA **1.2**
- Honest finding: HR does **not** improve risk AUC on this MM-Fit setup. See `reports/m3_results.md`.

## RQ3 · Personal vs general (B4 vs B2) and vs rule (B1)

- B2 (general+HR) AUC **0.8843037706912052** — better than B4 here.
- B1 rule AUC **0.5002538473742661**, lead **0.0**.
- Within-session personal baseline does not beat general normalisation on these labels.

## RQ4 · Raw MM-Fit video

- w20 MediaPipe vs dataset pose: M1 pose-only 0.85→0.80, rep MAE 0.93→1.56 (`reports/rq4_first_look.md`).

## Big data

- Spark sync scaling: see week1 / `experiments/*_spark_scaling` if present.
- Kafka replay: pose 29.8 msg/s @ 48 ms; HR 0.79 msg/s @ 19 ms (`experiments/20261107_kafka_replay`).
- Rep table: 5,918 rows (`data/processed/rep_table_summary.json`).

## Runs indexed

- `20261022_m1_cnn_both`
- `20261022_m1_cnn_imu`
- `20261022_m1_cnn_pose`
- `20261022_m1_lstm_both`
- `20261022_m1_lstm_imu`
- `20261022_m1_lstm_pose`
- `20261022_m1_transformer_both`
- `20261022_m1_transformer_imu`
- `20261022_m1_transformer_pose`
- `20261022_rep_counting`
- `20261022_spark_scaling`
- `20261023_public_videos`
- `20261024_m1_cnn_pose_quick`
- `20261024_public_videos`
- `20261024_rep_counting`
- `20261104_m1_cnn_pose_quick`
- `20261104_m2_benchmark`
- `20261104_m2_lstm`
- `20261104_m2_rf`
- `20261104_m2_rf_quick`
- `20261104_m2_stgcn`
- `20261104_m2_stgcn_quick`
- `20261107_kafka_replay`
- `20261107_m1_cnn_pose_quick`
- `20261107_m2_rf_quick`
- `20261107_m2_stgcn_quick`
- `20261107_m3_b1`
- `20261107_m3_b2`
- `20261107_m3_b3`
- `20261107_m3_b4`
- `20261107_m3_b5`
- `20261107_rq4_mmfit_video`
