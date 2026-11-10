# Results · RQ1–RQ4 + Big Data

Numbers below are copied from `experiments/*/metrics.json` (and the week reports).
Model 3 numbers prefer `*_v2` (Step 3.2b: fair B1, speed-only labels, 3 seeds).

## RQ1 · Recognition on unseen people

- Best Model 1: transformer pose+IMU, macro-F1 **0.880** on `test_unseen` (`reports/m1_benchmark.md`).
- Rep counting macro MAE **0.49** on 616 MM-Fit sets (`reports/rep_counting.md`).

## RQ2 · Heart rate vs movement (B4 vs B3)

- B4 (personal+HR) risk AUC **0.640992046365374**
- B3 (personal, no HR) risk AUC **0.688541217091066**
- Honest finding: HR does **not** improve risk AUC on this MM-Fit setup. See `reports/m3_results.md`.

## RQ3 · Personal vs general (B4 vs B2) and vs rule (B1)

- B2 (general+HR) AUC **0.7289578066876298** — better than B4 here.
- B5 (general, no HR) AUC **0.760** ± 0.009 (restored after quick-train overwrite; was wrongly 0.600).
- Fair B1 continuous AUC **0.7506381404525556** (alert @25% / @20% in `reports/m3_results.md`).
- Within-session personal baseline does not beat general normalisation on these labels.
- Exercise-only control AUC **0.773** ≥ learned variants — see `reports/m3_results.md`.

## RQ4 · Raw MM-Fit video

- Four sessions (w00/w12/w13/w20); mean keeps sync≥70% only (`reports/rq4_first_look.md`).
- RQ4 run keys: `['sessions']` …

## Big data / coach

- Spark sync scaling: see week1 / `experiments/*_spark_scaling` if present.
- Kafka replay: pose 29.8 msg/s @ 48 ms; HR 0.79 msg/s @ 19 ms (`experiments/20261107_kafka_replay`).
- Kafka coach: median model **0.834** ms, e2e **74.566** ms; slow@rep1 still present=True (`experiments/20261107_coach_kafka`).
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
- `20261107_coach_kafka`
- `20261107_coach_replay`
- `20261107_kafka_replay`
- `20261107_m1_cnn_pose_quick`
- `20261107_m2_rf_quick`
- `20261107_m2_stgcn_quick`
- `20261107_m3_b1`
- `20261107_m3_b1_v2`
- `20261107_m3_b2`
- `20261107_m3_b2_v2`
- `20261107_m3_b3`
- `20261107_m3_b3_v2`
- `20261107_m3_b4`
- `20261107_m3_b4_v2`
- `20261107_m3_b5`
- `20261107_m3_b5_v2`
- `20261107_rq4_mmfit_video`
