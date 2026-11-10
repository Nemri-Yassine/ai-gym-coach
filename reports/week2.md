# Week 2 validation package · updated 10 Nov 2026

**Presentation (HTML, offline):** [`reports/validation2/AI_Workout_Coach_Week2.html`](validation2/AI_Workout_Coach_Week2.html) · rebuild with `python scripts/docs/build_week2.py` (template + `week2_data.js`). Keys: ← →, F fullscreen.

## Own recordings (Nov 10 — closes the old “no own data” gap)

Gym station: iPhone Safari → laptop HTTPS + Galaxy Watch 7 BLE. Manifest `data/own_manifest.csv`: **20 eval + 2 test** sets, **2 people**, **9** of 10 MM-Fit exercises (no lunges in eval). Session folders: yassine **9**, partner **6** (one day, several short sessions — not multi-week fatigue yet).

RPE ignored (≈10 everywhere). Weight typed as body weight — ignored. Notes corrected 3 wrong buttons. Full RQ4: `reports/rq4_own.md`.

| Metric | Value | n |
|---|---:|---:|
| M1 set accuracy (pose-only transformer) | **0.70** (Wilson 95% [0.48, 0.85]) | 20 |
| M1 macro-F1 (9 exercises) | **0.511** | 20 |
| M1 accuracy yassine / partner | 0.750 / 0.625 | 12 / 8 |
| Rep MAE (true exercise) | **3.17** (exact 22%, ±1 50%) — **typed default, not a count** until verified | 18 |
| Rep MAE (M1 exercise) | 3.67 (typed default) | 18 |
| M2 mean P(correct) on correct squats | **0.06** (% classified correct 0%) | 20 reps |
| HR mean coverage yassine / partner (eval) | 0.54 / 0.52 (hr_ok 7/12 · 4/8) | 12 / 8 eval sets |
| Spark own rep rows | **269** (20 eval + 2 test; 4 sets `segmentation_failed`) | — |
| Live latency **cached** pose (old claim) | 0.84 / 0.03 / 0.03 s | 3 sets |
| Live latency **full MediaPipe** (honest) | mean **13.1 s** (pose ~13.1 / M1 ~0.01 / reps ~0.01 / M2 0) | 3 sets |
| Live latency **live_fast** (stride 2 + 480px) | mean **7.8 s** (pose ~7.8); exercise match 3/3 vs cache; reps Δ −1 on 1 set | 3 sets |

Honest RQ4: domain shift (portrait gym phone vs MM-Fit). Tricep extensions and situps/lateral/rows are weak; pushups/squats/shoulder press/jumping jacks/biceps mostly fine. M2 P(correct)≈0.06 on our correct squats — **live suggestion hides form** (message: not calibrated; opt-in `--show-form`). HR gaps: watch silent 20–80 s on several sets.

Live demo: after `video_saved`, background worker runs MediaPipe with live_fast, writes `data/interim/own_live/…/analysis_setNN.json`, shows suggestion on the control page (never into `sets.csv`). Metrics: `experiments/20261110_station_latency/`. **Docker Kafka daemon was not running** — station `--kafka` not re-measured. Commands: `docs/demo.md`.

## Data-source note (history)

Until Nov 10, one teammate was injured and validation used MM-Fit RGB as a stand-in. That work stays below. Own data is **test-only** here (no training / threshold tuning).

## Model 2 (EC3D, step 2.2)

Honest conclusion (also in `m2_benchmark.md`): ST-GCN squat 55.6 ± 12.4% vs 2D RF 53.8 ± 6.1% — the squat lead is within fold noise. RF wins on lunge and plank. Do not treat ST-GCN as the overall winner. Authors' 90.9% is a different setup (context only).

## Kafka (step 2.3, replay)

Broker that worked: **Docker `apache/kafka`**. KRaft binaries blocked (`wmic` missing).

| Topic | Median msg/s | Median latency |
|---|---:|---:|
| `pose.keypoints` (w20 RGB) | 29.8 | 48 ms |
| `watch.hr` (w20 `sw_l_hr`) | 0.79 | 19 ms |

Sink → `data/interim/replay/` only. Metrics: `experiments/20261107_kafka_replay/metrics.json`. Station can optionally publish with `--kafka` (logs once and continues if the broker is down). **Nov 10 check:** Docker Desktop installed but daemon not running — no new station Kafka metrics; do not claim a live station Kafka demo.

## RQ4 · MM-Fit raw video (step 2.4, w20)

| Pose source | M1 macro-F1 pose / both | Rep MAE | Sync ≤3 fr. | M2 P(correct) |
|---|---:|---:|---:|---:|
| Dataset pose | 0.85 / 0.66 | 0.93 | 8/10 | 0.08 |
| MediaPipe from RGB | 0.80 / 0.48 | 1.56 | (same map) | 0.14 |

No form labels on MM-Fit. See also four-session table in `reports/rq4_first_look.md`.

## Spark rep table (step 2.5 + 2.5b)

**MM-Fit:** 5,918 rep rows, 616 sets, 21 sessions; within-session baselines; `has_baseline` rate 0.67; HR mask forced 0 on w05/w10/w18.

**Own:** 269 rep rows, **20 eval + 2 test** sets, 2 people → `data/interim/own_rep_table.parquet`. Sets with counted reps <0.5× or >1.5× truth are `segmentation_failed` (4 sets) and excluded from velocity-loss medians. Median velocity loss at last rep (eval, non-failed): squats 0.222, pushups 0.065, tricep_extensions −1.146 (remaining sets still noisy vs typed-10 truth). Summary: `data/interim/own_rep_table_summary.json`.
