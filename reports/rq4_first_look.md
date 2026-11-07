# RQ4 first look · MM-Fit raw video vs dataset pose

Own recordings were not available (injury). Comparison uses MM-Fit RGB (Zenodo 7672767)
→ MediaPipe → same Model 1 / rep counter / Model 2 RF as the dataset pose pipeline.

**No form labels** on MM-Fit: Model 2 column is mean P(correct) only, not accuracy.

| Session | Sync within 3 fr. | M1 macro-F1 dataset (pose / both) | M1 macro-F1 MediaPipe (pose / both) | Rep MAE dataset | Rep MAE MediaPipe | M2 P(correct) ds / mp |
|---|---:|---:|---:|---:|---:|---:|
| w00 † | 0/12 (0.00) | 1.000 / 0.800 | 0.891 / 0.818 | 0.53 | 0.87 | 0.39 / 0.27 |
| w12 † | 2/12 (0.17) | 0.778 / 0.778 | 0.778 / 0.778 | 0.32 | 1.00 | 0.25 / 0.28 |
| w13 † | 2/12 (0.17) | 1.000 / 0.827 | 0.891 / 0.800 | 0.73 | 0.63 | 0.24 / 0.28 |
| w20 | 10/12 (0.83) | 0.850 / 0.660 | 0.800 / 0.480 | 0.93 | 1.37 | 0.08 / 0.13 |
| **mean** (sync≥70%) | — | 0.850 / 0.660 | 0.800 / 0.480 | 0.93 | 1.37 | 0.08 / 0.13 |

† sync < 70% of sets within 3 frames — shown but excluded from the mean.

**Domain shift (RQ4 finding, not a bug):** the EC3D-trained Model 2 random forest does **not** carry over to MM-Fit. Even on dataset pose, mean P(correct) on squats is about **0.08**. Different skeleton source, camera and people.

**Auto sync validation (w20):** fitted `source=auto` vs config measured map — mean |Δ| **1.42** frames on sampled video frames (frac_within_3=0.8333333333333334).

Metrics: `experiments/20261107_rq4_mmfit_video/metrics.json`.
