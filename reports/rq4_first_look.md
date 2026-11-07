# RQ4 first look · MM-Fit raw video vs dataset pose

Own recordings were not available (injury). Comparison uses MM-Fit RGB (Zenodo 7672767)
→ MediaPipe → same Model 1 / rep counter / Model 2 RF as the dataset pose pipeline.

**No form labels** on MM-Fit: Model 2 column is mean P(correct) only, not accuracy.

| Session | Sync within 3 fr. | M1 macro-F1 dataset (pose / both) | M1 macro-F1 MediaPipe (pose / both) | Rep MAE dataset | Rep MAE MediaPipe | M2 P(correct) ds / mp |
|---|---:|---:|---:|---:|---:|---:|
| w20 | 8/10 (0.80) | 0.850 / 0.660 | 0.800 / 0.480 | 0.93 | 1.56 | 0.08 / 0.14 |

Metrics: `experiments/20261107_rq4_mmfit_video/metrics.json`. Extra `test_unseen` videos (w00, w12, w13) were started from Zenodo 7672767 but not finished in time; this table is **w20 only**.
