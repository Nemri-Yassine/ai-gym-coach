# RQ4 · Models 1–2 on our own recordings

Gym station (iPhone portrait 720×1280 @ 30 fps + Galaxy Watch 7 HR), two people, one day.
Labels from `data/own_manifest.csv` (note wins over the exercise button). **Test data only — no training or threshold tuning.**

Eval sets: **20** (partner 8 + yassine 12). Test sets (excluded from main numbers): **2** (yassine 2026-11-09_s01 lunges; yassine 2026-11-10_s01 squats, note=`tt`).
Exercises covered in eval: 9 of 10 MM-Fit (no lunges except the test set). All form labels = correct.
RPE ignored (almost every set typed 10). Weight typed as body weight — ignored.

## Label corrections

Wrong button pressed on 3 eval sets; the note (and folder name for 2 unsaved sets) is the truth:

| Set | Button | Note / folder | Truth |
|---|---|---|---|
| `yassine/2026-11-10_s007/set01` | squats | PUSHU¨YASSINE | **pushups** |
| `yassine/2026-11-10_PUHSUP2/set01` | — | 2026-11-10_PUHSUP2 | **pushups** |
| `yassine/2026-11-10_JUMPJACK/set01` | — | 2026-11-10_JUMPJACK | **jumping_jacks** |
| `yassine/2026-11-10_TRICEPS/set01` | squats | TRICEPS | **tricep_extensions** |
| `yassine/2026-11-10_TRICEPS/set04` | lateral_shoulder_raises | BICEP 2 | **bicep_curls** |

## Model 1 (pose only)

Primary checkpoint: `m1_transformer_pose`. Windows 90 / hop 45, majority vote excluding `non_activity`.
Set accuracy: **0.7** (14/20), Wilson 95% CI [0.48, 0.85]. Macro-F1 (9 exercises present): **0.5111111111111111**.
Window accuracy: **0.6287625418060201** (n=299).

Differences of a few sets are noise at this n.

| Person | n | Accuracy | Wilson 95% |
|---|---:|---:|---|
| yassine | 12 | 0.750 | [0.47, 0.91] |
| partner | 8 | 0.625 | [0.31, 0.86] |

| Exercise | n | Accuracy |
|---|---:|---:|
| bicep_curls | 2 | 1.000 |
| dumbbell_rows | 1 | 0.000 |
| dumbbell_shoulder_press | 4 | 1.000 |
| jumping_jacks | 2 | 1.000 |
| lateral_shoulder_raises | 1 | 0.000 |
| pushups | 4 | 1.000 |
| situps | 1 | 0.000 |
| squats | 2 | 1.000 |
| tricep_extensions | 3 | 0.000 |

### Wrong-button check

Did Model 1 agree with the note (truth) or the wrong button?

| Set | Button | Truth | M1 | Agrees with |
|---|---|---|---|---|
| `yassine/2026-11-10_s007/set01` | squats | pushups | pushups | **note/truth** |
| `yassine/2026-11-10_TRICEPS/set01` | squats | tricep_extensions | non_activity | **neither** |
| `yassine/2026-11-10_TRICEPS/set04` | lateral_shoulder_raises | bicep_curls | bicep_curls | **note/truth** |

### Comparison (same pose-only transformer, set/video accuracy)

| Source | Metric | Value | n |
|---|---|---:|---:|
| MM-Fit test_unseen (dataset pose) | macro-F1 | 0.85 | — |
| MM-Fit w20 MediaPipe | macro-F1 | 0.80 | — |
| RepCount public videos | video accuracy (4 ex.) | 0.808 | — |
| **Our videos** | set accuracy | 0.7 | 20 |
| **Our videos** | macro-F1 (9 ex.) | 0.5111111111111111 | 20 |

Confusion: `reports/figures/m1_confusion_own.png`.

## Rep counting

No peak-parameter tuning on these sets.

True exercise: MAE **3.17**, exact **22.2%**, within ±1 **50.0%** (n=18).
M1-predicted exercise: MAE **3.67**, exact **16.7%**, within ±1 **50.0%** (n=18).

| Exercise | n | MAE (true ex) |
|---|---:|---:|
| bicep_curls | 2 | 11.50 |
| dumbbell_rows | 1 | 0.00 |
| dumbbell_shoulder_press | 4 | 3.50 |
| jumping_jacks | 1 | 1.00 |
| lateral_shoulder_raises | 1 | 1.00 |
| pushups | 3 | 2.00 |
| situps | 1 | 2.00 |
| squats | 2 | 0.00 |
| tricep_extensions | 3 | 3.33 |

| Person | n | MAE (true ex) |
|---|---:|---:|
| partner | 8 | 2.88 |
| yassine | 10 | 3.40 |

Figure: `reports/figures/rep_counting_own.png`.

## Model 2 (form)

Only **correct** squats recorded (2 eval + 1 test). No lunges/plank mistakes. This measures **false alarms**, not mistake detection.
Eval squats: mean P(correct) **0.06000000000000001**, % reps classified correct **0.0** (n=20).
Compare MM-Fit RQ4 MediaPipe P(correct) on squat/lunge: **0.08 / 0.14**.

## Heart rate

| Person | sets | hr_ok | mean coverage | median coverage | mean HR rise (end−start) |
|---|---:|---:|---:|---:|---:|
| yassine | 14 | 7 | 0.50 | 0.43 | 2.3 |
| partner | 8 | 4 | 0.52 | 0.59 | -1.5 |

Limitation: the watch went silent for 20–80 s on several sets (hr_coverage 0 on many early sets).
Figure: `reports/figures/own_hr_session.png`.

## Limits

- Small n (20 eval sets), one day, two people, 9 exercises, no lunges in eval.
- Portrait phone framing and gym background differ from MM-Fit / RepCount.
- RPE not usable; HR gaps; two sets missing typed reps.
- Own data never used for training or threshold tuning in this report.

Metrics: `C:/Users/yassi/OneDrive/Bureau/projet DL BIG DATA/experiments/20261110_own_rq4/metrics.json`.
