# Public videos: Model 1 and rep counting on unseen cameras

RepCount-A test split (TransRAC, CVPR 2022): YouTube gym videos with a rep count per video.
Skeleton from MediaPipe Pose Landmarker (full), mapped to COCO-18, resampled to 30 fps.
Pose only: these videos have no watch data. Models trained with IMU get a zero IMU and mask 0.
Only the span from the first labelled rep to the last is used.

Videos considered: 136. Skipped: 1.

## Model 1 (squats, pushups, situps, jumping jacks)

| Checkpoint | Videos | Video accuracy | Windows | Window accuracy |
|---|---:|---:|---:|---:|
| m1_transformer_both | 78 | 67.9% | 1339 | 54.1% |
| m1_transformer_pose | 78 | 80.8% | 1339 | 71.8% |
| m1_cnn_both | 78 | 71.8% | 1339 | 56.9% |

Per exercise, `m1_transformer_pose` (video accuracy):

| Exercise | Videos | Video accuracy |
|---|---:|---:|
| squats | 16 | 62.5% |
| pushups | 16 | 81.2% |
| situps | 20 | 85.0% |
| jumping_jacks | 26 | 88.5% |

Most common window predictions: situps 374, jumping_jacks 374, pushups 222, non_activity 182, squats 111, tricep_extensions 41.

## Rep counting

Macro MAE on the four MM-Fit exercises: 1.95. Over every exercise: 2.42.

| Exercise | Videos | Exact | Within ±1 | MAE |
|---|---:|---:|---:|---:|
| bench_press | 19 | 10.5% | 36.8% | 6.42 |
| front_raise | 17 | 52.9% | 82.4% | 1.12 |
| jumping_jacks | 26 | 30.8% | 84.6% | 3.23 |
| pull_up | 19 | 42.1% | 73.7% | 1.58 |
| pushups | 16 | 43.8% | 75.0% | 1.81 |
| situps | 20 | 25.0% | 70.0% | 2.05 |
| squats | 18 | 66.7% | 83.3% | 0.72 |

Skipped videos:

- `test534`: person found in 0% of frames
