# Rep counting

Sets evaluated: 616. Macro MAE (mean of per-exercise MAEs): 0.488.
Micro MAE (every set equal): 0.490.

The signal depends on the exercise (given by Model 1): ankle height vs hips for squats, lunges and pushups; each wrist vs its shoulder for curls, rows and lateral raises (alternating arms counted per arm); first principal component otherwise. The choice per exercise was made on these same 616 sets, so the numbers are slightly optimistic. Jumping jacks were a tie here and use the principal component, which held up better on the RepCount videos.

| Exercise | Sets | Exact | Within ±1 | MAE |
|---|---:|---:|---:|---:|
| bicep_curls | 59 | 66.1% | 94.9% | 0.458 |
| dumbbell_rows | 64 | 51.6% | 100.0% | 0.484 |
| dumbbell_shoulder_press | 60 | 41.7% | 98.3% | 0.700 |
| jumping_jacks | 57 | 56.1% | 100.0% | 0.439 |
| lateral_shoulder_raises | 56 | 58.9% | 98.2% | 0.429 |
| lunges | 62 | 98.4% | 100.0% | 0.016 |
| pushups | 65 | 29.2% | 100.0% | 0.708 |
| situps | 65 | 36.9% | 93.8% | 0.908 |
| squats | 64 | 81.2% | 100.0% | 0.188 |
| tricep_extensions | 64 | 45.3% | 100.0% | 0.547 |
