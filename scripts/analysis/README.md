# Analysis scripts

Run from the project root. Each one reads `data/raw/` and writes a data file used by the presentation.

| Script | What it measures | Output |
|---|---|---|
| `extract_mmfit.py` | MM-Fit stats, rep timing, rep-count check, heart-rate spread | `presentation/mmfit_data.js` |
| `extract_ec3d.py` | EC3D stats, one measure per mistake, leave-one-person-out baseline | `presentation/ec3d_data.js` |
| `angle_experiment.py` | Form recognition vs simulated camera angle | `presentation/angle_data.js` |
