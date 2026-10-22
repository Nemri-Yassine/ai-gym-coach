# AI Workout Coach

**Most apps tell you what you did. Ours tells you when to stop.**

An AI coach that watches you train with a **phone camera** and a **smartwatch** (Galaxy Watch 7). It recognizes the exercise, counts reps, checks the form of every rep, and compares today's session with **your own past sessions** to say *stop* before fatigue turns into injury.

Course project · Deep Learning & Big Data · 2026–2027

---

## Progress

The step-by-step work plan (3 weeks, one validation per week) is in [docs/PLAN.md](docs/PLAN.md), with a designed PDF in `reports/Implementation_Plan.pdf`. Cursor follows it with the rules in `.cursor/rules/awc.mdc`.

| Step | What | Status |
|---|---|---|
| 0 | Project proposal and validation presentation | ✅ Done · Oct 7, 2026 |
| 1 | Project structure, git, dataset loaders | ✅ Done · Oct 7, 2026 |
| 2 | Data pipeline: sync streams, normalize skeletons, cut windows (Spark → Parquet) | ⏳ Next |
| 3 | Model 1: exercise recognition + rep counting (MM-Fit) | ⬜ |
| 4 | Model 2: form check, good rep or which mistake (EC3D) | ⬜ |
| 5 | Our own recordings: phone + Galaxy Watch 7 capture, Kafka streaming | ⬜ |
| 6 | Model 3: fatigue and risk with personal baseline (our contribution) | ⬜ |
| 7 | Live coach: models running on live streams, voice feedback | ⬜ |
| 8 | Evaluation and benchmarks (RQ1–RQ4) | ⬜ |
| 9 | Final report and demo | ⬜ |

### Log

- **2026-10-22 · Planning.** Three-week implementation plan with tests and acceptance checks per step (`docs/PLAN.md`, rebuild the PDF with `python scripts/docs/build_plan_pdf.py`).
- **2026-10-07 · Step 1.** Clean project structure, `awc` Python package, MM-Fit and EC3D loaders, data sanity tests, analysis scripts moved to `scripts/analysis/`; one-page overview in `reports/Project_Structure.pdf`.
- **2026-10-07 · Step 0.** Validation presentation (`presentation/index.html`) and briefing (`reports/AI_Workout_Coach_Briefing.pdf`). First measurements on the real data, see *Results so far*.

---

## How it works

```
Phone camera ──► skeleton (18 joints, MediaPipe)  ─┐
Galaxy Watch 7 ─► heart rate + wrist motion  ───────┤  Kafka (live)
                                                    ▼
          Model 1: which exercise, where each rep starts/ends
                                                    ▼
          Model 2: is this rep good, or which mistake?
                                                    ▼
          Model 3: fatigue / injury risk vs your own history   ◄── Spark (session history)
                                                    ▼
               Voice: "Keep going" · "Slow down" · "Stop"
```

## Research questions

1. **RQ1**: Does a model trained on public data recognize exercises and count reps on people it has never seen?
2. **RQ2**: Does the watch's heart rate detect fatigue better than movement alone?
3. **RQ3**: Does a personal baseline (your own past sessions) warn earlier than a general model? *(main contribution)*
4. **RQ4**: Do the models still work on our own phone and watch?

## Data

| Dataset | Job | Size |
|---|---|---|
| [MM-Fit](https://mmfit.github.io) | Exercise, reps, heart rate | 21 sessions, 6,160 reps, 70.8 M sensor rows |
| [EC3D](https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises) | Good vs bad reps | 371 reps, 29,789 3D frames, 8 mistakes |
| Our recordings | Fatigue over weeks | 2 people, 3–6 weeks (to record) |

Download instructions: [data/README.md](data/README.md).

## Project structure

```
.
├── README.md              this file, updated at every step
├── requirements.txt       Python dependencies
├── pyproject.toml         makes src/awc installable, configures pytest
├── data/
│   ├── raw/               original datasets, never modified (not in git)
│   │   ├── mm-fit/        w00 … w20
│   │   ├── ec3d/          data_3D.pickle, data.pickle
│   │   └── own_recordings/
│   ├── interim/           synced + normalized streams (Parquet)
│   └── processed/         model-ready windows and rep tensors
├── src/awc/               the project's Python package
│   ├── config.py          all paths and constants
│   ├── data/              loaders: mmfit.py, ec3d.py, recordings.py
│   ├── features/          sync, skeleton normalization, windows, rep segmentation
│   ├── models/            m1_recognition.py, m2_form.py, m3_fatigue.py
│   ├── training/          training loops
│   ├── evaluation/        metrics and benchmarks
│   ├── streaming/         Kafka producers and live consumer
│   ├── bigdata/           Spark batch jobs
│   └── coach/             live feedback (voice)
├── scripts/analysis/      one-off analyses (dataset stats, baselines, camera angle)
├── notebooks/             exploration notebooks
├── experiments/           one folder per experiment: config + metrics
├── models/                trained checkpoints (not in git)
├── tests/                 sanity checks on data and code
├── reports/               briefing PDF, figures, final report
├── presentation/          validation web app (open index.html)
└── docs/                  course documents and notes
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .                # makes `import awc` work everywhere
python -m pytest                # checks that the datasets are in place
```

## Results so far

First measurements (quick tests, not final results):

- **Fatigue is visible**: in 45 of 64 MM-Fit squat sets, the last reps are slower than the first ones.
- **Rep counting from the skeleton**: exact in 63/64 squat sets and 62/62 lunge sets.
- **Form mistakes on an unseen person** (random forest, train on 3 people, test on the 4th): squat 53 %, lunge 82 %, plank 90 % (chance 20–33 %). EC3D authors: 90.9 %.
- **Camera angle**: squat mistakes are best recognized at about 45° in front (71 %), vs ~55 % from the front or the side.

Reproduce with the scripts in `scripts/analysis/`.

## Tools

All free: Python, PyTorch, MediaPipe, Apache Kafka, Apache Spark, pandas, scikit-learn.
