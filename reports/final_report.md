# AI Workout Coach · Final report

**Course:** Deep Learning & Big Data · 2026–2027  
**Team:** Yassine Nemri, Iheb Lamouchi  
**Date:** 7 Nov 2026 · tag v1.1

## Objective

Build a coach that recognizes the exercise, counts reps, checks form, and warns before fatigue becomes risky — using a phone camera and a smartwatch, with a **personal baseline** as the main contribution.

## Research questions

1. **RQ1:** Public-data models on unseen people?  
2. **RQ2:** Does heart rate beat movement alone for fatigue risk?  
3. **RQ3:** Does a personal baseline warn earlier than a general model?  
4. **RQ4:** Do the models still work on raw RGB (phone-like) video?

## Context and existing apps

Commercial apps (e.g. velocity-based training) often stop a set when bar speed drops by a fixed fraction (≈25%). Our B1 rule reproduces that idea on skeleton speed. The fair comparison uses a **continuous** speed-loss score for AUC, not a 0/1 step that was mismatched to the risk window.

## Data

- **MM-Fit:** 21 sessions, pose + watch IMU + HR (missing in w05/w10/w18).  
- **EC3D:** form mistakes for Model 2.  
- **Own recordings:** none (injury). Replaced by MM-Fit RGB (Zenodo 7672767) + **within-session** baselines. No RPE.

## Approach

Kafka (replay) → MediaPipe / dataset pose → Model 1 → rep counter → Model 2 (form RF, input only) → Spark rep history → Model 3 (GRU, B1–B5) → coach policy. Breakdown labels are **speed-only** after v1.1 (Model 2 does not define the label).

## Results

| RQ | Finding (v1.1) |
|---|---|
| RQ1 | Transformer pose+IMU macro-F1 **0.880** unseen; rep MAE **0.49** |
| RQ2 | B4 **0.641 ± 0.022** < B3 **0.689** — HR does **not** help |
| RQ3 | B2 **0.729** > B4 **0.641** — within-session personal does **not** beat general |
| Existing apps | Fair B1 AUC **0.751** > B4 **0.641** — “beats apps” claim **withdrawn** |
| Control | Exercise-only logistic AUC **0.773** ≥ all learned variants |
| RQ4 | Four RGB sessions processed; mean uses sync≥70% only (w20). M1 pose 0.85→0.80; rep MAE 0.93→1.37; Model 2 P(correct)≈0.08 (domain shift). Auto sync on w20 within **1.4** frames of measured. |
| Spark | 5,918 rep rows; within-session baselines |
| Kafka coach | Two w20 demos: median model **0.83 ms**, e2e **75 ms**. “Slow down” near rep 1 still present. |

Details: `reports/m3_results.md`, `reports/rq4_first_look.md`, `reports/results.md`.

## Limits

- No own recordings (injury); no multi-week personal history  
- No RPE; Spearman uses `set_idx_in_exercise` (sign negative; not explained by shorter later sets)  
- Within-session baseline only  
- Breakdown labels share the speed signal with the inputs (and with B1)  
- Live Galaxy Watch link not tested  
- MM-Fit rep counts assumed correct (teacher’s question — open)  
- Model 2 does not transfer from EC3D to MM-Fit  
- Youden-tuned alert thresholds give high false-alarm rates; use the 0.7 column for a coach operating point  

## References

- Strömbäck et al., MM-Fit, IMWUT 2020.  
- Zhao et al., EC3D form feedback.  
- Dritsas 2025 (Spark scaling context for the course).  
