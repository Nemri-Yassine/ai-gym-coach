# Validation 2 presentation (October 28, 2026)

**Main file: `AI_Workout_Coach_Week1.html`**: one self-contained file, same design as the first presentation, with live simulations on real data (Model 1's real predictions on an unseen person, both rep counters on the same set, Spark sync, a public YouTube video, the gym recorder walkthrough, EC3D in 3D). Double-click to open. Keys: ← → to move, F fullscreen. Rebuild with `python scripts/docs/build_progress_data.py` then `python scripts/docs/build_progress.py`.

The earlier slide-deck version is below.

The deck lives online: https://claude.ai/artifact/Rs7TWWwQcgeozhssWb7S8s
Present it from there, or export it as PowerPoint or PDF from the deck's download menu.

**One-file version:** `AI_Workout_Coach_Validation2.html`: open it in any browser. Keys: → / Space next, ← previous, F fullscreen, N speaker notes. Rebuild it after editing a slide with `python scripts/docs/build_validation2_html.py`.

This folder also keeps the slide sources:

- `deck.json`: title, slide order and sections
- `slides/*.html`: one file per slide, with speaker notes in the `<aside>` at the end

Slide order: cover, idea, questions, w1-glance, w1-spark, w1-model1, w1-reps, w1-public, w1-recorder, next, risks, end.

To fill in after the gym session: `slides/w1-recorder.html` has two placeholders, `[result of the watch test]` and `[number]` of sessions recorded.
