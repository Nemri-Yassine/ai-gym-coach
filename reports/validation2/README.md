# Validation 2 presentation (October 28, 2026)

**Two files for Wednesday:**

- `AI_Workout_Coach_Week1.html`: the presentation. One self-contained file, same design as the first presentation. Its headline slide plays the full pipeline, frame by frame, on four public gym videos never seen in training (squats 15/14 reps, push-ups 20/19, sit-ups 12/11, jumping jacks 26/26, about 60–70 fps on CPU). Keys: ← → to move, F fullscreen.
- `AI_Workout_Coach_Week1_Guide.html`: our technical notes: the flow, every input and output, each parameter and why we chose it, diagrams and a glossary.

Rebuild: `python scripts/demo_video.py <video> --truth-reps N` for a demo video, then `python scripts/docs/build_progress.py` (builds both files).

The earlier slide-deck version is below.

The deck lives online: https://claude.ai/artifact/Rs7TWWwQcgeozhssWb7S8s
Present it from there, or export it as PowerPoint or PDF from the deck's download menu.

**One-file version:** `AI_Workout_Coach_Validation2.html`: open it in any browser. Keys: → / Space next, ← previous, F fullscreen, N speaker notes. Rebuild it after editing a slide with `python scripts/docs/build_validation2_html.py`.

This folder also keeps the slide sources:

- `deck.json`: title, slide order and sections
- `slides/*.html`: one file per slide, with speaker notes in the `<aside>` at the end

Slide order: cover, idea, questions, w1-glance, w1-spark, w1-model1, w1-reps, w1-public, w1-recorder, next, risks, end.

To fill in after the gym session: `slides/w1-recorder.html` has two placeholders, `[result of the watch test]` and `[number]` of sessions recorded.
