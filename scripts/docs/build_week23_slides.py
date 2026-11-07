"""Build Week 2–3 validation slides (self-contained HTML).

    python scripts/docs/build_week23_slides.py

Output: reports/validation2/AI_Workout_Coach_Week2-3.html
Keys: ← → navigate, F fullscreen.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "validation2" / "AI_Workout_Coach_Week2-3.html"

SLIDES = [
    (
        "What changed",
        "Injury → no own recordings. MM-Fit RGB (Zenodo 7672767) + watch IMU/HR replace them. "
        "Baselines are within-session (earlier sets of the same exercise). RPE dropped. "
        "Live Watch link remains untested.",
    ),
    (
        "Kafka replay",
        "Docker apache/kafka. Pose 29.8 msg/s @ 48 ms median; HR 0.79 msg/s @ 19 ms. "
        "Sink writes under data/interim/replay/ only. Setup: docs/kafka_setup.md · demo: docs/demo.md.",
    ),
    (
        "RQ4 · raw video",
        "MediaPipe from RGB vs dataset pose. Domain shift: EC3D Model 2 P(correct)≈0.08 on MM-Fit squats. "
        "Four sessions; mean keeps sync≥70% only. w20 auto sync within 1.4 frames of measured. "
        "Table: reports/rq4_first_look.md.",
    ),
    (
        "Spark rep table",
        "5,918 MM-Fit reps · 616 sets. Within-session baselines via "
        "Window.partitionBy(session, exercise).orderBy(set_idx).rowsBetween(unboundedPreceding, −1). "
        "has_baseline rate 0.67. No RPE.",
    ),
    (
        "Model 3 labels",
        "Speed-only breakdown (use_form=False). Risk = 1 from breakdown−2 to end of set. "
        "Squats breakdown rate 0.42 (was ~100% when form polluted the label). "
        "jumping_jacks rate 0.018 — reported, not hidden.",
    ),
    (
        "Model 3 results",
        "Fair B1 continuous AUC 0.751. B4 0.641 ± 0.022 — does not beat B1 or the exercise-only control (0.773). "
        "RQ2/RQ3 negative for personal+HR. Old “beats apps” claim withdrawn. reports/m3_results.md.",
    ),
    (
        "Limits",
        "No own data · no RPE · within-session baseline only · shared speed in labels · "
        "Watch untested · MM-Fit reps assumed correct · Model 2 does not transfer EC3D→MM-Fit.",
    ),
    (
        "Demo",
        "Kafka: pose_producer (--start-frame 2100) + hr_producer + live.py --kafka (docs/demo.md). "
        "Median model 0.83 ms · e2e 75 ms on w20. Offline fallback is model-only. "
        "Honest: “Slow down” still fires near rep 1 (risk ~0.40–0.45 > slow_thresh) — not the old form-polluted every-squat@rep2 pattern.",
    ),
]


def build():
    sections = []
    for i, (title, body) in enumerate(SLIDES, 1):
        sections.append(
            f'<section class="slide" id="s{i}">'
            f'<div class="num">{i} / {len(SLIDES)}</div>'
            f"<h1>{title}</h1>"
            f"<p>{body}</p>"
            f"</section>"
        )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Workout Coach · Week 2–3</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root {{ --bg:#0B1016; --fg:#E8EEF5; --muted:#8FA3B8; --accent:#3DDC97; --card:#141B24; }}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; height:100%; background:var(--bg); color:var(--fg);
  font-family:"DM Sans",system-ui,sans-serif; overflow:hidden; }}
#stage {{ position:absolute; inset:0; }}
.slide {{ position:absolute; inset:0; padding:8vh 10vw; display:none; flex-direction:column; justify-content:center;
  background: radial-gradient(1200px 600px at 10% -10%, #1a2736 0%, var(--bg) 55%); }}
.slide.on {{ display:flex; animation:in .35s ease; }}
@keyframes in {{ from {{ opacity:0; transform:translateY(12px); }} to {{ opacity:1; transform:none; }} }}
.num {{ font-family:"IBM Plex Mono",monospace; color:var(--muted); letter-spacing:.08em; margin-bottom:2vh; }}
h1 {{ font-size:clamp(2rem,5vw,3.4rem); margin:0 0 3vh; max-width:18ch; line-height:1.1; }}
p {{ font-size:clamp(1.05rem,2.1vw,1.45rem); line-height:1.55; color:var(--muted); max-width:52ch; margin:0; }}
.brand {{ position:fixed; left:10vw; bottom:4vh; font-family:"IBM Plex Mono",monospace; font-size:.85rem; color:var(--accent); }}
.hint {{ position:fixed; right:10vw; bottom:4vh; font-size:.85rem; color:var(--muted); }}
</style>
</head>
<body>
<div id="stage">
{''.join(sections)}
</div>
<div class="brand">AI Workout Coach · Week 2–3</div>
<div class="hint">← → · F fullscreen</div>
<script>
const slides=[...document.querySelectorAll('.slide')];
let i=0;
function show(n){{ i=(n+slides.length)%slides.length; slides.forEach((s,k)=>s.classList.toggle('on',k===i)); }}
show(0);
window.addEventListener('keydown',e=>{{
  if(e.key==='ArrowRight'||e.key===' ') show(i+1);
  if(e.key==='ArrowLeft') show(i-1);
  if(e.key==='f'||e.key==='F'){{
    if(!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  }}
}});
</script>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
