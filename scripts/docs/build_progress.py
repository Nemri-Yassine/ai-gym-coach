"""Build the week 1 progress presentation as ONE self-contained HTML file.

    python scripts/docs/build_progress_data.py   # real predictions and rep comparisons (once)
    python scripts/docs/build_progress.py

Run scripts/demo_video.py first for the annotated real-video demo (embedded in the page).
Takes reports/validation2/src/week1_template.html and fills it with, from presentation/index.html
(read only: the main presentation is never modified),
the styles, the "idea" and "how it works" slides and the drawing code (so both presentations
look and behave the same), then inlines the data files. Output:
reports/validation2/AI_Workout_Coach_Week1.html
"""
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "presentation"                       # main presentation: only read, never changed
SRC = ROOT / "reports" / "validation2" / "src"     # week 1 template and data
OUT = ROOT / "reports" / "validation2" / "AI_Workout_Coach_Week1.html"
GUIDE = ROOT / "reports" / "validation2" / "AI_Workout_Coach_Week1_Guide.html"   # technical notes for us
DEMO_DIR = ROOT / "reports" / "validation2" / "demo"               # made by scripts/demo_video.py
# one typical public video per exercise (rep error at or below the median), in tab order
DEMOS = [("stu9_64", "squats", "The extra rep comes from the static hold at the end of the video."),
         ("stu8_46", "pushups", ""),
         ("stu3_53", "situps", "The person stands up at the end; the result keeps the exercise done the longest."),
         ("stu6_24", "jumping_jacks", "")]


def between(text, start, end, include_start=True):
    a = text.index(start)
    b = text.index(end, a + len(start))
    return text[a if include_start else a + len(start):b]


def section(index, sid):
    m = re.search(rf'<section class="step" id="{sid}".*?</section>', index, re.S)
    if not m:
        raise ValueError(f"{sid} not found in index.html")
    return m.group(0)


def data_js(path):
    return path.read_text(encoding="utf-8").strip()


def build():
    index = (PRES / "index.html").read_text(encoding="utf-8")
    page = (SRC / "week1_template.html").read_text(encoding="utf-8")

    style = between(index, "<style>", "</style>", include_start=False)
    lib = between(index, "const EX = ", "/* ---------------- step inits")
    lib3d = between(index, "/* ---------------- EC3D: 3D skeleton viewer", "function initEC3D(")

    page = page.replace("/*INDEX_STYLE*/", style)
    page = page.replace("<!--SECTION:s-idea-->", section(index, "s-idea"))
    page = page.replace("<!--SECTION:s-how-->", section(index, "s-how"))
    page = page.replace("/*INDEX_LIB*/", lib)
    page = page.replace("/*INDEX_3D*/", lib3d)
    demos = []
    for vid, label, note in DEMOS:
        summary = json.loads((DEMO_DIR / f"{vid}.json").read_text(encoding="utf-8"))
        summary.pop("timeline", None)
        video = DEMO_DIR / summary.get("file", f"{vid}.mp4")
        mime = "video/mp4" if video.suffix == ".mp4" else "video/webm"
        demos.append({"label": label, "note": note, "summary": summary,
                      "src": f"data:{mime};base64," + base64.b64encode(video.read_bytes()).decode()})
    page = page.replace("/*DEMOS*/", json.dumps(demos))
    page = page.replace("/*DATA*/", "\n".join(data_js(f) for f in (PRES / "mmfit_data.js", PRES / "ec3d_data.js", SRC / "week1_data.js")))
    left = re.findall(r"/\*(INDEX_\w+|DATA|DEMOS)\*/|<!--SECTION:[\w-]+-->", page)
    if left:
        raise ValueError(f"unfilled placeholders: {left}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    build_guide(demos)


NAMES = {"squats": "Squats", "pushups": "Push-ups", "situps": "Sit-ups", "jumping_jacks": "Jumping jacks"}


def build_guide(demos):
    """Technical guide: same numbers as the presentation; the video table comes from the demo JSON files."""
    rows = []
    for d in demos:
        r = d["summary"]
        rows.append(f"<tr><td>{r['video']} · {NAMES.get(d['label'], d['label'])}</td><td>{NAMES.get(r['exercise'], r['exercise'])}</td>"
                    f"<td class=\"n\">{r['reps_counted']} / {r['truth_reps']}</td><td class=\"n\">{round(r['processing_fps_mean'])} fps</td></tr>")
    page = (SRC / "guide_template.html").read_text(encoding="utf-8").replace("/*DEMO_ROWS*/", "\n    ".join(rows))
    GUIDE.write_text(page, encoding="utf-8")
    print(f"wrote {GUIDE} ({GUIDE.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
