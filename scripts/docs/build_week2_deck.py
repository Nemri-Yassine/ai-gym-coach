"""Build the Week 2 presentation: reports/validation2/AI_Workout_Coach_Week2.html.

    python scripts/docs/build_week2_deck.py

Figures are inlined as base64; demo videos stay as files in reports/validation2/demo/week2/
(open the HTML from its folder, keep demo/ next to it). Re-render the videos with
scripts/docs/render_week2_demos.py.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "validation2" / "src" / "week2_deck_template.html"
OUT = ROOT / "reports" / "validation2" / "AI_Workout_Coach_Week2.html"


def inline(match: re.Match) -> str:
    path = ROOT / match.group(1)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    html = SRC.read_text(encoding="utf-8")
    html = re.sub(r"\{\{IMG:([^}]+)\}\}", inline, html)
    OUT.write_text(html, encoding="utf-8")
    vids = sorted(set(re.findall(r'(demo/week2/[\w.-]+\.mp4)', html)))
    missing = [v for v in vids if not (OUT.parent / v).is_file()]
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), {len(vids)} videos referenced")
    if missing:
        print("MISSING videos:", ", ".join(missing))


if __name__ == "__main__":
    main()
