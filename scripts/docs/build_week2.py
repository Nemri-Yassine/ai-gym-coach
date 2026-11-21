"""Build the week 2 progress presentation (demo-first).

    python scripts/docs/build_week2.py

Takes reports/validation2/src/week2_template.html + week2_data.js, pulls CSS from
presentation/index.html (same look as week 1), inlines the diagnosis figure as base64.
Videos stay as relative paths under demo/week2/ (keep that folder next to the HTML).
Output: reports/validation2/AI_Workout_Coach_Week2.html
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "presentation"
SRC = ROOT / "reports" / "validation2" / "src"
OUT = ROOT / "reports" / "validation2" / "AI_Workout_Coach_Week2.html"
FIGURES = ROOT / "reports" / "figures"
DEMO = ROOT / "reports" / "validation2" / "demo" / "week2"


def between(text: str, start: str, end: str, include_start: bool = True) -> str:
    a = text.index(start)
    b = text.index(end, a + len(start))
    return text[a if include_start else a + len(start):b]


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build() -> Path:
    index = (PRES / "index.html").read_text(encoding="utf-8")
    page = (SRC / "week2_template.html").read_text(encoding="utf-8")
    data = (SRC / "week2_data.js").read_text(encoding="utf-8").strip()

    style = between(index, "<style>", "</style>", include_start=False)
    page = page.replace("/*INDEX_STYLE*/", style)
    page = page.replace("/*DATA*/", data)

    diag = FIGURES / "own_m1_diagnosis.png"
    if not diag.is_file():
        raise FileNotFoundError(diag)
    page = page.replace("/*FIG_DIAGNOSIS*/", data_uri(diag, "image/png"))

    left = re.findall(r"/\*(INDEX_\w+|DATA|FIG_\w+)\*/", page)
    if left:
        raise ValueError(f"unfilled placeholders: {left}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    if DEMO.is_dir():
        vids = sorted(DEMO.glob("*.mp4"))
        print(f"demo videos beside HTML: {len(vids)} in {DEMO.relative_to(ROOT)}")
        for v in vids:
            print(f"  {v.name}: {v.stat().st_size / 1e6:.2f} MB")
    else:
        print("WARNING: demo/week2/ missing — open the HTML from reports/validation2/ after rendering")
    return OUT


if __name__ == "__main__":
    build()
