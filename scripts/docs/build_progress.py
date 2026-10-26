"""Build the week 1 progress presentation as ONE self-contained HTML file.

    python scripts/docs/build_progress_data.py   # real predictions and rep comparisons (once)
    python scripts/docs/build_progress.py

Takes presentation/progress_template.html and fills it with, from presentation/index.html,
the styles, the "idea" and "how it works" slides and the drawing code (so both presentations
look and behave the same), then inlines the data files. Output:
reports/validation2/AI_Workout_Coach_Week1.html
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "presentation"
OUT = ROOT / "reports" / "validation2" / "AI_Workout_Coach_Week1.html"


def between(text, start, end, include_start=True):
    a = text.index(start)
    b = text.index(end, a + len(start))
    return text[a if include_start else a + len(start):b]


def section(index, sid):
    m = re.search(rf'<section class="step" id="{sid}".*?</section>', index, re.S)
    if not m:
        raise ValueError(f"{sid} not found in index.html")
    return m.group(0)


def data_js(name):
    return (PRES / name).read_text(encoding="utf-8").strip()


def build():
    index = (PRES / "index.html").read_text(encoding="utf-8")
    page = (PRES / "progress_template.html").read_text(encoding="utf-8")

    style = between(index, "<style>", "</style>", include_start=False)
    lib = between(index, "const EX = ", "/* ---------------- step inits")
    lib3d = between(index, "/* ---------------- EC3D: 3D skeleton viewer", "function initEC3D(")

    page = page.replace("/*INDEX_STYLE*/", style)
    page = page.replace("<!--SECTION:s-idea-->", section(index, "s-idea"))
    page = page.replace("<!--SECTION:s-how-->", section(index, "s-how"))
    page = page.replace("/*INDEX_LIB*/", lib)
    page = page.replace("/*INDEX_3D*/", lib3d)
    page = page.replace("/*DATA*/", "\n".join(data_js(n) for n in ("mmfit_data.js", "ec3d_data.js", "progress_data.js")))
    left = re.findall(r"/\*(INDEX_\w+|DATA)\*/|<!--SECTION:[\w-]+-->", page)
    if left:
        raise ValueError(f"unfilled placeholders: {left}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
