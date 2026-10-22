"""Build reports/Implementation_Plan.pdf from docs/PLAN.md (single source of truth).
Run from the project root:  python scripts/docs/build_plan_pdf.py
Needs `markdown`, `pymdown-extensions` and Google Chrome."""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "PLAN.md"
HTML = ROOT / "reports" / "plan" / "plan.html"
PDF = ROOT / "reports" / "Implementation_Plan.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
:root{--ink:#101A23;--muted:#56646F;--line:#D5DCE3;--sunk:#EEF1F4;--accent:#1E4BD2;--accent-soft:#E3E9FC;
--ok:#1F7F52;--ok-soft:#DFF2E7;--warn:#B06F0C;--warn-soft:#FBEFD9;--w1:#1E4BD2;--w2:#1F7F52;--w3:#7A4FD0}
@page{size:A4;margin:14mm 14mm 16mm}
*{-webkit-print-color-adjust:exact;print-color-adjust:exact;box-sizing:border-box}
body{font:10pt/1.5 "Public Sans","Segoe UI",Arial,sans-serif;color:var(--ink);margin:0}
h1,h2,h3,h4{font-family:"Archivo","Arial Narrow",Arial,sans-serif;text-transform:uppercase;line-height:1;margin:0}
h1{font-stretch:66%;font-weight:900;font-size:34pt;margin:2mm 0 4mm}
h2{font-stretch:72%;font-weight:900;font-size:22pt;color:#fff;background:var(--ink);padding:5mm 6mm;border-radius:3mm;margin:0 0 5mm;break-before:page}
h2.w1{background:var(--w1)} h2.w2{background:var(--w2)} h2.w3{background:var(--w3)}
h3{font-stretch:76%;font-weight:850;font-size:14pt;margin:7mm 0 3mm;padding:3mm 4mm;border-left:2mm solid var(--accent);background:var(--accent-soft);border-radius:0 2mm 2mm 0;break-after:avoid}
h3 .tag{font:700 7.5pt "IBM Plex Mono",Consolas,monospace;padding:1mm 2mm;border-radius:1.5mm;margin-left:2mm;vertical-align:2pt;color:#fff}
h3 .must{background:var(--accent)} h3 .should{background:var(--warn)}
h4{font-size:10pt;letter-spacing:.04em;margin:4mm 0 1.5mm;color:var(--muted)}
p{margin:1.5mm 0}
p strong:first-child{color:var(--accent)}
ul,ol{margin:1mm 0 2mm;padding-left:6mm}
li{margin:.8mm 0}
code{font:8.8pt "IBM Plex Mono",Consolas,monospace;background:var(--sunk);padding:.2mm 1.2mm;border-radius:1mm}
pre{background:#0F1720;color:#E6EDF3;border-radius:2mm;padding:3mm 4mm;font-size:8.4pt;line-height:1.45;white-space:pre-wrap;break-inside:avoid}
pre code{background:none;color:inherit;padding:0}
table{border-collapse:collapse;width:100%;margin:2mm 0 3mm;font-size:9pt;break-inside:avoid}
th{background:var(--sunk);text-align:left;font:600 8pt "IBM Plex Mono",Consolas,monospace;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
th,td{border-bottom:1px solid var(--line);padding:1.8mm 2.4mm;vertical-align:top}
blockquote{margin:3mm 0;padding:3mm 4mm;background:var(--warn-soft);border-left:2mm solid var(--warn);border-radius:0 2mm 2mm 0}
blockquote p{margin:0}
hr{border:0;border-top:1px solid var(--line);margin:5mm 0}
li.check{list-style:none;margin-left:-5mm}
li.check::before{content:"";display:inline-block;width:3.4mm;height:3.4mm;border:1.3pt solid var(--ok);border-radius:.8mm;margin-right:2mm;vertical-align:-.5mm;background:#fff}
.cover{height:265mm;display:flex;flex-direction:column;gap:6mm}
.cover .brand{font:900 13pt "Archivo",Arial;font-stretch:70%;text-transform:uppercase}
.cover .brand b{color:var(--accent)}
.cover .lede{font-size:13pt;color:var(--muted);max-width:160mm}
.weeks{display:grid;grid-template-columns:repeat(3,1fr);gap:4mm;margin-top:6mm}
.wk{border-radius:3mm;padding:5mm;color:#fff;display:flex;flex-direction:column;gap:2mm}
.wk b{font:900 20pt/1 "Archivo",Arial;font-stretch:70%;text-transform:uppercase}
.wk span{font-size:10pt;opacity:.92}
.wk small{font:600 8pt "IBM Plex Mono",monospace;opacity:.85}
.howto{margin-top:auto;background:var(--sunk);border-radius:3mm;padding:5mm;font-size:10pt}
"""

COVER = """
<div class="cover">
  <div class="brand">AI Workout <b>Coach</b></div>
  <h1>Implementation plan</h1>
  <p class="lede">Three weeks, one validation per week. Every step is small, tested on its own and confirmed before the next one starts. Written to be followed step by step in Cursor.</p>
  <div class="weeks">
    <div class="wk" style="background:#1E4BD2"><small>Week 1 · steps 1.1–1.6</small><b>Data pipeline + Model 1</b><span>Environment, capture decision, recording starts, Spark sync to Parquet, unified skeleton, Model 1 benchmark, rep counting.</span></div>
    <div class="wk" style="background:#1F7F52"><small>Week 2 · steps 2.1–2.6</small><b>Model 2 + live capture</b><span>EC3D rep dataset, ST-GCN vs baselines, Kafka streaming of pose and heart rate, our videos through Models 1–2, Spark history job.</span></div>
    <div class="wk" style="background:#7A4FD0"><small>Week 3 · steps 3.1–3.5</small><b>Model 3 + coach + results</b><span>Fatigue and risk labels, Model 3 vs fixed-threshold rule (RQ2, RQ3), live coach with voice, RQ1–RQ4 evaluation, final report.</span></div>
  </div>
  <div class="howto"><b>How to use it.</b> Give Cursor <code>docs/PLAN.md</code> (this document's source) and the rules file <code>.cursor/rules/awc.mdc</code>. Ask for one step at a time: "Do step 1.3". Each step ends with tests, a README update and a commit. Steps marked <b>MUST</b> are required; <b>SHOULD</b> can be cut if only two weeks remain (section 4).</div>
</div>
"""


def build():
    md = SRC.read_text(encoding="utf-8")
    md = md.split("\n---\n", 1)[1]                     # the cover replaces the title block
    html = markdown.markdown(md, extensions=["tables", "pymdownx.superfences", "sane_lists"])
    html = re.sub(r"<li>\[ \]\s*", '<li class="check">', html)
    for i, cls in ((1, "w1"), (2, "w2"), (3, "w3")):
        html = html.replace(f"<h2>Week {i}", f'<h2 class="{cls}">Week {i}')
    html = re.sub(r"(<h3>[^<]*?)· MUST([^<]*)</h3>", r'\1<span class="tag must">MUST</span>\2</h3>', html)
    html = re.sub(r"(<h3>[^<]*?)· SHOULD([^<]*)</h3>", r'\1<span class="tag should">SHOULD</span>\2</h3>', html)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>AI Workout Coach · Implementation plan</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style></head><body>{COVER}{html}</body></html>"""
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(page, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True)
    print("wrote", PDF)


if __name__ == "__main__":
    build()
