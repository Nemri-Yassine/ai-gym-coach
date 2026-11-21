"""Build reports/AI_Workout_Coach_Final_Report.pdf from reports/final_report.md.

    python scripts/docs/build_final_report_pdf.py

Same Chrome-print method as build_plan_pdf.py; styling inspired by reports/briefing/.
"""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "final_report.md"
HTML = ROOT / "reports" / "final_report" / "final_report.html"
PDF = ROOT / "reports" / "AI_Workout_Coach_Final_Report.pdf"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

CSS = """
:root{--ink:#101A23;--muted:#56646F;--line:#D5DCE3;--sunk:#EEF1F4;--accent:#1E4BD2}
@page{size:A4;margin:16mm}
*{print-color-adjust:exact;-webkit-print-color-adjust:exact;box-sizing:border-box}
body{font:11pt/1.55 "Segoe UI",Arial,sans-serif;color:var(--ink);margin:0;max-width:180mm}
h1{font:900 26pt/1.1 Arial,sans-serif;margin:0 0 4mm}
h2{font:800 14pt/1.2 Arial,sans-serif;margin:8mm 0 3mm;padding:2.5mm 3mm;background:var(--ink);color:#fff;border-radius:2mm}
p,li{margin:1.5mm 0}
table{border-collapse:collapse;width:100%;margin:3mm 0;font-size:9.5pt}
th,td{border-bottom:1px solid var(--line);padding:2mm 2.5mm;text-align:left;vertical-align:top}
th{background:var(--sunk);font-size:8.5pt;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
code{font:9pt Consolas,monospace;background:var(--sunk);padding:.2mm 1mm;border-radius:1mm}
ul{padding-left:5mm}
.lede{color:var(--muted);font-size:12pt}
"""


def build():
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    # drop the top H1 duplicate styling slightly
    page = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>AI Workout Coach · Final report</title>
<style>{CSS}</style></head><body>{body}</body></html>"""
    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(page, encoding="utf-8")
    if not Path(CHROME).is_file():
        print(f"Chrome not found at {CHROME}; wrote HTML only: {HTML}")
        return HTML
    cmd = [
        CHROME, "--headless", "--disable-gpu",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {PDF} ({PDF.stat().st_size // 1024} KB)")
    return PDF


if __name__ == "__main__":
    build()
