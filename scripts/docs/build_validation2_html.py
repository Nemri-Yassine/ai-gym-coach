"""Build one self-contained HTML presentation from reports/validation2/ (deck.json + slides/*.html).

    python scripts/docs/build_validation2_html.py

Output: reports/validation2/AI_Workout_Coach_Validation2.html
Keys: → / Space next, ← previous, Home / End, F fullscreen, N speaker notes. Click right/left half to move.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "reports" / "validation2"
OUT = SRC / "AI_Workout_Coach_Validation2.html"

# Inline SVG versions of the deck's icons (stroke icons, 24x24 grid, colour from currentColor).
ICONS = {
    "Database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "Users": '<circle cx="9" cy="8" r="4"/><path d="M2 21c0-4 3-6 7-6s7 2 7 6"/><circle cx="17" cy="9" r="3"/><path d="M17 15c3 0 5 2 5 5"/>',
    "Chart": '<path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/>',
    "CheckCircle": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
    "Globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c2.5 2.5 3.8 5.5 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.5-3.8-9S9.5 5.5 12 3z"/>',
    "Lightning": '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
    "Play": '<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5-6-3.5z"/>',
    "Activity": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "Clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "Check": '<path d="M5 12.5l4.5 4.5L19 7"/>',
}


def icon_svg(match):
    name = re.search(r'name="([^"]+)"', match.group(0)).group(1)
    style = re.search(r'style="([^"]*)"', match.group(0))
    body = ICONS.get(name, '<circle cx="12" cy="12" r="9"/>')
    return (f'<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" style="flex:none;{style.group(1) if style else ""}" '
            f'aria-hidden="true">{body}</svg>')


def build():
    deck = json.loads((SRC / "deck.json").read_text(encoding="utf-8"))
    slides = []
    for sid in deck["order"]:
        html = (SRC / "slides" / f"{sid}.html").read_text(encoding="utf-8").strip()
        html = re.sub(r"<x-icon[^>]*>\s*</x-icon>", icon_svg, html)
        slides.append(html)
    fonts = "&".join(f["href"].split("?", 1)[1].replace("&display=swap", "") for f in deck["faces"].values())
    title = deck["title"]
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?{fonts}&display=swap">
<style>
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; height: 100%; background: #0B1016; overflow: hidden; }}
#stage {{ position: absolute; left: 50%; top: 50%; width: 1920px; height: 1080px; transform-origin: 0 0; }}
#stage > section {{ position: absolute; inset: 0; width: 1920px; height: 1080px; overflow: hidden;
  opacity: 0; visibility: hidden; transition: opacity .35s ease, visibility .35s; }}
#stage > section.on {{ opacity: 1; visibility: visible; }}
section h1, section h2, section h3, section p {{ margin: 0; }}
section h1 {{ font-size: 96px; }} section h2 {{ font-size: 64px; }} section h3 {{ font-size: 44px; }} section p {{ font-size: 32px; }}
section table {{ border-collapse: collapse; width: 100%; }}
section th, section td {{ padding: .35em .6em; border-bottom: 1px solid #D5DCE3; text-align: left; vertical-align: top; }}
section th {{ font-weight: 600; color: #56646F; }}
section aside {{ display: none; }}
#bar {{ position: fixed; left: 0; right: 0; bottom: 0; display: flex; justify-content: space-between; align-items: center;
  padding: 8px 14px; font: 13px system-ui, sans-serif; color: #8C9AA7; background: linear-gradient(transparent, rgba(11,16,22,.85));
  opacity: 0; transition: opacity .3s; }}
body:hover #bar {{ opacity: 1; }}
#bar button {{ background: #1C2733; color: #EEF2F6; border: 1px solid #26313C; border-radius: 6px; padding: 4px 10px; font: inherit; cursor: pointer; }}
#notes {{ position: fixed; left: 50%; bottom: 44px; transform: translateX(-50%); width: min(900px, 92vw); background: #EEF2F6; color: #111A24;
  font: 18px/1.45 system-ui, sans-serif; padding: 14px 18px; border-radius: 10px; box-shadow: 0 8px 30px rgba(0,0,0,.4); }}
@media print {{ #bar, #notes {{ display: none !important; }} }}
</style>
</head>
<body>
<div id="stage">
{chr(10).join(slides)}
</div>
<div id="notes" hidden></div>
<div id="bar">
  <span>← → to move · F fullscreen · N notes</span>
  <span><button id="prev">←</button> <span id="count"></span> <button id="next">→</button></span>
</div>
<script>
const slides = [...document.querySelectorAll('#stage > section')];
let cur = 0;
function fit() {{
  const s = Math.min(innerWidth / 1920, innerHeight / 1080);
  document.getElementById('stage').style.transform = `scale(${{s}}) translate(-50%, -50%)`;
}}
function show(i) {{
  cur = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((el, k) => el.classList.toggle('on', k === cur));
  document.getElementById('count').textContent = (cur + 1) + ' / ' + slides.length;
  const aside = slides[cur].querySelector('aside');
  document.getElementById('notes').textContent = aside ? aside.textContent : 'No notes for this slide.';
  try {{ history.replaceState(null, '', '#' + (cur + 1)); }} catch (e) {{}}
}}
addEventListener('resize', fit);
addEventListener('keydown', e => {{
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) {{ e.preventDefault(); show(cur + 1); }}
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) {{ e.preventDefault(); show(cur - 1); }}
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(slides.length - 1);
  else if (e.key === 'f' || e.key === 'F') {{ document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen().catch(() => {{}}); }}
  else if (e.key === 'n' || e.key === 'N') {{ const n = document.getElementById('notes'); n.hidden = !n.hidden; }}
}});
document.getElementById('stage').addEventListener('click', e => show(cur + (e.clientX > innerWidth / 2 ? 1 : -1)));
document.getElementById('prev').onclick = () => show(cur - 1);
document.getElementById('next').onclick = () => show(cur + 1);
fit();
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
</script>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(slides)} slides)")


if __name__ == "__main__":
    build()
