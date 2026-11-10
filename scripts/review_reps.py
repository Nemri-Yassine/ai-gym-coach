"""Hand-count reps for every own-recording set (Fix 1).

    python scripts/review_reps.py

Opens http://127.0.0.1:8765 — count each set, Save. Writes data/own_reps_verified.csv
(tracked; never under data/raw). About 15 minutes for 22 sets.
"""
from __future__ import annotations

import csv
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

from awc.config import OWN_DIR, OWN_MANIFEST, OWN_REPS_VERIFIED, MMFIT_CLASSES
from awc.data.own_manifest import EXPECTED_CORE, load_manifest, write_manifest
from awc.data.recordings import EXERCISES

PORT = 8765
VERIFIED_COLS = [
    "person", "session_folder", "set", "reps_verified", "exercise_verified",
    "cant_tell", "comment", "reviewer", "reviewed_at",
]


def _load_verified() -> dict[tuple[str, str, int], dict]:
    if not OWN_REPS_VERIFIED.is_file():
        return {}
    out = {}
    with OWN_REPS_VERIFIED.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["person"], r["session_folder"], int(r["set"]))
            out[key] = r
    return out


def _save_verified(rows: dict[tuple[str, str, int], dict]) -> None:
    OWN_REPS_VERIFIED.parent.mkdir(parents=True, exist_ok=True)
    with OWN_REPS_VERIFIED.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VERIFIED_COLS)
        w.writeheader()
        for key in sorted(rows.keys()):
            r = rows[key]
            w.writerow({c: r.get(c, "") for c in VERIFIED_COLS})


def _sets_for_ui():
    # Prefer current manifest; fall back to EXPECTED_CORE order if missing.
    try:
        rows = load_manifest()
    except Exception:
        rows = []
    if not rows:
        # Build minimal list from EXPECTED_CORE
        for person, folder, set_n, _btn, ex, reps, use, _note in EXPECTED_CORE:
            rows.append({
                "person": person, "session_folder": folder, "set": set_n,
                "exercise": ex, "reps": reps, "use": use,
                "video": f"sets/set{set_n:02d}.mp4",
            })
    verified = _load_verified()
    out = []
    for r in rows:
        key = (r["person"], r["session_folder"], int(r["set"]))
        v = verified.get(key, {})
        video = OWN_DIR / r["person"] / r["session_folder"] / r.get("video", f"sets/set{int(r['set']):02d}.mp4")
        out.append({
            "person": r["person"],
            "session_folder": r["session_folder"],
            "set": int(r["set"]),
            "exercise": r["exercise"],
            "use": r.get("use", "eval"),
            "reps_typed": r.get("reps"),
            "reps_verified": v.get("reps_verified", ""),
            "exercise_verified": v.get("exercise_verified", ""),
            "cant_tell": v.get("cant_tell", ""),
            "comment": v.get("comment", ""),
            "has_video": video.is_file() and video.stat().st_size > 0,
            "done": bool(v.get("reps_verified") not in (None, "") or str(v.get("cant_tell", "")).lower() in ("1", "true", "yes")),
        })
    return out


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><title>AWC · verify reps</title>
<style>
:root{--bg:#0B1016;--fg:#E8EEF5;--muted:#8FA3B8;--accent:#3DDC97;--card:#141B24;--warn:#E8A838}
*{box-sizing:border-box} body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{display:grid;grid-template-columns:280px 1fr;min-height:100vh}
nav{background:var(--card);padding:16px;overflow:auto;border-right:1px solid #243040}
nav h1{font-size:1rem;margin:0 0 12px}
nav button{display:block;width:100%;text-align:left;margin:4px 0;padding:8px 10px;border:0;border-radius:6px;
  background:#1a2430;color:var(--fg);cursor:pointer}
nav button.on{background:var(--accent);color:#0B1016;font-weight:600}
nav button.done{opacity:.85}
nav button.done:not(.on){border-left:3px solid var(--accent)}
.panel{padding:20px 28px;max-width:900px}
.ex{font-size:2rem;font-weight:700;margin:0 0 8px;text-transform:capitalize}
.meta{color:var(--muted);margin-bottom:12px}
video{width:100%;max-height:55vh;background:#000;border-radius:8px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}
.btn{padding:10px 14px;border:0;border-radius:6px;background:#243040;color:var(--fg);cursor:pointer;font-size:1rem}
.btn.primary{background:var(--accent);color:#0B1016;font-weight:700}
.stepper b,input[type=number]{font-size:2rem;min-width:3ch;text-align:center}
input,select,textarea{background:#1a2430;border:1px solid #334;color:var(--fg);border-radius:6px;padding:8px}
textarea{width:100%;min-height:60px}
.progress{color:var(--muted);margin-bottom:8px}
.flash{color:var(--accent);min-height:1.2em}
</style></head><body>
<main>
<nav>
  <h1>Verify reps</h1>
  <div class="progress" id="prog"></div>
  <div id="list"></div>
</nav>
<div class="panel" id="panel">
  <p class="meta">Load a set from the left. Count the reps while watching. Save after each set.</p>
</div>
</main>
<script>
const EX = %EXERCISES%;
let sets = [];
let i = 0;
let reps = null;

async function api(path, body){
  const r = await fetch(path, body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  return r.json();
}
function renderList(){
  const done = sets.filter(s=>s.done).length;
  document.getElementById("prog").textContent = `${done} / ${sets.length} done`;
  document.getElementById("list").innerHTML = sets.map((s,k)=>
    `<button class="${k===i?"on":""} ${s.done?"done":""}" data-i="${k}">
      ${s.person.slice(0,1)} · ${s.session_folder.replace(/^\\d{4}-\\d{2}-\\d{2}_/,"")} set${String(s.set).padStart(2,"0")}
      <br/><small>${s.exercise.replaceAll("_"," ")} · ${s.use}</small>
    </button>`).join("");
  document.querySelectorAll("#list button").forEach(b=>b.onclick=()=>{i=+b.dataset.i; show();});
}
function show(){
  const s = sets[i];
  reps = s.reps_verified!=="" && s.reps_verified!=null ? +s.reps_verified : null;
  const url = `/video/${s.person}/${s.session_folder}/${s.set}`;
  document.getElementById("panel").innerHTML = `
    <p class="ex">${(s.exercise_verified||s.exercise).replaceAll("_"," ")}</p>
    <p class="meta">${s.person} / ${s.session_folder} / set ${s.set} · typed default: ${s.reps_typed ?? "—"} · ${s.use}
      ${s.has_video?"":" · <b style=color:var(--warn)>no video</b>"}</p>
    <video id="vid" controls src="${url}"></video>
    <div class="row">
      <button class="btn" id="half">0.5×</button>
      <button class="btn" id="full">1×</button>
    </div>
    <div class="row">
      <span>Reps counted</span>
      <button class="btn" id="minus">−</button>
      <b id="repsBox">${reps===null?"—":reps}</b>
      <button class="btn" id="plus">+</button>
      <input type="number" id="repsNum" min="0" max="99" style="width:5rem" placeholder="n" value="${reps===null?"":reps}"/>
    </div>
    <div class="row">
      <label><input type="checkbox" id="cant" ${s.cant_tell==="true"||s.cant_tell==="1"?"checked":""}/> can't tell</label>
    </div>
    <div class="row">
      <label>Wrong exercise?
        <select id="exfix"><option value="">(keep ${s.exercise})</option>
          ${EX.map(e=>`<option value="${e}" ${s.exercise_verified===e?"selected":""}>${e}</option>`).join("")}
        </select>
      </label>
    </div>
    <div class="row"><textarea id="comment" placeholder="optional comment">${s.comment||""}</textarea></div>
    <div class="row">
      <button class="btn primary" id="save">Save</button>
      <button class="btn" id="next">Next →</button>
      <span class="flash" id="flash"></span>
    </div>`;
  const v = document.getElementById("vid");
  document.getElementById("half").onclick=()=>{v.playbackRate=0.5};
  document.getElementById("full").onclick=()=>{v.playbackRate=1};
  const paint=()=>{document.getElementById("repsBox").textContent=reps===null?"—":reps;
    document.getElementById("repsNum").value=reps===null?"":reps;};
  document.getElementById("minus").onclick=()=>{reps=Math.max(0,(reps??1)-1);paint();};
  document.getElementById("plus").onclick=()=>{reps=(reps??0)+1;paint();};
  document.getElementById("repsNum").oninput=e=>{const t=e.target.value;reps=t===""?null:+t;};
  document.getElementById("save").onclick=save;
  document.getElementById("next").onclick=()=>{i=Math.min(sets.length-1,i+1);show();renderList();};
  renderList();
}
async function save(){
  const s = sets[i];
  const cant = document.getElementById("cant").checked;
  if (!cant && (reps===null || reps==="")) {
    document.getElementById("flash").textContent = "Enter a count or tick can't tell";
    return;
  }
  const body = {
    person: s.person, session_folder: s.session_folder, set: s.set,
    reps_verified: cant ? "" : reps,
    exercise_verified: document.getElementById("exfix").value || "",
    cant_tell: cant, comment: document.getElementById("comment").value,
    reviewer: "local",
  };
  const r = await api("/api/save", body);
  document.getElementById("flash").textContent = r.ok ? "saved" : (r.error||"error");
  if (r.ok) {
    Object.assign(s, {done:true, reps_verified: body.reps_verified, exercise_verified: body.exercise_verified,
      cant_tell: cant?"true":"", comment: body.comment});
    renderList();
  }
}
(async()=>{
  sets = await api("/api/sets");
  show();
})();
</script></body></html>
""".replace("%EXERCISES%", repr([c for c in MMFIT_CLASSES if c != "non_activity"] + ["plank"]))


def create_app() -> FastAPI:
    app = FastAPI(title="AWC verify reps")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/sets")
    def api_sets():
        return _sets_for_ui()

    @app.get("/video/{person}/{session_folder}/{set_n}")
    def video(person: str, session_folder: str, set_n: int):
        path = OWN_DIR / person / session_folder / "sets" / f"set{int(set_n):02d}.mp4"
        if not path.is_file():
            return JSONResponse({"error": "missing"}, status_code=404)
        return FileResponse(path, media_type="video/mp4")

    @app.post("/api/save")
    def save(body: dict = Body(...)):
        try:
            person = body["person"]
            folder = body["session_folder"]
            set_n = int(body["set"])
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        cant = bool(body.get("cant_tell"))
        reps_v = body.get("reps_verified")
        if not cant and reps_v in (None, ""):
            return JSONResponse({"error": "reps_verified required (or cant_tell)"}, status_code=400)
        if not cant:
            try:
                reps_v = int(reps_v)
            except (TypeError, ValueError):
                return JSONResponse({"error": "reps_verified must be an integer"}, status_code=400)
            if reps_v < 0:
                return JSONResponse({"error": "reps_verified must be >= 0"}, status_code=400)
        else:
            reps_v = ""
        ex_v = (body.get("exercise_verified") or "").strip()
        if ex_v and ex_v not in EXERCISES:
            return JSONResponse({"error": f"unknown exercise {ex_v}"}, status_code=400)
        rows = _load_verified()
        key = (person, folder, set_n)
        rows[key] = {
            "person": person,
            "session_folder": folder,
            "set": set_n,
            "reps_verified": reps_v,
            "exercise_verified": ex_v,
            "cant_tell": "true" if cant else "",
            "comment": (body.get("comment") or "").replace("\n", " "),
            "reviewer": body.get("reviewer") or "local",
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        _save_verified(rows)
        # Rebuild manifest so reps_source updates immediately
        write_manifest()
        return {"ok": True, "n_verified": len(rows), "path": str(OWN_REPS_VERIFIED)}

    return app


def main():
    print(f"Verify reps UI → http://127.0.0.1:{PORT}/")
    print(f"Saves to {OWN_REPS_VERIFIED}")
    print("Count all 22 sets (~15 min), then re-run: python scripts/eval_own_recordings.py")
    webbrowser.open(f"http://127.0.0.1:{PORT}/")
    uvicorn.run(create_app(), host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
