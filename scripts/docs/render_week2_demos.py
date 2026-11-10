"""Render Week 2 annotated demos into reports/validation2/demo/week2/.

    python scripts/docs/render_week2_demos.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_video import compress, resolve_own, run, run_side_by_side  # noqa: E402

OUT = ROOT / "reports" / "validation2" / "demo" / "week2"
LOPO_SUMMARY = ROOT / "experiments" / "20261110_m1_lopo_own_summary" / "metrics.json"

DEMOS = [
    {
        "file": "squats_yassine",
        "own": "yassine/2026-11-10_SQUAT/1",
        "model": "m1_transformer_pose",
        "shows": "clean success: squats",
    },
    {
        "file": "pushups_partner",
        "own": "partner/2026-11-10_PUSHUP/1",
        "model": "m1_transformer_pose",
        "shows": "success on partner",
    },
    {
        "file": "shoulder_press_partner",
        "own": "partner/2026-11-10_SHOULDERPRESS/1",
        "model": "m1_transformer_pose",
        "shows": "success shoulder press",
    },
    {
        "file": "jumping_jacks_partner",
        "own": "partner/2026-11-10_EX3IHEB/1",
        "model": "m1_transformer_pose",
        "shows": "success jumping jacks",
    },
    {
        "file": "bicep_curls_yassine",
        "own": "yassine/2026-11-10_TRICEPS/3",
        "model": "m1_transformer_pose",
        "shows": "bicep curls success (reps may be off)",
    },
    {
        "file": "wrong_button_pushups",
        "own": "yassine/2026-11-10_s007/1",
        "model": "m1_transformer_pose",
        "shows": "button squats, model push-ups",
        "banner": "Button pressed: Squats · Model says: Push-ups ✓",
    },
    {
        "file": "failure_triceps",
        "own": "yassine/2026-11-10_TRICEPS/2",
        "model": "m1_transformer_pose",
        "shows": "honest failure triceps",
        "end_reason": "Ours stay low-amplitude near the hip — MM-Fit triceps are overhead standing.",
    },
]


def pick_lopo_compare():
    """Yassine set: zero-shot wrong, adapted (on partner) right. Else None."""
    if not LOPO_SUMMARY.is_file():
        return None
    data = json.loads(LOPO_SUMMARY.read_text(encoding="utf-8"))
    for d in data.get("directions", []):
        if d.get("test_person") != "yassine":
            continue
        z = {p["key"]: p for p in d["zero_shot"]["preds"]}
        a = {p["key"]: p for p in d["adapted"]["preds"]}
        for key, zp in z.items():
            ap = a.get(key)
            if zp and ap and (not zp["ok"]) and ap["ok"]:
                # key like yassine/2026-11-10_TRICEPS/set01
                person, session, set_tok = key.split("/")
                set_n = int(set_tok.replace("set", ""))
                return {
                    "own": f"{person}/{session}/{set_n}",
                    "zero": "m1_transformer_pose",
                    "adapted": Path(d["checkpoint"]).stem,
                    "key": key,
                    "zero_pred": zp["pred"],
                    "adapted_pred": ap["pred"],
                    "truth": zp["truth"],
                }
    return None


def make_montage(clips: list[Path], out: Path, titles: list[str]):
    """60–90 s highlight: title → short clips → end card numbers."""
    try:
        import imageio_ffmpeg
        from PIL import Image, ImageDraw, ImageFont
        import cv2
        import numpy as np
    except ImportError as e:
        print("montage skipped:", e)
        return None

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = out.parent / "_montage_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []

    def card(text_lines, seconds, name):
        img = Image.new("RGB", (1920, 1080), (11, 16, 22))
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype("segoeuib.ttf", 48)
            fs = ImageFont.truetype("segoeui.ttf", 28)
        except OSError:
            f = ImageFont.load_default()
            fs = f
        y = 360
        for i, line in enumerate(text_lines):
            d.text((100, y), line, font=f if i == 0 else fs, fill=(238, 242, 246) if i == 0 else (140, 154, 167))
            y += 70 if i == 0 else 44
        path = tmp / f"{name}.mp4"
        w = cv2.VideoWriter(str(path.with_suffix(".webm")), cv2.VideoWriter_fourcc(*"VP80"), 30, (1920, 1080))
        frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for _ in range(int(30 * seconds)):
            w.write(frame)
        w.release()
        final = compress(path.with_suffix(".webm"), path, width=1920, crf=28)
        parts.append(Path(final))
        return Path(final)

    card(["AI Workout Coach", "Week 2 · our own gym videos"], 3.0, "title")
    for clip, title in zip(clips, titles):
        if not clip.is_file():
            continue
        # take ~6 s from mid
        mid = tmp / f"clip_{clip.stem}.mp4"
        # probe duration
        probe = subprocess.run(
            [ffmpeg, "-i", str(clip)], capture_output=True, text=True,
        )
        # ffmpeg prints duration on stderr
        dur = 12.0
        for line in (probe.stderr or "").splitlines():
            if "Duration:" in line:
                # Duration: 00:00:22.10
                try:
                    hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
                    dur = float(hh) * 3600 + float(mm) * 60 + float(ss)
                except Exception:
                    pass
                break
        start = max(0.0, min(dur * 0.25, max(0.0, dur - 7.0)))
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(clip),
            "-t", "6.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(mid),
        ], check=False)
        if mid.is_file():
            # title strip overlay via card then clip
            card([title], 1.2, f"t_{clip.stem}")
            parts.append(mid)

    card([
        "14 / 20 sets recognised zero-shot",
        "Wrong-button caught · triceps still fail",
        "Adapted 75% → 92% (n=12) — separate experiment",
    ], 4.0, "end")

    lst = tmp / "concat.txt"
    lines = []
    for p in parts:
        if p.is_file():
            lines.append(f"file '{p.resolve().as_posix()}'")
    lst.write_text("\n".join(lines), encoding="utf-8")
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(lst), "-c:v", "libx264", "-preset", "slow", "-crf", "28",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
    ], check=False)
    if out.is_file() and out.stat().st_size > 8e6:
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error", "-i", str(out),
            "-c:v", "libx264", "-preset", "slow", "-crf", "32",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
        ], check=False)
    return out if out.is_file() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    rendered = []

    for d in DEMOS:
        out = OUT / d["file"]
        print(f"\n=== {d['file']} ===")
        meta = resolve_own(d["own"])
        path, summary = run(
            meta["video"], out, model_name=d["model"], own_meta=meta,
            banner=d.get("banner"), end_reason=d.get("end_reason"), crf=28,
        )
        path = Path(path)
        entry = {
            "file": path.name,
            "shows": d["shows"],
            "own": d["own"],
            "seconds": summary.get("seconds"),
            "megabytes": summary.get("megabytes"),
            "exercise": summary.get("exercise"),
            "truth": summary.get("truth_exercise"),
            "ok": summary.get("exercise_ok"),
            "reps_counted": summary.get("reps_counted"),
            "truth_reps": summary.get("truth_reps"),
        }
        index.append(entry)
        rendered.append((path, d["shows"]))
        mark = "✓" if entry["ok"] else "✗"
        print(f"  {path.name}: {entry['seconds']}s · {entry['megabytes']} MB · "
              f"{entry['exercise']} {mark} (truth {entry['truth']})")

    compare = pick_lopo_compare()
    if compare:
        print(f"\n=== zero_shot_vs_adapted ({compare['key']}) ===")
        print(f"  zero-shot {compare['zero_pred']} ✗ → adapted {compare['adapted_pred']} ✓ "
              f"(truth {compare['truth']})")
        path, summary = run_side_by_side(
            compare["own"], compare["zero"], compare["adapted"],
            OUT / "zero_shot_vs_adapted", crf=28,
        )
        path = Path(path)
        index.append({
            "file": path.name,
            "shows": f"side-by-side LOPO on {compare['key']}",
            "own": compare["own"],
            "seconds": summary.get("seconds"),
            "megabytes": summary.get("megabytes"),
            "left": summary.get("left"),
            "right": summary.get("right"),
            "ok": summary.get("right", {}).get("ok"),
        })
        rendered.append((path, "zero-shot vs adapted"))
        print(f"  {path.name}: {summary.get('seconds')}s · {summary.get('megabytes')} MB")
    else:
        print("\n=== zero_shot_vs_adapted SKIPPED ===")
        print("  No yassine set where zero-shot fails and adapted succeeds.")
        index.append({"file": None, "shows": "skipped — no qualifying LOPO pair", "ok": None})

    print("\n=== montage_week2 ===")
    montage_clips = [OUT / f"{d['file']}.mp4" for d in DEMOS]
    titles = [d["shows"] for d in DEMOS]
    # prefer wrong-button + failure near the end already in list order
    mout = make_montage(montage_clips, OUT / "montage_week2.mp4", titles)
    if mout and mout.is_file():
        # duration via ffprobe-ish
        import imageio_ffmpeg
        probe = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(mout)],
            capture_output=True, text=True,
        )
        dur = None
        for line in (probe.stderr or "").splitlines():
            if "Duration:" in line:
                try:
                    hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
                    dur = float(hh) * 3600 + float(mm) * 60 + float(ss)
                except Exception:
                    pass
                break
        index.append({
            "file": mout.name,
            "shows": "highlight reel",
            "seconds": round(dur, 1) if dur else None,
            "megabytes": round(mout.stat().st_size / 1e6, 2),
            "ok": None,
        })
        print(f"  {mout.name}: {dur}s · {mout.stat().st_size / 1e6:.2f} MB")
    else:
        print("  montage failed")

    # station demo placeholder note
    station = OUT / "station_demo.mp4"
    index.append({
        "file": "station_demo.mp4" if station.is_file() else None,
        "shows": "live station screen recording",
        "ok": None,
        "note": "not auto-recorded — see docs/demo.md (Win+Alt+R)",
    })

    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("\nWrote", OUT / "index.json")
    for e in index:
        print(f"  {e.get('file')}: {e.get('seconds')}s {e.get('megabytes')}MB "
              f"{'✓' if e.get('ok') else ('✗' if e.get('ok') is False else '·')} {e.get('shows')}")


if __name__ == "__main__":
    main()
