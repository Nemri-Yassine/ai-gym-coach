"""Run the pipeline on a real video, frame by frame, as it would run live, and save an annotated video.

    python scripts/demo_video.py path/to/video.mp4 --truth-reps 14 --title "RepCount-A · squats"
    python scripts/demo_video.py --own yassine/2026-11-10_SQUAT/1 --out reports/validation2/demo/week2/squats

Each frame only uses the past: MediaPipe skeleton -> every 0.1 s Model 1 (camera only) reads the
last 3 s; a new exercise is accepted once it stays the top class for 2 s -> the rep counter counts
since that exercise started. Own phone videos use a 1920×1080 portrait layout (video left, info
panel right). Output: <out>.mp4 (H.264) and <out>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from awc.config import (
    MMFIT_CLASSES,
    MODELS,
    OWN_DIR,
    OWN_MANIFEST,
    OWN_REPS_VERIFIED,
    POSE_MODEL,
    REPORTS,
)
from awc.features.pose_video import landmarks_to_pixels, resample_times
from awc.features.reps import count_reps
from awc.features.skeleton import BONES, normalize
from awc.models.m1_recognition import M1Model

WINDOW_S = 3.0
EVERY = 3                    # run Model 1 every 3 frames (0.1 s at 30 fps)
EMA = 0.35                   # smoothing of the displayed probabilities
SWITCH_STEPS = 20           # a new exercise must stay the top class for 20 model steps (2 s) in a row
NA = MMFIT_CLASSES.index("non_activity")
NAMES = {
    "squats": "Squats", "lunges": "Lunges", "bicep_curls": "Biceps curls", "situps": "Sit-ups",
    "pushups": "Push-ups", "tricep_extensions": "Triceps extensions", "dumbbell_rows": "Dumbbell rows",
    "jumping_jacks": "Jumping jacks", "dumbbell_shoulder_press": "Shoulder press",
    "lateral_shoulder_raises": "Lateral raises", "non_activity": "No exercise",
    "no_exercise_detected": "No exercise",
}

INK, MUTED, ACCENT, OK, STOP, BG = (
    (238, 242, 246), (140, 154, 167), (115, 147, 255),
    (76, 196, 135), (255, 90, 95), (11, 16, 22),
)
LEFT_C, RIGHT_C, MID_C = (80, 200, 255), (255, 140, 90), (238, 242, 246)
# COCO-18: right = 2–4, 8–10, 14, 16; left = 5–7, 11–13, 15, 17
RIGHT_J = {2, 3, 4, 8, 9, 10, 14, 16}
LEFT_J = {5, 6, 7, 11, 12, 13, 15, 17}
OUT_W, OUT_H = 1920, 1080


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_SYM = "✓✗"
_SYM_CACHE = {}


def sym_font(size):
    """Segoe UI has no ✓ / ✗ glyphs; Segoe UI Symbol does."""
    if size not in _SYM_CACHE:
        try:
            _SYM_CACHE[size] = ImageFont.truetype("seguisym.ttf", size)
        except OSError:
            _SYM_CACHE[size] = None
    return _SYM_CACHE[size]


def draw_text(d, xy, s, font=None, fill=None):
    """d.text that draws ✓ / ✗ with a font that has them."""
    x, y = xy
    s = str(s)
    if not any(c in s for c in _SYM):
        d.text((x, y), s, font=font, fill=fill)
        return
    size = getattr(font, "size", 20)
    for ch_run in _split_sym(s):
        f = (sym_font(size) or font) if ch_run in _SYM else font
        d.text((x, y), ch_run, font=f, fill=fill)
        x += d.textlength(ch_run, font=f)


def _split_sym(s):
    out, cur = [], ""
    for c in s:
        if c in _SYM:
            if cur:
                out.append(cur)
            out.append(c)
            cur = ""
        else:
            cur += c
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- Model 2 (form), real per-rep scoring
_M2 = {"pack": None}
_M2_EX = {"squats": "squat", "lunges": "lunge"}


def m2_score_reps(seq_norm, bounds, exercise):
    """Model 2 random forest (models/m2_rf.pickle) on each finished rep. Returns [(pred_name, p_pred, [(name, p)])]."""
    key = _M2_EX.get(exercise)
    if key is None:
        return []
    if _M2["pack"] is None:
        import pickle
        with open(MODELS / "m2_rf.pickle", "rb") as fh:
            _M2["pack"] = pickle.load(fh)
    forest = _M2["pack"]["forests"].get(key)
    if forest is None:
        return []
    from awc.data.ec3d import MISTAKES
    from awc.features.form_dataset import MISTAKE_CODES, resample
    from awc.features.form_features import feature_vector, features_2d
    names = [MISTAKES[c] for c in MISTAKE_CODES[key]]
    out = []
    for a, b in bounds:
        seg = seq_norm[a:b + 1]
        if len(seg) < 8 or not np.isfinite(seg).all():
            continue
        proba = forest.predict_proba(feature_vector(features_2d(resample(seg, 64))).reshape(1, -1))[0]
        full = np.zeros(len(names))
        full[forest.classes_] = proba
        i = int(np.argmax(full))
        out.append((names[i], float(full[i]), sorted(zip(names, full.tolist()), key=lambda kv: -kv[1])))
    return out


def load_model(name):
    stem = name[:-3] if str(name).endswith(".pt") else name
    ckpt = torch.load(MODELS / f"{stem}.pt", map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["inputs"], stem


def window(poses, times, now, seconds):
    """Last `seconds` of skeletons resampled to 30 fps (90 frames for 3 s); None if incomplete."""
    keep = [i for i, t in enumerate(times) if now - seconds * 1000 - 40 <= t <= now]
    if len(keep) < 2 or (now - times[keep[0]]) < seconds * 1000 - 80:
        return None
    seq = np.asarray([poses[i] for i in keep], dtype=np.float64)
    if not np.isfinite(seq).all():
        return None
    t = np.asarray([times[i] for i in keep], dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        return None
    out = resample_times(seq, t)
    n = int(round(seconds * 30))
    return out[-n:] if len(out) >= n else None


def resolve_own(spec: str) -> dict:
    """Parse person/session/set → video + clocks + manifest truth."""
    parts = Path(str(spec).replace("\\", "/")).parts
    if len(parts) < 3:
        raise ValueError(f"--own expects person/session/set, got {spec!r}")
    person, session, set_tok = parts[0], parts[1], parts[2]
    set_n = int(str(set_tok).lower().replace("set", ""))
    folder = OWN_DIR / person / session
    video = folder / "sets" / f"set{set_n:02d}.mp4"
    if not video.is_file():
        raise FileNotFoundError(video)
    frames_csv = folder / f"frames_set{set_n:02d}.csv"
    hr_csv = folder / "hr.csv"
    frame_times = None
    if frames_csv.is_file():
        with frames_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        frame_times = [float(r["pc_ts_ms"]) for r in rows]
    hr = []
    if hr_csv.is_file():
        with hr_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    hr.append((float(r["pc_ts_ms"]), float(r["bpm"])))
                except (KeyError, ValueError):
                    continue
        hr.sort()

    truth_ex, truth_reps, reps_note, button = None, None, "typed, unverified", None
    if OWN_MANIFEST.is_file():
        with OWN_MANIFEST.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["person"] == person and r["session_folder"] == session and int(r["set"]) == set_n:
                    truth_ex = r["exercise"]
                    button = r.get("exercise_button") or None
                    truth_reps = int(float(r["reps"])) if r.get("reps") not in ("", None) else None
                    if r.get("reps_source") == "verified":
                        reps_note = "verified"
                    break
    if OWN_REPS_VERIFIED.is_file():
        with OWN_REPS_VERIFIED.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("person") == person and r.get("session_folder") == session and int(r["set"]) == set_n:
                    if r.get("reps_verified") not in ("", None):
                        truth_reps = int(float(r["reps_verified"]))
                        reps_note = "verified"
                    break

    return {
        "person": person,
        "session": session,
        "set": set_n,
        "video": video,
        "frame_times": frame_times,
        "hr": hr,
        "truth_exercise": truth_ex,
        "truth_reps": truth_reps,
        "reps_note": reps_note,
        "button": button,
        "key": f"{person}/{session}/set{set_n:02d}",
    }


def hr_at(hr_series, t_ms, max_age_ms=3000.0):
    if not hr_series:
        return None
    best = None
    for ts, bpm in hr_series:
        if ts > t_ms:
            break
        if t_ms - ts <= max_age_ms:
            best = bpm
    return best


def hr_spark(hr_series, now_ms, window_ms=30000.0, w=220, h=36):
    pts = [(t, b) for t, b in hr_series if now_ms - window_ms <= t <= now_ms]
    img = Image.new("RGBA", (w, h), (20, 28, 36, 255))
    d = ImageDraw.Draw(img)
    if len(pts) < 2:
        d.line([(4, h // 2), (w - 4, h // 2)], fill=MUTED + (160,), width=1)
        return img
    ys = [b for _, b in pts]
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 1.0)
    t0, t1 = pts[0][0], pts[-1][0]
    dt = max(t1 - t0, 1.0)
    poly = []
    for t, b in pts:
        x = 4 + (t - t0) / dt * (w - 8)
        y = h - 4 - (b - lo) / span * (h - 8)
        poly.append((x, y))
    d.line(poly, fill=STOP + (220,), width=2)
    return img


def joint_color(j):
    if j in LEFT_J:
        return LEFT_C
    if j in RIGHT_J:
        return RIGHT_C
    return MID_C


def draw_skeleton(d, pose_px, ox, oy, scale, thick=5):
    if pose_px is None or not np.isfinite(pose_px).all():
        return
    # pose_px is y-up image pixels; convert to y-down for drawing on the placed video
    pts = [(ox + float(x) * scale, oy + float(-y) * scale) for x, y in pose_px]
    for a, b in BONES:
        if a >= 14 or b >= 14:
            continue
        col = joint_color(a) if a in LEFT_J or b in LEFT_J else joint_color(b)
        if a in LEFT_J and b in LEFT_J:
            col = LEFT_C
        elif a in RIGHT_J and b in RIGHT_J:
            col = RIGHT_C
        else:
            col = MID_C
        d.line([pts[a], pts[b]], fill=col + (235,), width=max(3, int(thick)))
    for j, p in enumerate(pts[:14]):
        r = 6 if j not in (8, 11) else 8
        c = joint_color(j)
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=c + (255,))


def topk(probs, k=3):
    if probs is None:
        return []
    idx = np.argsort(probs)[::-1][:k]
    return [(MMFIT_CLASSES[i], float(probs[i])) for i in idx]


def draw_landscape_overlay(frame, pose_px, state, scale):
    """Week-1 style overlay (text on video) for landscape / public demos."""
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    draw_skeleton(d, pose_px, 0, 0, scale, thick=4)
    pw, ph = 380, 236
    d.rounded_rectangle([16, 16, 16 + pw, 16 + ph], 16, fill=BG + (205,))
    draw_text(d, (32, 28), "AI WORKOUT COACH · LIVE", font=font(15, True), fill=ACCENT)
    ex = state["exercise"]
    draw_text(d, (32, 54), NAMES.get(ex, ex), font=font(34, True), fill=INK if ex != "non_activity" else MUTED)
    if state.get("final"):
        draw_text(d, (32, 98), "RESULT FOR THE WHOLE VIDEO", font=font(15, True), fill=OK)
    else:
        conf = state["conf"]
        d.rounded_rectangle([32, 104, 32 + 280, 116], 6, fill=(38, 49, 60, 255))
        d.rounded_rectangle([32, 104, 32 + max(12, int(280 * conf)), 116], 6, fill=ACCENT + (255,))
        draw_text(d, (322, 98), f"{conf * 100:.0f}%", font=font(18, True), fill=INK)
    draw_text(d, (32, 130), "REPS", font=font(15, True), fill=MUTED)
    draw_text(d, (32, 146), str(state["reps"]), font=font(64, True), fill=OK if state["reps"] else INK)
    draw_text(d, (150, 158), state["sub"], font=font(15), fill=MUTED)
    draw_text(d, (150, 180), f"{state['fps']:.0f} fps on CPU", font=font(15), fill=MUTED)
    draw_text(d, (150, 202), "Model 1, camera only, last 3 s", font=font(15), fill=MUTED)
    if state.get("footer"):
        tw = d.textlength(state["footer"], font=font(17))
        d.rounded_rectangle([W - tw - 40, H - 46, W - 16, H - 14], 10, fill=BG + (205,))
        draw_text(d, (W - tw - 28, H - 42), state["footer"], font=font(17), fill=INK)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_portrait_canvas(bgr, pose_px, state, meta, probs, model_name, banner=None):
    """1920×1080: portrait video left (full height), live info panel on the right."""
    src_h, src_w = bgr.shape[:2]
    scale = OUT_H / src_h
    vw = int(round(src_w * scale)) // 2 * 2
    vh = OUT_H
    video = cv2.resize(bgr, (vw, vh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
    canvas[:] = (11, 16, 22)
    canvas[:, :vw] = video
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    draw_skeleton(d, pose_px, 0, 0, scale, thick=6)

    px = vw + 36
    draw_text(d, (px, 28), "AI WORKOUT COACH · LIVE", font=font(18, True), fill=ACCENT)
    if banner:
        d.rounded_rectangle([px, 58, OUT_W - 28, 108], 12, fill=(40, 20, 24, 240))
        draw_text(d, (px + 16, 70), banner, font=font(20, True), fill=STOP)

    y0 = 130 if banner else 70
    ex = state["exercise"]
    mark = ""
    truth = meta.get("truth_exercise")
    if truth and ex not in ("non_activity", None) and state.get("decided"):
        mark = "  ✓" if ex == truth else "  ✗"
    label = NAMES.get(ex, ex or "—") if state.get("decided") else "Detecting…"
    draw_text(d, (px, y0), label + mark, font=font(48, True),
           fill=INK if state.get("decided") else MUTED)
    conf = float(state.get("conf") or 0.0)
    bar_w = OUT_W - px - 48
    d.rounded_rectangle([px, y0 + 64, px + bar_w, y0 + 80], 8, fill=(38, 49, 60, 255))
    d.rounded_rectangle([px, y0 + 64, px + max(14, int(bar_w * conf)), y0 + 80], 8, fill=ACCENT + (255,))
    draw_text(d, (px + bar_w - 70, y0 + 58), f"{conf * 100:.0f}%", font=font(20, True), fill=INK)

    yy = y0 + 100
    for name, p in topk(probs, 3):
        draw_text(d, (px, yy), NAMES.get(name, name), font=font(16), fill=MUTED)
        bw = int((OUT_W - px - 200) * p)
        d.rounded_rectangle([px + 200, yy + 4, px + 200 + max(4, bw), yy + 16], 4, fill=ACCENT + (200,))
        draw_text(d, (OUT_W - 90, yy), f"{p * 100:.0f}%", font=font(15, True), fill=INK)
        yy += 26

    yy += 10
    draw_text(d, (px, yy), "REPS", font=font(16, True), fill=MUTED)
    rep_col = OK if state.get("rep_flash") else (OK if state["reps"] else INK)
    draw_text(d, (px, yy + 22), str(state["reps"]), font=font(92, True), fill=rep_col)

    hr = state.get("hr")
    draw_text(d, (px + 220, yy), "HEART RATE", font=font(16, True), fill=MUTED)
    draw_text(d, (px + 220, yy + 28), "—" if hr is None else f"{hr:.0f}", font=font(56, True),
           fill=STOP if hr is not None else MUTED)
    draw_text(d, (px + 220, yy + 100), "bpm", font=font(18), fill=MUTED)
    spark = hr_spark(meta.get("hr") or [], state.get("pc_ts") or 0.0)
    img.paste(spark, (px + 220, yy + 130), spark)

    yy = yy + 200
    draw_text(d, (px, yy), "GROUND TRUTH", font=font(14, True), fill=MUTED)
    treps = meta.get("truth_reps")
    treps_s = "—" if treps is None else f"{treps} ({meta.get('reps_note', 'typed, unverified')})"
    draw_text(d, (px, yy + 22), f"Exercise: {NAMES.get(truth, truth) if truth else '—'}", font=font(18), fill=MUTED)
    draw_text(d, (px, yy + 48), f"Reps: {treps_s}", font=font(18), fill=MUTED)
    if meta.get("button") and truth and meta["button"] != truth:
        draw_text(d, (px, yy + 74), f"Button pressed: {NAMES.get(meta['button'], meta['button'])}",
               font=font(17, True), fill=STOP)

    if state.get("form_on"):
        fy = yy + 112
        draw_text(d, (px, fy), "FORM · MODEL 2 (per finished rep)", font=font(16, True), fill=ACCENT)
        reps_f = state.get("form_reps") or []
        if not reps_f:
            draw_text(d, (px, fy + 28), "waiting for the first full rep…", font=font(22), fill=MUTED)
        else:
            name, p, ranked = reps_f[-1]
            good = name == "Correct"
            draw_text(d, (px, fy + 26), f"Rep {len(reps_f)}: {name}", font=font(40, True), fill=OK if good else STOP)
            by = fy + 82
            for nm, pv in ranked[:3]:
                draw_text(d, (px, by), nm, font=font(17), fill=MUTED)
                bw = int((OUT_W - px - 260) * pv)
                d.rounded_rectangle([px + 190, by + 5, px + 190 + max(4, bw), by + 19], 5,
                                    fill=(OK if nm == "Correct" else STOP) + (220,))
                draw_text(d, (OUT_W - 80, by), f"{pv * 100:.0f}%", font=font(16, True), fill=INK)
                by += 28
            dx = px
            draw_text(d, (px, by + 8), "all reps:", font=font(16), fill=MUTED)
            for nm, _, _ in reps_f:
                d.ellipse([dx + 90, by + 12, dx + 108, by + 30], fill=(OK if nm == "Correct" else STOP) + (255,))
                dx += 26
    elif state.get("form_line"):
        draw_text(d, (px, yy + 110), state["form_line"], font=font(17), fill=MUTED)

    foot = (f"{meta.get('person', '')} · set {meta.get('set', '')} · {model_name} · "
            f"{state['fps']:.0f} fps")
    draw_text(d, (px, OUT_H - 48), foot, font=font(16), fill=MUTED)
    draw_text(d, (px, OUT_H - 28), state.get("sub") or "", font=font(15), fill=MUTED)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def end_card(state, meta, model_name, reason=None, size=(OUT_W, OUT_H)):
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    ex = state.get("exercise") or "non_activity"
    truth = meta.get("truth_exercise")
    ok = truth is not None and ex == truth
    mark = "✓" if ok else "✗"
    reps = state.get("reps", 0)
    treps = meta.get("truth_reps")
    hr0, hr1 = state.get("hr_start"), state.get("hr_end")
    hr_bit = ""
    if hr0 is not None and hr1 is not None:
        hr_bit = f" · HR {hr0:.0f} → {hr1:.0f} bpm"
    elif hr1 is not None:
        hr_bit = f" · HR {hr1:.0f} bpm"
    line = f"Detected: {NAMES.get(ex, ex)} {mark} · Reps counted {reps}"
    if treps is not None:
        line += f" / true {treps}"
    line += hr_bit
    draw_text(d, (80, 360), "RESULT", font=font(28, True), fill=ACCENT)
    draw_text(d, (80, 420), line, font=font(36, True), fill=OK if ok else STOP)
    if reason:
        draw_text(d, (80, 500), reason, font=font(24), fill=MUTED)
    draw_text(d, (80, 980), f"{meta.get('key', '')} · {model_name}", font=font(18), fill=MUTED)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def compress(webm, mp4, width=1920, crf=28):
    """Re-encode OpenCV's large VP8 file to a small H.264 MP4. Returns the file to keep."""
    try:
        import subprocess
        import imageio_ffmpeg
    except ImportError:
        return webm
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(webm),
           "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(mp4)]
    if subprocess.run(cmd).returncode == 0 and Path(mp4).exists():
        Path(webm).unlink(missing_ok=True)
        return Path(mp4)
    return webm


def _score_step(model, inputs, poses, times, ts, probs_s, state, seg_start, seg_ex, candidate, streak,
                segments, timeline, fps):
    w = window(poses, times, ts, WINDOW_S)
    if w is None:
        return probs_s, state, seg_start, seg_ex, candidate, streak
    x = torch.from_numpy(normalize(w).reshape(1, 90, 36).astype(np.float32))
    with torch.no_grad():
        logits = model(x) if inputs == "pose" else model(x, torch.zeros(1, 90, 6), torch.zeros(1))
        p = torch.softmax(logits, 1).numpy()[0]
    probs_s = p if probs_s is None else EMA * p + (1 - EMA) * probs_s
    top = MMFIT_CLASSES[int(np.argmax(probs_s))]
    # Same rule as the reported numbers (eval_own_recordings / eval_public_videos):
    # majority vote of the exercise windows seen so far, "no exercise" windows excluded.
    # An exercise is shown once there are 2 s of exercise windows (SWITCH_STEPS model steps).
    votes = state.setdefault("votes", {})
    first = state.setdefault("vote_t0", {})
    if top != "non_activity":
        votes[top] = votes.get(top, 0) + 1
        first.setdefault(top, ts)
    if sum(votes.values()) >= SWITCH_STEPS:
        leader = max(votes, key=lambda k: (votes[k], -first[k]))
        if leader != state["exercise"]:
            state["exercise"] = leader
            state["decided"] = True
            if leader != seg_ex:
                seg_ex, seg_start, state["reps"] = leader, max(0.0, first[leader] - WINDOW_S * 1000), 0
                segments.append({"exercise": leader, "start_s": round(seg_start / 1000, 1), "end_s": None, "reps": 0})
    state["conf"] = float(probs_s[int(np.argmax(probs_s))])
    state["sub"] = "reps since the exercise started" if seg_ex else "looking for an exercise"
    timeline.append([round(ts / 1000, 2), MMFIT_CLASSES[int(np.argmax(probs_s))], round(state["conf"], 3)])
    if seg_ex and state["exercise"] == seg_ex:
        keep = [i for i, t in enumerate(times) if t >= seg_start and np.isfinite(poses[i]).all()]
        if len(keep) > 30:
            seq = resample_times(np.asarray([poses[i] for i in keep]), np.asarray([times[i] for i in keep]))
            seq_n = normalize(seq)
            n, bounds = count_reps(seq_n, fps=30, exercise=seg_ex)
            if state.get("form_on") and seg_ex in _M2_EX and len(bounds) > len(state.get("form_reps") or []):
                state["form_reps"] = m2_score_reps(seq_n, bounds, seg_ex)
            prev = state["reps"]
            state["reps"] = max(state["reps"], int(n))
            state["rep_flash"] = state["reps"] > prev
            segments[-1]["reps"], segments[-1]["end_s"] = state["reps"], round(ts / 1000, 1)
    return probs_s, state, seg_start, seg_ex, candidate, streak


def run(
    video,
    out,
    model_name="m1_transformer_pose",
    truth_reps=None,
    title="",
    width=1280,
    max_seconds=None,
    own_meta=None,
    show_form=False,
    banner=None,
    end_reason=None,
    crf=28,
):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    torch.set_num_threads(4)
    model, inputs, model_stem = load_model(model_name)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    portrait = own_meta is not None or src_h > src_w
    if portrait:
        size = (OUT_W, OUT_H)
        scale = OUT_H / src_h
    else:
        scale = width / src_w
        size = (width, int(round(src_h * scale)) // 2 * 2)

    meta = dict(own_meta or {})
    if truth_reps is not None:
        meta["truth_reps"] = truth_reps
    if title and not meta.get("key"):
        meta["key"] = title

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out.with_suffix(".webm")), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
    )
    poses, times, pc_times = [], [], []
    probs_s = None
    state = {
        "exercise": "non_activity", "conf": 0.0, "reps": 0, "sub": "warming up: 3 s of history needed",
        "fps": 0.0, "footer": title, "decided": False, "rep_flash": False, "hr": None,
        "hr_start": None, "hr_end": None, "form_line": None, "pc_ts": None,
    }
    seg_start, seg_ex = None, None
    candidate, streak = None, 0
    segments, timeline, proc = [], [], []
    index, last_ts = 0, -1
    frame_times = meta.get("frame_times")
    hr_series = meta.get("hr") or []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_times is not None and index < len(frame_times):
                pc_ts = frame_times[index]
                ts = pc_ts - frame_times[0]
            else:
                pc_ts = None
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = ts if ts > last_ts else last_ts + 1000.0 / fps
            last_ts = ts
            if max_seconds and ts > max_seconds * 1000:
                break
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts),
            )
            if res.pose_landmarks:
                pose = landmarks_to_pixels([(p.x, p.y) for p in res.pose_landmarks[0]], src_w, src_h)
            else:
                pose = np.full((18, 2), np.nan, np.float32)
            poses.append(pose)
            times.append(ts)
            pc_times.append(pc_ts)

            state["rep_flash"] = False
            if index % EVERY == 0:
                probs_s, state, seg_start, seg_ex, candidate, streak = _score_step(
                    model, inputs, poses, times, ts, probs_s, state, seg_start, seg_ex,
                    candidate, streak, segments, timeline, fps,
                )
            if pc_ts is not None:
                bpm = hr_at(hr_series, pc_ts)
                state["hr"] = bpm
                state["pc_ts"] = pc_ts
                if bpm is not None:
                    if state["hr_start"] is None:
                        state["hr_start"] = bpm
                    state["hr_end"] = bpm
            if show_form and state["exercise"] in _M2_EX:
                state["form_on"] = True
            else:
                state["form_line"] = None

            proc.append(time.perf_counter() - t0)
            state["fps"] = 1.0 / max(np.mean(proc[-30:]), 1e-6)
            if portrait:
                writer.write(draw_portrait_canvas(
                    bgr, pose, state, meta, probs_s, model_stem, banner=banner,
                ))
            else:
                frame = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                writer.write(draw_landscape_overlay(frame, pose, state, scale))
            index += 1
    cap.release()

    main = max(segments, key=lambda g: (g["end_s"] or g["start_s"]) - g["start_s"]) if segments else None
    if index:
        if main:
            state["exercise"], state["reps"] = main["exercise"], main["reps"]
            state["decided"] = True
            state["final"] = True
            state["sub"] = f"longest bout: {main['start_s']:.0f}–{main['end_s'] or main['start_s']:.0f} s"
        if portrait:
            card = end_card(state, meta, model_stem, reason=end_reason)
            for _ in range(int(fps * 2)):
                writer.write(card)
        else:
            state["footer"] = f"Counted {state['reps']} reps" + (
                f" · ground truth {meta.get('truth_reps')}" if meta.get("truth_reps") is not None else ""
            )
            last = draw_landscape_overlay(
                cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), poses[-1], state, scale,
            )
            for _ in range(int(fps * 2)):
                writer.write(last)
    writer.release()

    pred = main["exercise"] if main else None
    truth = meta.get("truth_exercise")
    summary = {
        "video": Path(video).name,
        "own": meta.get("key"),
        "frames": index,
        "seconds": round(times[-1] / 1000, 1) if times else 0,
        "model": model_stem,
        "exercise": pred,
        "truth_exercise": truth,
        "exercise_ok": bool(truth and pred == truth),
        "reps_counted": main["reps"] if main else 0,
        "truth_reps": meta.get("truth_reps"),
        "reps_note": meta.get("reps_note"),
        "segments": segments,
        "processing_fps_mean": round(1 / float(np.mean(proc)), 1) if proc else None,
        "realtime_capable": bool(proc) and 1 / float(np.mean(proc)) >= fps,
        "pose_found_ratio": round(float(np.mean([np.isfinite(p).all() for p in poses])), 3) if poses else 0,
        "timeline": timeline,
        "size": list(size),
        "banner": banner,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    final = compress(out.with_suffix(".webm"), out.with_suffix(".mp4"), width=size[0], crf=crf)
    # shrink further if still > 8 MB
    if Path(final).suffix == ".mp4" and Path(final).stat().st_size > 8e6 and crf < 34:
        tmp = out.with_name(out.stem + "_tmp.webm")
        # re-read mp4 as source for tighter encode
        import subprocess
        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(final),
               "-c:v", "libx264", "-preset", "slow", "-crf", "32",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out.with_suffix(".mp4"))]
        subprocess.run(cmd)
        final = out.with_suffix(".mp4")
    summary["file"] = Path(final).name
    summary["megabytes"] = round(Path(final).stat().st_size / 1e6, 2)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return final, summary


def run_side_by_side(own_spec, model_left, model_right, out, max_seconds=None, crf=28):
    """Same video twice: left zero-shot, right adapted. Pose extracted once."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    meta = resolve_own(own_spec)
    video = meta["video"]
    torch.set_num_threads(4)
    mL, inL, stemL = load_model(model_left)
    mR, inR, stemR = load_model(model_right)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (OUT_W, OUT_H)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out.with_suffix(".webm")), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
    )

    def fresh_state():
        return {
            "exercise": "non_activity", "conf": 0.0, "reps": 0, "sub": "", "fps": 0.0,
            "decided": False, "rep_flash": False, "hr": None, "hr_start": None, "hr_end": None,
            "form_line": None, "pc_ts": None,
        }

    poses, times = [], []
    packs = [
        {"model": mL, "inputs": inL, "stem": stemL, "probs": None, "state": fresh_state(),
         "seg_start": None, "seg_ex": None, "cand": None, "streak": 0, "segments": [], "timeline": []},
        {"model": mR, "inputs": inR, "stem": stemR, "probs": None, "state": fresh_state(),
         "seg_start": None, "seg_ex": None, "cand": None, "streak": 0, "segments": [], "timeline": []},
    ]
    index, last_ts, proc = 0, -1, []
    frame_times = meta.get("frame_times")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_times is not None and index < len(frame_times):
                pc_ts = frame_times[index]
                ts = pc_ts - frame_times[0]
            else:
                pc_ts = None
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = ts if ts > last_ts else last_ts + 1000.0 / fps
            last_ts = ts
            if max_seconds and ts > max_seconds * 1000:
                break
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts),
            )
            pose = (
                landmarks_to_pixels([(p.x, p.y) for p in res.pose_landmarks[0]], src_w, src_h)
                if res.pose_landmarks else np.full((18, 2), np.nan, np.float32)
            )
            poses.append(pose)
            times.append(ts)
            for pack in packs:
                pack["state"]["rep_flash"] = False
                if index % EVERY == 0:
                    (pack["probs"], pack["state"], pack["seg_start"], pack["seg_ex"],
                     pack["cand"], pack["streak"]) = _score_step(
                        pack["model"], pack["inputs"], poses, times, ts, pack["probs"],
                        pack["state"], pack["seg_start"], pack["seg_ex"], pack["cand"],
                        pack["streak"], pack["segments"], pack["timeline"], fps,
                    )
                if pc_ts is not None:
                    pack["state"]["hr"] = hr_at(meta.get("hr") or [], pc_ts)
                    pack["state"]["pc_ts"] = pc_ts
            proc.append(time.perf_counter() - t0)
            fps_now = 1.0 / max(np.mean(proc[-30:]), 1e-6)
            for pack in packs:
                pack["state"]["fps"] = fps_now

            # compose half-width portrait panels
            half = OUT_W // 2
            canvas = np.zeros((OUT_H, OUT_W, 3), np.uint8)
            canvas[:] = (11, 16, 22)
            scale = (OUT_H - 80) / src_h
            vw = int(round(src_w * scale)) // 2 * 2
            vh = int(round(src_h * scale)) // 2 * 2
            vid = cv2.resize(bgr, (vw, vh), interpolation=cv2.INTER_AREA)
            for i, pack in enumerate(packs):
                ox = i * half + (half - vw) // 2
                canvas[60:60 + vh, ox:ox + vw] = vid
                img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(img, "RGBA")
                draw_skeleton(d, pose, ox, 60, scale, thick=5)
                label = "Zero-shot" if i == 0 else "Adapted (other person)"
                ex = pack["state"]["exercise"]
                truth = meta.get("truth_exercise")
                mark = ""
                if truth and pack["state"].get("decided") and ex != "non_activity":
                    mark = " ✓" if ex == truth else " ✗"
                d.rectangle([i * half, 0, (i + 1) * half, 56], fill=BG + (255,))
                draw_text(d, (i * half + 20, 12), f"{label}: {NAMES.get(ex, ex) if pack['state'].get('decided') else 'Detecting…'}{mark}",
                       font=font(22, True), fill=INK)
                draw_text(d, (i * half + 20, OUT_H - 40), f"reps {pack['state']['reps']} · {pack['stem']}",
                       font=font(16), fill=MUTED)
                canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            writer.write(canvas)
            index += 1
    cap.release()

    results = []
    for pack in packs:
        main = max(pack["segments"], key=lambda g: (g["end_s"] or g["start_s"]) - g["start_s"]) if pack["segments"] else None
        pred = main["exercise"] if main else None
        results.append({
            "model": pack["stem"],
            "exercise": pred,
            "reps": main["reps"] if main else 0,
            "ok": bool(meta.get("truth_exercise") and pred == meta["truth_exercise"]),
        })
    # end card
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    draw_text(d, (80, 320), "Zero-shot vs adapted", font=font(36, True), fill=ACCENT)
    draw_text(d, (80, 400), f"Truth: {NAMES.get(meta.get('truth_exercise'), meta.get('truth_exercise'))}",
           font=font(28), fill=INK)
    draw_text(d, (80, 460), f"Left  {results[0]['model']}: {NAMES.get(results[0]['exercise'], results[0]['exercise'])} "
           f"{'✓' if results[0]['ok'] else '✗'}", font=font(28, True), fill=OK if results[0]["ok"] else STOP)
    draw_text(d, (80, 520), f"Right {results[1]['model']}: {NAMES.get(results[1]['exercise'], results[1]['exercise'])} "
           f"{'✓' if results[1]['ok'] else '✗'}", font=font(28, True), fill=OK if results[1]["ok"] else STOP)
    draw_text(d, (80, 600), meta["key"], font=font(20), fill=MUTED)
    card = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for _ in range(int(fps * 2)):
        writer.write(card)
    writer.release()

    summary = {
        "own": meta["key"], "frames": index, "seconds": round(times[-1] / 1000, 1) if times else 0,
        "truth_exercise": meta.get("truth_exercise"),
        "left": results[0], "right": results[1],
        "size": list(size),
        "mode": "side_by_side",
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    final = compress(out.with_suffix(".webm"), out.with_suffix(".mp4"), width=OUT_W, crf=crf)
    if Path(final).is_file() and Path(final).stat().st_size > 8e6:
        import subprocess
        import imageio_ffmpeg
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(final),
            "-c:v", "libx264", "-preset", "slow", "-crf", "32",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out.with_suffix(".mp4")),
        ])
        final = out.with_suffix(".mp4")
    summary["file"] = Path(final).name
    summary["megabytes"] = round(Path(final).stat().st_size / 1e6, 2)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return final, summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Annotated demo video: skeleton, exercise, rep count")
    p.add_argument("video", nargs="?", default=None)
    p.add_argument("--own", default=None, help="person/session/set — own phone recording")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="m1_transformer_pose")
    p.add_argument("--side-by-side-model", default=None,
                   help="second model for side-by-side compare (use with --own)")
    p.add_argument("--truth-reps", type=int, default=None)
    p.add_argument("--title", default="")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--show-form", action="store_true")
    p.add_argument("--banner", default=None)
    p.add_argument("--end-reason", default=None)
    p.add_argument("--crf", type=int, default=28)
    a = p.parse_args(argv)

    if a.own and a.side_by_side_model:
        out = a.out or REPORTS / "validation2" / "demo" / "week2" / "compare"
        path, s = run_side_by_side(a.own, a.model, a.side_by_side_model, out, a.max_seconds, a.crf)
        print(f"wrote {path} ({s['megabytes']} MB)")
        print(f"  left  {s['left']}")
        print(f"  right {s['right']}")
        return

    own_meta = None
    video = a.video
    if a.own:
        own_meta = resolve_own(a.own)
        video = str(own_meta["video"])
        if a.truth_reps is None:
            a.truth_reps = own_meta.get("truth_reps")
        if not a.title:
            a.title = own_meta["key"]
    if not video:
        p.error("provide a video path or --own person/session/set")

    out = a.out or REPORTS / "validation2" / "demo" / Path(video).stem
    path, s = run(
        video, out, a.model, a.truth_reps, a.title, a.width, a.max_seconds,
        own_meta=own_meta, show_form=a.show_form, banner=a.banner,
        end_reason=a.end_reason, crf=a.crf,
    )
    print(f"wrote {path} ({s['megabytes']} MB)")
    mark = "✓" if s.get("exercise_ok") else "✗"
    print(f"  exercise {s['exercise']} {mark} (truth {s.get('truth_exercise')}) · "
          f"reps counted {s['reps_counted']} · truth {s['truth_reps']}")
    print(f"  {s['processing_fps_mean']} fps on CPU · size {s.get('size')} · "
          f"pose found {s['pose_found_ratio']:.0%}")


if __name__ == "__main__":
    main()
