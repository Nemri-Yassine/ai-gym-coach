"""Run the pipeline on a real video, frame by frame, as it would run live, and save an annotated video.

    python scripts/demo_video.py path/to/video.mp4 --truth-reps 14 --title "RepCount-A · squats"

Each frame only uses the past: MediaPipe skeleton -> every 0.1 s Model 1 (camera only) reads the
last 3 s; a new exercise is accepted once it stays the top class for 2 s -> the rep counter counts since that exercise started, with the
exercise Model 1 chose. The overlay shows the skeleton, the exercise with its confidence, the rep
count and the processing speed. Output: <out>.mp4 (H.264, small, plays everywhere; needs the
imageio-ffmpeg package, else a large <out>.webm is kept) and <out>.json.
Works on public videos and on our own phone recordings (their frame times come from the file).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from awc.config import MMFIT_CLASSES, MODELS, POSE_MODEL, REPORTS
from awc.features.pose_video import landmarks_to_pixels, resample_times
from awc.features.reps import count_reps
from awc.features.skeleton import BONES, normalize
from awc.models.m1_recognition import M1Model

WINDOW_S = 3.0
EVERY = 3                    # run Model 1 every 3 frames (0.1 s at 30 fps)
EMA = 0.35                   # smoothing of the displayed probabilities
SWITCH_STEPS = 20           # a new exercise must stay the top class for 20 model steps (2 s) in a row
NA = MMFIT_CLASSES.index("non_activity")
NAMES = {"squats": "Squats", "lunges": "Lunges", "bicep_curls": "Biceps curls", "situps": "Sit-ups",
         "pushups": "Push-ups", "tricep_extensions": "Triceps extensions", "dumbbell_rows": "Dumbbell rows",
         "jumping_jacks": "Jumping jacks", "dumbbell_shoulder_press": "Shoulder press",
         "lateral_shoulder_raises": "Lateral raises", "non_activity": "No exercise"}

INK, MUTED, ACCENT, OK, BG = (238, 242, 246), (140, 154, 167), (115, 147, 255), (76, 196, 135), (11, 16, 22)


def font(size, bold=False):
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_model(name):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location="cpu", weights_only=False)
    model = M1Model(ckpt["arch"], ckpt["inputs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["inputs"]


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


def draw_overlay(frame, pose_px, state, scale):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    if pose_px is not None and np.isfinite(pose_px).all():
        pts = [(float(x) * scale, float(-y) * scale) for x, y in pose_px]   # back to image y-down
        for a, b in BONES:
            if a >= 14 or b >= 14:
                continue
            d.line([pts[a], pts[b]], fill=(238, 242, 246, 230), width=max(3, int(4 * scale / 2)))
        for j, p in enumerate(pts[:14]):
            r = 5 if j not in (8, 11) else 7
            d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=ACCENT + (255,))
    # panel
    pw, ph = 380, 236
    d.rounded_rectangle([16, 16, 16 + pw, 16 + ph], 16, fill=BG + (205,))
    d.text((32, 28), "AI WORKOUT COACH · LIVE", font=font(15, True), fill=ACCENT)
    ex = state["exercise"]
    d.text((32, 54), NAMES.get(ex, ex), font=font(34, True), fill=INK if ex != "non_activity" else MUTED)
    if state.get("final"):
        d.text((32, 98), "RESULT FOR THE WHOLE VIDEO", font=font(15, True), fill=OK)
    else:
        conf = state["conf"]
        d.rounded_rectangle([32, 104, 32 + 280, 116], 6, fill=(38, 49, 60, 255))
        d.rounded_rectangle([32, 104, 32 + max(12, int(280 * conf)), 116], 6, fill=ACCENT + (255,))
        d.text((322, 98), f"{conf * 100:.0f}%", font=font(18, True), fill=INK)
    d.text((32, 130), "REPS", font=font(15, True), fill=MUTED)
    d.text((32, 146), str(state["reps"]), font=font(64, True), fill=OK if state["reps"] else INK)
    d.text((150, 158), state["sub"], font=font(15), fill=MUTED)
    d.text((150, 180), f"{state['fps']:.0f} fps on CPU", font=font(15), fill=MUTED)
    d.text((150, 202), "Model 1, camera only, last 3 s", font=font(15), fill=MUTED)
    # footer
    if state.get("footer"):
        tw = d.textlength(state["footer"], font=font(17))
        d.rounded_rectangle([W - tw - 40, H - 46, W - 16, H - 14], 10, fill=BG + (205,))
        d.text((W - tw - 28, H - 42), state["footer"], font=font(17), fill=INK)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def compress(webm, mp4, width=960, crf=30):
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
        Path(webm).unlink()
        return Path(mp4)
    return webm


def run(video, out, model_name="m1_transformer_pose", truth_reps=None, title="", width=1280, max_seconds=None):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    torch.set_num_threads(4)
    model, inputs = load_model(model_name)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = width / src_w
    size = (width, int(round(src_h * scale)) // 2 * 2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out.with_suffix(".webm")), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    options = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
                                           running_mode=vision.RunningMode.VIDEO, num_poses=1)
    poses, times = [], []
    probs_s = None
    state = {"exercise": "non_activity", "conf": 0.0, "reps": 0, "sub": "warming up: 3 s of history needed", "fps": 0.0, "footer": title}
    seg_start, seg_ex = None, None
    candidate, streak = None, 0
    segments = []            # one entry per exercise bout: exercise, start/end seconds, reps
    timeline, proc = [], []
    index, last_ts = 0, -1
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = ts if ts > last_ts else last_ts + 1000.0 / fps       # strictly increasing clock
            last_ts = ts
            if max_seconds and ts > max_seconds * 1000:
                break
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(ts))
            if res.pose_landmarks:
                pose = landmarks_to_pixels([(p.x, p.y) for p in res.pose_landmarks[0]], src_w, src_h)
            else:
                pose = np.full((18, 2), np.nan, np.float32)
            poses.append(pose)
            times.append(ts)

            if index % EVERY == 0:
                w = window(poses, times, ts, WINDOW_S)
                if w is not None:
                    x = torch.from_numpy(normalize(w).reshape(1, 90, 36).astype(np.float32))
                    with torch.no_grad():
                        logits = model(x) if inputs == "pose" else model(x, torch.zeros(1, 90, 6), torch.zeros(1))
                        p = torch.softmax(logits, 1).numpy()[0]
                    probs_s = p if probs_s is None else EMA * p + (1 - EMA) * probs_s
                    top = MMFIT_CLASSES[int(np.argmax(probs_s))]
                    # switch only when the same new class stays on top for 1 s: one noisy step cannot flip it
                    if top == state["exercise"]:
                        candidate, streak = None, 0
                    else:
                        streak = streak + 1 if top == candidate else 1
                        candidate = top
                        if streak >= SWITCH_STEPS:
                            state["exercise"], candidate, streak = top, None, 0
                            if top != "non_activity" and top != seg_ex:
                                back = (WINDOW_S + SWITCH_STEPS * EVERY / fps) * 1000
                                seg_ex, seg_start, state["reps"] = top, max(0.0, ts - back), 0
                                segments.append({"exercise": top, "start_s": round(seg_start / 1000, 1), "end_s": None, "reps": 0})
                    state["conf"] = float(probs_s[int(np.argmax(probs_s))])
                    state["sub"] = "reps since the exercise started" if seg_ex else "looking for an exercise"
                    timeline.append([round(ts / 1000, 2), MMFIT_CLASSES[int(np.argmax(probs_s))], round(state["conf"], 3)])
                if seg_ex and state["exercise"] == seg_ex:
                    keep = [i for i, t in enumerate(times) if t >= seg_start and np.isfinite(poses[i]).all()]
                    if len(keep) > 30:
                        seq = resample_times(np.asarray([poses[i] for i in keep]), np.asarray([times[i] for i in keep]))
                        n, _ = count_reps(normalize(seq), fps=30, exercise=seg_ex)
                        state["reps"] = max(state["reps"], int(n))     # a confirmed rep never disappears
                        segments[-1]["reps"], segments[-1]["end_s"] = state["reps"], round(ts / 1000, 1)
            proc.append(time.perf_counter() - t0)
            state["fps"] = 1.0 / max(np.mean(proc[-30:]), 1e-6)
            frame = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            writer.write(draw_overlay(frame, pose, state, scale))
            index += 1
    cap.release()
    # the result is the exercise done the longest; a bout at the very end (standing up) does not erase it
    main = max(segments, key=lambda g: (g["end_s"] or g["start_s"]) - g["start_s"]) if segments else None
    if index:
        if main:
            state["exercise"], state["reps"] = main["exercise"], main["reps"]
            state["final"], state["sub"] = True, f"longest bout: {main['start_s']:.0f}–{main['end_s'] or main['start_s']:.0f} s"
        state["footer"] = f"Counted {state['reps']} reps" + (f" · ground truth {truth_reps}" if truth_reps else "")
        last = draw_overlay(frame, poses[-1], state, scale)
        for _ in range(int(fps * 2)):
            writer.write(last)
    writer.release()
    summary = {
        "video": Path(video).name, "frames": index, "seconds": round(times[-1] / 1000, 1) if times else 0,
        "model": model_name, "exercise": main["exercise"] if main else None,
        "reps_counted": main["reps"] if main else 0, "truth_reps": truth_reps, "segments": segments,
        "processing_fps_mean": round(1 / float(np.mean(proc)), 1) if proc else None,
        "realtime_capable": bool(proc) and 1 / float(np.mean(proc)) >= fps,
        "pose_found_ratio": round(float(np.mean([np.isfinite(p).all() for p in poses])), 3) if poses else 0,
        "timeline": timeline,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    final = compress(out.with_suffix(".webm"), out.with_suffix(".mp4"))
    summary["file"] = Path(final).name
    summary["megabytes"] = round(Path(final).stat().st_size / 1e6, 1)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return final, summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Annotated demo video: skeleton, exercise, rep count")
    p.add_argument("video")
    p.add_argument("--out", default=None)
    p.add_argument("--model", default="m1_transformer_pose")
    p.add_argument("--truth-reps", type=int, default=None)
    p.add_argument("--title", default="")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--max-seconds", type=float, default=None)
    a = p.parse_args(argv)
    out = a.out or REPORTS / "validation2" / "demo" / Path(a.video).stem
    path, s = run(a.video, out, a.model, a.truth_reps, a.title, a.width, a.max_seconds)
    print(f"wrote {path} ({s['megabytes']} MB)")
    print(f"  exercise {s['exercise']} · reps counted {s['reps_counted']} · truth {s['truth_reps']}")
    print(f"  {s['processing_fps_mean']} fps on CPU · real-time capable: {s['realtime_capable']} · pose found {s['pose_found_ratio']:.0%}")


if __name__ == "__main__":
    main()
