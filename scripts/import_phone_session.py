"""Import a session recorded with the phone app (docs/gym/index.html) into data/raw/own_recordings.

    python scripts/import_phone_session.py "C:/Users/yassi/Downloads/awc_yassine_2026-10-27_s01_session.json"

The set videos (awc_..._setNN.mp4/webm) are looked up next to the JSON unless --videos-dir is given.
Output folder data/raw/own_recordings/<user>/<date>_<session>/:
    sets/setNN.<ext>      the video of each set
    frames_setNN.csv      frame, pc_ts_ms   (phone clock, one row per video frame)
    hr.csv                pc_ts_ms, bpm     (phone clock, whole session including rests)
    sets.csv              same columns as the PC recorder
    meta.json             devices, camera settings, clock note, import checks
Every timestamp is the phone clock (epoch ms), so heart rate and video line up without any sync step.
"""
import argparse
import json
import shutil
from pathlib import Path

from awc.config import OWN_DIR
from awc.data.recordings import FRAMES_HEADER, HR_HEADER, append_set, init_session_files, session_dir

APP = "awc-gym-recorder"


def video_frame_times(path):
    """Presentation time (ms from the first frame) of every frame, read from the file itself."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    times = []
    while True:
        ok = cap.grab()
        if not ok:
            break
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    if not times:
        raise RuntimeError(f"{path} has no frames")
    first = times[0]
    return [t - first for t in times]


def frame_clock(set_info, pts_ms):
    """Phone-clock time of every video frame.

    The file's own frame times give the spacing; the first captured frame logged by the app
    (frames_ms[0]) anchors them on the phone clock, falling back to the recorder start.
    Measured on a Chrome recording: the file can hold one frame more than the app logged,
    so the anchor is good to about one frame (33 ms).
    """
    anchor = set_info["frames_ms"][0] if set_info.get("frames_ms") else set_info["rec_start_ms"]
    return [int(round(anchor + t)) for t in pts_ms]


def import_session(json_path, videos_dir=None, out_root=OWN_DIR):
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if data.get("app") != APP:
        raise ValueError(f"{json_path.name} is not an {APP} session file")
    videos_dir = Path(videos_dir) if videos_dir else json_path.parent

    session_dir(data["user"], data["session_id"])          # validates user and session id
    folder = Path(out_root) / data["user"] / f"{data['date']}_{data['session_id']}"
    init_session_files(folder)
    (folder / "sets").mkdir(exist_ok=True)

    with (folder / "hr.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(HR_HEADER) + "\n")
        for sample in data.get("hr", []):
            f.write(f"{int(sample[0])},{float(sample[1]):.1f}\n")

    checks = []
    for s in data["sets"]:
        src = videos_dir / s["video_file"]
        if not src.exists():
            raise FileNotFoundError(f"missing video {src}")
        n = int(s["set"])
        dst = folder / "sets" / f"set{n:02d}{src.suffix}"
        shutil.copy2(src, dst)
        pts = video_frame_times(dst)
        clock = frame_clock(s, pts)
        with (folder / f"frames_set{n:02d}.csv").open("w", encoding="utf-8") as f:
            f.write(",".join(FRAMES_HEADER) + "\n")
            for i, ts in enumerate(clock):
                f.write(f"{i},{ts}\n")
        append_set(folder, n, s["exercise"], s["start_ms"], s["end_ms"], s["reps"], s["rpe"], s["form_label"])
        seconds = (clock[-1] - clock[0]) / 1000 if len(clock) > 1 else 0.0
        checks.append({
            "set": n,
            "video": dst.name,
            "frames_in_file": len(pts),
            "frames_logged_by_app": len(s.get("frames_ms", [])),
            "fps": round((len(pts) - 1) / seconds, 2) if seconds > 0 else None,
            "first_frame_after_set_start_ms": clock[0] - int(s["start_ms"]),
        })

    hr = data.get("hr", [])
    span_min = (hr[-1][0] - hr[0][0]) / 60000 if len(hr) > 1 else 0
    meta = {
        "user": data["user"],
        "date": data["date"],
        "session_id": data["session_id"],
        "source": "phone-app",
        "clock": "phone epoch ms for frames, heart rate and set marks (column name pc_ts_ms kept for compatibility)",
        "camera_position": data.get("camera_position"),
        "camera": data.get("camera"),
        "hr_source": data.get("hr_source"),
        "hr_device": data.get("hr_device"),
        "hr_samples": len(hr),
        "hr_samples_per_min": round(len(hr) / span_min, 1) if span_min else None,
        "started_at_ms": data.get("started_at_ms"),
        "ended_at_ms": data.get("ended_at_ms"),
        "sets": checks,
        "demo": bool(data.get("demo")),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder, meta


def main(argv=None):
    p = argparse.ArgumentParser(description="Import a phone-app session into data/raw/own_recordings")
    p.add_argument("session_json")
    p.add_argument("--videos-dir", default=None)
    args = p.parse_args(argv)
    folder, meta = import_session(args.session_json, args.videos_dir)
    print(f"imported {folder}")
    print(f"  heart rate: {meta['hr_samples']} samples ({meta['hr_samples_per_min']} per minute) from {meta['hr_device']}")
    for c in meta["sets"]:
        print(f"  set {c['set']}: {c['frames_in_file']} frames, {c['fps']} fps, first frame {c['first_frame_after_set_start_ms']} ms after start")


if __name__ == "__main__":
    main()
