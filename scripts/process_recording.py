"""Own recordings → MediaPipe pose + HR join → data/interim/own/.

    python scripts/process_recording.py --all
    python scripts/process_recording.py --all --workers 4
    python scripts/process_recording.py partner/2026-11-10_s01
    python scripts/process_recording.py --all --force

Uses frames_setNN.csv as the clock (extract_pose frame_times_ms). Never writes under data/raw.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from awc.config import FIGURES, OWN_DIR, OWN_INTERIM, OWN_MANIFEST, VIDEO_FPS
from awc.data.own_manifest import load_manifest
from awc.features.own_sync import join_hr_nearest
from awc.features.skeleton import BONES

POSE_X = [f"x{i}" for i in range(18)]
POSE_Y = [f"y{i}" for i in range(18)]


def interim_path(person: str, session_folder: str, set_n: int) -> Path:
    return OWN_INTERIM / person / f"{session_folder}_set{set_n:02d}.parquet"


def qa_path(person: str, session_folder: str, set_n: int) -> Path:
    return OWN_INTERIM / person / f"{session_folder}_set{set_n:02d}_qa.json"


def _load_frame_times(session_dir: Path, set_n: int) -> np.ndarray:
    path = session_dir / f"frames_set{set_n:02d}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df["pc_ts_ms"].to_numpy(dtype=np.float64)


def _load_hr(session_dir: Path) -> list[tuple[int, float]]:
    path = session_dir / "hr.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return [(int(t), float(b)) for t, b in zip(df["pc_ts_ms"], df["bpm"])]


def process_one(row: dict, force: bool = False) -> dict:
    """Process one manifest row. Safe to call from a worker process."""
    from awc.features.pose_video import extract_pose

    person = row["person"]
    session_folder = row["session_folder"]
    set_n = int(row["set"])
    out = interim_path(person, session_folder, set_n)
    qa_out = qa_path(person, session_folder, set_n)
    if out.exists() and qa_out.exists() and not force:
        return {"key": f"{person}/{session_folder}/set{set_n:02d}", "status": "cached"}

    session_dir = OWN_DIR / person / session_folder
    video = session_dir / row["video"]
    if not video.is_file() or video.stat().st_size <= 0:
        return {"key": f"{person}/{session_folder}/set{set_n:02d}", "status": "missing_video"}

    times = _load_frame_times(session_dir, set_n)
    t0 = time.perf_counter()
    pose, fps, n_src, qa = extract_pose(video, frame_times_ms=times, return_qa=True)
    # Resampled clock: start at first frame time, step 1000/30 ms
    t0_ms = float(times[0])
    pc_ts = t0_ms + np.arange(len(pose)) * (1000.0 / VIDEO_FPS)
    hr_rows = _load_hr(session_dir)
    hr, hr_mask = join_hr_nearest(pc_ts, hr_rows)

    data = {
        "frame": np.arange(len(pose), dtype=np.int32),
        "pc_ts_ms": pc_ts.astype(np.float64),
    }
    for j in range(18):
        data[f"x{j}"] = pose[:, j, 0].astype(np.float32)
        data[f"y{j}"] = pose[:, j, 1].astype(np.float32)
    data["hr"] = hr.astype(np.float32)
    data["hr_mask"] = hr_mask.astype(np.int32)
    data["exercise"] = row["exercise"]
    data["person"] = person
    data["session_folder"] = session_folder
    data["set"] = set_n
    df = pd.DataFrame(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    qa_row = {
        **qa,
        "person": person,
        "session_folder": session_folder,
        "set": set_n,
        "exercise": row["exercise"],
        "src_fps": fps,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "hr_mask_mean": float(hr_mask.mean()) if len(hr_mask) else 0.0,
        "parquet": str(out.relative_to(out.parents[2])),
    }
    qa_out.write_text(json.dumps(qa_row, indent=2), encoding="utf-8")
    return {
        "key": f"{person}/{session_folder}/set{set_n:02d}",
        "status": "ok",
        "n_out": len(pose),
        "pct_detected": qa["pct_detected"],
        "elapsed_s": qa_row["elapsed_s"],
    }


def draw_pose_check(row: dict, out_path: Path | None = None) -> Path:
    """Draw the skeleton on 3 frames of one set → reports/figures/own_pose_check.png."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from awc.features.pose_video import extract_pose

    person, session_folder, set_n = row["person"], row["session_folder"], int(row["set"])
    session_dir = OWN_DIR / person / session_folder
    video = session_dir / row["video"]
    times = _load_frame_times(session_dir, set_n)
    pose, _, n_src = extract_pose(video, frame_times_ms=times)

    cap = cv2.VideoCapture(str(video))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_src
    picks = [int(n_vid * f) for f in (0.25, 0.5, 0.75)]
    frames_bgr = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        frames_bgr.append(bgr if ok else None)
    cap.release()

    # Map source video index → resampled pose index by time fraction
    pose_idxs = [min(len(pose) - 1, int(round(i / max(n_vid - 1, 1) * (len(pose) - 1)))) for i in picks]

    out_path = out_path or (FIGURES / "own_pose_check.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, bgr, pi, src_i in zip(axes, frames_bgr, pose_idxs, picks):
        if bgr is None:
            ax.set_title(f"frame {src_i} (missing)")
            ax.axis("off")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        h, w = rgb.shape[:2]
        sk = pose[pi]
        # pose is y-up pixels; image y is down — flip for overlay
        xs = sk[:, 0]
        ys = -sk[:, 1]
        # If y was stored as y-up from image height scale, -y maps back near [0,h]
        # landmarks_to_pixels: y_pixel_up = - (norm_y * height) so -y_up = norm_y * height
        for a, b in BONES:
            if np.isfinite(xs[[a, b]]).all() and np.isfinite(ys[[a, b]]).all():
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], "c-", lw=1.5)
        ok = np.isfinite(xs) & np.isfinite(ys)
        ax.scatter(xs[ok], ys[ok], c="yellow", s=12, zorder=3)
        ax.set_title(f"{person} {session_folder} set{set_n:02d}\nsrc={src_i} pose={pi}")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")
    fig.suptitle("Own pose check (MediaPipe → COCO-18)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="MediaPipe pose for own recordings")
    p.add_argument("session", nargs="?", default=None,
                   help="person/session_folder (e.g. yassine/2026-11-10_SQUAT); omit with --all")
    p.add_argument("--all", action="store_true", help="process every eval/test manifest row")
    p.add_argument("--force", action="store_true", help="recompute even if parquet exists")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pose-check", action="store_true", help="draw skeleton QA figure")
    p.add_argument("--manifest", type=str, default=None)
    args = p.parse_args()

    manifest = load_manifest(Path(args.manifest) if args.manifest else OWN_MANIFEST)
    rows = [r for r in manifest if r["use"] in ("eval", "test")]
    if args.session:
        # accept "person/folder" or just folder
        part = args.session.replace("\\", "/").strip("/")
        if "/" in part:
            person, folder = part.split("/", 1)
            rows = [r for r in rows if r["person"] == person and r["session_folder"] == folder]
        else:
            rows = [r for r in rows if r["session_folder"] == part]
    elif not args.all:
        p.error("pass a session folder or --all")

    if not rows:
        raise SystemExit("no matching manifest rows")

    print(f"{len(rows)} sets to process (workers={args.workers})", flush=True)
    results = []
    if args.workers <= 1:
        for i, row in enumerate(rows, 1):
            r = process_one(row, force=args.force)
            results.append(r)
            print(f"[{i}/{len(rows)}] {r}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_one, row, args.force): row for row in rows}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                results.append(r)
                print(f"[{i}/{len(rows)}] {r}", flush=True)

    if args.pose_check or args.all:
        # Prefer a squat set with good detection for the figure
        pick = next((r for r in rows if r["exercise"] == "squats" and r["use"] == "eval"), rows[0])
        path = draw_pose_check(pick)
        print(f"pose check → {path}", flush=True)

    ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
    print(f"done: {ok}/{len(results)} ok/cached", flush=True)


if __name__ == "__main__":
    main()
