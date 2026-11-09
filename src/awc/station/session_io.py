"""Write gym-station sessions under data/raw/own_recordings (resume-safe)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from awc.data import recordings as rec
from awc.station.chunks import ChunkAssembler


def verify_video(path: Path) -> dict:
    """Open with OpenCV; remux with ffmpeg -c copy if needed. Returns frame stats."""
    path = Path(path)
    try:
        n, fps = _count_frames(path)
    except Exception:  # noqa: BLE001
        n, fps = 0, None
    remuxed = False
    if n <= 0 and path.stat().st_size > 32:
        remuxed = _remux(path)
        try:
            n, fps = _count_frames(path)
        except Exception:  # noqa: BLE001
            n, fps = 0, None
    return {"frames": n, "fps": fps, "ok": n > 0, "remuxed": remuxed, "path": str(path)}


def _count_frames(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, None
    # Prefer metadata when available (avoids hanging on corrupt/partial fMP4).
    meta_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    if meta_n > 0:
        cap.release()
        return meta_n, float(fps) if fps else None
    n = 0
    while n < 100_000:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n, float(fps) if fps else None


def _remux(path: Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(path), "-c", "copy", str(tmp),
    ]
    import subprocess

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
        return True
    except Exception:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def write_frames_csv(folder: Path, set_index: int, pc_ts_list: list[int], n_decoded: int | None = None):
    """Write frames_setNN.csv. If counts mismatch, spread timestamps linearly over decoded count."""
    path = folder / f"frames_set{set_index:02d}.csv"
    stamps = [int(t) for t in pc_ts_list]
    note = None
    if n_decoded is not None and n_decoded > 0 and stamps and n_decoded != len(stamps):
        note = f"frame count mismatch phone={len(stamps)} decoded={n_decoded}; linear spread"
        t0, t1 = stamps[0], stamps[-1]
        if n_decoded == 1:
            stamps = [t0]
        else:
            stamps = [int(round(t0 + (t1 - t0) * i / (n_decoded - 1))) for i in range(n_decoded)]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(rec.FRAMES_HEADER) + "\n")
        for i, ts in enumerate(stamps):
            f.write(f"{i},{ts}\n")
    return note


class ActiveSet:
    def __init__(self, folder: Path, set_index: int, ext: str = "mp4"):
        self.set_index = int(set_index)
        self.ext = ext.lstrip(".")
        self.video_path = Path(folder) / "sets" / f"set{self.set_index:02d}.{self.ext}"
        self.assembler = ChunkAssembler(self.video_path)
        self.phone_frame_ts: list[float] = []
        self.start_ms = None
        self.end_ms = None
        self.n_chunks_declared = None


class StationSession:
    def __init__(self, folder: Path, meta: dict | None = None):
        self.folder = Path(folder)
        rec.init_session_files(self.folder, station=True)
        self.meta = meta or {}
        self.active: ActiveSet | None = None
        self._write_meta()

    def _write_meta(self):
        rec.write_meta(self.folder, self.meta)

    def update_meta(self, **kwargs):
        self.meta.update(kwargs)
        self._write_meta()

    def next_set(self) -> int:
        return rec.next_set_index(self.folder)

    def start_set(self, set_index: int, start_ms: int, ext: str = "mp4") -> ActiveSet:
        self.active = ActiveSet(self.folder, set_index, ext=ext)
        self.active.start_ms = int(start_ms)
        rec.append_event(self.folder, start_ms, "set_start", f"set={set_index}")
        return self.active

    def add_chunk(self, seq: int, data: bytes) -> list[int]:
        if not self.active:
            raise RuntimeError("no active set")
        return self.active.assembler.add(seq, data)

    def add_phone_frames(self, phone_ts_list):
        if self.active:
            self.active.phone_frame_ts.extend(float(t) for t in phone_ts_list)

    def finish_upload(self, n_chunks: int, end_ms: int, phone_to_pc) -> dict:
        if not self.active:
            raise RuntimeError("no active set")
        active = self.active
        active.n_chunks_declared = int(n_chunks)
        active.end_ms = int(end_ms)
        # wait briefly for trailing chunks already in flight is caller's job
        stats = verify_video(active.video_path)
        pc_frames = [int(round(phone_to_pc(t))) for t in active.phone_frame_ts]
        note = write_frames_csv(self.folder, active.set_index, pc_frames, n_decoded=stats["frames"])
        if note:
            rec.append_event(self.folder, int(time.time() * 1000), "frame_mismatch", note)
        return {"set": active.set_index, "video": stats, "frame_note": note, "n_phone_frames": len(pc_frames)}

    def save_set_labels(self, set_index, exercise, start_ms, end_ms, reps, rpe, form_label, extra: dict):
        rec.append_set(self.folder, set_index, exercise, start_ms, end_ms, reps, rpe, form_label)
        rec.append_set_extra(self.folder, set_index, **extra)
        rec.append_event(self.folder, int(time.time() * 1000), "set_saved", f"set={set_index}")

    def append_hr(self, pc_ts_ms, bpm, rr_ms=None):
        rec.append_hr(self.folder / "hr.csv", pc_ts_ms, bpm)
        if rr_ms:
            path = self.folder / "hr_rr.csv"
            with path.open("a", encoding="utf-8") as f:
                for rr in rr_ms:
                    f.write(f"{int(pc_ts_ms)},{rr:.3f}\n")

    def append_clock(self, pc_ts_ms, offset_ms, rtt_ms):
        path = self.folder / "clock_sync.csv"
        with path.open("a", encoding="utf-8") as f:
            f.write(f"{int(pc_ts_ms)},{float(offset_ms):.3f},{float(rtt_ms):.3f}\n")

    def hr_stats(self, start_ms, end_ms):
        rows = []
        path = self.folder / "hr.csv"
        if path.is_file():
            import csv
            with path.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    t = int(float(r["pc_ts_ms"]))
                    if start_ms <= t <= end_ms:
                        rows.append(float(r["bpm"]))
        if not rows:
            return {"hr_mean": "", "hr_peak": "", "hr_start": "", "hr_end": ""}
        arr = np.asarray(rows, dtype=float)
        return {
            "hr_mean": round(float(arr.mean()), 1),
            "hr_peak": round(float(arr.max()), 1),
            "hr_start": round(float(arr[0]), 1),
            "hr_end": round(float(arr[-1]), 1),
        }
