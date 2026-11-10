"""Write gym-station sessions under data/raw/own_recordings (resume-safe)."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from awc.data import recordings as rec
from awc.station.chunks import ChunkAssembler


def verify_video(path: Path) -> dict:
    """Remux the concatenated MediaRecorder chunks into a normal file, then read
    every frame's presentation time (ms from the first frame).

    iOS uploads fragmented MP4 pieces; `ffmpeg -c copy` turns them into a
    regular MP4 (no re-encode) so OpenCV / MediaPipe read it reliably.
    """
    path = Path(path)
    remuxed = False
    if path.is_file() and path.stat().st_size > 32:
        remuxed = _remux(path)
    try:
        pts, fps = _frame_times(path)
    except Exception:  # noqa: BLE001
        pts, fps = [], None
    n = len(pts)
    return {"frames": n, "fps": fps, "ok": n > 0, "remuxed": remuxed, "path": str(path), "pts_ms": pts}


def _frame_times(path: Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], None
    pts: list[float] = []
    while len(pts) < 200_000:
        if not cap.grab():
            break
        pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    cap.release()
    if len(pts) >= 2 and pts[-1] > pts[0]:
        fps = (len(pts) - 1) / ((pts[-1] - pts[0]) / 1000.0)
    else:
        fps = None
    # Some containers report 0 for every frame: treat as unknown.
    if len(pts) >= 2 and max(pts) <= 0:
        pts = []
    return pts, fps


def _remux(path: Path) -> bool:
    try:
        import imageio_ffmpeg
    except ImportError:
        return False
    tmp = path.with_name(path.stem + ".remux" + path.suffix)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-c", "copy"]
    if path.suffix.lower() == ".mp4":
        cmd += ["-movflags", "+faststart"]
    cmd.append(str(tmp))
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        if tmp.is_file() and tmp.stat().st_size > 32:
            raw = path.with_name(path.stem + ".raw" + path.suffix)
            path.replace(raw)  # keep the original upload as a backup
            tmp.replace(path)
            return True
    except Exception:  # noqa: BLE001
        pass
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False


def write_frames_csv(folder: Path, set_index: int, pc_ts_list, n_decoded: int | None = None,
                     name: str | None = None):
    """Write frames_setNN.csv. If counts mismatch, spread timestamps linearly over decoded count."""
    path = folder / (name or f"frames_set{set_index:02d}.csv")
    stamps = [int(round(t)) for t in pc_ts_list]
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
        self.phone_rec_start: float | None = None  # phone clock, MediaRecorder "start"
        self.start_ms = None
        self.end_ms = None
        self.n_chunks_declared = None
        self.finalized: dict | None = None


class StationSession:
    def __init__(self, folder: Path, meta: dict | None = None):
        self.folder = Path(folder)
        rec.init_session_files(self.folder, station=True)
        self.meta = meta or {}
        self.active: ActiveSet | None = None
        self.sets: dict[int, ActiveSet] = {}
        self._write_meta()

    def _write_meta(self):
        rec.write_meta(self.folder, self.meta)

    def update_meta(self, **kwargs):
        self.meta.update({k: v for k, v in kwargs.items() if v is not None})
        self._write_meta()

    def next_set(self) -> int:
        n = rec.next_set_index(self.folder)
        # a recorded-but-not-yet-saved set also takes its number (never overwrite a video)
        if self.sets:
            n = max(n, max(self.sets) + 1)
        for p in (self.folder / "sets").glob("set*.*"):
            digits = p.name[3:5]
            if digits.isdigit() and p.stat().st_size > 0:
                n = max(n, int(digits) + 1)
        return n

    def start_set(self, set_index: int, start_ms: int, ext: str = "mp4") -> ActiveSet:
        a = ActiveSet(self.folder, set_index, ext=ext)
        a.start_ms = int(start_ms)
        self.active = a
        self.sets[int(set_index)] = a
        rec.append_event(self.folder, start_ms, "set_start", f"set={set_index}")
        return a

    def get_set(self, set_index: int) -> ActiveSet | None:
        return self.sets.get(int(set_index))

    def add_chunk(self, seq: int, data: bytes, set_index: int | None = None) -> list[int]:
        a = self.get_set(set_index) if set_index is not None else self.active
        if a is None:
            raise RuntimeError("no such set")
        return a.assembler.add(seq, data)

    def add_phone_frames(self, phone_ts_list, set_index: int | None = None):
        a = self.get_set(set_index) if set_index is not None else self.active
        if a is not None:
            a.phone_frame_ts.extend(float(t) for t in phone_ts_list)

    def finish_upload(self, n_chunks: int, end_ms: int, phone_to_pc, set_index: int | None = None) -> dict:
        a = self.get_set(set_index) if set_index is not None else self.active
        if a is None:
            raise RuntimeError("no such set")
        if a.finalized is not None:
            return a.finalized
        a.n_chunks_declared = int(n_chunks)
        a.end_ms = int(end_ms)
        stats = verify_video(a.video_path)
        pts = stats.pop("pts_ms", None) or []
        n_dec = int(stats["frames"])
        phone_pc = [phone_to_pc(t) for t in a.phone_frame_ts]
        note = None
        clock = "none"
        if pts and a.phone_rec_start is not None:
            # Best: real per-frame times from the file + recorder start on the laptop clock.
            base = phone_to_pc(a.phone_rec_start)
            stamps = [base + (p - pts[0]) for p in pts]
            write_frames_csv(self.folder, a.set_index, stamps)
            clock = "video_pts+rec_start"
        elif phone_pc:
            note = write_frames_csv(self.folder, a.set_index, phone_pc, n_decoded=n_dec or None)
            clock = "phone_rvfc"
        elif n_dec:
            t0 = a.start_ms or end_ms
            stamps = np.linspace(t0, end_ms, n_dec).tolist()
            write_frames_csv(self.folder, a.set_index, stamps)
            clock = "linear_start_end"
            note = "no frame timestamps from phone; linear spread between start and end"
        if phone_pc:  # keep the raw phone frame callbacks for checking
            write_frames_csv(self.folder, a.set_index, phone_pc,
                             name=f"frames_phone_set{a.set_index:02d}.csv")
        if note:
            rec.append_event(self.folder, int(time.time() * 1000), "frame_note", note)
        a.finalized = {
            "set": a.set_index, "video": stats, "frame_note": note,
            "n_phone_frames": len(phone_pc), "frame_clock": clock,
        }
        rec.append_event(self.folder, int(time.time() * 1000), "video_saved",
                         f"set={a.set_index} frames={n_dec} fps={stats.get('fps')} clock={clock}")
        if self.active is a:
            self.active = None
        return a.finalized

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
