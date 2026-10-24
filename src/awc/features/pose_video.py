"""Video file → COCO-18 skeleton at 30 fps with MediaPipe Pose Landmarker.

Output coordinates are pixels with y up, the same convention as MM-Fit after sync.
Frames where no person is found are NaN; short gaps are interpolated.
"""
import numpy as np

from awc.config import POSE_MODEL, VIDEO_FPS
from awc.features.skeleton import mediapipe_to_coco18

MAX_GAP_SECONDS = 0.5


def landmarks_to_pixels(landmarks, width, height) -> np.ndarray:
    """MediaPipe normalized landmarks (33, 2) → COCO-18 pixels, y up."""
    lm = np.asarray(landmarks, dtype=np.float64)[:, :2].copy()
    lm[:, 0] *= width
    lm[:, 1] *= height
    return mediapipe_to_coco18(lm, y_down=True)


def fill_gaps(seq, max_gap) -> np.ndarray:
    """Linearly interpolate NaN runs no longer than max_gap frames. Longer runs stay NaN."""
    seq = np.asarray(seq, dtype=np.float64).copy()
    missing = ~np.isfinite(seq).all(axis=(1, 2))
    if not missing.any() or missing.all():
        return seq
    t = np.arange(len(seq))
    good = ~missing
    flat = seq.reshape(len(seq), -1)
    for c in range(flat.shape[1]):
        flat[missing, c] = np.interp(t[missing], t[good], flat[good, c])
    # Undo the fill for long runs and for the edges outside the first/last detection.
    run_start = None
    for i in range(len(seq) + 1):
        is_missing = i < len(seq) and missing[i]
        if is_missing and run_start is None:
            run_start = i
        elif not is_missing and run_start is not None:
            at_edge = run_start == 0 or i == len(seq)
            if at_edge or i - run_start > max_gap:
                flat[run_start:i] = np.nan
            run_start = None
    return flat.reshape(seq.shape)


def resample(seq, fps_in, fps_out=VIDEO_FPS) -> np.ndarray:
    """Resample (T, 18, 2) in time with linear interpolation. NaN frames stay NaN."""
    seq = np.asarray(seq, dtype=np.float64)
    n_in = len(seq)
    if n_in == 0 or abs(fps_in - fps_out) < 1e-6:
        return seq.copy()
    duration = (n_in - 1) / fps_in
    n_out = int(np.floor(duration * fps_out)) + 1
    t_in = np.arange(n_in) / fps_in
    t_out = np.arange(n_out) / fps_out
    flat = seq.reshape(n_in, -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_out, t_in, flat[:, c])
    # A resampled frame is missing if either neighbouring source frame was missing.
    src_missing = ~np.isfinite(flat).all(axis=1)
    if src_missing.any():
        lo = np.clip(np.floor(t_out * fps_in).astype(int), 0, n_in - 1)
        hi = np.clip(lo + 1, 0, n_in - 1)
        out[src_missing[lo] | src_missing[hi]] = np.nan
    return out.reshape(n_out, *seq.shape[1:])


def resample_times(seq, times_ms, fps_out=VIDEO_FPS) -> np.ndarray:
    """Resample (T, 18, 2) onto a regular fps_out grid using real per-frame times.

    For our own recordings: times_ms comes from frames.csv (PC clock), so uneven phone
    frame rates do not distort rep durations. NaN frames stay NaN.
    """
    seq = np.asarray(seq, dtype=np.float64)
    t_in = (np.asarray(times_ms, dtype=np.float64) - float(times_ms[0])) / 1000.0
    if len(t_in) != len(seq):
        raise ValueError(f"{len(t_in)} timestamps for {len(seq)} frames")
    if len(seq) < 2:
        return seq.copy()
    if np.any(np.diff(t_in) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    flat = seq.reshape(len(seq), -1)
    out = np.empty((len(t_out), flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_out, t_in, flat[:, c])
    src_missing = ~np.isfinite(flat).all(axis=1)
    if src_missing.any():
        hi = np.clip(np.searchsorted(t_in, t_out), 0, len(t_in) - 1)
        lo = np.clip(hi - 1, 0, len(t_in) - 1)
        out[src_missing[lo] | src_missing[hi]] = np.nan
    return out.reshape(len(t_out), *seq.shape[1:])


def extract_pose(video_path, model_path=POSE_MODEL, frame_times_ms=None):
    """Run Pose Landmarker on every frame. Returns (pose (T, 18, 2) at 30 fps, source fps, source frames).

    frame_times_ms: real capture time of every frame (frames.csv of our recordings). When given,
    it is the clock: the video's fps is ignored. Without it (public videos), the video fps is used.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    if not np.isfinite(fps) or fps <= 1:
        fps = VIDEO_FPS
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    frames = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            height, width = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = (int(frame_times_ms[index] - frame_times_ms[0]) if frame_times_ms is not None
                  and index < len(frame_times_ms) else int(round(index * 1000 / fps)))
            result = landmarker.detect_for_video(image, ts)
            if result.pose_landmarks:
                lm = [(p.x, p.y) for p in result.pose_landmarks[0]]
                frames.append(landmarks_to_pixels(lm, width, height))
            else:
                frames.append(np.full((18, 2), np.nan, dtype=np.float32))
            index += 1
    cap.release()
    raw = np.asarray(frames, dtype=np.float64).reshape(-1, 18, 2)
    if frame_times_ms is not None:
        times = np.asarray(frame_times_ms[:len(raw)], dtype=np.float64)
        if len(times) != len(raw):
            raise ValueError(f"video has {len(raw)} frames but {len(times)} timestamps")
        real_fps = (len(times) - 1) / max((times[-1] - times[0]) / 1000.0, 1e-6)
        filled = fill_gaps(raw, max_gap=int(round(MAX_GAP_SECONDS * real_fps)))
        return resample_times(filled, times).astype(np.float32), float(real_fps), len(raw)
    filled = fill_gaps(raw, max_gap=int(round(MAX_GAP_SECONDS * fps)))
    return resample(filled, fps).astype(np.float32), float(fps), len(raw)
