"""RepCount-A test split (Hu et al., TransRAC, CVPR 2022), public gym videos from YouTube.

Stored as Lance in data/raw/repcount/test.lance, one row per video with the mp4 bytes.
Labels: action_type, count (reps), cycle_bounds (flat list of start/end frame pairs, source fps).
Academic use only. Not in git.
"""
import json

from awc.config import REPCOUNT_DIR

# RepCount spells some actions two ways.
TO_MMFIT = {
    "squat": "squats",
    "squant": "squats",
    "push_up": "pushups",
    "pushups": "pushups",
    "situp": "situps",
    "jump_jack": "jumping_jacks",
    "jumpjacks": "jumping_jacks",
}
# Workouts without an MM-Fit class. Used for rep counting only.
REPS_ONLY = {
    "bench_pressing": "bench_press",
    "benchpressing": "bench_press",
    "front_raise": "front_raise",
    "frontraise": "front_raise",
    "pull_up": "pull_up",
    "pullups": "pull_up",
}


def exercise_name(action_type):
    """Canonical exercise for a RepCount action, or None when we do not use it."""
    return TO_MMFIT.get(action_type) or REPS_ONLY.get(action_type)


def dataset(path=None):
    import lance
    return lance.dataset(str(path or REPCOUNT_DIR / "test.lance"))


def videos(path=None):
    """Metadata for the usable videos: video_id, exercise, mmfit (bool), count, cycle_bounds, row."""
    ds = dataset(path)
    cols = ["video_id", "action_type", "count", "cycle_bounds_json"]
    out = []
    for row, rec in enumerate(ds.to_table(columns=cols).to_pylist()):
        name = exercise_name(rec["action_type"])
        if name is None:
            continue
        out.append({
            "video_id": rec["video_id"],
            "action_type": rec["action_type"],
            "exercise": name,
            "mmfit": rec["action_type"] in TO_MMFIT,
            "count": int(rec["count"]),
            "cycle_bounds": json.loads(rec["cycle_bounds_json"] or "[]"),
            "row": row,
        })
    return out


def write_video(row, dest, path=None):
    """Copy the mp4 bytes of one row to dest."""
    ds = dataset(path)
    blob = ds.take_blobs("video_blob", indices=[row])[0]
    with blob as f, open(dest, "wb") as out:
        out.write(f.read())
    return dest
