"""Model 2 v2 at inference: judge a whole set from its 3-second windows.

Trained by awc.training.train_m2_gym on real gym people (Mendeley, 25-26 people per exercise).
Covers squats, push-ups, bicep curls, dumbbell rows, dumbbell shoulder press, tricep extensions
(+ side bends, which Model 1 does not recognise).

Personal calibration (optional): once per person, exercise and camera position, save a short
reference set done with good form (`save_reference`). Later sets are then judged relative to
that person's own movement, which removes body-shape and camera differences.
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np

from awc.config import MODELS, OWN_LIVE_DIR
from awc.features.form_windows import set_windows, window_features

MODEL_PATH = MODELS / "m2_gym.pickle"
# A window is flagged only when the model is fairly sure it is bad: fewer false alarms.
FLAG_P_BAD = 0.70
# Set verdict from the mean P(correct) over its windows.
GOOD_ABOVE, BAD_BELOW = 0.55, 0.40
# Personal calibration is OFF: on Mendeley the reference and the good test windows come from the
# same recording (one good set per person), so the calibrated model learned "different recording =
# bad". On our own sets it called only 23% of correct windows correct (plain model: 63%).
# See reports/m2_gym_windows.md. Turn on only once it is trained with references from another set.
USE_CALIBRATION = False
# Relative judgement: flag a set when its mean P(correct) is this much below the person's usual
# (median of their earlier sets of the same exercise). With the ensemble (RF + boosting + temporal
# CNN), ranking a person's two sets is right 85% of the time on unseen people vs 69% for the absolute
# verdict. With this drop: bad sets caught 70% (514 Mendeley pairs), our own correct sets flagged
# 28% (only 18 pairs; boosting alone: 67% / 17%). scripts/analysis/m2_relative.py, ensemble.json.
RELATIVE_DROP = 0.10


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    if not path.is_file():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def _ensemble_tcn(id_pack: int):
    """Rebuild the temporal CNN once per loaded pack."""
    import torch

    from awc.training.train_m2_gym import TCN

    pack = load_model()
    ens = pack["ensemble"]
    net = TCN(ens["tcn_channels"], len(pack["exercises"]))
    net.load_state_dict(ens["tcn_state"])
    return net.eval()


def ensemble_proba(pack, exercise: str, windows) -> np.ndarray:
    """Mean P(correct) of forest + boosting + temporal CNN per window."""
    import torch

    from awc.features.form_windows import angle_sequence

    rf, hgb = pack["ensemble"]["forests"][exercise]
    feats = window_features(windows)
    p = [rf.predict_proba(feats)[:, 1], hgb.predict_proba(feats)[:, 1]]
    from awc.training.train_m2_gym import TCN

    if pack is load_model():
        net = _ensemble_tcn(id(pack))
    else:  # pack passed in directly (training script): build the net from it
        net = TCN(pack["ensemble"]["tcn_channels"], len(pack["exercises"]))
        net.load_state_dict(pack["ensemble"]["tcn_state"])
        net.eval()
    ex = torch.full((len(windows),), list(pack["exercises"]).index(exercise), dtype=torch.long)
    onehot = torch.eye(len(pack["exercises"]))[ex]
    with torch.no_grad():
        p.append(torch.sigmoid(net(torch.from_numpy(angle_sequence(windows)), onehot)).numpy())
    return np.mean(p, axis=0)


def reference_path(person: str, exercise: str):
    return OWN_LIVE_DIR / person / "form_reference" / f"{exercise}.npy"


def save_reference(person: str, exercise: str, pose_n) -> int:
    """Store the mean features of the first K windows of a good-form set. Returns windows used."""
    pack = load_model()
    k = pack["k_ref"] if pack else 3
    w = set_windows(pose_n)[:k]
    if not len(w):
        return 0
    path = reference_path(person, exercise)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, window_features(w).mean(axis=0))
    return int(len(w))


def history_path(person: str):
    return OWN_LIVE_DIR / person / "form_history.json"


def _load_history(person: str) -> dict:
    import json

    path = history_path(person)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_history(person: str, history: dict):
    import json

    path = history_path(person)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=1), encoding="utf-8")


def relative_verdict(p_correct: float, earlier: list[float]):
    """Compare a set to the person's usual (median of earlier sets). None without history."""
    if not earlier:
        return None
    usual = float(np.median(earlier))
    worse = p_correct < usual - RELATIVE_DROP
    return {"usual": round(usual, 3), "n_earlier": len(earlier),
            "verdict": "worse than usual" if worse else "as usual"}


def check_set(exercise: str, pose_n, person: str | None = None, set_key: str | None = None) -> dict | None:
    """pose_n: normalized set skeleton (T, 18, 2) at 30 fps. None if the exercise is not covered.

    With person and set_key, the set's score is stored and later sets of the same exercise are
    also judged relative to this person's usual (set_key identifies the set, e.g. 'session/set01').
    """
    pack = load_model()
    if pack is None or exercise not in pack["plain"]:
        return None
    w = set_windows(pose_n)
    if not len(w):
        return None
    feats = window_features(w)
    ref = None
    if person is not None and USE_CALIBRATION:
        rp = reference_path(person, exercise)
        if rp.is_file():
            ref = np.load(rp)
    if ref is not None:
        x = np.concatenate([feats, feats - ref[None, :]], axis=1)
        p = pack["cal"][exercise].predict_proba(x)[:, 1]
    elif "ensemble" in pack:
        p = ensemble_proba(pack, exercise, w)
    else:
        p = pack["plain"][exercise].predict_proba(feats)[:, 1]
    mean_p = float(p.mean())
    flagged = int(((1 - p) >= FLAG_P_BAD).sum())
    if mean_p >= GOOD_ABOVE:
        verdict = "good"
    elif mean_p < BAD_BELOW:
        verdict = "check form"
    else:
        verdict = "unsure"

    relative = None
    if person is not None and set_key is not None:
        history = _load_history(person)
        sets = history.setdefault(exercise, {})
        relative = relative_verdict(mean_p, [v for k, v in sets.items() if k != set_key])
        sets[set_key] = round(mean_p, 4)
        _save_history(person, history)

    if relative is not None:
        # The person's own history decides; the absolute score is kept for reference.
        verdict = "check form" if relative["verdict"] == "worse than usual" else "good"
        message = (f"form {relative['verdict']} ({mean_p:.2f} vs your usual {relative['usual']:.2f}; "
                   f"{flagged} of {len(p)} moments look off)")
    else:
        message = (f"form {verdict} ({flagged} of {len(p)} moments look off"
                   f"{', personal reference' if ref is not None else ''}; first set of this exercise)")
    return {
        "verdict": verdict,
        "p_correct": round(mean_p, 3),
        "windows": int(len(p)),
        "flagged": flagged,
        "calibrated": ref is not None,
        "relative": relative,
        "message": message,
    }
