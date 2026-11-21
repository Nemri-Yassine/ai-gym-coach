"""Model 2 (form) demo video on EC3D, the dataset Model 2 is trained on.

    python scripts/docs/render_form_demo.py

EC3D ships 3D motion-capture skeletons, not RGB video, so the demo animates the skeleton
the model sees. Honest setup, same as the reported benchmark (reports/m2_benchmark.md):
leave-one-person-out. For each person, the 2D random forest (Model 2, 2D features, all
train yaws 0..90 + mirror) is trained on the OTHER people only, then predicts that person's
reps. Each clip shows a correct rep next to a mistake rep of the same unseen person,
drawn from the camera angle where the mistake is visible; the model predicts from that
same 2D view. One clip shows a miss on purpose.

Output: reports/validation2/demo/week2/form_demo.mp4 (+ form_demo.json with every prediction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from demo_video import BG, INK, MUTED, OK, ACCENT, compress, draw_text, font  # noqa: E402

from awc.features.form_dataset import (  # noqa: E402
    EXERCISES, MISTAKE_CODES, TRAIN_YAWS, fold_indices, load_m2_reps, loso_folds, rep_pose,
    skeleton_from_rep,
)
from awc.features.skeleton import BONES, normalize, project_ec3d  # noqa: E402
from awc.data.ec3d import MISTAKES  # noqa: E402
from awc.training.train_m2 import _feature_matrix, _fit_forests, _labels  # noqa: E402

OUT_DIR = ROOT / "reports" / "validation2" / "demo" / "week2"
W, H, FPS = 1920, 1080, 30
STOP = (255, 90, 95)
LEFT_C, RIGHT_C, MID_C = (255, 150, 80), (90, 180, 255), (238, 242, 246)
LEFT_J = {5, 6, 7, 11, 12, 13, 15, 17}
RIGHT_J = {2, 3, 4, 8, 9, 10, 14, 16}
SEED = 42

# (exercise, mistake code, display yaw, highlight joints, why the angle)
PAIRS = [
    ("squat", "3", 0, (9, 10, 12, 13), "front view: watch the knees"),
    ("squat", "4", 90, (8, 9, 11, 12), "side view: watch how low the hips go"),
    ("squat", "5", 90, (1, 8, 11), "side view: watch the back"),
    ("lunge", "6", 90, (9, 10, 12, 13), "side view: watch the front knee"),
    ("plank", "7", 90, (1, 8, 11), "side view: watch the hips and back"),
]


def name_of(exercise, idx):
    return MISTAKES[MISTAKE_CODES[exercise][idx]]


def loso_predictions(pack, yaws):
    """{(rep, yaw): (pred_idx, proba list)} with a forest that never saw that rep's person."""
    ex_idx, mistake, _ = _labels(pack)
    n = len(ex_idx)
    views = {i: {y: skeleton_from_rep(rep_pose(pack, i), y) for y in TRAIN_YAWS} for i in range(n)}
    mirr = {i: {y: skeleton_from_rep(rep_pose(pack, i), y, True) for y in TRAIN_YAWS} for i in range(n)}
    out, folds = {}, []
    for fold in loso_folds(pack["subject"]):
        train_ids, test_ids = fold_indices(pack["subject"], fold["test"])
        poses, ex, y = [], [], []
        for r in train_ids:
            for yaw in TRAIN_YAWS:
                for v in (views[r][yaw], mirr[r][yaw]):
                    poses.append(v); ex.append(ex_idx[r]); y.append(mistake[r])
        forests = _fit_forests(_feature_matrix(poses), np.asarray(y), np.asarray(ex), SEED)
        correct = 0
        for r in test_ids:
            name = EXERCISES[ex_idx[r]]
            for yaw in yaws:
                feats = _feature_matrix([views[r][yaw]])
                proba = forests[name].predict_proba(feats)[0]
                full = np.zeros(len(MISTAKE_CODES[name]))
                full[forests[name].classes_] = proba
                out[(int(r), yaw)] = (int(np.argmax(full)), full.tolist())
            correct += out[(int(r), 45)][0] == mistake[r]
        folds.append({"held_out": fold["test"], "n": len(test_ids), "acc_45": correct / max(1, len(test_ids))})
        print(f"  fold held-out {fold['test']}: {correct}/{len(test_ids)} at 45°")
    return out, folds


def pick_pairs(pack, preds):
    ex_idx, mistake, _ = _labels(pack)
    subj = pack["subject"].astype(str)
    chosen = []
    for exercise, code, yaw, hl, why in PAIRS:
        e = EXERCISES.index(exercise)
        m = MISTAKE_CODES[exercise].index(code)
        found = None
        for person in sorted(set(subj)):
            good = [r for r in range(len(subj)) if subj[r] == person and ex_idx[r] == e and mistake[r] == 0
                    and preds[(r, yaw)][0] == 0]
            bad = [r for r in range(len(subj)) if subj[r] == person and ex_idx[r] == e and mistake[r] == m
                   and preds[(r, yaw)][0] == m]
            if good and bad:
                found = (person, good[0], bad[0])
                break
        if found:
            chosen.append({"exercise": exercise, "code": code, "yaw": yaw, "hl": hl, "why": why,
                           "person": found[0], "good": found[1], "bad": found[2], "kind": "success"})
        else:
            print(f"  no clean pair for {exercise} {MISTAKES[code]} at {yaw}° — skipped")
    # one honest miss: a squat mistake the model calls correct
    for r in range(len(subj)):
        if EXERCISES[ex_idx[r]] == "squat" and mistake[r] != 0:
            for yaw in (90, 0):
                if preds[(r, yaw)][0] == 0:
                    goods = [g for g in range(len(subj)) if subj[g] == subj[r] and ex_idx[g] == ex_idx[r]
                             and mistake[g] == 0 and preds[(g, yaw)][0] == 0]
                    if goods:
                        chosen.append({"exercise": "squat", "code": MISTAKE_CODES["squat"][mistake[r]],
                                       "yaw": yaw, "hl": (), "why": "a miss: the model calls this mistake correct",
                                       "person": subj[r], "good": goods[0], "bad": r, "kind": "miss"})
                        return chosen
    return chosen


def draw_skel(d, seq_t, ox, oy, size, hl, bad):
    """seq_t: (18, 2) normalized, y up. Draw inside a size×size box at (ox, oy)."""
    pts = np.asarray(seq_t, dtype=np.float64)
    s = size / 4.0
    cx, cy = ox + size / 2, oy + size * 0.42
    xy = [(cx + p[0] * s, cy - p[1] * s) for p in pts]
    floor_y = max(y for _, y in xy) + 8
    d.line([(ox + 30, floor_y), (ox + size - 30, floor_y)], fill=(40, 52, 66), width=3)
    for a, b in BONES:
        col = LEFT_C if (a in LEFT_J and b in LEFT_J) else RIGHT_C if (a in RIGHT_J and b in RIGHT_J) else MID_C
        if a in hl and b in hl:
            col = STOP if bad else OK
        d.line([xy[a], xy[b]], fill=col, width=16)
    for j, (x, y) in enumerate(xy):
        r = 15 if j in hl else 9
        fill = (STOP if bad else OK) if j in hl else (245, 245, 245)
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def display_seq(pack, rep, yaw):
    return normalize(project_ec3d(rep_pose(pack, rep), yaw))


def render_pair(pack, preds, item, writer, seconds=7.0):
    exercise = item["exercise"]
    left, right = item["good"], item["bad"]
    seqs = [display_seq(pack, r, item["yaw"]) for r in (left, right)]
    m_true = [0, MISTAKE_CODES[exercise].index(item["code"])]
    n_frames = int(FPS * seconds)
    half = W // 2
    for f in range(n_frames):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img, "RGBA")
        title = f"Model 2 · {exercise.capitalize()} · person {item['person']} (never seen in training)"
        draw_text(d, (60, 26), title, font=font(36, True), fill=INK)
        draw_text(d, (60, 74), f"{item['why']} · {item['yaw']}° view · the model sees this 2D skeleton",
                  font=font(24), fill=MUTED)
        for k, (r, seq) in enumerate(zip((left, right), seqs)):
            ox = k * half + 60
            t = (f * 1.0) % len(seq)  # loop each rep at its own length
            frame = seq[int(t)]
            pred, proba = preds[(r, item["yaw"])]
            truth = m_true[k]
            is_bad_pred = pred != 0
            draw_skel(d, frame, ox + 70, 120, 700, item["hl"], is_bad_pred)
            px = ox + 0
            draw_text(d, (px, 836), "TRUTH", font=font(20, True), fill=MUTED)
            draw_text(d, (px + 130, 826), name_of(exercise, truth), font=font(32, True), fill=INK)
            ok = pred == truth
            col = OK if ok else STOP
            draw_text(d, (px, 886), "MODEL 2", font=font(20, True), fill=MUTED)
            draw_text(d, (px + 130, 872), f"{name_of(exercise, pred)} {'✓' if ok else '✗'}",
                      font=font(40, True), fill=col)
            yy = 938
            for ci, p in sorted(enumerate(proba), key=lambda kv: -kv[1])[:3]:
                draw_text(d, (px, yy), name_of(exercise, ci), font=font(20), fill=MUTED)
                bw = int(480 * p)
                d.rounded_rectangle([px + 200, yy + 5, px + 200 + max(4, bw), yy + 23], 6,
                                    fill=(ACCENT if ci == pred else (70, 84, 100)) + (255,))
                draw_text(d, (px + 700, yy), f"{p * 100:.0f}%", font=font(20, True), fill=INK)
                yy += 34
        d.line([(half, 140), (half, H - 40)], fill=(36, 48, 64), width=2)
        writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))


def card(writer, lines, seconds=2.2):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = 400
    for i, line in enumerate(lines):
        draw_text(d, (120, y), line, font=font(54 if i == 0 else 30, i == 0), fill=INK if i == 0 else MUTED)
        y += 90 if i == 0 else 50
    frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    for _ in range(int(FPS * seconds)):
        writer.write(frame)


def main():
    pack = load_m2_reps()
    print("leave-one-person-out predictions (2D random forest, Model 2)…")
    preds, folds = loso_predictions(pack, yaws=(0, 45, 90))
    pairs = pick_pairs(pack, preds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    webm = OUT_DIR / "form_demo.webm"
    writer = cv2.VideoWriter(str(webm), cv2.VideoWriter_fourcc(*"VP80"), FPS, (W, H))
    card(writer, ["Model 2 · is the form correct?",
                  "EC3D motion capture · each person is tested by a model trained without them",
                  "Left: a correct rep · right: a rep with a mistake, same person"], 3.5)
    for item in pairs:
        head = f"{item['exercise'].capitalize()} · {MISTAKES[item['code']]}"
        card(writer, [head if item["kind"] == "success" else f"A miss · {head}",
                      item["why"]], 1.8)
        render_pair(pack, preds, item, writer)
    accs = ", ".join(f"{f['held_out']} {f['acc_45'] * 100:.0f}%" for f in folds)
    card(writer, ["It sees the mistakes it was trained on", f"Held-out accuracy at 45°, per person: {accs}",
                  "On our own phone videos it is not calibrated yet (P(correct) 0.06), so the live demo hides it"],
         4.0)
    writer.release()
    path = compress(webm, OUT_DIR / "form_demo.mp4", width=1920, crf=28)
    summary = {
        "file": Path(path).name,
        "setup": "leave-one-person-out, 2D random forest (Model 2), trained on yaws 0-90 + mirror of the other people",
        "folds": folds,
        "pairs": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()} for p in pairs],
        "pair_predictions": [
            {"rep": int(r), "yaw": p["yaw"], "truth": name_of(p["exercise"], t), "pred": name_of(p["exercise"], preds[(r, p["yaw"])][0])}
            for p in pairs for r, t in ((p["good"], 0), (p["bad"], MISTAKE_CODES[p["exercise"]].index(p["code"])))
        ],
    }
    (OUT_DIR / "form_demo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path} · {len(pairs)} pairs")
    for p in summary["pair_predictions"]:
        print("  ", p)


if __name__ == "__main__":
    main()
