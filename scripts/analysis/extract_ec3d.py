"""Extract real statistics, samples and a quick baseline from EC3D for the presentation app.
Run from the project root:  python scripts/analysis/extract_ec3d.py
Writes presentation/ec3d_data.js (a global const EC3D used by index.html).
data_3D.pickle only holds numpy arrays (checked with pickletools before loading)."""
import pickle, os, json, collections
import numpy as np

PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')   # project root
HERE = os.path.join(PROJ, 'presentation')            # output folder
PICKLE = os.path.join(PROJ, 'data', 'raw', 'ec3d', 'data_3D.pickle')
d = pickle.load(open(PICKLE, 'rb'))
L, P = d['labels'], d['poses']          # labels: exercise, subject, mistake code, rep index, frame ; poses: (N, xyz, 25)
NAMES = {'1': 'Correct', '2': 'Feet too wide', '3': 'Knees inward', '4': 'Not low enough', '5': 'Front bent',
         '6': 'Knee passes toe', '7': 'Arched back', '8': 'Hunched back', '10': 'Unknown'}   # from the authors' evaluation.py
EXN = {'SQUAT': 'Squat', 'Lunges': 'Lunge', 'Plank': 'Plank'}
JOINTS = ['nose', 'neck', 'r_shoulder', 'r_elbow', 'r_wrist', 'l_shoulder', 'l_elbow', 'l_wrist', 'mid_hip', 'r_hip',
          'r_knee', 'r_ankle', 'l_hip', 'l_knee', 'l_ankle', 'r_eye', 'l_eye', 'r_ear', 'l_ear', 'l_big_toe',
          'l_small_toe', 'l_heel', 'r_big_toe', 'r_small_toe', 'r_heel']
r = lambda a, n=3: [round(float(v), n) for v in a]

seqs = collections.OrderedDict()
for i, k in enumerate(map(tuple, L[:, :4])):
    seqs.setdefault(k, []).append(i)

def dist(p, a, b): return np.linalg.norm(p[:, :, a] - p[:, :, b], axis=1)

def features(p):
    """One number per known mistake, computed on one rep. z is vertical, mid-hip is the origin."""
    hipw, shw = dist(p, 9, 12), dist(p, 2, 5)
    ank = (p[:, 2, 11] + p[:, 2, 14]) / 2
    leg = dist(p, 9, 11).mean() + 1e-9
    # EC3D keeps the trunk upright (neck fixed above mid-hip), so measure the trunk angle against the shins
    trunk = p[:, :, 1] - p[:, :, 8]
    shin = (p[:, :, 10] + p[:, :, 13]) / 2 - (p[:, :, 11] + p[:, :, 14]) / 2
    cos = np.einsum('ij,ij->i', trunk, shin) / (np.linalg.norm(trunk, axis=1) * np.linalg.norm(shin, axis=1) + 1e-9)
    lean = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # body line for planks: distance of mid-hip from the shoulder-ankle line, signed by vertical side
    sh = (p[:, :, 2] + p[:, :, 5]) / 2; an = (p[:, :, 11] + p[:, :, 14]) / 2
    t = np.clip(np.einsum('ij,ij->i', -sh, an - sh) / (np.einsum('ij,ij->i', an - sh, an - sh) + 1e-9), 0, 1)
    proj = sh + t[:, None] * (an - sh)
    sag = -proj[:, 2] / (dist(p, 2, 11).mean() + 1e-9)          # mid-hip (origin) above (+) or below (-) the line
    # knee beyond toe, horizontal, for the front leg (the knee furthest from the hip)
    kt = []
    for kn, toe in ((10, 22), (13, 19)):
        kt.append(np.linalg.norm(p[:, :2, kn], axis=1) - np.linalg.norm(p[:, :2, toe], axis=1))
    return dict(knee_width=float((dist(p, 10, 13) / hipw).min()),
                feet_width=float((dist(p, 11, 14) / shw).mean()),
                depth=float((-ank).min() / leg),          # hip height above the ankles at the bottom of the rep
                lean=float(lean.max()),
                knee_toe=float(np.max(kt) / leg),
                sag=float(np.median(sag)))

rows = []
for (ex, sub, lab, rep), idx in seqs.items():
    f = features(P[idx])
    rows.append(dict(ex=ex, sub=sub, lab=lab, rep=int(rep), n=len(idx), **f))

# ---------- summary ----------
counts = collections.defaultdict(collections.Counter)
for x in rows: counts[x['ex']][x['lab']] += 1
subjects = sorted({x['sub'] for x in rows})
per_sub = {s: sum(1 for x in rows if x['sub'] == s) for s in subjects}
lens = np.array([x['n'] for x in rows])

# ---------- feature evidence: one feature per mistake vs correct reps of the same exercise ----------
EVID = [('SQUAT', '2', 'feet_width', 'Ankle width ÷ shoulder width (mean)'),
        ('SQUAT', '3', 'knee_width', 'Knee width ÷ hip width (narrowest point)'),
        ('SQUAT', '4', 'depth', 'Lowest hip height ÷ leg length'),
        ('SQUAT', '5', 'lean', 'Max trunk angle vs shins (°)'),
        ('Lunges', '4', 'depth', 'Lowest hip height ÷ leg length'),
        ('Lunges', '6', 'knee_toe', 'Knee ahead of toe ÷ leg length'),
        ('Plank', '7', 'sag', 'Hip offset from shoulder–ankle line'),
        ('Plank', '8', 'sag', 'Hip offset from shoulder–ankle line')]
def q(v): return r(np.percentile(v, [10, 25, 50, 75, 90]), 3)
evidence = []
for ex, lab, feat, desc in EVID:
    c = [x[feat] for x in rows if x['ex'] == ex and x['lab'] == '1']
    e = [x[feat] for x in rows if x['ex'] == ex and x['lab'] == lab]
    evidence.append(dict(ex=EXN[ex], err=NAMES[lab], feat=desc, correct=q(c), error=q(e), nc=len(c), ne=len(e)))

# ---------- quick baseline: leave-one-subject-out, hand-made features, per exercise ----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
FEATS = ['knee_width', 'feet_width', 'depth', 'lean', 'knee_toe', 'sag']
baseline = {}
for ex in ['SQUAT', 'Lunges', 'Plank']:
    data = [x for x in rows if x['ex'] == ex and x['lab'] != '10']
    accs, yt, yp = [], [], []
    for s in subjects:
        tr = [x for x in data if x['sub'] != s]; te = [x for x in data if x['sub'] == s]
        clf = RandomForestClassifier(n_estimators=300, random_state=0).fit([[x[f] for f in FEATS] for x in tr], [x['lab'] for x in tr])
        pred = clf.predict([[x[f] for f in FEATS] for x in te])
        accs.append(accuracy_score([x['lab'] for x in te], pred)); yt += [x['lab'] for x in te]; yp += list(pred)
    labs = sorted({x['lab'] for x in data}, key=int)
    cm = [[sum(1 for a, b in zip(yt, yp) if a == i and b == j) for j in labs] for i in labs]
    baseline[EXN[ex]] = dict(acc=round(accuracy_score(yt, yp) * 100, 1), f1=round(f1_score(yt, yp, average='macro') * 100, 1),
                             per_subject=[round(a * 100, 1) for a in accs], labels=[NAMES[l] for l in labs], cm=cm,
                             chance=round(100 / len(labs), 1))

# ---------- animation clips: one rep per class, subject Sena (front + side visible via 3D rotation) ----------
clips = {}
for ex in ['SQUAT', 'Lunges', 'Plank']:
    for lab in sorted({x['lab'] for x in rows if x['ex'] == ex and x['lab'] != '10'}, key=int):
        k = next(k for k in seqs if k[0] == ex and k[2] == lab and k[1] == 'Sena')
        p = P[seqs[k]]
        clips[f'{EXN[ex]}|{NAMES[lab]}'] = [[r(p[t, 0], 3), r(p[t, 1], 3), r(p[t, 2], 3)] for t in range(len(p))]

# ---------- raw sample rows ----------
i0 = seqs[next(iter(seqs))][0]
raw_labels = [list(map(str, L[i])) for i in range(i0, i0 + 6)]
raw_pose = dict(label=list(map(str, L[i0])), x=r(P[i0, 0], 4), y=r(P[i0, 1], 4), z=r(P[i0, 2], 4))

data = dict(summary=dict(frames=int(len(L)), reps=len(rows), subjects=len(subjects), joints=25,
                         per_subject=per_sub, frames_per_rep=[int(lens.min()), int(np.median(lens)), int(lens.max())],
                         mb=round(os.path.getsize(PICKLE) / 1e6, 1)),
            counts={EXN[e]: {NAMES[k]: v for k, v in sorted(c.items(), key=lambda kv: int(kv[0]))} for e, c in counts.items()},
            evidence=evidence, baseline=baseline, clips=clips, joints=JOINTS,
            bones=[[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13],
                   [13, 14], [0, 15], [0, 16], [15, 17], [16, 18], [11, 24], [14, 21], [14, 19], [19, 20], [11, 22], [22, 23]],
            raw_labels=raw_labels, raw_pose=raw_pose)
open(os.path.join(HERE, 'ec3d_data.js'), 'w', encoding='utf-8').write('const EC3D = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print(json.dumps(data['summary']))
print(json.dumps(data['counts']))
for e in evidence: print(e['ex'], e['err'], 'correct', e['correct'][2], 'error', e['error'][2])
for k, v in baseline.items(): print(k, v['acc'], v['f1'], v['per_subject'], 'chance', v['chance'])
print(os.path.getsize(os.path.join(HERE, 'ec3d_data.js')) // 1024, 'KB')
