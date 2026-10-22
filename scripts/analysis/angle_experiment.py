"""Does the phone's camera angle matter? Simulate a phone at different angles around the person
using EC3D's 3D skeletons, then classify squat/lunge mistakes from the 2D view only.
Run from the project root:  python scripts/analysis/angle_experiment.py
Writes presentation/angle_data.js (a global const ANGLE used by index.html)."""
import pickle, os, json, collections
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')   # project root
HERE = os.path.join(PROJ, 'presentation')            # output folder
d = pickle.load(open(os.path.join(PROJ, 'data', 'raw', 'ec3d', 'data_3D.pickle'), 'rb'))   # numpy arrays only
L, P = d['labels'], d['poses']
seqs = collections.OrderedDict()
for i, k in enumerate(map(tuple, L[:, :4])):
    seqs.setdefault(k, []).append(i)

# which way does the person face? toes minus heels, averaged over the dataset (horizontal plane)
toe = (P[:, :2, 19] + P[:, :2, 22]) / 2; heel = (P[:, :2, 21] + P[:, :2, 24]) / 2
f = (toe - heel).mean(axis=0); facing = np.arctan2(f[1], f[0])

def view(p, angle_deg):
    """2D image seen by a camera at hip height, angle_deg around the person (0 = facing them, 90 = side)."""
    a = facing + np.radians(angle_deg)
    right = np.array([-np.sin(a), np.cos(a)])            # image x axis, horizontal
    u = np.einsum('tcj,c->tj', p[:, :2, :], right)        # horizontal image coordinate
    v = p[:, 2, :]                                        # vertical stays vertical
    return u, v

def feats(u, v):
    """Generic 2D features: joints relative to mid-hip, scaled by torso length, summarized over the rep."""
    torso = np.hypot(u[:, 1] - u[:, 8], v[:, 1] - v[:, 8]).mean() + 1e-9
    U, V = (u - u[:, [8]]) / torso, (v - v[:, [8]]) / torso
    J = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 19, 22]
    out = []
    for M in (U[:, J], V[:, J]):
        out += [M.min(0), M.max(0), M.mean(0)]
    return np.concatenate(out)

subjects = sorted(set(L[:, 1]))
res = {}
ANGLES = list(range(0, 181, 15))
for ex, name in (('SQUAT', 'Squat'), ('Lunges', 'Lunge')):
    keys = [k for k in seqs if k[0] == ex and k[2] != '10']
    accs = []
    for ang in ANGLES:
        X = np.array([feats(*view(P[seqs[k]], ang)) for k in keys]); y = np.array([k[2] for k in keys]); g = np.array([k[1] for k in keys])
        yt, yp = [], []
        for s in subjects:
            clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X[g != s], y[g != s])
            yt += list(y[g == s]); yp += list(clf.predict(X[g == s]))
        accs.append(round(accuracy_score(yt, yp) * 100, 1))
    # train on many angles, test on angles never used in training (does 3D augmentation help?)
    train_angles, test_angles = [0, 45, 90, 135, 180], [15, 30, 60, 75, 105, 120, 150, 165]
    yt, yp = [], []
    for s in subjects:
        tr = [k for k in keys if k[1] != s]; te = [k for k in keys if k[1] == s]
        Xtr = np.array([feats(*view(P[seqs[k]], a)) for k in tr for a in train_angles]); ytr = [k[2] for k in tr for a in train_angles]
        clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
        for a in test_angles:
            yt += [k[2] for k in te]; yp += list(clf.predict(np.array([feats(*view(P[seqs[k]], a)) for k in te])))
    res[name] = dict(angles=ANGLES, acc=accs, multi=round(accuracy_score(yt, yp) * 100, 1),
                     chance=round(100 / len(set(k[2] for k in keys)), 1))
    print(name, dict(zip(ANGLES, accs)), 'multi-angle model on unseen angles:', res[name]['multi'])

open(os.path.join(HERE, 'angle_data.js'), 'w', encoding='utf-8').write('const ANGLE = ' + json.dumps(res) + ';\n')
