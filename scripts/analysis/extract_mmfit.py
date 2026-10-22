"""Extract real statistics and small samples from MM-Fit for the presentation app.
Run from the project root:  python scripts/analysis/extract_mmfit.py
Writes presentation/mmfit_data.js (a global const MMFIT used by index.html)."""
import numpy as np, glob, os, json, csv

PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')   # project root
ROOT = os.path.join(PROJ, 'data', 'raw', 'mm-fit')
OUT = os.path.join(PROJ, 'presentation', 'mmfit_data.js')
JOINTS = ['nose','neck','r_shoulder','r_elbow','r_wrist','l_shoulder','l_elbow','l_wrist',
          'r_hip','r_knee','r_ankle','l_hip','l_knee','l_ankle','r_eye','l_eye','r_ear','l_ear']

def r(x, n=3): return [round(float(v), n) for v in x]

sessions, ex_count, ex_reps, ex_sets = [], {}, {}, {}
total_bytes, total_rows = 0, 0
for d in sorted(glob.glob(os.path.join(ROOT, 'w*'))):
    w = os.path.basename(d)
    files = sorted(os.listdir(d))
    labels = list(csv.reader(open(os.path.join(d, f'{w}_labels.csv'))))
    size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
    total_bytes += size
    rows = 0
    for f in files:
        if f.endswith('.npy'):
            a = np.load(os.path.join(d, f), mmap_mode='r')
            rows += a.shape[1] if a.ndim == 3 else a.shape[0]
    total_rows += rows
    p2 = np.load(os.path.join(d, f'{w}_pose_2d.npy'), mmap_mode='r')
    frames = p2.shape[1]
    hr_files = [f for f in files if '_hr' in f]
    hr = None
    if hr_files:
        h = np.load(os.path.join(d, hr_files[0]))
        hr = [round(float(h[:, 2].min()), 1), round(float(h[:, 2].max()), 1)]
    for s, e, reps, ex in labels:
        ex_count[ex] = ex_count.get(ex, 0) + 1
        ex_reps[ex] = ex_reps.get(ex, 0) + int(reps)
    sessions.append(dict(id=w, minutes=round(frames / 30 / 60, 1), sets=len(labels),
                         reps=sum(int(l[2]) for l in labels), files=files,
                         mb=round(size / 1e6), hr=hr, rows=rows))

# ---------- detailed samples from w00 ----------
d = os.path.join(ROOT, 'w00')
labels = [[int(a), int(b), int(c), e] for a, b, c, e in csv.reader(open(os.path.join(d, 'w00_labels.csv')))]
L = lambda n: np.load(os.path.join(d, f'w00_{n}.npy'), mmap_mode='r')

start = labels[0][0]  # first squat set
def rows_at(arr, frame, n=8):
    i = int(np.searchsorted(arr[:, 0], frame))
    return [[int(x[0]), int(x[1])] + r(x[2:], 4) for x in np.array(arr[i:i + n])]

raw = {}
for n in ['sw_l_acc', 'sw_l_gyr', 'sp_r_acc', 'eb_l_acc']:
    raw[n] = rows_at(L(n), start)
hr_l = L('sw_l_hr')
raw['sw_l_hr'] = [[int(x[0]), int(x[1]), round(float(x[2]), 1)] for x in np.array(hr_l[:8])]
p2 = L('pose_2d'); p3 = L('pose_3d')
i = int(np.searchsorted(p2[0, :, 0], start))
raw['pose_2d'] = dict(frame=int(p2[0, i, 0]), x=r(p2[0, i, 1:], 1), y=r(p2[1, i, 1:], 1))
raw['pose_3d'] = dict(frame=int(p3[0, i, 0]), x=r(p3[0, i, 1:], 1), y=r(p3[1, i, 1:], 1), z=r(p3[2, i, 1:], 1))

# skeleton animation for one squat set (every 2nd frame)
s0, s1 = labels[0][0], labels[0][1]
idx = np.where((p2[0, :, 0] >= s0) & (p2[0, :, 0] <= s1))[0][::2]
skel = [[r(p2[0, k, 1:], 0), r(p2[1, k, 1:], 0)] for k in idx]

# missing joints (OpenPose writes 0 when a joint is not found)
zeros = float((np.array(p2[0, :, 1:]) == 0).mean())

# watch accel during that set, 100 Hz -> show raw slice + 25 Hz resampled
acc = np.array(L('sw_l_acc'))
m = (acc[:, 0] >= s0) & (acc[:, 0] <= s0 + 150)     # first 5 s of the set
seg = acc[m]
mag_raw = np.linalg.norm(seg[:, 2:5], axis=1)
t_raw = (seg[:, 1] - seg[0, 1]) / 1000
t_new = np.arange(0, t_raw[-1], 1 / 30)   # align to the 30 fps video clock
mag_new = np.interp(t_new, t_raw, mag_raw)
k = 5; mag_smooth = np.convolve(np.pad(mag_new, 2, mode='edge'), np.ones(k) / k, mode='valid')
mag_z = (mag_smooth - mag_smooth.mean()) / mag_smooth.std()
clean = dict(t_raw=r(t_raw, 3), raw=r(mag_raw, 2), t_new=r(t_new, 3), smooth=r(mag_smooth, 2), z=r(mag_z, 3),
             raw_hz=round(1 / np.median(np.diff(t_raw)), 1))

# heart rate across the session with set markers
h = np.array(hr_l)
hr_series = [[int(x[0]), round(float(x[2]), 1)] for x in h[::5]]

# weak fatigue labels: rep durations per set from hip height (pose y of mid-hip), squats/lunges only
def rep_durations(s, e):
    k = np.where((p2[0, :, 0] >= s) & (p2[0, :, 0] <= e))[0]
    y = (np.array(p2[1, k, 9]) + np.array(p2[1, k, 12])) / 2   # r_hip, l_hip
    y = np.convolve(np.pad(y, 4, mode='edge'), np.ones(9) / 9, mode='valid')
    thr = (np.percentile(y, 10) + np.percentile(y, 90)) / 2
    low = y > thr                                               # image y grows downward: hips low = squat bottom
    edges = np.where(np.diff(low.astype(int)) == 1)[0]
    f = p2[0, k, 0]
    return [round(float(f[b] - f[a]) / 30, 2) for a, b in zip(edges[:-1], edges[1:])]

def set_hr(s, e):
    m = (h[:, 0] >= s) & (h[:, 0] <= e + 300)
    return round(float(h[m, 2].max()), 1) if m.any() else None

fatigue = []
for s, e, reps, ex in labels:
    item = dict(ex=ex, start=s, end=e, reps=reps, hr_peak=set_hr(s, e))
    if ex in ('squats', 'lunges'):
        item['rep_s'] = rep_durations(s, e)
    fatigue.append(item)

# one short clip per exercise (first set in w00): pose every 3rd frame + watch accel magnitude at 10 Hz
clips = {}
for s, e, reps, ex in labels:
    if ex in clips: continue
    k = np.where((p2[0, :, 0] >= s) & (p2[0, :, 0] <= min(e, s + 240)))[0][::3]
    m = (acc[:, 0] >= s) & (acc[:, 0] <= min(e, s + 240))
    a = np.linalg.norm(acc[m][:, 2:5], axis=1)[::10]
    clips[ex] = dict(frames=[[r(p2[0, j, 1:], 0), r(p2[1, j, 1:], 0)] for j in k], acc=r(a, 2), reps=reps)

# within-set slowdown across ALL sessions (squats + lunges, hip height from 2D pose)
slow = {'squats': [], 'lunges': []}
repcount = {'squats': [], 'lunges': []}
for dd in sorted(glob.glob(os.path.join(ROOT, 'w*'))):
    w = os.path.basename(dd)
    P = np.load(os.path.join(dd, f'{w}_pose_2d.npy'), mmap_mode='r')
    F = np.array(P[0, :, 0])
    for s, e, reps, ex in csv.reader(open(os.path.join(dd, f'{w}_labels.csv'))):
        if ex not in slow: continue
        k = np.where((F >= int(s)) & (F <= int(e)))[0]
        if len(k) < 60: continue
        y = (np.array(P[1, k, 9]) + np.array(P[1, k, 12])) / 2
        y = np.convolve(np.pad(y, 4, mode='edge'), np.ones(9) / 9, mode='valid')
        thr = (np.percentile(y, 10) + np.percentile(y, 90)) / 2
        ed = np.where(np.diff((y > thr).astype(int)) == 1)[0]
        dur = np.diff(F[k][ed]) / 30
        repcount[ex].append((len(ed), int(reps)))
        if len(dur) < 6: continue
        slow[ex].append(dict(w=w, first=round(float(dur[:3].mean()), 2), last=round(float(dur[-3:].mean()), 2)))
slowdown = {}
for ex, v in slow.items():
    ch = np.array([x['last'] / x['first'] - 1 for x in v])
    slowdown[ex] = dict(sets=len(v), slower=int((ch > 0).sum()), median_pct=round(float(np.median(ch) * 100), 1), sets_detail=v)

# replay for the live demo: third squat set of w00 (clear slowdown)
s, e = labels[2][0], labels[2][1]
k = np.where((p2[0, :, 0] >= s) & (p2[0, :, 0] <= e))[0]
y = (np.array(p2[1, k, 9]) + np.array(p2[1, k, 12])) / 2
y = np.convolve(np.pad(y, 4, mode='edge'), np.ones(9) / 9, mode='valid')
thr = (np.percentile(y, 10) + np.percentile(y, 90)) / 2
ed = np.where(np.diff((y > thr).astype(int)) == 1)[0]
F = p2[0, k, 0]
mh = (h[:, 0] >= s - 60) & (h[:, 0] <= e + 60)
demo = dict(start=s, end=e, frames=[[int(F[j])] + [r(p2[0, k[j], 1:], 0), r(p2[1, k[j], 1:], 0)] for j in range(0, len(k), 2)],
            rep_frames=[int(F[x]) for x in ed], hr=[[int(a), round(float(b), 1)] for a, b in h[mh][:, [0, 2]]])

repeval = {ex: dict(sets=len(v), exact=sum(1 for d_, t in v if d_ == t), within1=sum(1 for d_, t in v if abs(d_ - t) <= 1),
                   mae=round(float(np.mean([abs(d_ - t) for d_, t in v])), 2)) for ex, v in repcount.items()}
hr_all = [s_['hr'] for s_ in sessions if s_['hr']]
hrstats = dict(min_low=min(h[0] for h in hr_all), max_low=max(h[0] for h in hr_all), min_high=min(h[1] for h in hr_all), max_high=max(h[1] for h in hr_all))

data = dict(repeval=repeval, hrstats=hrstats, demo=demo, clips=clips, slowdown=slowdown,
    summary=dict(sessions=len(sessions), sets=sum(s['sets'] for s in sessions),
                 reps=sum(s['reps'] for s in sessions), minutes=round(sum(s['minutes'] for s in sessions)),
                 gb=round(total_bytes / 1e9, 2), rows=total_rows,
                 with_hr=sum(1 for s in sessions if s['hr'])),
    exercises=[dict(name=k, sets=ex_count[k], reps=ex_reps[k]) for k in sorted(ex_count, key=lambda k: -ex_count[k])],
    sessions=sessions, labels=labels, raw=raw, joints=JOINTS, skeleton=skel,
    missing_joint_ratio=round(zeros, 4), clean=clean, hr=hr_series, fatigue=fatigue)
open(OUT, 'w').write('const MMFIT = ' + json.dumps(data, separators=(',', ':')) + ';\n')
print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')
print(json.dumps(data['summary']), json.dumps(data['exercises']))
print([ (s['id'], s['minutes'], s['sets'], s['hr']) for s in sessions])
print('missing', zeros, 'raw_hz', clean['raw_hz'])
print({k:(v['sets'],v['slower'],v['median_pct']) for k,v in slowdown.items()}); print(repeval, hrstats)
