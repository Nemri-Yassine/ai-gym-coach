# Demo commands (Week 1–2 with own recordings)

## A · Live gym station (iPhone + Galaxy Watch 7)

Laptop control page + iPhone Safari camera + BLE heart rate. After each set the station
runs pose → Model 1 → rep count → Model 2 (squat/lunge/plank) → HR in a **background**
worker and shows a suggestion under the set form (never written into `sets.csv`).

```powershell
# From the project root, venv active. Firewall: allow TCP 8443 on Public networks.
.venv\Scripts\python.exe -m awc.station
# Optional Kafka (Docker broker must be up; if down, logs once and continues):
.venv\Scripts\python.exe -m awc.station --kafka
```

Open the control page (browser opens automatically). Scan the QR on the iPhone, unlock
with the PIN, connect the watch via *Heart for Bluetooth*. Record a set; after upload
the blue “Model: …” line appears under the form.

### Screen-record the live station (Windows Game Bar)

We could not capture `reports/validation2/demo/week2/station_demo.mp4` automatically.
Record it by hand:

1. Start the station (`python -m awc.station`) and open the control page full screen.
2. Press **Win+G** to open Xbox Game Bar, then **Win+Alt+R** to start/stop recording.
3. Film: live iPhone view → Start set → End set → wait for the suggestion line (~8 s with live_fast).
4. Save the clip as `reports/validation2/demo/week2/station_demo.mp4` (H.264, muted is fine).

Check analysis JSON (interim only):

```powershell
Get-Content "data\interim\own_live\<person>\<session>\analysis_set01.json"
```

## B · Offline replay of one own session

```powershell
# Pose (skips cached sets unless --force)
.venv\Scripts\python.exe scripts/process_recording.py yassine/2026-11-10_SQUAT --workers 1

# Models 1–2 + reps + HR report for that session
.venv\Scripts\python.exe scripts/eval_own_recordings.py --session yassine/2026-11-10_SQUAT

# Full own RQ4 (all eval sets)
.venv\Scripts\python.exe scripts/eval_own_recordings.py

# Spark own rep table
.venv\Scripts\python.exe -m awc.bigdata.history --own --cores local[2]
```

## C · MM-Fit Kafka replay coach (existing)

Needs Docker Kafka (`docs/kafka_setup.md`). Four terminals from the project root with the venv active:

```powershell
# 0. Broker (if not already up)
docker start awc-kafka
# or: docker run -d --name awc-kafka -p 9092:9092 apache/kafka:latest

# 1. Monitor
.venv\Scripts\python.exe -m awc.streaming.monitor --seconds 120

# 2. Pose producer (w20 first squat set starts ~frame 2178 ≈ 72.6 s)
.venv\Scripts\python.exe -m awc.streaming.pose_producer --video w20 --session w20_replay --start-frame 2100 --max-seconds 50

# 3. HR producer (aligned to the same video time)
.venv\Scripts\python.exe -m awc.streaming.hr_producer --source replay --session w20_replay --start-seconds 70 --max-seconds 50

# 4. Coach (consumes pose/HR, publishes coach.events)
.venv\Scripts\python.exe -m awc.coach.live --kafka --replay w20_replay --no-voice --max-seconds 55
```

Latencies logged by the coach:

- **model**: rep end → message (inference only)
- **e2e**: frame `pc_ts_ms` → message (pose extraction in the producer + Kafka + models)

## D · Offline coach fallback (no broker)

Uses MM-Fit interim Parquet. Latency is **model-only** (no MediaPipe, no Kafka).

```powershell
.venv\Scripts\python.exe -m awc.coach.live --replay w20 --no-voice --max-sets 3
```
