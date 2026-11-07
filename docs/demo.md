# Demo commands (Week 2–3)

## A · Kafka replay coach (preferred)

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

## B · Offline fallback (no broker)

Uses MM-Fit interim Parquet. Latency is **model-only** (no MediaPipe, no Kafka).

```powershell
.venv\Scripts\python.exe -m awc.coach.live --replay w20 --no-voice --max-sets 3
```

Live Galaxy Watch link is optional and untested.
