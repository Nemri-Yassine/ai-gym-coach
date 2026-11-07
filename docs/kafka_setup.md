# Kafka setup (Windows)

Scratch logs live under `C:/awc_tmp/` (never inside OneDrive). Bootstrap: `localhost:9092` (`KAFKA_BOOTSTRAP` in `src/awc/config.py`).

Topics (`KAFKA_TOPICS`), **1 partition** each: `pose.keypoints`, `watch.hr`, `watch.imu`, `coach.events`.

## What worked (2026-11-07)

**Docker Desktop + `apache/kafka`** — this is the working option on this machine.

```powershell
# start Docker Desktop, then:
docker rm -f awc-kafka 2>$null
docker run -d --name awc-kafka -p 9092:9092 apache/kafka:latest

# create topics (one partition each)
foreach ($t in @("pose.keypoints","watch.hr","watch.imu","coach.events")) {
  docker exec awc-kafka /opt/kafka/bin/kafka-topics.sh `
    --bootstrap-server localhost:9092 --create --if-not-exists `
    --topic $t --partitions 1 --replication-factor 1
}

docker exec awc-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
```

Stop: `docker stop awc-kafka` (or `docker rm -f awc-kafka`).

Python probe:

```powershell
.venv\Scripts\python.exe -c "from awc.streaming.kafka_io import broker_up; print(broker_up())"
```

## Tried but blocked: Kafka 3.9 binaries (KRaft)

Binaries are unpacked at `C:/awc_tmp/kafka_dist/kafka_2.13-3.9.0` (junction `C:\k` avoids the Windows classpath length limit). Config for KRaft combined mode:

- properties: `C:/awc_tmp/kafka_server.properties` (copy of `config/kraft/server.properties` with `log.dirs=C:/awc_tmp/kafka`)
- format: from `C:\k`, `bin\windows\kafka-storage.bat format -t <uuid> -c C:\awc_tmp\kafka_server.properties`

**Blocker:** `kafka-server-start.bat` calls `wmic` (architecture check), which is not available on this Windows build, so the broker never starts from the bat scripts. Prefer Docker until that is fixed (or start the JVM by hand without `wmic`).

```powershell
# intended binary start (blocked here by missing wmic)
cd C:\k
bin\windows\kafka-server-start.bat C:\awc_tmp\kafka_server.properties
```

## Replay producers (demo / tests)

```powershell
# terminal 1
.venv\Scripts\python.exe -m awc.streaming.monitor --seconds 120

# terminal 2
.venv\Scripts\python.exe -m awc.streaming.sink --session w20_replay --replay --seconds 120

# terminal 3
.venv\Scripts\python.exe -m awc.streaming.pose_producer --video w20 --session w20_replay --max-seconds 120

# terminal 4
.venv\Scripts\python.exe -m awc.streaming.hr_producer --source replay --session w20_replay --max-seconds 120
```

Live Galaxy Watch link is **optional and untested** (Pulsoid / HypeRate / Sensor Logger code exists, not confirmed on Watch 7).
