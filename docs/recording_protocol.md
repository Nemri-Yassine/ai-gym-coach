# Recording protocol

Two people (Yassine and one partner). Every session is filmed with the same phone placement and labelled on the PC while it happens. Model 3 needs several sessions per person, so recording starts in week 1 and continues through week 3.

## Phone

- Free Android app **IP Webcam** on the Poco X3 Pro. The PC reads `http://<phone-ip>:8080/video`.
- Place the phone about **45° in front** of the person, at **hip height**, **2–3 m** away, with the **whole body** in frame.
- Mark the tripod spot on the floor with tape and use that spot every session.
- Target at least **25 fps at 720p**. If IP Webcam stays under that, switch to DroidCam (free) over USB and pass that URL to the recorder.

## Watch heart rate

Try these in order. Keep the first one that is free, sends at least one value every 2 seconds, and gives a timestamp. No custom watch code.

1. **Pulsoid** — Wear OS app + websocket. `--hr-source pulsoid --hr-token <token>`
2. **HypeRate** — Wear OS app + websocket. `--hr-source hyperate --hr-token <id>`
3. **Sensor Logger** — Wear OS companion, HTTP push to the PC. `--hr-source sensorlogger`
4. **Fallback** — export Samsung Health / Health Connect after the session and align it on the PC clock using the start of `frames.csv`. The recorder still writes `hr.csv` with PC receive time when a live source is used.

If the chosen app does not also stream wrist accelerometer and gyroscope, live Model 1 runs on the skeleton only. That is supported (the model is trained with IMU dropout).

## How to record

```bash
python scripts/record_session.py --user yassine --session-id s01 --camera-url http://PHONE:8080/video --hr-source pulsoid --hr-token TOKEN
```

Preview window (click it first):

- `s` — set starts
- `e` — set ends; then press the RPE as a digit in the preview (1–9, `0` = 10). Filming never pauses.
- `q` — session ends; the terminal then asks, set by set, the exercise, reps and form label (`correct` or an EC3D mistake name).
- `q` — stop and write `meta.json`

Files in `data/raw/own_recordings/<user>/<YYYY-MM-DD>_<session-id>/`:

| File | Contents |
|---|---|
| `video.mp4` | phone video |
| `frames.csv` | `frame, pc_ts_ms` |
| `hr.csv` | `pc_ts_ms, bpm` |
| `sets.csv` | `set, exercise, start_ms, end_ms, reps, rpe, form_label` |
| `meta.json` | user, date, phone position, camera angle, optional body weight |

## Normal session (about 30 min)

1. 5 min warm-up.
2. 3 sets of squats and 3 sets of lunges.
3. Each set goes to **RPE 8–9**: hard, but safe. Bodyweight or a light load. Stop immediately on pain.
4. 90 s rest between sets.
5. Press the RPE digit right after each set (`e`, then `1`–`9` or `0` for 10).

## Form session (once a week)

Squat, 8 reps each, labelled in `sets.csv`: correct, feet too wide, knees inward, not low enough, front bent.

Lunge, 8 reps each: correct, not low enough, knee passes toe.

## Targets by the end of week 3

At least **6 normal sessions** and **2 form sessions** per person.

By validation 1 (end of week 1): at least **2 real sessions per person**.
