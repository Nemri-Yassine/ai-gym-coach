# Gym Station · iPhone Safari → laptop (live)

Film on the iPhone, control on the laptop. Each set's video is uploaded live (1-second pieces) and saved
on the laptop in `data/raw/own_recordings/<user>/<date>_<session>/`, with the watch heart rate.
No cable, no AirDrop afterwards. The laptop shows a smooth live view of the iPhone camera.

```
iPhone Safari /phone ──HTTPS (PIN)──► laptop: python -m awc.station ──► own_recordings/…
   · records each set (MediaRecorder, 30 fps) and uploads it while recording
   · live view: WebRTC video → laptop (≈ 30 fps), falls back to images (≈ 12 fps)
Watch ── Bluetooth (Heart for Bluetooth)  or  Pulsoid (internet) ──► laptop
```

Free tools only. No native iOS / watch app: Safari + **Heart for Bluetooth** (Wear OS) or **Pulsoid**.

---

## 1 · One-time setup (already done on this laptop, except 1.3 if you change phone)

### 1.1 Packages

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1.2 HTTPS certificate: nothing to do any more

Safari only opens the camera on HTTPS. The iPhone trusts our **mkcert root CA** (`rootCA.pem`, done once).
At every start, `python -m awc.station` **re-issues the laptop certificate automatically** for whatever IP
the laptop has now (home Wi-Fi, gym Wi-Fi, iPhone hotspot `172.20.10.1–15`), signed by that same root CA.
So changing network needs **nothing on the iPhone**: just restart the station.

- mkcert binary: `C:\awc_tmp\bin\mkcert.exe` · certs: `C:\awc_tmp\station_certs\`
- Force a new certificate: `python -m awc.station --renew-cert`

### 1.3 Trust the root CA on a (new) iPhone, once

1. Send `C:\awc_tmp\station_certs\rootCA.pem` to the iPhone (AirDrop / Mail).
2. On the iPhone, open it → **Install Profile** → passcode → Install.
3. **Settings → General → About → Certificate Trust Settings** → turn on full trust for **mkcert**.

### 1.4 Windows firewall: allow **all** network types (important)

The phone hotspot and gym Wi-Fi are usually **Public** networks in Windows. The old rule only allowed
Private networks, so the phone could not connect there. Run once in an **Administrator** PowerShell:

```powershell
netsh advfirewall firewall add rule name="AWC Gym Station 8443 any" dir=in action=allow protocol=TCP localport=8443 profile=any
```

`scripts/station_check.py` checks this.

### 1.5 Which network

- **Recommended at the gym: iPhone Personal Hotspot.** iPhone → Settings → Personal Hotspot → on;
  connect the laptop to it. Works in any gym, no gym Wi-Fi needed, laptop gets `172.20.10.x`
  (already in the certificate).
- Same Wi-Fi also works if the gym Wi-Fi doesn't block device-to-device traffic.
- The camera path needs no internet. Pulsoid (option B) does (the hotspot gives it).

### 1.6 PIN

The phone page needs a 4-digit **PIN**, shown on the laptop setup screen and included in the QR code
(kept in `C:\awc_tmp\station.json`; change it with `python -m awc.station --pin 1234`). The control
page only opens on the laptop itself.

---

## 2 · Watch heart rate

| Option | Watch | Path | Internet? |
|---|---|---|---|
| **A · Bluetooth (recommended)** | Galaxy Watch 7 + free Wear OS app **Heart for Bluetooth** | laptop `bleak` | no |
| **B · Pulsoid** | Galaxy Watch or Apple Watch + Pulsoid app | cloud websocket | yes |

A Galaxy Watch 7 cannot pair with an iPhone; once set up from Android it broadcasts HR on its own.

**Option A:** open Heart for Bluetooth on the watch first and keep it open → control page **Scan** → **Connect HR**.
The station reconnects automatically (≈ 1 s) and counts drops ("drops N" under the bpm).
On Nov 9 the link dropped about every 30 s. If that happens again:
1. keep the watch screen on / app in front, laptop within 2–3 m;
2. pair the watch once in **Windows Settings → Bluetooth → Add device**, then Connect again;
3. otherwise use **option B (Pulsoid)** with the watch on the iPhone hotspot Wi-Fi.

RR intervals (if sent) go to `hr_rr.csv`. Watch IMU is not live (optional: `scripts/import_watch_export.py`).

---

## 3 · Run the station

```powershell
.venv\Scripts\python.exe -m awc.station
```

1. The control page opens on `https://localhost:8443/` with the **QR code + PIN**.
2. iPhone: scan the QR with the Camera app → opens in Safari → choose **30 fps** → **Connect camera**.
   (If the laptop has several IPs, tap the other address buttons under the QR.)
3. The live view appears on the laptop ("live video NN fps" in green; yellow "images" = fallback).
4. Connect the watch HR, pick user / session id / type → **Start session**.
5. Per set: pick the exercise → **Space** (start) → **Space** (end) → reps, RPE, form → **Save set**.
   Save unlocks when "video saved on laptop ✓ N frames @ 30 fps" shows (a few seconds).
6. **Clap** (button or C) with the watch hand in front of the camera at the start.

If the phone loses Wi-Fi for a moment, its upload queue keeps every piece and resends it; nothing is lost
as long as the Safari page stays open. "Also save last set on this phone" keeps a backup copy on the iPhone.

### Pre-gym check (at home)

```powershell
.venv\Scripts\python.exe -m awc.station          # terminal 1
.venv\Scripts\python.exe scripts\station_check.py  # terminal 2
```

Then record 2 short test sets and check the table shows ✓ for the video.

### Camera fallback (if Safari fails)

Free iPhone webcam apps (Camo, Iriun, DroidCam) show the phone as a Windows webcam; record with
`scripts/record_session.py --camera-url <index or url>` (laptop-timestamped frames).

---

## 4 · Session files

```
data/raw/own_recordings/<user>/<YYYY-MM-DD>_<session>/
  sets/setNN.mp4            remuxed, playable (setNN.raw.mp4 = original upload, backup)
  frames_setNN.csv          frame, pc_ts_ms   (laptop clock: video frame times + recorder start)
  frames_phone_setNN.csv    phone frame callbacks (reference only)
  sets.csv                  set,exercise,start_ms,end_ms,reps,rpe,form_label
  sets_extra.csv            weight, note, HR stats, duration
  hr.csv / hr_rr.csv
  events.csv                set start/end, video_saved, clap, hr drops
  clock_sync.csv            pc_ts_ms, offset_ms, rtt_ms
  meta.json                 source: "gym-station-iphone", phone resolution / fps / mime, HR source
```

Same session id = continue that session. Set numbers never reuse an existing video.
Load with `awc.data.recordings.load(session_folder)`.

---

## 5 · Gym checklist

- [ ] Everything charged; laptop plugged in if possible; ≥ 5 GB free disk (≈ 40 MB per minute of video)
- [ ] iPhone Personal Hotspot on, laptop joined to it, station started **after** joining
- [ ] Firewall rule for **all** profiles (1.4)
- [ ] Phone on a tripod, **45°** front, **hip height**, **2–3 m**, whole body in frame
- [ ] Safari page stays open in front (don't lock the phone)
- [ ] Watch: Heart for Bluetooth open, bpm visible on the laptop
- [ ] **Clap** at the start
- [ ] After each set: reps, RPE, form → Save set (wait for ✓)
