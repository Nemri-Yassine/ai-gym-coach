# Gym Station · iPhone Safari → laptop (live)

Film on the iPhone, control on the laptop. Video chunks and heart rate land directly in
`data/raw/own_recordings/<user>/<date>_<session>/` — no cable, no AirDrop afterwards.

```
iPhone Safari  ──HTTPS + WebSocket (same Wi-Fi)──►  laptop: python -m awc.station
  /phone                                              /  (control page)
Watch ── BLE Heart Rate  or  Pulsoid cloud ─────────►  laptop HR thread
```

Free tools only. No native iOS / watch app: Safari + **Heart for Bluetooth** (Wear OS) or **Pulsoid**.

---

## 1 · One-time laptop setup

### 1.1 Install packages

```powershell
.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]" bleak qrcode pillow httpx
```

(`requirements.txt` lists them.)

### 1.2 mkcert (HTTPS — required for the iPhone camera)

Safari only unlocks `getUserMedia` on a **secure origin**.

1. Install [mkcert](https://github.com/FiloSottile/mkcert) (Windows scoop/choco or the release binary).
2. In PowerShell:

```powershell
mkcert -install
mkdir C:\awc_tmp\station_certs -Force
cd C:\awc_tmp\station_certs
# Replace 192.168.x.y with your laptop LAN IP from `ipconfig`
mkcert -cert-file station.pem -key-file station-key.pem 192.168.x.y localhost 127.0.0.1
```

Certs stay in `C:\awc_tmp\station_certs\` (never in git / OneDrive).
`python -m awc.station` also tries to generate them if missing.

### 1.3 Trust the CA on the iPhone (screenshots-level)

1. On the laptop, find the mkcert root CA:

```powershell
mkcert -CAROOT
# e.g. C:\Users\<you>\AppData\Local\mkcert\rootCA.pem
```

2. Send `rootCA.pem` to the iPhone (AirDrop, Mail, or iCloud Drive).
3. On the iPhone, open the file → **Install Profile** (Settings may open automatically) → enter the passcode → Install.
4. Go to **Settings → General → About → Certificate Trust Settings**.
5. Enable full trust for the **mkcert** root certificate.
6. Open `https://<laptop-ip>:8443/phone` — the lock icon should be OK (not a scary warning).

### 1.4 Windows firewall (private networks)

Run **once** as Administrator:

```powershell
netsh advfirewall firewall add rule name="AWC Gym Station 8443" dir=in action=allow protocol=TCP localport=8443 profile=private
```

Check:

```powershell
netsh advfirewall firewall show rule name="AWC Gym Station 8443"
```

### 1.5 Gym Wi-Fi client isolation → Personal Hotspot fallback

Many gym APs block phone↔laptop traffic. Fallback:

1. On the iPhone: **Personal Hotspot** → allow others to join.
2. Connect the **laptop** to that hotspot (laptop gets something like `172.20.10.2`).
3. Use the laptop IP shown on the control page / QR (`https://172.20.10.x:8443/phone`).
4. Camera path does **not** need internet. Pulsoid (option B) does.

---

## 2 · Watch heart rate

| Option | Watch | Path | Internet? |
|---|---|---|---|
| **A · BLE (recommended)** | Galaxy Watch 7 + free Wear OS app **Heart for Bluetooth** (service `0x180D`) | laptop `bleak` | no |
| **B · Pulsoid** | Galaxy Watch or Apple Watch + Pulsoid app | cloud websocket | yes |

**Note:** a Galaxy Watch 7 cannot pair with an iPhone for setup; once set up from Android it can still broadcast HR on its own.

### Option A

1. On the watch, open **Heart for Bluetooth** → start broadcasting.
2. On the laptop control page: HR source **Bluetooth LE** → **Connect HR**.
3. Keep the watch screen on the first time Windows pairs.

### Option B

1. Create a free Pulsoid token.
2. Control page: HR source **Pulsoid** → paste token → Connect.
3. Cloud delay is ~1 s; noted in `meta.json` when measured.

RR intervals (if present in `0x2A37`) go to `hr_rr.csv`.

Watch **IMU is not live**. Optional after-session: Sensor Logger export → `scripts/import_watch_export.py`.

---

## 3 · Run the station

```powershell
.venv\Scripts\python.exe -m awc.station
```

- Control page: `https://localhost:8443/`
- Phone page: QR on the setup screen → `https://<laptop-ip>:8443/phone`
- Space bar = Start / End set

### Pre-gym check

```powershell
# Terminal 1
.venv\Scripts\python.exe -m awc.station

# Terminal 2
.venv\Scripts\python.exe scripts/station_check.py
```

Expect: phone connected, RTT &lt; 50 ms, test clip ~30 fps, HR OK (A or B).

### Camera fallback (if Safari fails)

Install **Camo**, **Iriun** or **DroidCam** (free tier; check resolution / watermark). They appear as a Windows webcam. On the setup screen choose **Webcam fallback**. Frames are then timestamped on the laptop (no phone chunk upload).

---

## 4 · Session files

Same layout as the phone-app importer:

```
data/raw/own_recordings/<user>/<YYYY-MM-DD>_<session>/
  sets/setNN.mp4          (or .webm)
  frames_setNN.csv        frame, pc_ts_ms   (laptop clock)
  sets.csv                set,exercise,start_ms,end_ms,reps,rpe,form_label
  sets_extra.csv          weight, note, HR stats, …
  hr.csv / hr_rr.csv
  events.csv              clap, disconnects, mismatches
  clock_sync.csv          pc_ts_ms, offset_ms, rtt_ms
  meta.json               source: "gym-station-iphone"
```

Resume with the same session id continues at the next set number and never overwrites.

Load with `awc.data.recordings.load(session_folder)`.

---

## 5 · Gym checklist

- [ ] Everything charged (phone, watch, laptop plugged in)
- [ ] mkcert trusted on the iPhone; firewall rule present
- [ ] Same Wi-Fi **or** iPhone Personal Hotspot
- [ ] Phone on a tripod, **45°** front, **hip height**, **2–3 m**, whole body in frame
- [ ] Phone screen stays on (Wake Lock / don’t switch apps)
- [ ] Watch broadcasting (Heart for Bluetooth) or Pulsoid streaming
- [ ] **Clap** once at the start of the session (IMU alignment later)
- [ ] After each set: reps, RPE, form → Save set (waits for upload complete)

---

## 6 · Demo / offline note

The older single-page recorder `docs/gym/index.html` still downloads files on the phone.
This station **replaces that path for live gym work** by uploading to the laptop.
