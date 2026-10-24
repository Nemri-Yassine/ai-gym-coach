# Gym session with the phone app

Phone (Poco X3 Pro) films each set, the Galaxy Watch 7 sends heart rate over Bluetooth, everything is stamped by the phone clock. No laptop at the gym.

App: **https://nemri-yassine.github.io/ai-gym-coach/gym/** (Chrome on the phone). Rehearse at home with `?demo=1` at the end of the address (fake camera and fake watch).

## Once, before Monday (≈ 15 min)

1. **Watch:** on the Galaxy Watch 7, open the Play Store and install **Heart for Bluetooth** (free). Open it, allow body sensors, start broadcasting.
2. **Phone:** open the app address in **Chrome**, tap *Connect heart rate*, pick the watch. The status must say `streaming from …` and the bpm must update every second or two.
3. If the watch is not found or the bpm freezes: keep the watch screen on (or turn on Always On Display) and retry. Still nothing → switch to **Pulsoid** (Wear OS app + personal token from pulsoid.net, needs mobile data), or **None** (video and sets only).
4. Tap *Start camera*, allow the camera. Do a 1-minute test set and check that a video lands in **Downloads**.
5. Optional: Chrome menu → *Add to Home screen*, so the app opens full screen.

## At the gym

1. Tripod: about **45° in front**, **hip height**, **2–3 m**, whole body visible. Same spot every time (tape mark).
2. Watch: open Heart for Bluetooth, start broadcasting. Phone: open the app, choose the person, check the session id (`s01`, `s02`… it goes up by itself).
3. Connect heart rate → start camera → tick the placement box → **Start session**. The screen stays on.
4. For each set:
    - tap **Start set** just before the first rep, **End set** right after the last one
    - pick the exercise, adjust the reps, tap the **RPE** (1–10), choose the form (`correct`, or the mistake you did on purpose), **Save set**
    - the set video downloads immediately (a few MB); heart rate keeps logging during the rest
5. Normal session: 3 sets of squats + 3 sets of lunges at RPE 8–9, 90 s rest. Form session (once a week): 8 reps correct + 8 reps of each mistake, each saved with its form label.
6. At the end tap **Finish session and export** → `…_session.json` lands in Downloads. If the browser closed by accident, open the app again and tap *Export last session*.

## Back home

Copy the files from the phone's **Downloads** to the PC (same folder for the JSON and the set videos), then:

```bash
python scripts/import_phone_session.py "C:/path/to/awc_yassine_2026-10-27_s01_session.json"
```

It prints the heart-rate rate (expect about 30–60 samples per minute) and, for every set, the number of frames and the frame rate. The session lands in `data/raw/own_recordings/<user>/<date>_<session>/`.

## What gets saved

| File | Contents |
|---|---|
| `awc_<user>_<date>_<session>_setNN.mp4` | video of one set (H.264, or WebM if the phone cannot do MP4) |
| `awc_<user>_<date>_<session>_session.json` | heart rate for the whole session, every set (exercise, reps, RPE, form, start/end), the capture time of every video frame, camera settings |

Everything uses the phone clock (epoch milliseconds), so heart rate and video line up without any sync step.
