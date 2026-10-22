# Decisions

One row per important choice. Date, what was decided, which options were on the table, and why this one.

| Date | Decision | Options considered | Why |
|---|---|---|---|
| 2026-10-22 | JDK is Eclipse Temurin 17.0.20.1, unpacked at `C:\Java\jdk-17.0.20.1+1`. `JAVA_HOME` and `C:\hadoop` (winutils 3.3.6) are set for the user. Spark scratch stays in `C:\awc_tmp`. | System installer (needs an admin prompt) or a zip JDK on a path outside OneDrive | The zip JDK does not need an administrator install, and Spark temp files must not sit inside OneDrive. |
| 2026-10-22 | Phone video is read from IP Webcam (`http://<phone>:8080/video`). Heart rate is pluggable: Pulsoid, HypeRate, Sensor Logger, or `none`. **No option has been confirmed on the Poco X3 Pro or the Galaxy Watch 7 yet** — those devices were not connected while this step was built, so no latency number is claimed. Until a live source passes the checks, sessions are recorded with `--hr-source none`. | (A) Pulsoid websocket (B) HypeRate websocket (C) Sensor Logger HTTP push (D) Samsung Health export after the session | All four meet the "no custom watch code" rule on paper. The recorder implements A–C and writes `hr.csv` with the PC clock, which is also what fallback D has to line up with. The first source that is free, timestamped and at least one value every 2 s on the real watch wins; that test is still to do. |
