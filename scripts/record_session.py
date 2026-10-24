"""Record one training session from the phone camera and an optional heart-rate source.

    python scripts/record_session.py --user yassine --session-id s01 --camera-url http://PHONE:8080/video --hr-source none

Keys while the preview is focused: s = set starts, e = set ends, then a digit = RPE of that set
(1-9, 0 means 10), q = stop the session. Filming never pauses: exercise, reps and form label
are typed in the terminal after q, set by set.

Timing: a reader thread stamps every frame with the PC clock as soon as it arrives
(frames.csv). The video is written at the frame rate measured over the first 2 seconds,
but frames.csv is the reference clock: downstream code must take time from it, never
from the video's fps.
"""
import argparse
import json
import queue
import threading
import time
from datetime import date, datetime
from urllib.parse import urlparse

from awc.data.recordings import (
    EXERCISES,
    FORM_LABELS,
    append_frame,
    append_hr,
    append_set,
    init_session_files,
    session_dir,
    write_meta,
)


def _ask(prompt, default=None, valid=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        if valid is None or raw in valid:
            return raw
        print(f"  choose one of: {', '.join(valid)}")


def _hr_loop(source, token, port, stop, hr_path):
    """Append bpm values until stop is set. Failures are printed, not fatal."""
    try:
        if source == "pulsoid":
            stream = _iter_pulsoid(token)
        elif source == "hyperate":
            stream = _iter_hyperate(token)
        elif source == "sensorlogger":
            stream = _iter_sensorlogger(port, stop)
        else:
            return
        for bpm in stream:
            if stop.is_set():
                break
            try:
                append_hr(hr_path, time.time() * 1000, bpm)
            except ValueError as exc:
                print(f"hr skipped: {exc}")
    except Exception as exc:  # noqa: BLE001 — a dead HR source must not kill the video
        print(f"heart-rate source stopped: {exc}")


def _iter_pulsoid(token):
    url = f"wss://dev.pulsoid.net/api/v1/data/real_time?access_token={token}"
    for text in _websocket_texts(url):
        msg = json.loads(text)
        data = msg.get("data", msg)
        bpm = data.get("heart_rate", data.get("heartRate", data.get("bpm")))
        if bpm is not None:
            yield bpm


def _iter_hyperate(token):
    """HypeRate uses a Phoenix websocket. token is the HypeRate ID."""
    url = "wss://app.hyperate.io/socket/websocket?vsn=2.0.0"
    join = json.dumps({"topic": f"hr:{token}", "event": "phx_join", "payload": {}, "ref": "1"})
    for text in _websocket_texts(url, first_message=join):
        msg = json.loads(text)
        payload = msg.get("payload") or {}
        bpm = payload.get("hr") or payload.get("heart_rate") or payload.get("bpm")
        if bpm is not None:
            yield bpm


def _iter_sensorlogger(port, stop):
    """Tiny HTTP server. Sensor Logger (or a curl) POSTs JSON containing bpm."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import queue
    found = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n)
            self.send_response(204)
            self.end_headers()
            try:
                msg = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return
            bpm = _find_bpm(msg)
            if bpm is not None:
                found.put(bpm)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Sensor Logger push URL: http://<this-pc>:{port}/")
    try:
        while not stop.is_set():
            try:
                yield found.get(timeout=0.5)
            except queue.Empty:
                continue
    finally:
        server.shutdown()


def _find_bpm(msg):
    if isinstance(msg, dict):
        for key in ("bpm", "heart_rate", "heartRate", "hr"):
            if key in msg and isinstance(msg[key], (int, float)):
                return msg[key]
        for value in msg.values():
            found = _find_bpm(value)
            if found is not None:
                return found
    if isinstance(msg, list):
        for item in msg:
            found = _find_bpm(item)
            if found is not None:
                return found
    return None


def _websocket_texts(url, first_message=None):
    """Minimal websocket client: yield server text frames. No extra dependency."""
    import base64
    import os
    import socket
    import ssl

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    raw = socket.create_connection((host, port), timeout=10)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if parsed.scheme == "wss" else raw
    key = base64.b64encode(os.urandom(16)).decode()
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    sock.sendall((
        f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    ).encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("websocket handshake failed")
        buf += chunk
    head, buf = buf.split(b"\r\n\r\n", 1)
    if b" 101 " not in head.split(b"\r\n", 1)[0]:
        raise ConnectionError(head.decode("utf-8", errors="replace"))
    if first_message:
        _ws_send(sock, first_message)
    sock.settimeout(5)
    while True:
        while len(buf) < 2:
            buf += _recv(sock)
        opcode = buf[0] & 0x0F
        length = buf[1] & 0x7F
        used = 2
        if length == 126:
            while len(buf) < 4:
                buf += _recv(sock)
            length = int.from_bytes(buf[2:4], "big")
            used = 4
        elif length == 127:
            while len(buf) < 10:
                buf += _recv(sock)
            length = int.from_bytes(buf[2:10], "big")
            used = 10
        while len(buf) < used + length:
            buf += _recv(sock)
        payload, buf = buf[used:used + length], buf[used + length:]
        if opcode == 1:
            yield payload.decode("utf-8")
        elif opcode == 8:
            break
        elif opcode == 9:
            _ws_send(sock, payload, opcode=0xA)


def _recv(sock):
    chunk = sock.recv(65536)
    if not chunk:
        raise ConnectionError("websocket closed")
    return chunk


def _ws_send(sock, data, opcode=0x1):
    import os
    if isinstance(data, str):
        data = data.encode()
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    header = bytes([0x80 | opcode])
    n = len(data)
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + n.to_bytes(2, "big")
    else:
        header += bytes([0x80 | 127]) + n.to_bytes(8, "big")
    sock.sendall(header + mask + masked)


FPS_PROBE_SECONDS = 2.0


def estimate_fps(timestamps_ms, fallback=30.0):
    """Frame rate from PC timestamps: (n - 1) / elapsed. Falls back when there is too little data."""
    if len(timestamps_ms) < 2:
        return fallback
    elapsed = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if elapsed <= 0:
        return fallback
    return (len(timestamps_ms) - 1) / elapsed


def rpe_from_key(key):
    """'1'..'9' -> 1..9, '0' -> 10, anything else -> None."""
    if ord("1") <= key <= ord("9"):
        return key - ord("0")
    if key == ord("0"):
        return 10
    return None


def _reader(cap, frames, stop):
    """Read the camera as fast as it delivers and stamp each frame on arrival."""
    misses = 0
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            misses += 1
            if misses > 50:
                frames.put(None)
                return
            time.sleep(0.01)
            continue
        misses = 0
        frames.put((frame, int(time.time() * 1000)))


def _ask_set_details(folder, sets):
    """After the session: exercise, reps and form label for every set, RPE if it was not keyed."""
    if not sets:
        print("no sets recorded")
        return
    print(f"\n{len(sets)} set(s) to describe.")
    previous = "squats"
    for i, st in enumerate(sets, 1):
        print(f"set {i}: {(st['end'] - st['start']) / 1000:.1f} s")
        exercise = _ask("  exercise", previous, EXERCISES)
        reps = int(_ask("  reps"))
        rpe = st["rpe"] if st["rpe"] is not None else int(_ask("  RPE 1-10"))
        form_label = _ask("  form label", "correct", FORM_LABELS)
        append_set(folder, i, exercise, st["start"], st["end"], reps, rpe, form_label)
        previous = exercise


def _status_text(set_running, sets):
    if set_running:
        return f"SET {len(sets) + 1} RUNNING  (e = end)"
    if sets and sets[-1]["rpe"] is None:
        return f"set {len(sets)} ended: press RPE 1-9, 0 = 10"
    return f"{len(sets)} set(s)  (s = start)"


def record(user, session_id, camera_url, hr_source, hr_token="", hr_port=8081,
           phone_position="45 deg front, hip height, 2-3 m", camera_angle="45",
           body_weight="", max_seconds=None):
    import cv2

    started_at = datetime.now().isoformat(timespec="seconds")
    folder = session_dir(user, session_id, date.today())
    init_session_files(folder)
    stop = threading.Event()
    if hr_source != "none":
        threading.Thread(
            target=_hr_loop, args=(hr_source, hr_token, hr_port, stop, folder / "hr.csv"), daemon=True,
        ).start()

    cap = cv2.VideoCapture(int(camera_url) if str(camera_url).isdigit() else camera_url)
    if not cap.isOpened():
        stop.set()
        raise RuntimeError(f"could not open camera {camera_url}")
    frames = queue.Queue()
    reader = threading.Thread(target=_reader, args=(cap, frames, stop), daemon=True)
    reader.start()

    # Measure the real frame rate before opening the writer.
    probe = []
    t_probe = time.perf_counter()
    while time.perf_counter() - t_probe < FPS_PROBE_SECONDS or len(probe) < 2:
        try:
            item = frames.get(timeout=5)
        except queue.Empty:
            item = None
        if item is None:
            stop.set()
            cap.release()
            raise RuntimeError("camera opened but returned no frame")
        probe.append(item)
    fps = estimate_fps([ts for _, ts in probe])
    height, width = probe[0][0].shape[:2]
    writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"camera at {fps:.1f} fps. Focus the preview. s = set start, e = set end, digit = RPE, q = quit.")

    index = 0
    sets = []                 # {"start", "end", "rpe"}, PC milliseconds
    set_start = None
    first_ts = last_ts = None
    t0 = time.perf_counter()
    pending = list(probe)
    try:
        while True:
            if pending:
                item = pending.pop(0)
            else:
                try:
                    item = frames.get(timeout=1)
                except queue.Empty:
                    continue
            if item is None:
                print("camera stopped sending frames")
                break
            frame, ts = item
            first_ts = ts if first_ts is None else first_ts
            last_ts = ts
            writer.write(frame)
            append_frame(folder / "frames.csv", index, ts)
            index += 1

            preview = frame.copy()
            cv2.putText(preview, _status_text(set_start is not None, sets), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("awc record", preview)
            key = cv2.waitKey(1) & 0xFF
            now_ms = ts           # set marks use the clock of the frame on screen, like the video
            if key in (ord("q"), 27):
                break
            if key == ord("s") and set_start is None:
                set_start = now_ms
                print(f"set {len(sets) + 1} started")
            elif key == ord("e"):
                if set_start is None:
                    print("press s before e")
                else:
                    sets.append({"start": set_start, "end": now_ms, "rpe": None})
                    set_start = None
                    print(f"set {len(sets)} ended, press its RPE")
            elif sets and sets[-1]["rpe"] is None and rpe_from_key(key) is not None:
                sets[-1]["rpe"] = rpe_from_key(key)
                print(f"set {len(sets)} RPE {sets[-1]['rpe']}")
            if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
                break
    finally:
        stop.set()
        reader.join(timeout=2)
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        measured = fps
        if index > 1 and last_ts != first_ts:
            measured = (index - 1) / ((last_ts - first_ts) / 1000.0)
        meta = {
            "user": user,
            "date": date.today().isoformat(),
            "session_id": session_id,
            "phone_position": phone_position,
            "camera_angle": camera_angle,
            "body_weight": body_weight or None,
            "camera_url": camera_url,
            "hr_source": hr_source,
            "frames": index,
            "video_fps": round(fps, 2),
            "measured_fps": round(measured, 2),
            "clock": "frames.csv pc_ts_ms is the reference; never time anything from the video fps",
            "started_at": started_at,
        }
        write_meta(folder, meta)
        print(f"saved {folder}  frames={index}  fps={meta['measured_fps']}")
    if set_start is not None:
        print("the last set was started but never ended; it is not saved")
    _ask_set_details(folder, sets)
    return folder


def main(argv=None):
    p = argparse.ArgumentParser(description="Record a phone + watch session")
    p.add_argument("--user", required=True, choices=["yassine", "partner"])
    p.add_argument("--session-id", required=True)
    p.add_argument("--camera-url", required=True)
    p.add_argument("--hr-source", required=True, choices=["pulsoid", "hyperate", "sensorlogger", "none"])
    p.add_argument("--hr-token", default="", help="Pulsoid access token or HypeRate id")
    p.add_argument("--hr-port", type=int, default=8081)
    p.add_argument("--phone-position", default="45 deg front, hip height, 2-3 m")
    p.add_argument("--camera-angle", default="45")
    p.add_argument("--body-weight", default="")
    p.add_argument("--max-seconds", type=float, default=None, help="Stop on its own, for a timed dry run")
    args = p.parse_args(argv)
    record(
        args.user, args.session_id, args.camera_url, args.hr_source, args.hr_token, args.hr_port,
        args.phone_position, args.camera_angle, args.body_weight, args.max_seconds,
    )


if __name__ == "__main__":
    main()
