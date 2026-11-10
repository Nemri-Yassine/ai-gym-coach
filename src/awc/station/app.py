"""FastAPI HTTPS gym station: / (control, laptop only) and /phone (iPhone Safari, PIN).

Video path: the phone records each set with MediaRecorder and uploads 1-s
chunks in order (retry until acked) to /api/upload; set_done is posted through
the same queue, so it always arrives after the last chunk.

Live preview: WebRTC phone → laptop browser (signalling relayed over the two
websockets), with a JPEG-over-websocket fallback.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import date
from pathlib import Path

from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from awc.config import STATION_PORT
from awc.data import recordings as rec
from awc.station.session_io import StationSession
from awc.station.state import RUNTIME

STATIC = Path(__file__).resolve().parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
PHONE_PATHS = ("/phone", "/api/upload/", "/api/phone/", "/static/", "/favicon.ico")


def _is_local(host: str | None) -> bool:
    return (host or "") in LOCAL_HOSTS


def create_app(pin: str | None = None, port: int = STATION_PORT) -> FastAPI:
    from awc.station.netutil import station_pin

    app = FastAPI(title="AWC Gym Station")
    app.state.pin = str(pin) if pin else station_pin()
    app.state.controls = set()  # control-page websockets
    app.state.phone_ws = None
    app.state.cert_names = []
    app.state.port = int(port)
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    def pin_ok(request_or_ws) -> bool:
        q = request_or_ws.query_params.get("pin")
        h = request_or_ws.headers.get("x-awc-pin")
        c = request_or_ws.cookies.get("awc_pin")
        return app.state.pin in (q, h, c)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.client.host if request.client else None
        if _is_local(host):
            return await call_next(request)
        path = request.url.path
        if not path.startswith(PHONE_PATHS):
            return JSONResponse({"error": "control page is laptop-only"}, status_code=403)
        if path.startswith("/static/") or path == "/favicon.ico":
            return await call_next(request)
        if not pin_ok(request):
            if path == "/phone":
                return HTMLResponse(_PIN_PAGE, status_code=401)
            return JSONResponse({"error": "bad pin"}, status_code=401)
        resp = await call_next(request)
        if path == "/phone":
            resp.set_cookie("awc_pin", app.state.pin, max_age=30 * 86400, secure=True, samesite="strict")
        return resp

    # ------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    def control():
        return (STATIC / "control.html").read_text(encoding="utf-8")

    @app.get("/phone", response_class=HTMLResponse)
    def phone():
        return (STATIC / "phone.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status():
        return RUNTIME.snapshot()

    @app.get("/api/setup")
    def setup_info():
        from awc.station.netutil import lan_ips, phone_url, qr_data_url

        ips = lan_ips()
        urls = [phone_url(ip, app.state.pin, app.state.port) for ip in ips]
        names = set(app.state.cert_names or [])
        return {
            "ips": ips,
            "phone_url": urls[0],
            "phone_urls": urls,
            "qr_data_url": qr_data_url(urls[0]),
            "pin": app.state.pin,
            "port": app.state.port,
            "cert_missing_ips": [ip for ip in ips if names and ip not in names],
            "exercises": list(rec.EXERCISES),
            "form_by_exercise": {k: list(v) for k, v in rec.FORM_BY_EXERCISE.items()},
            "users": list(rec.USERS),
        }

    @app.get("/api/qr")
    def qr(ip: str):
        from awc.station.netutil import phone_url, qr_data_url

        url = phone_url(ip, app.state.pin, app.state.port)
        return {"phone_url": url, "qr_data_url": qr_data_url(url)}

    # ------------------------------------------------------------ session
    @app.post("/api/session/start")
    def session_start(body: dict | None = Body(None)):
        body = body or {}
        user = body.get("user", "yassine")
        session_id = (body.get("session_id") or "s01").strip()
        session_type = body.get("session_type", "fatigue")
        try:
            on_date = date.fromisoformat(body["date"]) if body.get("date") else date.today()
            folder = rec.session_dir(user, session_id, on_date=on_date)
        except ValueError as exc:
            return JSONResponse({"error": f"{exc} (no spaces or slashes, e.g. s02)"}, status_code=400)
        meta = {
            "user": user,
            "date": on_date.isoformat(),
            "session_id": session_id,
            "session_type": session_type,
            "source": "gym-station-iphone",
            "video_mode": "upload",
            "started_at_ms": int(time.time() * 1000),
            "hr_source": RUNTIME.hr_source,
            "cam_source": "iphone",
            "phone_position": body.get("phone_position") or "45° front, hip height, 2-3 m",
            **{k: v for k, v in RUNTIME.phone_info.items() if v is not None},
        }
        if folder.exists() and (folder / "meta.json").exists():
            existing = {}
            try:
                existing = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
            meta = {**existing, **meta, "started_at_ms": existing.get("started_at_ms", meta["started_at_ms"]),
                    "resumed_at_ms": int(time.time() * 1000)}
        sess = StationSession(folder, meta=meta)
        nxt = sess.next_set()
        with RUNTIME.lock:
            RUNTIME.session = sess
            RUNTIME.pending_labels = None
            RUNTIME.recording = False
            RUNTIME.upload = {}
        rec.append_event(folder, int(time.time() * 1000), "session_start", session_id)
        return {"folder": str(folder), "next_set": nxt, "resumed": nxt > 1}

    @app.post("/api/session/end")
    def session_end():
        with RUNTIME.lock:
            sess = RUNTIME.session
            if sess:
                sess.update_meta(ended_at_ms=int(time.time() * 1000), hr_drops=RUNTIME.hr_drops)
                rec.append_event(sess.folder, int(time.time() * 1000), "session_end", "")
            RUNTIME.session = None
            RUNTIME.recording = False
            RUNTIME.pending_labels = None
        return {"ok": True}

    # ------------------------------------------------------------ sets
    @app.post("/api/set/start")
    def set_start(body: dict | None = Body(None)):
        body = body or {}
        with RUNTIME.lock:
            if RUNTIME.session is None:
                return JSONResponse({"error": "no session"}, status_code=400)
            if RUNTIME.recording:
                return JSONResponse({"error": "already recording"}, status_code=400)
            if RUNTIME.pending_labels is not None:
                return JSONResponse({"error": "save the previous set first"}, status_code=400)
            if not RUNTIME.phone_connected and not body.get("allow_no_phone"):
                return JSONResponse({"error": "iPhone not connected"}, status_code=400)
            n = RUNTIME.session.next_set()
            start_ms = int(time.time() * 1000)
            ext = "webm" if "webm" in (RUNTIME.phone_info.get("mime") or "") else "mp4"
            RUNTIME.session.start_set(n, start_ms, ext=ext)
            RUNTIME.recording = True
            RUNTIME.rec_confirmed = False
            RUNTIME.current_set = n
            RUNTIME.upload[n] = {"received": 0, "acked": 0, "pending": 0, "done": False, "bytes": 0}
            RUNTIME.push_cmd({"type": "start_set", "set": n, "pc_start_ms": start_ms,
                              "session_id": RUNTIME.session.meta.get("session_id", "s01")})
        return {"set": n, "start_ms": start_ms}

    @app.post("/api/set/end")
    def set_end():
        with RUNTIME.lock:
            if not RUNTIME.recording or RUNTIME.session is None:
                return JSONResponse({"error": "not recording"}, status_code=400)
            end_ms = int(time.time() * 1000)
            n = RUNTIME.current_set
            a = RUNTIME.session.get_set(n)
            if a is not None:
                a.end_ms = end_ms
            RUNTIME.push_cmd({"type": "end_set", "set": n, "pc_end_ms": end_ms})
            RUNTIME.recording = False
            RUNTIME.pending_labels = {
                "set": n, "start_ms": a.start_ms if a else end_ms, "end_ms": end_ms,
                "upload_complete": False, "video": None, "n_chunks": None,
            }
            rec.append_event(RUNTIME.session.folder, end_ms, "set_end", f"set={n}")
        return {"ok": True, "end_ms": end_ms, "set": n}

    @app.post("/api/set/save")
    def set_save(body: dict = Body(...)):
        with RUNTIME.lock:
            sess = RUNTIME.session
            pending = RUNTIME.pending_labels
            if sess is None or pending is None:
                return JSONResponse({"error": "nothing to save"}, status_code=400)
            if not pending.get("upload_complete") and not body.get("force"):
                return JSONResponse({"error": "video still uploading", "upload": RUNTIME.upload.get(pending["set"])},
                                    status_code=409)
            n = int(pending["set"])
            start_ms = int(pending["start_ms"])
            end_ms = int(pending["end_ms"])
            hr = sess.hr_stats(start_ms, end_ms)
            video = pending.get("video") or {}
            note = (body.get("note") or "").replace(",", ";")
            if not pending.get("upload_complete"):
                note = (note + " | video incomplete").strip(" |")
            extra = {
                "session_type": sess.meta.get("session_type", ""),
                "weight_kg": body.get("weight_kg", ""),
                "hold_s": body.get("hold_s", ""),
                "note": note,
                "duration_s": round((end_ms - start_ms) / 1000.0, 1),
                **hr,
            }
            sess.save_set_labels(
                n, body["exercise"], start_ms, end_ms,
                body["reps"], body["rpe"], body["form_label"], extra,
            )
            RUNTIME.pending_labels = None
            RUNTIME.current_set = None
        vstats = (video or {}).get("video") or {}
        return {"ok": True, "set": n, "extra": extra,
                "video_ok": bool(vstats.get("ok")), "frames": vstats.get("frames"), "fps": vstats.get("fps")}

    @app.post("/api/set/discard")
    def set_discard():
        """Drop the labels of the pending set (video file stays on disk, never reused)."""
        with RUNTIME.lock:
            pending = RUNTIME.pending_labels
            if RUNTIME.session and pending:
                rec.append_event(RUNTIME.session.folder, int(time.time() * 1000), "set_discarded",
                                 f"set={pending['set']}")
            RUNTIME.pending_labels = None
            RUNTIME.current_set = None
        return {"ok": True}

    @app.post("/api/clap")
    def clap():
        with RUNTIME.lock:
            if RUNTIME.session:
                rec.append_event(RUNTIME.session.folder, int(time.time() * 1000), "clap", "")
        return {"ok": True}

    # ------------------------------------------------------------ phone uploads (PIN)
    @app.post("/api/upload/{set_index}/{seq}")
    async def upload_chunk(set_index: int, seq: int, request: Request):
        data = await request.body()
        with RUNTIME.lock:
            sess = RUNTIME.session
            a = sess.get_set(set_index) if sess else None
            if a is None:
                # unknown set (e.g. station restarted): ack so the phone doesn't retry forever
                return {"acked": [seq], "unknown_set": True}
            if a.finalized is not None:
                return {"acked": [seq], "late": True}
            newly = a.assembler.add(seq, data)
            u = RUNTIME.upload.setdefault(int(set_index), {"received": 0, "acked": 0, "pending": 0, "done": False,
                                                           "bytes": 0})
            u["received"] = a.assembler.received_count
            u["acked"] = len(a.assembler._acked)
            u["pending"] = a.assembler.pending
            u["bytes"] = u.get("bytes", 0) + len(data)
        return {"acked": newly or [seq], "next": a.assembler.next_expected}

    @app.post("/api/phone/set_done/{set_index}")
    async def phone_set_done(set_index: int, body: dict = Body(...)):
        n = int(set_index)
        with RUNTIME.lock:
            sess = RUNTIME.session
            a = sess.get_set(n) if sess else None
            if a is None:
                return {"ok": True, "unknown_set": True}
            if a.finalized is not None:
                return {"ok": True, "already": True}
            a.phone_rec_start = body.get("phone_rec_start_ms")
            if body.get("frames"):
                a.phone_frame_ts = [float(t) for t in body["frames"]]
            n_chunks = int(body.get("n_chunks") or 0)
            complete = a.assembler.is_complete(n_chunks) if n_chunks else False
            end_ms = a.end_ms or int(time.time() * 1000)
            if not complete:
                return JSONResponse({"ok": False, "reason": "chunks missing", "next": a.assembler.next_expected,
                                     "n_chunks": n_chunks}, status_code=409)
        info = await asyncio.to_thread(sess.finish_upload, n_chunks, end_ms, RUNTIME.clock.phone_to_pc, n)
        with RUNTIME.lock:
            RUNTIME.upload[n] = {**RUNTIME.upload.get(n, {}), "done": True, "acked": n_chunks,
                                 "received": n_chunks, "pending": 0,
                                 "frames": info["video"]["frames"], "fps": info["video"]["fps"]}
            if RUNTIME.pending_labels and int(RUNTIME.pending_labels["set"]) == n:
                RUNTIME.pending_labels["upload_complete"] = True
                RUNTIME.pending_labels["video"] = info
                RUNTIME.pending_labels["n_chunks"] = n_chunks
            if body.get("fps_actual"):
                sess.update_meta(fps_actual=body.get("fps_actual"))
        return {"ok": True, "frames": info["video"]["frames"], "fps": info["video"]["fps"]}

    # ------------------------------------------------------------ HR
    @app.post("/api/hr/source")
    def hr_source(body: dict | None = Body(None)):
        body = body or {}
        RUNTIME.hr_source = body.get("source", "none")
        RUNTIME.hr_status = "idle"
        return {"ok": True, "source": RUNTIME.hr_source}

    @app.get("/api/hr/devices")
    def hr_devices():
        from awc.station.hr_ble import load_last_device, scan_hr_devices

        RUNTIME.hr_status = "scanning BLE…"
        try:
            found = scan_hr_devices(timeout=5.0)
        except Exception as exc:  # noqa: BLE001
            RUNTIME.hr_status = f"scan error: {exc}"
            return JSONResponse({"error": str(exc), "devices": [], "last": load_last_device()}, status_code=500)
        RUNTIME.hr_status = f"found {len(found)} HR device(s)"
        return {"devices": found, "last": load_last_device()}

    @app.post("/api/hr/connect")
    def hr_connect(body: dict | None = Body(None)):
        body = body or {}
        source = body.get("source", RUNTIME.hr_source)
        _stop_hr(app)
        RUNTIME.hr_source = source
        RUNTIME.hr_bpm = None
        RUNTIME.hr_drops = 0
        if RUNTIME.session:
            RUNTIME.session.update_meta(hr_source=source)
        if source == "ble":
            from awc.station.hr_ble import BleHR, load_last_device

            def on_sample(bpm, rr, ts):
                RUNTIME.note_hr(bpm, ts, rr=rr)
                RUNTIME.hr_status = "streaming (watch)"

            def on_status(state, detail):
                RUNTIME.hr_status = f"{state}: {detail}"
                if state == "disconnected":
                    RUNTIME.hr_drops += 1
                with RUNTIME.lock:
                    if RUNTIME.session and state in ("disconnected", "error", "connected"):
                        rec.append_event(RUNTIME.session.folder, int(time.time() * 1000), "hr_" + state, detail)

            address = body.get("address")
            if not address:
                address = (load_last_device() or {}).get("address")
            ble = BleHR(on_sample, on_status)
            ble.start(address=address, name_substr=body.get("name"))
            app.state.ble = ble  # type: ignore[attr-defined]
            return {"ok": True, "source": "ble", "address": address,
                    "note": "Open Heart for Bluetooth first, keep the watch screen on"}
        if source == "pulsoid":
            token = body.get("token", "")
            if not token:
                return JSONResponse({"error": "token required"}, status_code=400)
            _start_pulsoid(app, token)
            return {"ok": True, "source": "pulsoid"}
        if source == "demo":
            _start_demo_hr(app)
            return {"ok": True, "source": "demo", "warning": "FAKE heart rate (~70–120). Not the watch."}
        return {"ok": True, "source": "none"}

    @app.get("/api/preview.jpg")
    def preview():
        with RUNTIME.lock:
            data = RUNTIME.preview_jpeg
        if not data:
            return Response(status_code=204)
        return Response(content=data, media_type="image/jpeg")

    # ------------------------------------------------------------ websockets
    async def to_controls(text: str | None = None, data: bytes | None = None):
        dead = []
        for c in list(app.state.controls):
            try:
                async with c.state.send_lock:
                    if data is not None:
                        await c.send_bytes(data)
                    else:
                        await c.send_text(text)
            except Exception:  # noqa: BLE001
                dead.append(c)
        for c in dead:
            app.state.controls.discard(c)

    async def to_phone(text: str) -> bool:
        ws = app.state.phone_ws
        if ws is None:
            return False
        try:
            async with ws.state.send_lock:
                await ws.send_text(text)
            return True
        except Exception:  # noqa: BLE001
            return False

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket):
        host = ws.client.host if ws.client else None
        if not _is_local(host) and not pin_ok(ws):
            await ws.close(code=4401)
            return
        await ws.accept()
        ws.state.send_lock = asyncio.Lock()
        old = app.state.phone_ws
        app.state.phone_ws = ws
        if old is not None:
            try:
                await old.close(code=4000)
            except Exception:  # noqa: BLE001
                pass
        with RUNTIME.lock:
            RUNTIME.phone_connected = True
        RUNTIME.pop_cmds()  # stale commands from before the reconnect
        await to_controls(json.dumps({"type": "phone_connected"}))

        async def sender():
            n_ping = 0
            next_ping = 0.0
            while True:
                for cmd in RUNTIME.pop_cmds():
                    await to_phone(json.dumps(cmd))
                now = time.time()
                if now >= next_ping:
                    await to_phone(json.dumps({"type": "ping", "t0": now * 1000.0}))
                    n_ping += 1
                    next_ping = now + (0.3 if n_ping < 8 else 5.0)  # burst at connect, then every 5 s
                await asyncio.sleep(0.03)

        task = asyncio.create_task(sender())
        try:
            while True:
                raw = await ws.receive()
                if raw.get("type") == "websocket.disconnect":
                    break
                if raw.get("bytes") is not None:  # JPEG preview fallback
                    data = raw["bytes"]
                    with RUNTIME.lock:
                        RUNTIME.preview_jpeg = data
                        RUNTIME.preview_ts = time.time() * 1000.0
                    await to_controls(data=data)
                    continue
                text = raw.get("text")
                if not text:
                    continue
                msg = json.loads(text)
                if str(msg.get("type", "")).startswith("rtc_"):
                    await to_controls(text)
                    continue
                _handle_phone_msg(msg)
                if msg.get("type") in ("rec_started", "rec_error"):
                    await to_controls(text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            task.cancel()
            if app.state.phone_ws is ws:
                app.state.phone_ws = None
                with RUNTIME.lock:
                    RUNTIME.phone_connected = False
                await to_controls(json.dumps({"type": "phone_disconnected"}))

    @app.websocket("/ws/control")
    async def ws_control(ws: WebSocket):
        host = ws.client.host if ws.client else None
        if not _is_local(host):
            await ws.close(code=4403)
            return
        await ws.accept()
        ws.state.send_lock = asyncio.Lock()
        app.state.controls.add(ws)
        try:
            while True:
                text = await ws.receive_text()
                msg = json.loads(text)
                if str(msg.get("type", "")).startswith("rtc_") or msg.get("type") == "jpeg":
                    await to_phone(text)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            app.state.controls.discard(ws)

    return app


def _handle_phone_msg(msg: dict):
    typ = msg.get("type")
    if typ == "pong":
        t1 = time.time() * 1000.0
        RUNTIME.clock.add(float(msg["t0"]), float(msg["phone_ts"]), t1)
        offset, rtt = RUNTIME.clock.estimate()
        RUNTIME.last_offset_ms = offset
        RUNTIME.last_rtt_ms = rtt
        with RUNTIME.lock:
            if RUNTIME.session and offset is not None and rtt is not None:
                RUNTIME.session.append_clock(int(t1), offset, rtt)
        return
    if typ == "hello":
        info = {
            "phone_ua": msg.get("ua", ""),
            "mime": msg.get("mime"),
            "resolution": msg.get("resolution"),
            "fps_requested": msg.get("fps_requested"),
            "fps_actual": msg.get("fps_actual"),
        }
        with RUNTIME.lock:
            RUNTIME.phone_ua = info["phone_ua"]
            RUNTIME.phone_battery = msg.get("battery")
            RUNTIME.phone_info = info
            if RUNTIME.session:
                RUNTIME.session.update_meta(**info)
        return
    if typ == "status":
        with RUNTIME.lock:
            RUNTIME.phone_battery = msg.get("battery", RUNTIME.phone_battery)
        return
    if typ == "rec_started":
        with RUNTIME.lock:
            if RUNTIME.current_set == msg.get("set"):
                RUNTIME.rec_confirmed = True
            sess = RUNTIME.session
            a = sess.get_set(int(msg.get("set", -1))) if sess else None
            if a is not None:
                a.phone_rec_start = msg.get("phone_ts")
        return
    if typ == "rec_error":
        with RUNTIME.lock:
            if RUNTIME.session:
                rec.append_event(RUNTIME.session.folder, int(time.time() * 1000), "phone_rec_error",
                                 str(msg.get("detail", ""))[:200].replace(",", ";"))
        return


def _stop_hr(app):
    stop = getattr(app.state, "hr_stop", None)
    if stop is not None:
        stop.set()
    ble = getattr(app.state, "ble", None)
    if ble is not None:
        try:
            ble.stop()
        except Exception:  # noqa: BLE001
            pass
        app.state.ble = None


def _start_pulsoid(app, token: str):
    import threading

    from awc.streaming.hr_sources import iter_pulsoid

    stop = threading.Event()
    app.state.hr_stop = stop

    def run():
        while not stop.is_set():
            RUNTIME.hr_status = "connecting pulsoid"
            try:
                for bpm in iter_pulsoid(token):
                    if stop.is_set():
                        break
                    RUNTIME.note_hr(float(bpm), time.time() * 1000.0)
                    RUNTIME.hr_status = "streaming (pulsoid)"
            except Exception as exc:  # noqa: BLE001
                RUNTIME.hr_status = f"pulsoid error: {exc} · retrying"
                RUNTIME.hr_drops += 1
            stop.wait(2.0)

    threading.Thread(target=run, daemon=True).start()


def _start_demo_hr(app):
    import math
    import threading

    stop = threading.Event()
    app.state.hr_stop = stop

    def run():
        RUNTIME.hr_status = "DEMO fake HR (not the watch)"
        i = 0
        while not stop.is_set():
            bpm = 95 + 25 * math.sin(i / 20) + (i % 3)
            RUNTIME.note_hr(bpm, time.time() * 1000.0)
            i += 1
            time.sleep(1.0)

    threading.Thread(target=run, daemon=True).start()


_PIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>AWC PIN</title>
<style>body{margin:0;background:#0B1016;color:#EEF2F6;font:18px system-ui;display:grid;place-items:center;height:100vh}
form{display:flex;flex-direction:column;gap:12px;width:260px}input,button{font-size:28px;padding:14px;border-radius:12px;border:0}
input{text-align:center;letter-spacing:.3em}button{background:#7393FF;font-weight:800}</style></head>
<body><form method="get" action="/phone"><b>AWC station PIN</b>
<input name="pin" inputmode="numeric" autocomplete="off" autofocus><button>Open camera page</button>
<small>The PIN is shown on the laptop screen.</small></form></body></html>"""
