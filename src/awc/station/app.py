"""FastAPI HTTPS gym station: / (control) and /phone (iPhone Safari)."""
from __future__ import annotations

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


def create_app() -> FastAPI:
    app = FastAPI(title="AWC Gym Station")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

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
        url = phone_url(ips[0] if ips else "127.0.0.1")
        return {
            "ips": ips,
            "phone_url": url,
            "qr_data_url": qr_data_url(url),
            "port": STATION_PORT,
            "exercises": list(rec.EXERCISES),
            "form_by_exercise": {k: list(v) for k, v in rec.FORM_BY_EXERCISE.items()},
            "users": list(rec.USERS),
        }

    @app.post("/api/session/start")
    def session_start(body: dict | None = Body(None)):
        body = body or {}
        user = body.get("user", "yassine")
        session_id = (body.get("session_id") or "s01").strip()
        session_type = body.get("session_type", "fatigue")
        on_date = date.fromisoformat(body["date"]) if body.get("date") else date.today()
        folder = rec.session_dir(user, session_id, on_date=on_date)
        meta = {
            "user": user,
            "date": on_date.isoformat(),
            "session_id": session_id,
            "session_type": session_type,
            "source": "gym-station-iphone",
            "started_at_ms": int(time.time() * 1000),
            "hr_source": RUNTIME.hr_source,
            "cam_source": RUNTIME.cam_source,
        }
        if folder.exists() and (folder / "sets.csv").exists():
            # resume — never wipe
            existing = {}
            try:
                existing = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
            meta = {**existing, **meta, "resumed_at_ms": int(time.time() * 1000)}
        sess = StationSession(folder, meta=meta)
        nxt = sess.next_set()
        with RUNTIME.lock:
            RUNTIME.session = sess
            RUNTIME.pending_labels = None
        rec.append_event(folder, int(time.time() * 1000), "session_start", session_id)
        return {"folder": str(folder), "next_set": nxt, "resumed": nxt > 1}

    @app.post("/api/session/end")
    def session_end():
        with RUNTIME.lock:
            sess = RUNTIME.session
            if sess:
                sess.update_meta(ended_at_ms=int(time.time() * 1000))
                rec.append_event(sess.folder, int(time.time() * 1000), "session_end", "")
            RUNTIME.session = None
            RUNTIME.recording = False
        return {"ok": True}

    @app.post("/api/set/start")
    def set_start(body: dict | None = Body(None)):
        body = body or {}
        with RUNTIME.lock:
            if RUNTIME.session is None:
                return JSONResponse({"error": "no session"}, status_code=400)
            if RUNTIME.recording:
                return JSONResponse({"error": "already recording"}, status_code=400)
            n = RUNTIME.session.next_set()
            ext = body.get("ext", "mp4")
            start_ms = int(time.time() * 1000)
            RUNTIME.session.start_set(n, start_ms, ext=ext)
            RUNTIME.recording = True
            RUNTIME.current_set = n
            RUNTIME.upload[n] = {"received": 0, "acked": 0, "pending": 0, "done": False}
            RUNTIME.pending_labels = None
            RUNTIME.push_cmd({"type": "start_set", "set": n, "pc_start_ms": start_ms})
        return {"set": n, "start_ms": start_ms}

    @app.post("/api/set/end")
    def set_end():
        with RUNTIME.lock:
            if not RUNTIME.recording:
                return JSONResponse({"error": "not recording"}, status_code=400)
            end_ms = int(time.time() * 1000)
            RUNTIME.push_cmd({"type": "end_set", "pc_end_ms": end_ms})
            RUNTIME.recording = False
        return {"ok": True, "end_ms": end_ms}

    @app.post("/api/set/save")
    def set_save(body: dict = Body(...)):
        with RUNTIME.lock:
            sess = RUNTIME.session
            pending = RUNTIME.pending_labels
            if sess is None or pending is None:
                return JSONResponse({"error": "nothing to save"}, status_code=400)
            if not pending.get("upload_complete"):
                return JSONResponse({"error": "upload not complete"}, status_code=400)
            n = int(pending["set"])
            start_ms = int(pending["start_ms"])
            end_ms = int(pending["end_ms"])
            hr = sess.hr_stats(start_ms, end_ms)
            extra = {
                "session_type": sess.meta.get("session_type", ""),
                "weight_kg": body.get("weight_kg", ""),
                "hold_s": body.get("hold_s", ""),
                "note": (body.get("note") or "").replace(",", ";"),
                "duration_s": round((end_ms - start_ms) / 1000.0, 1),
                **hr,
            }
            sess.save_set_labels(
                n, body["exercise"], start_ms, end_ms,
                body["reps"], body["rpe"], body["form_label"], extra,
            )
            RUNTIME.pending_labels = None
            RUNTIME.current_set = None
        return {"ok": True, "set": n, "extra": extra}

    @app.post("/api/clap")
    def clap():
        with RUNTIME.lock:
            if RUNTIME.session:
                rec.append_event(RUNTIME.session.folder, int(time.time() * 1000), "clap", "")
        return {"ok": True}

    @app.post("/api/hr/source")
    def hr_source(body: dict | None = Body(None)):
        body = body or {}
        RUNTIME.hr_source = body.get("source", "none")
        RUNTIME.hr_status = "idle"
        return {"ok": True, "source": RUNTIME.hr_source}

    @app.post("/api/hr/connect")
    def hr_connect(body: dict | None = Body(None)):
        body = body or {}
        source = body.get("source", RUNTIME.hr_source)
        RUNTIME.hr_source = source
        if source == "ble":
            from awc.station.hr_ble import BleHR

            def on_sample(bpm, rr, ts):
                RUNTIME.note_hr(bpm, ts, rr=rr)
                RUNTIME.hr_status = "streaming"

            def on_status(state, detail):
                RUNTIME.hr_status = f"{state}: {detail}"
                with RUNTIME.lock:
                    if RUNTIME.session and state in ("disconnected", "error", "reconnect"):
                        rec.append_event(RUNTIME.session.folder, int(time.time() * 1000), "hr_" + state, detail)

            ble = BleHR(on_sample, on_status)
            ble.start(address=body.get("address"), name_substr=body.get("name"))
            app.state.ble = ble  # type: ignore[attr-defined]
            return {"ok": True, "source": "ble"}
        if source == "pulsoid":
            token = body.get("token", "")
            if not token:
                return JSONResponse({"error": "token required"}, status_code=400)
            _start_pulsoid(token)
            return {"ok": True, "source": "pulsoid"}
        if source == "demo":
            _start_demo_hr()
            return {"ok": True, "source": "demo"}
        return {"ok": True, "source": "none"}

    @app.get("/api/preview.jpg")
    def preview():
        with RUNTIME.lock:
            data = RUNTIME.preview_jpeg
        if not data:
            return Response(status_code=204)
        return Response(content=data, media_type="image/jpeg")

    @app.post("/api/upload/{set_index}/{seq}")
    async def upload_chunk(set_index: int, seq: int, request: Request):
        data = await request.body()
        with RUNTIME.lock:
            if RUNTIME.session is None or RUNTIME.session.active is None:
                return JSONResponse({"error": "no active set"}, status_code=400)
            if RUNTIME.session.active.set_index != int(set_index):
                return JSONResponse({"error": "set mismatch"}, status_code=400)
            newly = RUNTIME.session.add_chunk(seq, data)
            u = RUNTIME.upload.setdefault(int(set_index), {"received": 0, "acked": 0, "pending": 0, "done": False})
            u["received"] = RUNTIME.session.active.assembler.received_count
            u["acked"] = len(RUNTIME.session.active.assembler._acked)
            u["pending"] = RUNTIME.session.active.assembler.pending
        return {"acked": newly, "next": RUNTIME.session.active.assembler.next_expected}

    @app.websocket("/ws/phone")
    async def ws_phone(ws: WebSocket):
        import asyncio

        await ws.accept()
        with RUNTIME.lock:
            RUNTIME.phone_connected = True
        last_ping = 0.0
        try:
            while True:
                for cmd in RUNTIME.pop_cmds():
                    await ws.send_text(json.dumps(cmd))
                now = time.time()
                if now - last_ping >= 10.0:
                    await ws.send_text(json.dumps({"type": "ping", "t0": now * 1000.0}))
                    last_ping = now
                try:
                    raw = await asyncio.wait_for(ws.receive(), timeout=0.4)
                except asyncio.TimeoutError:
                    continue
                if raw.get("type") == "websocket.disconnect":
                    break
                if raw.get("bytes") is not None:
                    with RUNTIME.lock:
                        RUNTIME.preview_jpeg = raw["bytes"]
                        RUNTIME.preview_ts = time.time() * 1000.0
                    continue
                text = raw.get("text")
                if not text:
                    continue
                msg = json.loads(text)
                await _handle_phone_msg(ws, msg)
        except WebSocketDisconnect:
            pass
        finally:
            with RUNTIME.lock:
                RUNTIME.phone_connected = False

    return app


async def _handle_phone_msg(ws: WebSocket, msg: dict):
    typ = msg.get("type")
    if typ == "pong":
        t1 = time.time() * 1000.0
        t0 = float(msg["t0"])
        phone_ts = float(msg["phone_ts"])
        RUNTIME.clock.add(t0, phone_ts, t1)
        offset, rtt = RUNTIME.clock.estimate()
        RUNTIME.last_offset_ms = offset
        RUNTIME.last_rtt_ms = rtt
        with RUNTIME.lock:
            if RUNTIME.session and offset is not None and rtt is not None:
                RUNTIME.session.append_clock(int(t1), offset, rtt)
        return
    if typ == "hello":
        with RUNTIME.lock:
            RUNTIME.phone_ua = msg.get("ua", "")
            RUNTIME.phone_battery = msg.get("battery")
            if RUNTIME.session:
                RUNTIME.session.update_meta(
                    phone_ua=RUNTIME.phone_ua,
                    phone_model=msg.get("model"),
                    mime=msg.get("mime"),
                    resolution=msg.get("resolution"),
                )
        return
    if typ == "status":
        with RUNTIME.lock:
            RUNTIME.phone_battery = msg.get("battery", RUNTIME.phone_battery)
        return
    if typ == "preview":
        import base64

        b64 = msg.get("jpeg_b64", "")
        if b64:
            with RUNTIME.lock:
                RUNTIME.preview_jpeg = base64.b64decode(b64)
                RUNTIME.preview_ts = time.time() * 1000.0
        return
    if typ == "frames":
        with RUNTIME.lock:
            if RUNTIME.session and RUNTIME.session.active:
                RUNTIME.session.add_phone_frames(msg.get("frames") or [])
        return
    if typ == "set_done":
        n = int(msg["set"])
        n_chunks = int(msg.get("n_chunks", 0))
        phone_end = float(msg.get("phone_end_ms") or time.time() * 1000)
        with RUNTIME.lock:
            sess = RUNTIME.session
            if sess is None or sess.active is None:
                return
            # Laptop stamped start_ms on Start set; convert phone end → laptop clock.
            start_ms = int(sess.active.start_ms)
            end_ms = int(round(RUNTIME.clock.phone_to_pc(phone_end)))
            if end_ms < start_ms:
                end_ms = int(time.time() * 1000)
            complete = sess.active.assembler.is_complete(n_chunks) if n_chunks else (
                sess.active.assembler.pending == 0 and sess.active.assembler.next_expected > 0
            )
            if not n_chunks:
                n_chunks = sess.active.assembler.next_expected
                complete = sess.active.assembler.pending == 0 and n_chunks > 0
            info = None
            if complete:
                info = sess.finish_upload(n_chunks, end_ms, RUNTIME.clock.phone_to_pc)
                RUNTIME.upload[n] = {
                    **RUNTIME.upload.get(n, {}),
                    "done": True,
                    "acked": n_chunks,
                    "received": n_chunks,
                    "pending": 0,
                }
            RUNTIME.pending_labels = {
                "set": n,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "upload_complete": complete,
                "n_chunks": n_chunks,
                "video": info,
            }
            RUNTIME.recording = False
        await ws.send_text(json.dumps({"type": "set_done_ack", "set": n, "complete": complete}))
        return


def _start_pulsoid(token: str):
    import threading

    from awc.streaming.hr_sources import iter_pulsoid

    stop = threading.Event()

    def run():
        RUNTIME.hr_status = "connecting pulsoid"
        try:
            for bpm in iter_pulsoid(token):
                if stop.is_set():
                    break
                RUNTIME.note_hr(float(bpm), time.time() * 1000.0)
                RUNTIME.hr_status = "streaming"
        except Exception as exc:  # noqa: BLE001
            RUNTIME.hr_status = f"pulsoid error: {exc}"

    t = threading.Thread(target=run, daemon=True)
    t.start()


def _start_demo_hr():
    import math
    import threading

    stop = threading.Event()

    def run():
        RUNTIME.hr_status = "streaming"
        i = 0
        while not stop.is_set():
            bpm = 95 + 25 * math.sin(i / 20) + (i % 3)
            RUNTIME.note_hr(bpm, time.time() * 1000.0)
            i += 1
            time.sleep(1.0)

    threading.Thread(target=run, daemon=True).start()
