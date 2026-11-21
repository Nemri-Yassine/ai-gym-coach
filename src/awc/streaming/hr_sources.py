"""Heart-rate iterators reused by the recorder and the Kafka producer.

Pulsoid / HypeRate / Sensor Logger are implemented, but none has been confirmed
on the Galaxy Watch 7 yet. Do not claim a live watch link works.
"""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def iter_pulsoid(token):
    url = f"wss://dev.pulsoid.net/api/v1/data/real_time?access_token={token}"
    for text in _websocket_texts(url):
        msg = json.loads(text)
        data = msg.get("data", msg)
        bpm = data.get("heart_rate", data.get("heartRate", data.get("bpm")))
        if bpm is not None:
            yield float(bpm)


def iter_hyperate(token):
    """HypeRate Phoenix websocket. token is the HypeRate ID."""
    url = "wss://app.hyperate.io/socket/websocket?vsn=2.0.0"
    join = json.dumps({"topic": f"hr:{token}", "event": "phx_join", "payload": {}, "ref": "1"})
    for text in _websocket_texts(url, first_message=join):
        msg = json.loads(text)
        payload = msg.get("payload") or {}
        bpm = payload.get("hr") or payload.get("heart_rate") or payload.get("bpm")
        if bpm is not None:
            yield float(bpm)


def iter_sensorlogger(port, stop):
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
                found.put(float(bpm))

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Sensor Logger push URL: http://<this-pc>:{port}/", flush=True)
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
    import hashlib
    import os
    import socket
    import ssl
    from urllib.parse import urlparse

    parsed = urlparse(url)
    secure = parsed.scheme == "wss"
    host = parsed.hostname
    port = parsed.port or (443 if secure else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    raw = socket.create_connection((host, port), timeout=30)
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host) if secure else raw
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        f"GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    sock.sendall(req.encode("ascii"))
    header = b""
    while b"\r\n\r\n" not in header:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("websocket handshake closed")
        header += chunk
    accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
    if accept not in header:
        sock.close()
        raise ConnectionError("websocket handshake failed")
    if first_message is not None:
        _ws_send(sock, first_message)
    try:
        while True:
            text = _ws_recv_text(sock)
            if text is None:
                break
            yield text
    finally:
        sock.close()


def _ws_send(sock, text):
    data = text.encode("utf-8")
    mask = os.urandom(4) if False else __import__("os").urandom(4)
    header = bytearray([0x81])
    n = len(data)
    if n < 126:
        header.append(0x80 | n)
    elif n < 65536:
        header.append(0x80 | 126)
        header.extend(n.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(n.to_bytes(8, "big"))
    header.extend(mask)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
    sock.sendall(header + masked)


def _ws_recv_text(sock):
    header = _recvexact(sock, 2)
    if not header:
        return None
    opcode = header[0] & 0x0F
    masked = bool(header[1] & 0x80)
    length = header[1] & 0x7F
    if length == 126:
        length = int.from_bytes(_recvexact(sock, 2), "big")
    elif length == 127:
        length = int.from_bytes(_recvexact(sock, 8), "big")
    mask = _recvexact(sock, 4) if masked else b""
    payload = bytearray(_recvexact(sock, length))
    if masked:
        payload = bytearray(b ^ mask[i % 4] for i, b in enumerate(payload))
    if opcode == 0x8:
        return None
    if opcode != 0x1:
        return ""
    return payload.decode("utf-8")


def _recvexact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf
