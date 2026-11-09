"""LAN helpers for the gym station setup screen."""
from __future__ import annotations

import io
import socket
from urllib.request import urlopen

from awc.config import STATION_PORT


def lan_ips() -> list[str]:
    ips = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    # UDP trick: discover the outbound interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip not in ips and not ip.startswith("127."):
            ips.insert(0, ip)
    except OSError:
        pass
    return ips or ["127.0.0.1"]


def phone_url(ip: str | None = None) -> str:
    ip = ip or (lan_ips()[0] if lan_ips() else "127.0.0.1")
    return f"https://{ip}:{STATION_PORT}/phone"


def qr_data_url(url: str) -> str:
    try:
        import qrcode
    except ImportError:
        return ""
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64

    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def firewall_rule_cmd(port: int = STATION_PORT) -> str:
    return (
        f'netsh advfirewall firewall add rule name="AWC Gym Station {port}" '
        f"dir=in action=allow protocol=TCP localport={port} profile=private"
    )


def port_listening(port: int = STATION_PORT, host="127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False
