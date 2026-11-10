"""LAN, certificate and PIN helpers for the gym station.

Certificates: the iPhone trusts the mkcert **root CA** once. The station then
re-issues its own leaf certificate at every start for whatever IPs the laptop
has right now (plus the whole iPhone hotspot range 172.20.10.x), so changing
Wi-Fi never needs anything new on the phone.
"""
from __future__ import annotations

import base64
import io
import json
import os
import secrets
import shutil
import socket
import subprocess
from pathlib import Path

from awc.config import (
    STATION_CERT_DIR,
    STATION_CERT_FILE,
    STATION_KEY_FILE,
    STATION_PORT,
    TMP,
)

HOTSPOT_IPS = [f"172.20.10.{i}" for i in range(1, 16)]  # iPhone Personal Hotspot subnet
CERT_NAMES_FILE = STATION_CERT_DIR / "station_names.json"
STATION_SETTINGS = TMP / "station.json"


# ---------------------------------------------------------------- IPs
def _is_usable(ip: str) -> bool:
    return not (ip.startswith("127.") or ip.startswith("169.254."))


def all_ipv4() -> list[str]:
    """Every IPv4 address on every interface that is up (psutil if present)."""
    ips: list[str] = []
    try:
        import psutil

        stats = psutil.net_if_stats()
        for name, addrs in psutil.net_if_addrs().items():
            if name in stats and not stats[name].isup:
                continue
            for a in addrs:
                if a.family == socket.AF_INET and _is_usable(a.address) and a.address not in ips:
                    ips.append(a.address)
    except Exception:  # noqa: BLE001
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _is_usable(ip) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def lan_ips() -> list[str]:
    """Usable IPs, the one with the default route first (works offline too)."""
    ips = all_ipv4()
    primary = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent
        primary = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    if primary and _is_usable(primary):
        ips = [primary] + [i for i in ips if i != primary]
    # Hotspot address is the most likely one the phone can reach: put it first.
    ips.sort(key=lambda ip: 0 if ip.startswith("172.20.10.") else 1)
    return ips or ["127.0.0.1"]


# ---------------------------------------------------------------- PIN
def load_settings() -> dict:
    try:
        return json.loads(STATION_SETTINGS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(d: dict):
    TMP.mkdir(parents=True, exist_ok=True)
    STATION_SETTINGS.write_text(json.dumps(d, indent=2), encoding="utf-8")


def station_pin(override: str | None = None) -> str:
    """4-digit PIN kept in C:/awc_tmp/station.json so the phone bookmark keeps working."""
    s = load_settings()
    if override:
        s["pin"] = str(override)
        save_settings(s)
    if not s.get("pin"):
        s["pin"] = f"{secrets.randbelow(10_000):04d}"
        save_settings(s)
    return str(s["pin"])


def phone_url(ip: str | None = None, pin: str | None = None, port: int | None = None) -> str:
    ip = ip or lan_ips()[0]
    q = f"?pin={pin}" if pin else ""
    return f"https://{ip}:{port or STATION_PORT}/phone{q}"


def qr_data_url(url: str) -> str:
    try:
        import qrcode
    except ImportError:
        return ""
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- certificates
def find_mkcert() -> str | None:
    for cand in (shutil.which("mkcert"), str(TMP / "bin" / "mkcert.exe")):
        if cand and Path(cand).is_file():
            return cand
    return None


def cert_names() -> list[str]:
    try:
        return list(json.loads(CERT_NAMES_FILE.read_text(encoding="utf-8")).get("names", []))
    except (OSError, json.JSONDecodeError):
        return []


def wanted_names() -> list[str]:
    host = socket.gethostname()
    names = ["localhost", "127.0.0.1", host, f"{host}.local"] + all_ipv4() + HOTSPOT_IPS
    return list(dict.fromkeys(n for n in names if n))


def ensure_certs(force: bool = False) -> tuple[Path, Path, list[str]]:
    """Re-issue the leaf cert if any current IP is missing from it.

    The root CA (already trusted on the iPhone) is never changed, so the phone
    needs nothing new. Returns (cert, key, names_in_cert).
    """
    STATION_CERT_DIR.mkdir(parents=True, exist_ok=True)
    have = set(cert_names())
    want = wanted_names()
    ok_files = STATION_CERT_FILE.is_file() and STATION_KEY_FILE.is_file()
    if ok_files and not force and have and set(want) <= have:
        return STATION_CERT_FILE, STATION_KEY_FILE, sorted(have)
    names = list(dict.fromkeys(want + sorted(have)))  # keep old IPs too (home Wi-Fi etc.)
    mk = find_mkcert()
    if mk is None:
        if ok_files:
            print("WARNING: mkcert not found; keeping the old certificate "
                  f"(names: {sorted(have) or 'unknown'})", flush=True)
            return STATION_CERT_FILE, STATION_KEY_FILE, sorted(have)
        raise SystemExit(
            "mkcert not found. Put mkcert.exe in C:/awc_tmp/bin/ "
            "(https://github.com/FiloSottile/mkcert/releases) and re-run."
        )
    env = dict(os.environ)
    cmd = [mk, "-cert-file", str(STATION_CERT_FILE), "-key-file", str(STATION_KEY_FILE), *names]
    subprocess.run(cmd, check=True, cwd=str(STATION_CERT_DIR), env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    CERT_NAMES_FILE.write_text(json.dumps({"names": names}, indent=2), encoding="utf-8")
    # Keep a copy of the root CA next to the certs (the file to AirDrop to a new phone).
    try:
        caroot = subprocess.run([mk, "-CAROOT"], capture_output=True, text=True, check=True).stdout.strip()
        src = Path(caroot) / "rootCA.pem"
        if src.is_file():
            shutil.copyfile(src, STATION_CERT_DIR / "rootCA.pem")
    except Exception:  # noqa: BLE001
        pass
    print(f"Certificate re-issued for {len(names)} names (same root CA, nothing to do on the iPhone).",
          flush=True)
    return STATION_CERT_FILE, STATION_KEY_FILE, names


# ---------------------------------------------------------------- firewall
def firewall_rule_cmd(port: int = STATION_PORT) -> str:
    return (
        f'netsh advfirewall firewall add rule name="AWC Gym Station {port}" '
        f"dir=in action=allow protocol=TCP localport={port} profile=any"
    )


def port_listening(port: int = STATION_PORT, host="127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False
