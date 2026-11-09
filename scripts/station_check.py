"""Pre-gym device check for the iPhone → laptop station.

    python scripts/station_check.py
    python scripts/station_check.py --skip-phone --skip-hr
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from awc.config import STATION_CERT_FILE, STATION_KEY_FILE, STATION_PORT  # noqa: E402
from awc.station.netutil import firewall_rule_cmd, lan_ips, phone_url, qr_data_url  # noqa: E402


def check(name, ok, fix=""):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok and fix:
        print(f"       fix: {fix}")
    return bool(ok)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--skip-phone", action="store_true")
    p.add_argument("--skip-hr", action="store_true")
    p.add_argument("--pulsoid-token", default="")
    p.add_argument("--wait-phone-s", type=float, default=60)
    args = p.parse_args(argv)

    results = []
    ips = lan_ips()
    url = phone_url(ips[0] if ips else "127.0.0.1")
    print("LAN IPs:", ", ".join(ips))
    print("Phone URL:", url)
    qr = qr_data_url(url)
    print("QR data URL length:", len(qr), "(open the control page for the image)")
    print("Firewall command:\n ", firewall_rule_cmd())

    results.append(check(
        "mkcert cert files",
        STATION_CERT_FILE.is_file() and STATION_KEY_FILE.is_file(),
        f"run: mkcert -install && mkcert -cert-file {STATION_CERT_FILE} -key-file {STATION_KEY_FILE} "
        f"{' '.join(ips + ['localhost', '127.0.0.1'])}",
    ))

    sock_ok = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", STATION_PORT))
            sock_ok = True
    except OSError as exc:
        # port in use by station already counts as "reachable path exists"
        sock_ok = "10048" in str(exc) or "Address already in use" in str(exc)
    results.append(check(
        f"port {STATION_PORT} bindable or already serving",
        sock_ok,
        firewall_rule_cmd() + " · then python -m awc.station",
    ))

    if not args.skip_phone:
        print(f"Waiting up to {args.wait_phone_s:.0f}s for iPhone on /api/status (start the station first)…")
        print("  On phone: open", url, "→ Connect camera")
        deadline = time.time() + args.wait_phone_s
        phone_ok = False
        rtt = None
        while time.time() < deadline:
            try:
                import json
                import ssl
                from urllib.request import urlopen

                ctx = ssl._create_unverified_context()
                with urlopen(f"https://127.0.0.1:{STATION_PORT}/api/status", context=ctx, timeout=2) as resp:
                    st = json.loads(resp.read().decode())
                if st.get("phone_connected"):
                    phone_ok = True
                    rtt = st.get("clock_rtt_ms")
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)
        results.append(check("iPhone WebSocket connected", phone_ok, "trust mkcert CA on iPhone; same Wi-Fi or Personal Hotspot"))
        if phone_ok:
            results.append(check(
                f"clock RTT < 50 ms (got {rtt})",
                rtt is not None and rtt < 50,
                "move closer to the AP / use Personal Hotspot",
            ))
            # trigger a short set if session exists — otherwise just note
            print("  Tip: Start session → Start set for 5 s → End set, then check sets/setNN.mp4 decodes.")
    else:
        print("[SKIP] phone")

    if not args.skip_hr:
        if args.pulsoid_token:
            try:
                from awc.streaming.hr_sources import iter_pulsoid
                bpm = next(iter_pulsoid(args.pulsoid_token))
                results.append(check(f"Pulsoid bpm={bpm}", 30 <= float(bpm) <= 230, "check token / internet"))
            except Exception as exc:  # noqa: BLE001
                results.append(check("Pulsoid", False, str(exc)))
        else:
            try:
                from awc.station.hr_ble import scan_hr_devices
                found = scan_hr_devices(timeout=8.0)
                results.append(check(
                    f"BLE Heart Rate devices: {found}",
                    bool(found),
                    "open Heart for Bluetooth on the Galaxy Watch and start broadcasting",
                ))
                if found:
                    # read 30 s
                    from awc.station.hr_ble import BleHR
                    samples = []

                    def on_sample(bpm, rr, ts):
                        samples.append(bpm)

                    ble = BleHR(on_sample, lambda *a: None)
                    ble.start(address=found[0]["address"])
                    time.sleep(30)
                    ble.stop()
                    results.append(check(
                        f"BLE HR samples in 30 s: {len(samples)} (last={samples[-1] if samples else None})",
                        len(samples) >= 5,
                        "keep watch screen on; stay within a few metres",
                    ))
            except Exception as exc:  # noqa: BLE001
                results.append(check("BLE HR", False, f"{exc} · pip install bleak"))
    else:
        print("[SKIP] HR")

    print()
    ok = all(results)
    print("SUMMARY:", "ALL OK" if ok else "SOME FAILED — see fixes above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
