"""Run: python -m awc.station

Serves HTTPS on 0.0.0.0:8443. Certs from C:/awc_tmp/station_certs/ (mkcert).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path

from awc.config import STATION_CERT_DIR, STATION_CERT_FILE, STATION_KEY_FILE, STATION_PORT
from awc.station.netutil import firewall_rule_cmd, lan_ips, phone_url


def ensure_certs(extra_ips=None):
    STATION_CERT_DIR.mkdir(parents=True, exist_ok=True)
    if STATION_CERT_FILE.is_file() and STATION_KEY_FILE.is_file():
        return STATION_CERT_FILE, STATION_KEY_FILE
    ips = list(dict.fromkeys((extra_ips or []) + lan_ips() + ["localhost", "127.0.0.1"]))
    print("Generating mkcert certificates for:", ", ".join(ips), flush=True)
    try:
        subprocess.run(["mkcert", "-install"], check=False)
        cmd = ["mkcert", "-cert-file", str(STATION_CERT_FILE), "-key-file", str(STATION_KEY_FILE), *ips]
        subprocess.run(cmd, check=True, cwd=str(STATION_CERT_DIR))
    except FileNotFoundError:
        print(
            "mkcert not found. Install from https://github.com/FiloSottile/mkcert "
            "then re-run, or place station.pem + station-key.pem in "
            f"{STATION_CERT_DIR}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return STATION_CERT_FILE, STATION_KEY_FILE


def main(argv=None):
    p = argparse.ArgumentParser(description="AWC gym station (iPhone → laptop)")
    p.add_argument("--port", type=int, default=STATION_PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--http", action="store_true", help="Dev only: plain HTTP (iOS camera will refuse)")
    args = p.parse_args(argv)

    print("Firewall (run once as Admin if needed):", flush=True)
    print(" ", firewall_rule_cmd(args.port), flush=True)
    ips = lan_ips()
    print("LAN IPs:", ", ".join(ips), flush=True)
    print("Phone URL:", phone_url(ips[0] if ips else "127.0.0.1"), flush=True)

    import uvicorn
    from awc.station.app import create_app

    app = create_app()
    kwargs = {"host": "0.0.0.0", "port": args.port, "log_level": "info"}
    if not args.http:
        cert, key = ensure_certs()
        kwargs["ssl_certfile"] = str(cert)
        kwargs["ssl_keyfile"] = str(key)
    if not args.no_browser:
        url = f"{'http' if args.http else 'https'}://localhost:{args.port}/"
        webbrowser.open(url)
    uvicorn.run(app, **kwargs)


if __name__ == "__main__":
    main()
