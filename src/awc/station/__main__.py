"""Run: python -m awc.station

Serves HTTPS on 0.0.0.0:8443. The leaf certificate is re-issued automatically
for the laptop's current IPs (same mkcert root CA, so the iPhone needs nothing
new when the network changes). The phone needs the PIN shown here / in the QR.
"""
from __future__ import annotations

import argparse
import webbrowser

from awc.config import STATION_PORT
from awc.station.netutil import ensure_certs, firewall_rule_cmd, lan_ips, phone_url, station_pin


def main(argv=None):
    p = argparse.ArgumentParser(description="AWC gym station (iPhone → laptop)")
    p.add_argument("--port", type=int, default=STATION_PORT)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--pin", help="set a new 4+ digit phone PIN (kept in C:/awc_tmp/station.json)")
    p.add_argument("--renew-cert", action="store_true", help="force a new leaf certificate")
    p.add_argument("--http", action="store_true", help="Dev only: plain HTTP (iOS camera will refuse)")
    p.add_argument("--kafka", action="store_true",
                   help="Optional: publish watch.hr / coach.events if Docker Kafka is up")
    p.add_argument("--show-form", action="store_true",
                   help="Show Model 2 form scores (off by default: uncalibrated on our camera)")
    args = p.parse_args(argv)

    pin = station_pin(args.pin)
    print("Firewall (run once in an Admin PowerShell if the phone can't connect):", flush=True)
    print(" ", firewall_rule_cmd(args.port), flush=True)
    ips = lan_ips()
    print("Laptop IPs:", ", ".join(ips), flush=True)
    for ip in ips:
        print("Phone URL:", phone_url(ip, pin, args.port), flush=True)
    print("PIN:", pin, flush=True)

    import uvicorn

    from awc.station.app import create_app
    from awc.station.state import RUNTIME

    RUNTIME.kafka_enabled = bool(args.kafka)
    RUNTIME.show_form = bool(args.show_form)
    if args.kafka:
        print("Kafka: enabled (will log once and continue if the broker is down)", flush=True)
    from awc.models.m2_gym import load_model as load_form_model
    if load_form_model() is not None:
        print("Form model: Model 2 v2 (gym-trained) for squats, push-ups, curls, rows, shoulder press, "
              "triceps.", flush=True)
    elif not args.show_form:
        print("Form model: hidden (models/m2_gym.pickle missing). Pass --show-form to force the old one.",
              flush=True)
    app = create_app(pin=pin, port=args.port)
    kwargs = {"host": "0.0.0.0", "port": args.port, "log_level": "warning", "ws_max_size": 32 * 1024 * 1024}
    if not args.http:
        cert, key, names = ensure_certs(force=args.renew_cert)
        app.state.cert_names = names
        kwargs["ssl_certfile"] = str(cert)
        kwargs["ssl_keyfile"] = str(key)
    if not args.no_browser:
        url = f"{'http' if args.http else 'https'}://localhost:{args.port}/"
        webbrowser.open(url)
    uvicorn.run(app, **kwargs)


if __name__ == "__main__":
    main()
