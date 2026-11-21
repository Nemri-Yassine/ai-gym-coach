"""Galaxy Watch / Heart-for-Bluetooth via bleak (Heart Rate Service 0x180D)."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Callable

from awc.config import TMP
from awc.station.hr_parse import parse_hr_measurement

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
LAST_DEVICE_PATH = TMP / "station_hr_device.json"


def load_last_device() -> dict | None:
    if not LAST_DEVICE_PATH.is_file():
        return None
    try:
        return json.loads(LAST_DEVICE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_last_device(address: str, name: str | None = None):
    TMP.mkdir(parents=True, exist_ok=True)
    LAST_DEVICE_PATH.write_text(
        json.dumps({"address": address, "name": name, "saved_at": time.time()}, indent=2),
        encoding="utf-8",
    )


def _is_hr_adv(adv) -> bool:
    uuids = [u.lower() for u in (getattr(adv, "service_uuids", None) or [])]
    return HR_SERVICE in uuids or any("180d" in u for u in uuids)


class BleHR:
    def __init__(self, on_sample: Callable[[float, list[float], float], None], on_status=None):
        self.on_sample = on_sample
        self.on_status = on_status or (lambda *_: None)
        self._stop = threading.Event()
        self._thread = None
        self.address = None
        self.name = None

    def start(self, address: str | None = None, name_substr: str | None = None):
        self.address = address
        self.name = name_substr
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="awc-ble-hr")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                asyncio.run(self._session())
            except Exception as exc:  # noqa: BLE001
                self.on_status("error", str(exc))
            if self._stop.is_set():
                break
            self.on_status("reconnect", "reconnecting - keep Heart for Bluetooth open on the watch")
            self._stop.wait(0.5)

    async def _resolve_address(self) -> str:
        from bleak import BleakScanner

        if self.address:
            return self.address

        last = load_last_device()
        if last and last.get("address"):
            self.on_status("connecting", f"last watch {last.get('name') or last['address']}")
            self.address = last["address"]
            self.name = last.get("name") or self.name
            return self.address

        self.on_status("scan", "scanning for Heart Rate (stops when found)…")

        def match(device, adv):
            if not _is_hr_adv(adv):
                return False
            if self.name and self.name.lower() not in (device.name or "").lower():
                return False
            return True

        # Returns as soon as a matching advert is seen (much faster than a fixed 8 s scan).
        device = await BleakScanner.find_device_by_filter(match, timeout=6.0)
        if device is None:
            raise RuntimeError(
                "no BLE heart-rate device — open Heart for Bluetooth on the watch and keep the screen on"
            )
        self.address = device.address
        self.name = device.name
        save_last_device(device.address, device.name)
        return device.address

    async def _session(self):
        from bleak import BleakClient

        address = await self._resolve_address()
        self.on_status("connecting", f"{self.name or address}")

        def handler(_sender, data: bytearray):
            try:
                bpm, rr = parse_hr_measurement(data)
            except ValueError:
                return
            self.on_sample(bpm, rr, time.time() * 1000.0)

        gone = asyncio.Event()
        # services=[HR] makes Windows discover only the HR service: much faster (re)connects.
        # use_cached_services skips a full GATT re-read on each reconnect.
        async with BleakClient(
            address,
            disconnected_callback=lambda _c: gone.set(),
            services=[HR_SERVICE],
            timeout=8.0,
            winrt={"use_cached_services": True},
        ) as client:
            save_last_device(address, self.name)
            self.on_status("connected", self.name or address)
            await client.start_notify(HR_MEASUREMENT, handler)
            while not self._stop.is_set() and not gone.is_set():
                try:
                    await asyncio.wait_for(gone.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
            if not self._stop.is_set():
                self.on_status("disconnected", address)


def scan_hr_devices(timeout=5.0) -> list[dict]:
    async def _scan():
        from bleak import BleakScanner

        found = {}
        devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
        for dev, adv in devices.values():
            if not _is_hr_adv(adv):
                continue
            found[dev.address] = {
                "address": dev.address,
                "name": dev.name,
                "rssi": getattr(adv, "rssi", None),
            }
        return list(found.values())

    return asyncio.run(_scan())
