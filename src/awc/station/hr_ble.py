"""Galaxy Watch / Heart-for-Bluetooth via bleak (Heart Rate Service 0x180D)."""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Callable

from awc.station.hr_parse import parse_hr_measurement

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"


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
        self._thread = threading.Thread(target=self._run, daemon=True)
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
            self.on_status("reconnect", "retry in 3 s")
            self._stop.wait(3.0)

    async def _session(self):
        from bleak import BleakClient, BleakScanner

        address = self.address
        if not address:
            self.on_status("scan", "looking for Heart Rate Service…")
            devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
            for dev, adv in devices.values():
                uuids = [u.lower() for u in (adv.service_uuids or [])]
                if HR_SERVICE in uuids or any("180d" in u for u in uuids):
                    if self.name and self.name.lower() not in (dev.name or "").lower():
                        continue
                    address = dev.address
                    self.name = dev.name
                    break
            if not address:
                raise RuntimeError("no BLE heart-rate device found (is Heart for Bluetooth broadcasting?)")
        self.on_status("connecting", address)

        def handler(_sender, data: bytearray):
            try:
                bpm, rr = parse_hr_measurement(data)
            except ValueError:
                return
            self.on_sample(bpm, rr, time.time() * 1000.0)

        async with BleakClient(address, timeout=20.0) as client:
            self.on_status("connected", self.name or address)
            await client.start_notify(HR_MEASUREMENT, handler)
            while not self._stop.is_set() and client.is_connected:
                await asyncio.sleep(0.5)
            self.on_status("disconnected", address)


def scan_hr_devices(timeout=8.0) -> list[dict]:
    async def _scan():
        from bleak import BleakScanner

        found = []
        devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
        for dev, adv in devices.values():
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            if HR_SERVICE in uuids or any("180d" in u for u in uuids):
                found.append({"address": dev.address, "name": dev.name, "rssi": getattr(adv, "rssi", None)})
        return found

    return asyncio.run(_scan())
