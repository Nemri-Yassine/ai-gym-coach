"""Mutable runtime state for the gym station process."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from awc.station.clock import ClockSync
from awc.station.session_io import StationSession


@dataclass
class Runtime:
    lock: threading.RLock = field(default_factory=threading.RLock)
    cmd_lock: threading.Lock = field(default_factory=threading.Lock)
    clock: ClockSync = field(default_factory=ClockSync)
    session: StationSession | None = None
    phone_connected: bool = False
    phone_ua: str = ""
    phone_battery: float | None = None
    phone_info: dict = field(default_factory=dict)  # resolution, fps, mime
    preview_jpeg: bytes | None = None
    preview_ts: float = 0.0
    hr_bpm: float | None = None
    hr_ts: float = 0.0
    hr_source: str = "none"
    hr_status: str = "idle"
    hr_history: list = field(default_factory=list)  # (pc_ts_ms, bpm)
    hr_drops: int = 0
    recording: bool = False
    current_set: int | None = None
    rec_confirmed: bool = False  # phone said its MediaRecorder started
    upload: dict = field(default_factory=dict)  # set → {received, acked, pending, done}
    pending_labels: dict | None = None  # after set_done, waiting for Save set
    analyses: dict = field(default_factory=dict)  # set_index → analysis JSON (suggestion only)
    command_q: list = field(default_factory=list)  # commands for phone WS
    last_rtt_ms: float | None = None
    last_offset_ms: float | None = None
    kafka_enabled: bool = False
    kafka_warned: bool = False
    show_form: bool = False  # M2 uncalibrated on own camera; opt-in via --show-form

    def push_cmd(self, msg: dict):
        """Append a phone command. Safe to call while already holding self.lock."""
        with self.cmd_lock:
            self.command_q.append(msg)

    def pop_cmds(self) -> list:
        with self.cmd_lock:
            out = list(self.command_q)
            self.command_q.clear()
            return out

    def note_hr(self, bpm: float, pc_ts_ms: float, rr=None):
        with self.lock:
            self.hr_bpm = float(bpm)
            self.hr_ts = float(pc_ts_ms)
            self.hr_history.append((pc_ts_ms, bpm))
            cutoff = pc_ts_ms - 180_000
            self.hr_history = [h for h in self.hr_history if h[0] >= cutoff]
            sess = self.session
        if sess is not None:
            sess.append_hr(pc_ts_ms, bpm, rr_ms=rr)

    def snapshot(self) -> dict:
        with self.lock:
            offset, rtt = self.clock.estimate()
            hr_age = (time.time() * 1000 - self.hr_ts) / 1000 if self.hr_ts else None
            return {
                "phone_connected": self.phone_connected,
                "phone_ua": self.phone_ua,
                "phone_battery": self.phone_battery,
                "phone_info": dict(self.phone_info),
                "recording": self.recording,
                "rec_confirmed": self.rec_confirmed,
                "current_set": self.current_set,
                "upload": {str(k): v for k, v in self.upload.items()},
                "hr_bpm": self.hr_bpm,
                "hr_age_s": hr_age,
                "hr_source": self.hr_source,
                "hr_status": self.hr_status,
                "hr_drops": self.hr_drops,
                "hr_series": list(self.hr_history[-180:]),
                "clock_offset_ms": offset,
                "clock_rtt_ms": rtt,
                "pending_labels": self.pending_labels,
                "analyses": {str(k): v for k, v in self.analyses.items()},
                "session": None if self.session is None else str(self.session.folder),
                "next_set": None if self.session is None else self.session.next_set(),
                "preview_fresh_ms": (time.time() * 1000 - self.preview_ts) if self.preview_ts else None,
            }


RUNTIME = Runtime()
