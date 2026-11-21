"""Offline TTS via pyttsx3 in a background thread (never blocks inference)."""
from __future__ import annotations

import queue
import threading


class Voice:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._q = queue.Queue()
        self._thread = None
        if enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def say(self, text: str):
        if not self.enabled or not text:
            return
        self._q.put(text)

    def _loop(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            print(f"voice disabled: {exc}", flush=True)
            self.enabled = False
            return
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:  # noqa: BLE001
                print(f"voice error: {exc}", flush=True)

    def close(self):
        if self.enabled:
            self._q.put(None)
