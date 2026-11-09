"""Ordered chunk assembly for live MediaRecorder uploads (retries / out-of-order OK)."""
from __future__ import annotations

from pathlib import Path


class ChunkAssembler:
    """Write each seq exactly once, in ascending order, appending contiguous prefixes."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: dict[int, bytes] = {}
        self._next = 0
        self._acked: set[int] = set()
        if self.path.exists():
            self.path.unlink()
        self.path.touch()

    def add(self, seq: int, data: bytes) -> list[int]:
        """Store chunk; append newly contiguous bytes. Returns newly acked seqs."""
        seq = int(seq)
        if seq in self._acked or seq in self._buf:
            return []  # exact-once: ignore duplicates
        self._buf[seq] = bytes(data)
        newly = []
        with self.path.open("ab") as f:
            while self._next in self._buf:
                blob = self._buf.pop(self._next)
                f.write(blob)
                self._acked.add(self._next)
                newly.append(self._next)
                self._next += 1
        return newly

    @property
    def next_expected(self) -> int:
        return self._next

    @property
    def pending(self) -> int:
        return len(self._buf)

    @property
    def received_count(self) -> int:
        return len(self._acked) + len(self._buf)

    def is_complete(self, n_chunks: int) -> bool:
        return self._next >= int(n_chunks) and not self._buf
