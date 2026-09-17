from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading


class RotatingJsonlWriter:
    """Line-buffered JSONL writer with bounded, recoverable size rotation."""

    def __init__(self, path: Path, max_bytes: int = 100 * 1024 * 1024,
                 backups: int = 48) -> None:
        self.path = path
        self.max_bytes = max(1024, max_bytes)
        self.backups = max(1, backups)
        self._file = None
        self._lock = threading.Lock()

    def __enter__(self) -> "RotatingJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._repair_partial_tail()
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        return self

    def _repair_partial_tail(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        with self.path.open("rb+") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                return
            stream.seek(-1, os.SEEK_END)
            if stream.read(1) == b"\n":
                return
            position = stream.tell() - 1
            while position > 0:
                position -= 1
                stream.seek(position)
                if stream.read(1) == b"\n":
                    stream.truncate(position + 1)
                    return
            stream.seek(0)
            stream.truncate()

    def write(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            if self._file is None:
                raise RuntimeError("writer is not open")
            if self._file.tell() + len(line.encode("utf-8")) > self.max_bytes:
                self._rotate()
            self._file.write(line)

    def _rotate(self) -> None:
        assert self._file is not None
        self._file.close()
        oldest = self.path.with_suffix(self.path.suffix + f".{self.backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backups - 1, 0, -1):
            source = self.path.with_suffix(self.path.suffix + f".{index}")
            if source.exists():
                source.replace(self.path.with_suffix(self.path.suffix + f".{index + 1}"))
        if self.path.exists():
            self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        self._file = self.path.open("a", encoding="utf-8", buffering=1)

    def __exit__(self, *_args) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class RuntimeStateStore:
    """Atomic, non-secret runtime checkpoints used for restart diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def save(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


@dataclass
class RuntimeHealth:
    received_events: int = 0
    accepted_events: int = 0
    duplicate_or_old_events: int = 0
    dropped_queue_events: int = 0
    reconnects: int = 0
    last_event_at: str | None = None
    last_tick_latency_ms: float = 0.0
    max_tick_latency_ms: float = 0.0

    def received(self, accepted: bool) -> None:
        self.received_events += 1
        if accepted:
            self.accepted_events += 1
            self.last_event_at = datetime.now(timezone.utc).isoformat()
        else:
            self.duplicate_or_old_events += 1

    def tick(self, latency_ms: float) -> None:
        self.last_tick_latency_ms = latency_ms
        self.max_tick_latency_ms = max(self.max_tick_latency_ms, latency_ms)

    def snapshot(self) -> dict:
        return asdict(self)
