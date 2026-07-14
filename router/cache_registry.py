from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Entry:
    backend_id: str
    expires_at: float


class AffinityRegistry:
    def __init__(self, ttl_s: float = 3600, clock=time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.clock = clock
        self.prefixes: dict[str, Entry] = {}
        self.sessions: dict[str, Entry] = {}

    def _get(self, mapping: dict[str, Entry], key: str) -> str | None:
        entry = mapping.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self.clock():
            mapping.pop(key, None)
            return None
        return entry.backend_id

    def get_prefix(self, key: str) -> str | None:
        return self._get(self.prefixes, key)

    def get_session(self, key: str) -> str | None:
        return self._get(self.sessions, key)

    def commit(self, prefix_hash: str, session_id: str, backend_id: str) -> None:
        expires_at = self.clock() + self.ttl_s
        if prefix_hash:
            self.prefixes[prefix_hash] = Entry(backend_id, expires_at)
        if session_id:
            self.sessions[session_id] = Entry(backend_id, expires_at)

    def clear(self) -> None:
        self.prefixes.clear()
        self.sessions.clear()
