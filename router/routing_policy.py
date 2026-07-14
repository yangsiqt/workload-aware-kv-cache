from __future__ import annotations

import random
from dataclasses import dataclass

from router.cache_registry import AffinityRegistry


@dataclass
class Backend:
    id: str
    url: str
    active_requests: int = 0
    healthy: bool = True


@dataclass(frozen=True)
class Decision:
    backend_id: str
    reason: str


class RouterPolicy:
    def __init__(self, backends: list[Backend], registry: AffinityRegistry, seed: int = 42) -> None:
        if not backends:
            raise ValueError("At least one backend is required")
        self.backends = {backend.id: backend for backend in backends}
        self.registry = registry
        self.random = random.Random(seed)

    def _healthy(self) -> list[Backend]:
        healthy = [backend for backend in self.backends.values() if backend.healthy]
        if not healthy:
            raise RuntimeError("No healthy backends")
        return healthy

    def _affinity(self, backend_id: str | None, reason: str) -> Decision | None:
        backend = self.backends.get(backend_id or "")
        if backend and backend.healthy:
            return Decision(backend.id, reason)
        return None

    def choose(self, policy: str, prefix_hash: str, session_id: str) -> Decision:
        healthy = self._healthy()
        if policy == "random":
            backend = self.random.choice(healthy)
            return Decision(backend.id, "random")
        if policy == "prefix_affinity":
            hit = self._affinity(self.registry.get_prefix(prefix_hash), "prefix_hit")
            if hit:
                return hit
        elif policy == "session_affinity":
            hit = self._affinity(self.registry.get_session(session_id), "session_hit")
            if hit:
                return hit
        else:
            raise ValueError(f"Unknown routing policy: {policy}")
        backend = min(healthy, key=lambda value: (value.active_requests, value.id))
        return Decision(backend.id, "least_active_fallback")
