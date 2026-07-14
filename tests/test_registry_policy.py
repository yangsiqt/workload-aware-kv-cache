from router.cache_registry import AffinityRegistry
from router.routing_policy import Backend, RouterPolicy


def test_registry_ttl_and_commit() -> None:
    now = [10.0]
    registry = AffinityRegistry(ttl_s=5, clock=lambda: now[0])
    registry.commit("prefix", "session", "a")
    assert registry.get_prefix("prefix") == "a"
    assert registry.get_session("session") == "a"
    now[0] = 15.0
    assert registry.get_prefix("prefix") is None


def test_affinity_and_least_active() -> None:
    registry = AffinityRegistry()
    a, b = Backend("a", "http://a", active_requests=2), Backend("b", "http://b")
    policy = RouterPolicy([a, b], registry)
    assert policy.choose("prefix_affinity", "new", "s").backend_id == "b"
    registry.commit("known", "session", "a")
    assert policy.choose("prefix_affinity", "known", "other").reason == "prefix_hit"
    assert policy.choose("session_affinity", "other", "session").backend_id == "a"


def test_unhealthy_affinity_falls_back() -> None:
    registry = AffinityRegistry()
    a, b = Backend("a", "http://a"), Backend("b", "http://b")
    registry.commit("prefix", "session", "a")
    a.healthy = False
    decision = RouterPolicy([a, b], registry).choose("prefix_affinity", "prefix", "session")
    assert decision.backend_id == "b"
    assert decision.reason == "least_active_fallback"
