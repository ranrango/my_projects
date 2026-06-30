import time
from collections import defaultdict


class AccessController:
    """Decides grant/deny and manages gate open state and per-identity cooldowns."""

    GRANT = "GRANT"
    DENY = "DENY"

    def __init__(
        self,
        allow_unknown: bool = False,
        gate_open_duration: float = 3.0,
        cooldown_sec: float = 5.0,
        unknown_label: str = "UNKNOWN",
    ):
        self._allow_unknown = allow_unknown
        self._gate_open_until: float = 0.0
        self._cooldown_sec = cooldown_sec
        self._last_grant: dict[str, float] = {}
        self._unknown_label = unknown_label

    @property
    def gate_is_open(self) -> bool:
        return time.time() < self._gate_open_until

    def decide(self, identity: str, score: float) -> str:
        if identity == self._unknown_label and not self._allow_unknown:
            return self.DENY
        last = self._last_grant.get(identity, 0.0)
        if time.time() - last < self._cooldown_sec:
            return self.DENY
        return self.GRANT

    def open_gate(self, identity: str, gate_open_duration: float):
        self._gate_open_until = time.time() + gate_open_duration
        self._last_grant[identity] = time.time()
