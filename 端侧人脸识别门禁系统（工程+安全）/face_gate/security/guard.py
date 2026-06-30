import time
from collections import defaultdict


class SecurityGuard:
    """Tracks failed access attempts per identity/IP and enforces lockout."""

    def __init__(self, max_failed: int = 5, lockout_sec: float = 60.0):
        self._max_failed = max_failed
        self._lockout_sec = lockout_sec
        self._fail_counts: dict[str, int] = defaultdict(int)
        self._lockout_until: dict[str, float] = {}

    def is_locked(self, key: str) -> bool:
        until = self._lockout_until.get(key)
        if until is None:
            return False
        if time.time() < until:
            return True
        # Lock expired — reset
        del self._lockout_until[key]
        self._fail_counts[key] = 0
        return False

    def record_failure(self, key: str) -> bool:
        """Record one failed attempt. Returns True if this triggered a lockout."""
        self._fail_counts[key] += 1
        if self._fail_counts[key] >= self._max_failed:
            self._lockout_until[key] = time.time() + self._lockout_sec
            self._fail_counts[key] = 0
            return True
        return False

    def record_success(self, key: str):
        self._fail_counts.pop(key, None)
        self._lockout_until.pop(key, None)

    def lockout_remaining(self, key: str) -> float:
        until = self._lockout_until.get(key, 0.0)
        return max(0.0, until - time.time())
