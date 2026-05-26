"""Per-mpv-process singleton holding the expensive parts of the dual
chain — host_mp (with its 3 child python subprocesses + cuDNN/TRT
engines), queue_mgr, guest_mp (with its RDMA QP + worker connection) —
so they survive mpv's vapoursynth vf-rebuild on seek.

Why this works: mpv keeps one Python interpreter alive for the whole
mpv process and only re-execs the .vpy script on vf rebuild; the
module cache (sys.modules) keeps loaded modules between exec rounds,
so anything stored in this module's globals persists. Verified
experimentally — same id(sys.modules[__name__]) and same module-level
counter before and after a seek-triggered vf rebuild.

Lifecycle: tied to mpv's process lifetime (not systemd). When mpv
exits, the 3 child subprocesses get SIGHUP via the normal parent-exit
path; the rest is OS-reclaimed memory. No new resource-management
surface is added — we just stop pretending each rife.vpy invocation
is independent.

The singleton can be invalidated when params change (resolution
switch, mult cycle via F9) — call `reset()` after `shutdown_locked()`.
"""

import threading

# Single dict so callers can mutate via .update() without worrying
# about the module-attribute rebinding subtleties.
_state: dict = {}
_state_lock = threading.Lock()


def get() -> dict:
    """Return the singleton state dict. Caller acquires _state_lock if
    they're going to write."""
    return _state


def lock() -> threading.Lock:
    return _state_lock


def reset() -> None:
    """Drop all cached references. The actual host_mp/queue_mgr
    .shutdown() must be called by the caller BEFORE reset() — this
    function only clears the dict."""
    with _state_lock:
        _state.clear()
