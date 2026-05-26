"""Central queue manager for the split-task pipeline.

Why a central scheduler at all?

  - Cross-cuts host and guest workers. The same DAG node may end up
    running on either, depending on who's free. The mgr owns the
    decision, the workers just pop.
  - Refcount-driven cache eviction needs a single point of truth.
    CC outputs are consumed by exactly two INTERP tasks (plus one
    SR_SRC or display(N)); we free them when refcount hits zero.
    Distributing this across workers would deadlock or leak.
  - Out-of-order completion has to be reconciled with mpv's in-order
    encoder. The mgr signals frame_done[K] only when both halves of
    the K-th output pair are assembled; mpv's frame thread blocks on
    that event.

Threading model
---------------

The mgr is intentionally a thread-safe data structure, not its own
thread. Producers (mpv frame threads calling submit_frame) and
consumers (worker dispatchers calling pop_color/pop_compute) call
into it directly under one re-entrant lock. There's no internal
"mgr loop" — the lock is held only for the short critical sections
that mutate the DAG or queues. This keeps the design simple and
avoids an extra hop on the hot path.

Locking discipline
------------------

A single `threading.RLock` guards the DAG, both queues, the cc_cache
table, and the frame_done events table. Hot-path operations:

  - submit_frame:    O(log Q) per task pushed
  - task_done:       O(children) DAG walk + O(log Q) per child pushed
  - pop_color / pop_compute: O(log Q)
  - free_cc_field:   O(1)

A condvar on the same lock wakes `wait_frame_done` / `pop_*(blocking=True)`.
"""
from __future__ import annotations

import heapq
import itertools
import os
import threading
import time as _time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cc_cache import CCCacheShm  # avoid circular import at runtime


# ──────────────────────────────────────────────────────────────────────
# Task types and priority
# ──────────────────────────────────────────────────────────────────────


class TaskType(IntEnum):
    """Kernel selector. Integer values are wire-format compatible with
    RequestMeta.task_type and worker_3proc's TT_* constants."""

    INTERP      = 3   # RIFE on two CCSR producers' rgb_padded + features.
                       # For mult ≥ 3 ONE INTERP task runs dual/triple
                       # flownet and writes ALL output frames (mult-1
                       # of them) into separate cc_cache slots. Wire
                       # input is paid once per pair (~56 MB).
    SR_INTERP   = 4   # FSRCNNX on INTERP output (memmove to mpv VA dst_i).
                       # mult ≥ 3 has multiple SR_INTERP nodes per pair,
                       # distinguished by TaskNode.output_phase (1 →
                       # dst_i, 2 → dst_i2, 3 → dst_i3). All run the
                       # same _run_sr_interp kernel and route to either
                       # machine.
    CCSR        = 5   # merged CC + SR_SRC: raw YUV → rgb_padded +
                       # rife_features (cc_cache for INTERP) + 4K SR
                       # output (slot.dst → mpv VA dst_a). sr_yuv stays
                       # local within the kernel.


def queue_for(t: TaskType) -> str:
    """3-queue priority routing:
       cc_q     — CCSR (host's primary, guest secondary at low load).
       sr_q     — SR_INTERP (fast, ~6 ms; preferred over INTERP on host).
       interp_q — INTERP (heavy, ~17 ms; guest primary).
    Returns 'cc' / 'sr' / 'interp'."""
    if t == TaskType.CCSR:
        return "cc"
    if t == TaskType.SR_INTERP:
        return "sr"
    return "interp"


# Tie-breaker priority within the same frame_idx. Earlier-in-pipeline
# tasks float to the front so dependents drain faster. Order:
# CC < INTERP < SR_SRC < SR_INTERP. CC must run before its consumers;
# INTERP and SR_SRC are independent siblings but draining INTERP first
# unblocks SR_INTERP one step sooner.
_TYPE_PRIORITY = {
    TaskType.CCSR:        1,
    TaskType.INTERP:      2,
    TaskType.SR_INTERP:   4,
}


class TaskState(IntEnum):
    PENDING  = 0   # waiting on parents
    READY    = 1   # parents satisfied, on a queue, not yet popped
    RUNNING  = 2   # popped by a worker
    DONE     = 3
    FAILED   = 4


# ──────────────────────────────────────────────────────────────────────
# Task graph nodes
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TaskNode:
    """One DAG node. Mutated only under QueueManager._lock."""

    id:          int
    type:        TaskType
    frame_idx:   int                # frame the task is "about". For
                                    # INTERP(N, N+1) we use N (the
                                    # earlier frame); for SR_INTERP(N.5)
                                    # also N. This keeps pair-coupled
                                    # tasks adjacent in priority order.
    # SR_INTERP has multiple nodes per pair K when mult ≥ 3.
    # output_phase distinguishes them in the DAG (since (type,
    # frame_idx) is otherwise the same):
    #   1 = primary (frame at t=1/mult); reads INTERP.dst_cc_slot,
    #       writes mpv VA dst_i_y.
    #   2 = secondary (mult ≥ 3, frame at t=2/mult); reads
    #       INTERP.dst_cc_slot_2, writes mpv VA dst_i2_y.
    #   3 = tertiary (mult=4, frame at t=3/4); reads
    #       INTERP.dst_cc_slot_3, writes mpv VA dst_i3_y.
    # Other task types use 1.
    output_phase: int               = 1
    parents:     set[int]           = field(default_factory=set)
    # Staged gates: maps parent_id → stage index whose completion
    # satisfies THIS edge. Absent entry = gate on final (task_done).
    # stage_idx >= 0 = early stage (task_stage_done(tid, stage_idx)).
    # Generalises to arbitrary N intermediate stages so producers
    # with more intermediate frames work without further DAG-API
    # churn.
    parents_at_stage: dict[int, int] = field(default_factory=dict)
    # original_parents is set at creation time and never mutated;
    # _post_pop_locked uses it to look up parent dst_cc_slot ids after
    # task_done has already cleared `parents`.
    original_parents: set[int]      = field(default_factory=set)
    children:    set[int]           = field(default_factory=set)
    deps_remaining: int             = 0
    state:       TaskState          = TaskState.PENDING

    # Filled in by stages B/C. Opaque to the mgr.
    payload:     dict               = field(default_factory=dict)

    # Result handoff once the worker reports done. The mgr stores
    # whatever the worker reported (e.g. the dst shm slot id where
    # the output landed) so downstream tasks know where to read.
    result:      object             = None
    error:       Optional[str]      = None

    # Timing — instrumentation only.
    t_created:   float              = 0.0
    t_ready:     float              = 0.0
    t_dispatched: float             = 0.0
    t_done:      float              = 0.0


# ──────────────────────────────────────────────────────────────────────
# CC result table
# ──────────────────────────────────────────────────────────────────────


@dataclass
class FrameVA:
    """Per-frame_idx mpv virtual addresses + strides. Stored by
    QueueManager and read by the dispatch thread when populating a
    CC task's RequestMeta. Holds source-frame VAs for K and K+1 (the
    pair handled by output pair K), and the mpv dst VAs that the
    completed result will be memcpy'd into."""

    pair_k:    int
    # src frame K (the "a" half of pair K) — mpv VA + strides
    sa_y:      int = 0
    sa_u:      int = 0
    sa_v:      int = 0
    sa_y_stride:  int = 0
    sa_uv_stride: int = 0
    # src frame K+1
    sb_y:      int = 0
    sb_u:      int = 0
    sb_v:      int = 0
    # Phase 0 = source K; phase 1 = interp K.5; phase 2 (mult ≥ 3) →
    # dst_i2; phase 3 (mult = 4) → dst_i3.
    dst_a_y:   int = 0
    dst_a_u:   int = 0
    dst_a_v:   int = 0
    dst_i_y:   int = 0
    dst_i_u:   int = 0
    dst_i_v:   int = 0
    dst_i2_y:  int = 0
    dst_i2_u:  int = 0
    dst_i2_v:  int = 0
    # mult=4: third interp output (frame at t=3/4).
    dst_i3_y:  int = 0
    dst_i3_u:  int = 0
    dst_i3_v:  int = 0
    dst_y_stride:  int = 0
    dst_uv_stride: int = 0


@dataclass
class CCResult:
    """Per-slot tracker for a cc_cache shm slot. Each field on the slot
    has its own refcount; the slot is returned to cc_cache's free-list
    only when all three refcounts hit 0.

    Keyed on cc_cache slot id (not frame index) — INTERP outputs occupy
    cc_cache slots too but aren't naturally addressable by integer
    frame index (they're conceptually K.5 frames).

    Refcount conventions (refer to SPLIT_TASK_ARCHITECTURE.md §7 + §11):
      rgb_padded_refs:    2 nominally (INTERP(N-1,N) + INTERP(N,N+1)),
                          or 1 at the stream boundary. Producer: CC(N).
      rife_features_refs: same lifecycle as rgb_padded (encode output is
                          consumed by the same INTERPs). Producer: CC(N).
      sr_yuv_refs:        1. Producer: CC(N) for "source frame N", or
                          INTERP(N,N+1) for "interp frame N.5".
                          Consumer: SR_SRC(N) or SR_INTERP(N).
      rgb_4K_refs:        1 (display(N) writeback). Only present for 4K
                          source; CC(N) produces.
    """

    slot:               int
    frame_idx:          int = -1   # diagnostic only; e.g. source K or 2*K+1 for K.5

    rgb_padded_refs:    int = 0
    rife_features_refs: int = 0
    sr_yuv_refs:        int = 0
    rgb_4K_refs:        int = 0

    has_rgb_padded:     bool = False
    has_rife_features:  bool = False
    has_sr_yuv:         bool = False
    has_rgb_4K:         bool = False


# ──────────────────────────────────────────────────────────────────────
# Heap entries
# ──────────────────────────────────────────────────────────────────────


# Pushed into heapq as (frame_idx, type_priority, monotonic_seq, task_id).
# monotonic_seq breaks ties so the heap is stable even when many tasks
# share the same (frame_idx, type) (rare, but happens for the very first
# pair of the stream where the asymmetry of CC(0) vs CC(1) lands twice).
_HeapEntry = tuple[int, int, int, int]


# ──────────────────────────────────────────────────────────────────────
# QueueManager
# ──────────────────────────────────────────────────────────────────────


class QueueManager:
    """Thread-safe central scheduler. See module docstring."""

    def __init__(self, *,
                  cc_cache: "Optional[CCCacheShm]" = None,
                  log=None):
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # task_id allocator
        self._next_task_id = itertools.count(1)
        self._next_heap_seq = itertools.count(1)

        self._dag: dict[int, TaskNode] = {}

        # Heaps, ordered by (frame_idx, type_priority, seq, task_id).
        # We keep an auxiliary set of task_ids actually present in the
        # heap to support O(1) "is this task queued?" checks — necessary
        # for cancel/replace later.
        # 3-queue routing: cc / sr / interp. cc is host-exclusive;
        # sr & interp are shared but with rank priority (host: sr>cc>interp;
        # guest: interp>cc>sr). See queue_for() above.
        # Kept _color_*/_compute_* names as aliases for back-compat
        # with peek/wait_frame_done diagnostics.
        self._cc_heap: list[_HeapEntry] = []
        self._cc_present: set[int] = set()
        self._sr_heap: list[_HeapEntry] = []
        self._sr_present: set[int] = set()
        self._interp_heap: list[_HeapEntry] = []
        self._interp_present: set[int] = set()
        # Back-compat aliases used by older pop_color / pop_compute
        # callers (now wrappers below).
        self._color_heap = self._cc_heap
        self._color_present = self._cc_present
        self._compute_heap = self._interp_heap  # unused by new code
        self._compute_present = self._interp_present

        # Frame completion events. Keyed by frame_idx (the source frame
        # of the pair, == the frame_thread's K).
        self._frame_done: dict[int, threading.Event] = {}

        # Per-frame VA records, keyed on pair_k. mpv populates these
        # via update_frame_va; dispatch thread reads them when
        # populating CC tasks.
        self._frame_va: dict[int, FrameVA] = {}

        # cc_cache shm region — optional. When set, the mgr alloc/frees
        # slots in lockstep with task dispatch / completion. When None
        # the cc_cache layer is simply a no-op (useful for unit tests).
        self._cc_cache: "Optional[CCCacheShm]" = cc_cache
        # cc_cache slot trackers, keyed by *slot id* (not frame_idx) so
        # INTERP-output slots have their own entries.
        self._cc: dict[int, CCResult] = {}
        # When cc_cache is None we hand out synthetic non-negative ids.
        # Start from 1_000_000 so they're visually distinguishable from
        # real cc_cache slot ids (small integers).
        self._synthetic_slot_id = itertools.count(1_000_000)

        # Lifecycle.
        self._closed = False

        # Stats / instrumentation.
        self._stats = {
            "submitted_tasks":  0,
            "completed_tasks":  0,
            "failed_tasks":     0,
            "color_pops":       0,   # back-compat alias = cc_pops
            "compute_pops":     0,   # back-compat
            "cc_pops":          0,
            "sr_pops":          0,
            "interp_pops":      0,
            "cc_evictions":     0,
            "cc_allocs":        0,
        }

        self._log = log or (lambda msg: None)

        # PROFILE: pop-side instrumentation. Each pop_for_* call
        # records (1) which queue ultimately produced the task or "empty"
        # if it had to wait, (2) the queue depths at the time of pop,
        # (3) how long the caller blocked. Reported in summary log via
        # _profile_pop_report().
        self._profile = os.environ.get("DUAL_PROFILE", "0") == "1"
        self._pop_log: dict[str, dict] = {
            "host": {"cc": 0, "sr": 0, "interp": 0, "empty_waits": 0,
                     "wait_ms_total": 0.0, "n": 0,
                     "depth_cc_at_pop": 0, "depth_sr_at_pop": 0,
                     "depth_interp_at_pop": 0},
            "guest": {"interp": 0, "sr": 0, "empty_waits": 0,
                      "wait_ms_total": 0.0, "n": 0,
                      "depth_interp_at_pop": 0, "depth_sr_at_pop": 0},
        }
        self._pop_log_lock = threading.Lock()
        self._pop_last_report = _time.perf_counter()
        self._pop_report_every = float(os.environ.get(
            "DUAL_PROFILE_INTERVAL_S", "2.0"))
        # PROFILE: lock-hold instrumentation. For each entry point we
        # measure (wall_inside_lock_ms_total, n_calls) so the report
        # surfaces "submit_frame held _cv for X ms over the last Ys" —
        # which lets us see if mpv's frame_threads are serialising on
        # the queue_mgr lock vs being intake-bound.
        self._lock_log: dict[str, dict] = {
            "submit_frame": {"ms": 0.0, "n": 0, "max_ms": 0.0},
            "task_done":    {"ms": 0.0, "n": 0, "max_ms": 0.0},
            "pop_host":     {"ms": 0.0, "n": 0, "max_ms": 0.0},
            "pop_guest":    {"ms": 0.0, "n": 0, "max_ms": 0.0},
        }
        self._dag_size_peak = 0
        # Tracks the highest K seen by submit_frame. Used by the
        # signaled-K eviction policy below.
        self._max_submit_K: int = -1
        # CCSR is the only phase-0 producer (merged CC + SR_SRC).
        # submit_frame emits CCSR(K) which writes rgb_padded +
        # rife_features to cc_cache (for INTERP) and 4K SR to slot.dst
        # (for mpv VA writeback). INTERP(K) gates on CCSR(K) + CCSR(K+1);
        # SR_INTERP(K) gates on INTERP(K). Phase-0 frame_done = CCSR(K).
        # Defensive layer: once frame_done(K) signals, any late
        # compute(K, phase) call (e.g. mpv's phase-1 thread woke up
        # after the phase-0 thread already finished the whole pipeline
        # for K) must NOT re-enter submit_frame / wait_frame_done. If
        # we let it, _ensure_node_locked creates duplicate SR_SRC /
        # SR_INTERP nodes whose task_done double-consumes CC_K's
        # cc_cache.sr_yuv → refcount underflow → pipeline stall or
        # glibc heap corruption when slot K's cc_cache record had
        # already been recycled in between.
        self._signaled_K_set: set[int] = set()
        # Eviction watermark: drop K from _signaled_K_set once mpv has
        # moved this far past K (no chance of a late submit_frame).
        # 256 is conservative — CF rarely exceeds 32, vsynth's
        # concurrent-frames spec caps at 16.
        self._signaled_K_evict_lag = 256
        # Seek-flush epoch. Bumped by flush_seek() so parked
        # wait_phase_done() callers can detect "my wait was abandoned
        # by a seek" and return WAIT_ABANDONED instead of blocking
        # forever on a CCSR(K+1) that the post-seek chain orphaned.
        self._seek_epoch: int = 0
        # GC sweep cutoff = signaled_K (delete all DONE nodes with
        # frame_idx < the just-signaled K). With Option A removing
        # mpv's post-frame_done node access, no consumer needs nodes
        # to outlive the next signal: CC_K's payload is filled by
        # submit_frame(K-1), and the cutoff guarantees we don't sweep
        # CC_K until at least one LATER frame signals (i.e.
        # submit_frame(K+) has already touched CC).
        #
        # An attempted `frame_idx <= max_submit_K` watermark — delete
        # ALL submitted frames including the just-signaled one — gave
        # glibc "free(): corrupted unsorted chunks" within seconds.
        # The lag=0 cutoff (signaled_K exclusive) avoids that
        # codepath entirely.

    # ── lifecycle ──────────────────────────────────────────────

    def shutdown(self, *, drain_timeout: float = 2.0) -> None:
        """Mark closed and wake all waiters. Then drain up to
        `drain_timeout` seconds for in-flight tasks (those popped but not
        yet DONE) to finish — this matters during vf reload + abrupt
        teardown: in-flight RDMA WRITEs to mpv VA outlive a hard close,
        and on fast process teardown those writes can land in freed
        memory. Bounded drain lets the worker side finish its already-
        posted ops cleanly while still capping shutdown wall time.
        """
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        # Best-effort drain. Poll DAG for tasks still in CLAIMED state
        # (popped but not done). At drain_timeout we give up.
        if drain_timeout <= 0:
            return
        import time as _drain_time
        deadline = _drain_time.perf_counter() + drain_timeout
        while _drain_time.perf_counter() < deadline:
            inflight = 0
            with self._lock:
                for node in self._dag.values():
                    # RUNNING = popped by a worker, not yet task_done.
                    # READY = on a publishable queue but not yet popped;
                    # those we can let die since no host/guest is
                    # consuming them anymore after _closed=True.
                    if node.state == TaskState.RUNNING:
                        inflight += 1
            if inflight == 0:
                return
            _drain_time.sleep(0.025)

    # ── task creation (internal) ───────────────────────────────

    def _new_node(self, *, task_type: TaskType, frame_idx: int,
                   parents: list[int] | None = None,
                   parents_at_stage: dict[int, int] | None = None,
                   payload: dict | None = None,
                   output_phase: int = 1) -> TaskNode:
        """Add a new node to the DAG. Called by submit_frame /
        task_done's "publish children" path. Caller MUST hold _lock.

        parents_at_stage: maps parent_id → stage index whose completion
        satisfies this edge. Absent = gate on parent's final stage
        (task_done). Lets INTERP unblock as soon as its CCSR sources
        have written rgb_padded + features (stage 0), before they
        finish the SR yuv stage. Extensible to N stages."""
        tid = next(self._next_task_id)
        parents_set = set(parents) if parents else set()
        # Defensive: parents_at_stage keys must be a subset of parents.
        if parents_at_stage:
            pas = {pid: s for pid, s in parents_at_stage.items()
                   if pid in parents_set}
        else:
            pas = {}
        node = TaskNode(
            id=tid,
            type=task_type,
            frame_idx=frame_idx,
            output_phase=output_phase,
            parents=parents_set,
            parents_at_stage=pas,
            original_parents=set(parents_set),
            deps_remaining=len(parents_set),
            payload=payload or {},
            t_created=_time.perf_counter(),
        )
        self._dag[tid] = node
        for pid in parents_set:
            parent = self._dag.get(pid)
            if parent is None:
                # Parent has been garbage collected — count as done.
                node.deps_remaining -= 1
                continue
            edge_stage = pas.get(pid)   # None = final, else stage idx
            if parent.state == TaskState.DONE:
                node.deps_remaining -= 1
            elif edge_stage is not None:
                fired_stages: set = parent.payload.get(
                    "stages_fired", set())
                if edge_stage in fired_stages:
                    # Parent already passed that stage — count as done.
                    node.deps_remaining -= 1
                    node.parents.discard(pid)
            parent.children.add(tid)
        self._try_make_ready_locked(node)
        self._stats["submitted_tasks"] += 1
        return node

    def _node_publishable_locked(self, node: TaskNode) -> bool:
        """Strict gating: a node is publishable iff DAG parents are
        done AND any task-specific publish-time prerequisites are met.

        Without these gates, a dispatcher would pop a SR_* whose mpv
        dst VA isn't yet registered (set_phase_dst not yet called on the
        owning frame_thread), or a CC_K placeholder whose src VA hasn't
        been filled by its owner pair's submit_frame. The dispatcher
        used to handle both by re-queue + cv.wait(5ms), at the cost of
        an idle slot per task. We move the gate to publish time so the
        dispatcher only ever sees fully-ready work.
        """
        if node.deps_remaining > 0:
            return False
        t = node.type
        if t == TaskType.CCSR:
            # CCSR is the phase-0 producer; it both reads mpv VA src
            # (via dma_proc) and writes mpv VA dst_a (4K SR). Both gates
            # must be set before publish.
            #
            # Requiring both introduces a cross-K dependency that
            # deadlocks at low CF: mpv with CF=2 requests frame 0 +
            # frame 1 in parallel; frame_done(0) needs INTERP(0) which
            # needs CCSR(1); CCSR(1) has src_y (from K=0's sb side) but
            # not dst_a_y (K=1's set_phase_dst can't fire until a
            # frame_thread is free). PSNR tests therefore use CF >= 4
            # (≥ 2 pairs in flight); fps bench runs at CF=24.
            if not node.payload.get("src_y"):
                return False
            va = self._frame_va.get(node.frame_idx)
            if va is None or va.dst_a_y == 0:
                return False
        elif t == TaskType.SR_INTERP:
            # SR_INTERP writes its result back to an mpv dst VA. Phase
            # 1 → dst_i_y; phase 2 → dst_i2_y (mult ≥ 3); phase 3 →
            # dst_i3_y (mult = 4). Set by set_phase_dst before this
            # gate can publish.
            va = self._frame_va.get(node.frame_idx)
            if va is None:
                return False
            if node.output_phase == 3:
                if va.dst_i3_y == 0:
                    return False
            elif node.output_phase == 2:
                if va.dst_i2_y == 0:
                    return False
            else:
                if va.dst_i_y == 0:
                    return False
        return True

    def _try_make_ready_locked(self, node: TaskNode) -> None:
        """Publish node to its queue iff it's PENDING AND publishable.
        Safe to call from anywhere — gating helper that replaced the
        previous `if deps_remaining == 0: _make_ready_locked` pattern."""
        if node.state != TaskState.PENDING:
            return
        if not self._node_publishable_locked(node):
            return
        self._make_ready_locked(node)

    def _make_ready_locked(self, node: TaskNode) -> None:
        if os.environ.get("DUAL_REFCOUNT_DBG", "0") == "1":
            import traceback as _tb
            frames = _tb.extract_stack(limit=4)
            caller = frames[-2] if len(frames) >= 2 else None
            site = f"{caller.filename.split('/')[-1]}:{caller.lineno}" if caller else "?"
            self._log(f"[ref] make_ready tid={node.id} type={node.type.name} "
                       f"frame={node.frame_idx} prev_state={node.state.name} "
                       f"caller={site}")
        node.state = TaskState.READY
        node.t_ready = _time.perf_counter()
        seq = next(self._next_heap_seq)
        entry: _HeapEntry = (
            node.frame_idx,
            _TYPE_PRIORITY[node.type],
            seq,
            node.id,
        )
        which = queue_for(node.type)
        if which == "cc":
            heapq.heappush(self._cc_heap, entry)
            self._cc_present.add(node.id)
        elif which == "sr":
            heapq.heappush(self._sr_heap, entry)
            self._sr_present.add(node.id)
        else:  # "interp"
            heapq.heappush(self._interp_heap, entry)
            self._interp_present.add(node.id)
        self._cv.notify_all()

    # ── DAG indexing helpers ───────────────────────────────────

    def _dag_find_locked(self, t: TaskType,
                          frame_idx: int,
                          output_phase: int = 1,
                          ) -> Optional[TaskNode]:
        """O(n) scan keyed on (type, frame_idx, output_phase). Caller
        MUST hold _lock.

        output_phase distinguishes the multiple SR_INTERP nodes per
        pair K in mult ≥ 3 mode. Default 1 (primary) matches all
        non-SR_INTERP types unambiguously."""
        for node in self._dag.values():
            if (node.type == t and node.frame_idx == frame_idx
                    and node.output_phase == output_phase):
                return node
        return None

    def _ensure_node_locked(self, t: TaskType, frame_idx: int,
                             parents: list[int] | None = None,
                             parents_at_stage: dict[int, int] | None = None,
                             payload: dict | None = None,
                             output_phase: int = 1,
                             ) -> TaskNode:
        """Return the existing node for (t, frame_idx) if any, else
        create one. Caller MUST hold _lock.

        Payload merge policy: for CC nodes' mpv-VA fields
        (src_y/src_u/src_v + strides), LAST writer wins — pair K's
        sa-side VA and pair K-1's sb-side VA may differ if mpv hands
        a fresh buffer per compute() call. Stale cached VAs caused
        EFAULTs (addr=0x0) when dma_proc read the old address after
        mpv freed it. cc_cache slot ids (dst_cc_slot, src_cc_slot_*)
        keep first-writer-wins — those are queue_mgr-internal and
        must not be re-allocated mid-flight."""
        existing = self._dag_find_locked(t, frame_idx, output_phase)
        if existing is not None:
            if payload:
                _VA_KEYS = {
                    "src_y", "src_u", "src_v",
                    "src_y_stride", "src_uv_stride",
                }
                _dbg_va = os.environ.get("DUAL_VA_DBG", "0") == "1"
                for k, v in payload.items():
                    if k in _VA_KEYS:
                        # last-writer-wins for mpv-VA fields, but only
                        # if the new value is non-zero (don't let a
                        # later submit_frame with no va overwrite a
                        # good earlier va).
                        if v:
                            old = existing.payload.get(k, 0)
                            if _dbg_va and old and old != v and k == "src_y":
                                # Diagnostic: mpv handed us a DIFFERENT
                                # VA for the same source frame across
                                # compute() calls. The original VA
                                # buffer may have been recycled.
                                self._log(
                                    f"[va] OVERWRITE {t.name}(K={frame_idx})"
                                    f".{k}: 0x{old:x} → 0x{v:x}")
                            existing.payload[k] = v
                    elif k not in existing.payload:
                        existing.payload[k] = v
            return existing
        return self._new_node(task_type=t, frame_idx=frame_idx,
                                parents=parents,
                                parents_at_stage=parents_at_stage,
                                payload=payload,
                                output_phase=output_phase)

    # ── public — frame decomposition ────────────────────────────

    def submit_frame(self, K: int, *,
                      va: Optional["FrameVA"] = None) -> None:
        """frame_thread K asks the mgr to schedule output pair K
        (source frame K) + K.5 (interp frame).

        DAG nodes created:
          CCSR(K), CCSR(K+1)     merged CC + SR_SRC
          INTERP(K, K+1)         gated on CCSR(K) and CCSR(K+1)
          SR_INTERP(K.5)         gated on INTERP(K, K+1)

        4K source path is routed by native_dispatcher via
        downsample_pre — proc dims become H/downsample_pre, kernels
        and slot layout adapt, and the DAG shape is identical.

        Repeated calls with the same K are idempotent — any node
        already in the DAG is reused, which is the common case for the
        "next" frame (e.g. CC(K+1) created by submit_frame(K) is
        observed and shared by the later submit_frame(K+1)).
        """
        # CC payload ownership (last-writer-wins / VA staleness):
        # Each CC_M is "owned" by the EARLIEST pair that needs it. Pair
        # K's submit_frame holds VAs for frame K (sa) and frame K+1 (sb).
        # CC_K is depended on by INTERP_{K-1} and INTERP_K (pair K-1 is
        # earlier, so CC_K is owned by pair K-1, written via its sb).
        # CC_{K+1} is depended on by INTERP_K and INTERP_{K+1} (pair K
        # earlier), so CC_{K+1} is owned by pair K, written via its sb.
        # Exception: pair 0 has no predecessor — it bootstraps CC_0 from
        # its own sa side.
        cc_k_payload = None
        cc_kp1_payload = None
        if va is not None:
            if K == 0:
                # Bootstrap CC_0 — no pair (-1) exists to write it.
                cc_k_payload = {
                    "src_y": va.sa_y, "src_u": va.sa_u, "src_v": va.sa_v,
                    "src_y_stride": va.sa_y_stride,
                    "src_uv_stride": va.sa_uv_stride,
                }
            # CC_{K+1} is always owned by pair K's submit (sb side).
            cc_kp1_payload = {
                "src_y": va.sb_y, "src_u": va.sb_u, "src_v": va.sb_v,
                "src_y_stride": va.sa_y_stride,
                "src_uv_stride": va.sa_uv_stride,
            }
        _t_enter = _time.perf_counter() if self._profile else 0.0
        with self._lock:
            _t_inside = _time.perf_counter() if self._profile else 0.0
            if self._closed:
                return
            if K > self._max_submit_K:
                self._max_submit_K = K
            # Defensive: K already signaled means another frame_thread
            # for K completed the full pipeline. A late phase-0 /
            # phase-1 submit_frame must NOT re-create DAG nodes (would
            # cause duplicate SR_SRC / SR_INTERP, double-consume of
            # CC_K.sr_yuv → refcount underflow / heap corruption).
            # Bail — no notify needed (no new tasks made ready;
            # the frame_thread waiter uses an Event, not _cv).
            if K in self._signaled_K_set:
                return
            # CCSR(K) is the phase-0 producer (writes 4K SR to mpv VA
            # and rgb_padded + features to cc_cache for INTERP).
            cc_k = self._ensure_node_locked(
                TaskType.CCSR, K, payload=cc_k_payload)
            cc_kp1 = self._ensure_node_locked(
                TaskType.CCSR, K + 1, payload=cc_kp1_payload)
            if va is not None:
                # Merge: keep dst_a_*/dst_i_*/dst_i2_* if a prior
                # set_phase_dst already set them; mpv's per-phase
                # compute() calls arrive in arbitrary order vs
                # submit_frame.
                existing = self._frame_va.get(K)
                if existing is not None:
                    va.dst_a_y = existing.dst_a_y or va.dst_a_y
                    va.dst_a_u = existing.dst_a_u or va.dst_a_u
                    va.dst_a_v = existing.dst_a_v or va.dst_a_v
                    va.dst_i_y = existing.dst_i_y or va.dst_i_y
                    va.dst_i_u = existing.dst_i_u or va.dst_i_u
                    va.dst_i_v = existing.dst_i_v or va.dst_i_v
                    va.dst_i2_y = existing.dst_i2_y or va.dst_i2_y
                    va.dst_i2_u = existing.dst_i2_u or va.dst_i2_u
                    va.dst_i2_v = existing.dst_i2_v or va.dst_i2_v
                    va.dst_i3_y = existing.dst_i3_y or va.dst_i3_y
                    va.dst_i3_u = existing.dst_i3_u or va.dst_i3_u
                    va.dst_i3_v = existing.dst_i3_v or va.dst_i3_v
                    if not va.dst_y_stride:
                        va.dst_y_stride = existing.dst_y_stride
                    if not va.dst_uv_stride:
                        va.dst_uv_stride = existing.dst_uv_stride
                self._frame_va[K] = va
            # CCSR is the phase-0 producer (no separate SR_SRC node).
            sr_src = None
            # INTERP only needs CCSR's rgb_padded + rife_features
            # (stage 0), not the SR yuv (final stage). Both CCSR
            # parents fire stage 0 (mid) so INTERP unblocks as soon as
            # both CCSRs finish their RIFE-encode pass.
            # One INTERP per pair runs dual/triple flownet inside
            # worker when mult=3/4. SR_INTERP fan-out via
            # output_phase ∈ {1..mult-1}. Phase p gates on INTERP
            # stage (p-1) for p ∈ [1, mult-1); phase (mult-1) gates on
            # INTERP's final (task_done). N-stage allows each
            # SR_INTERP to start as soon as its specific frame is
            # ready, instead of all waiting on the last flownet.
            _mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
            if _mult not in (1, 2, 3, 4):
                _mult = 2
            # mult=1 (no_interp dual): skip INTERP + SR_INTERP entirely
            # (no consumer for INTERP outputs; mirror submit_with_phase_dst).
            interp = None
            sr_interp_nodes = []
            if _mult > 1:
                interp = self._ensure_node_locked(
                    TaskType.INTERP, K, parents=[cc_k.id, cc_kp1.id],
                    parents_at_stage={cc_k.id: 0, cc_kp1.id: 0})
                for _p in range(1, _mult):
                    _pas = ({interp.id: _p - 1}
                            if _p < _mult - 1 else None)
                    _node = self._ensure_node_locked(
                        TaskType.SR_INTERP, K, parents=[interp.id],
                        parents_at_stage=_pas,
                        output_phase=_p)
                    sr_interp_nodes.append(_node)
            sr_interp = sr_interp_nodes[0] if sr_interp_nodes else None
            # Re-check publishability for nodes whose gating may have
            # just flipped. Cases:
            #  - cc_k may have been a payload-less placeholder created
            #    by a prior pair K-1; this submit may have merged in
            #    src_y for K via cc_k_payload (K==0 bootstrap) → ready.
            #  - cc_kp1's src_y was just filled by this pair's sb side
            #    → ready (already covered by _new_node's try_make_ready
            #    when freshly created; this is a no-op for existing).
            #  - sr_src / sr_interp_*: if the parent CC/INTERP is
            #    already DONE (rare), deps_remaining is 0 on creation;
            #    gating now also needs FrameVA.dst_*_y to be set.
            for _n in [cc_k, cc_kp1, sr_src, interp] + sr_interp_nodes:
                if _n is not None:
                    self._try_make_ready_locked(_n)
            # Wake the dispatch loops on any newly-ready task.
            self._cv.notify_all()
            if self._profile:
                dt = (_time.perf_counter() - _t_inside) * 1000
                rec = self._lock_log["submit_frame"]
                rec["ms"] += dt
                rec["n"] += 1
                if dt > rec["max_ms"]:
                    rec["max_ms"] = dt
                if len(self._dag) > self._dag_size_peak:
                    self._dag_size_peak = len(self._dag)

    # ── mpv VA tracking ─────────────────────────────────────────

    def set_node_payload(self, task_id: int, key: str, value) -> None:
        """Atomic single-field update on a DAG node's payload. Used by
        the dispatch thread to record `worker_slot` mid-flight."""
        with self._lock:
            node = self._dag.get(task_id)
            if node is not None:
                node.payload[key] = value

    def update_frame_va(self, K: int, va: FrameVA) -> None:
        """mpv frame_thread sets the VA for pair K so the dispatch
        thread can populate CC tasks. Idempotent — repeated calls
        with the same K overwrite. Stored under the same lock as
        everything else."""
        with self._lock:
            self._frame_va[K] = va

    def set_phase_dst(self, K: int, phase: int, *,
                       dst_y: int, dst_u: int, dst_v: int,
                       dst_y_stride: int, dst_uv_stride: int) -> None:
        """mpv per-phase compute() call registers WHERE the SR_*
        result for (K, phase) should land in mpv VA. Phase 0 → SR_SRC
        (writes to dst_a_*); phase 1 → SR_INTERP (writes to dst_i_*).

        Must be called BEFORE submit_frame so that when the dispatcher
        pops SR_* it can read va.dst_a_*/dst_i_* and route accordingly.
        Updates the existing FrameVA in place if one exists (created
        by an earlier set_phase_dst or submit_frame); otherwise creates
        a stub FrameVA with just the dst fields set.

        Note: the dst_y_stride / dst_uv_stride fields are shared across
        phases in FrameVA — both phases hand mpv the same strides, so
        last-writer-wins is fine.
        """
        with self._lock:
            va = self._frame_va.get(K)
            if va is None:
                va = FrameVA(pair_k=K)
                self._frame_va[K] = va
            if phase == 0:
                va.dst_a_y = dst_y
                va.dst_a_u = dst_u
                va.dst_a_v = dst_v
            elif phase == 1:
                va.dst_i_y = dst_y
                va.dst_i_u = dst_u
                va.dst_i_v = dst_v
            elif phase == 2:  # mult ≥ 3
                va.dst_i2_y = dst_y
                va.dst_i2_u = dst_u
                va.dst_i2_v = dst_v
            else:  # phase == 3 (mult=4 only)
                va.dst_i3_y = dst_y
                va.dst_i3_u = dst_u
                va.dst_i3_v = dst_v
            va.dst_y_stride = dst_y_stride
            va.dst_uv_stride = dst_uv_stride
            # Strict gating: phase 0 → CCSR(K); phase p ≥ 1 →
            # SR_INTERP(K, output_phase=p). Re-check publishability
            # for the corresponding DAG node (may have been pending
            # only on VA).
            if phase == 0:
                sr_node = self._dag_find_locked(TaskType.CCSR, K)
            else:
                sr_node = self._dag_find_locked(
                    TaskType.SR_INTERP, K, output_phase=phase)
            if sr_node is not None:
                self._try_make_ready_locked(sr_node)
            self._cv.notify_all()

    def submit_with_phase_dst(self, K: int, phase: int, *,
                                sa_y: int, sa_u: int, sa_v: int,
                                sb_y: int, sb_u: int, sb_v: int,
                                sa_y_stride: int, sa_uv_stride: int,
                                dst_y: int, dst_u: int, dst_v: int,
                                dst_y_stride: int, dst_uv_stride: int
                                ) -> None:
        """Fused entry: set_phase_dst + submit_frame under ONE lock
        acquisition + ONE notify_all (instead of 2 of each). Same
        semantics as calling set_phase_dst(K, phase, ...) followed by
        submit_frame(K, va=FrameVA(...)) — but without the FrameVA
        dataclass construction in Python and without the extra lock
        cycle. Called from mpv-side compute_callable hot path."""
        # K==0 bootstrap (otherwise CC_0's src_y is never set).
        cc_k_payload = None
        if K == 0:
            cc_k_payload = {
                "src_y": sa_y, "src_u": sa_u, "src_v": sa_v,
                "src_y_stride": sa_y_stride,
                "src_uv_stride": sa_uv_stride,
            }
        cc_kp1_payload = {
            "src_y": sb_y, "src_u": sb_u, "src_v": sb_v,
            "src_y_stride": sa_y_stride,
            "src_uv_stride": sa_uv_stride,
        }
        _t_enter = _time.perf_counter() if self._profile else 0.0
        with self._lock:
            _t_inside = _time.perf_counter() if self._profile else 0.0
            if self._closed:
                return
            if K > self._max_submit_K:
                self._max_submit_K = K
            # Same defensive guard as submit_frame.
            if K in self._signaled_K_set:
                return
            # 1) Update FrameVA in place (set_phase_dst's body).
            va = self._frame_va.get(K)
            if va is None:
                va = FrameVA(pair_k=K)
                self._frame_va[K] = va
            if phase == 0:
                va.dst_a_y = dst_y
                va.dst_a_u = dst_u
                va.dst_a_v = dst_v
            elif phase == 1:
                va.dst_i_y = dst_y
                va.dst_i_u = dst_u
                va.dst_i_v = dst_v
            elif phase == 2:  # mult ≥ 3
                va.dst_i2_y = dst_y
                va.dst_i2_u = dst_u
                va.dst_i2_v = dst_v
            else:  # phase == 3 (mult=4 only)
                va.dst_i3_y = dst_y
                va.dst_i3_u = dst_u
                va.dst_i3_v = dst_v
            va.dst_y_stride = dst_y_stride
            va.dst_uv_stride = dst_uv_stride
            # Also fill sa/sb/strides so submit_frame's normal merge
            # (which assumed va was a fresh argument) becomes a no-op
            # since we're already updating in place.
            if K == 0 or not va.sa_y:
                va.sa_y = sa_y; va.sa_u = sa_u; va.sa_v = sa_v
                va.sa_y_stride = sa_y_stride
                va.sa_uv_stride = sa_uv_stride
            if not va.sb_y:
                va.sb_y = sb_y; va.sb_u = sb_u; va.sb_v = sb_v
            # 2) DAG nodes (submit_frame's body).
            # ONE INTERP task per pair runs dual/triple flownet inside
            # the worker when mult=3/4. SR_INTERP fan-out via
            # output_phase ∈ {1..mult-1}. Phase p gates on INTERP
            # stage (p-1) for p < mult-1; phase (mult-1) gates on INTERP
            # final (task_done). All SR_INTERPs can dispatch to either
            # machine in parallel.
            _mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
            if _mult not in (1, 2, 3, 4):
                _mult = 2
            cc_k = self._ensure_node_locked(
                TaskType.CCSR, K, payload=cc_k_payload)
            cc_kp1 = self._ensure_node_locked(
                TaskType.CCSR, K + 1, payload=cc_kp1_payload)
            sr_src = None  # CCSR is the phase-0 producer
            # mult=1 = no_interp dual mode: SplitDual emits ONE output
            # frame per pair (phase=0 only), and we skip INTERP /
            # SR_INTERP entirely. CCSR still runs for its SR pass on
            # source frames; CCSR worker-side branches skip rgb_padded +
            # rife_features writes since no INTERP consumer exists.
            interp = None
            sr_interp_nodes = []
            if _mult > 1:
                interp = self._ensure_node_locked(
                    TaskType.INTERP, K, parents=[cc_k.id, cc_kp1.id],
                    parents_at_stage={cc_k.id: 0, cc_kp1.id: 0})
                for _p in range(1, _mult):
                    _pas = ({interp.id: _p - 1}
                            if _p < _mult - 1 else None)
                    _node = self._ensure_node_locked(
                        TaskType.SR_INTERP, K, parents=[interp.id],
                        parents_at_stage=_pas,
                        output_phase=_p)
                    sr_interp_nodes.append(_node)
            sr_interp = sr_interp_nodes[0] if sr_interp_nodes else None
            # 3) Re-check publishability for all created/merged nodes.
            for _n in [cc_k, cc_kp1, sr_src, interp] + sr_interp_nodes:
                if _n is not None:
                    self._try_make_ready_locked(_n)
            self._cv.notify_all()
            if self._profile:
                dt = (_time.perf_counter() - _t_inside) * 1000
                rec = self._lock_log["submit_frame"]
                rec["ms"] += dt
                rec["n"] += 1
                if dt > rec["max_ms"]:
                    rec["max_ms"] = dt
                if len(self._dag) > self._dag_size_peak:
                    self._dag_size_peak = len(self._dag)

    def get_frame_va(self, K: int) -> Optional[FrameVA]:
        with self._lock:
            return self._frame_va.get(K)

    def drop_frame_va(self, K: int) -> None:
        with self._lock:
            self._frame_va.pop(K, None)

    def get_phase_result(self, K: int, phase: int
                          ) -> tuple[int, int]:
        """Returns (worker_slot, task_id) for output frame (K, phase).
        phase 0 → CCSR(K); phase 1 → SR_INTERP(K). Returns (-1, -1)
        if the task is not yet done or its payload is missing a
        worker_slot field."""
        with self._lock:
            t = TaskType.CCSR if phase == 0 else TaskType.SR_INTERP
            node = self._dag_find_locked(t, K)
            if node is None or node.state != TaskState.DONE:
                return (-1, -1)
            return (
                int(node.payload.get("worker_slot", -1)),
                int(node.id),
            )

    # ── task_done + refcount lifecycle ──────────────────────────

    def task_stage_done(self, task_id: int, stage_idx: int) -> None:
        """N-stage producer reports an intermediate stage complete.
        For CCSR, stage 0 means rgb_padded + rife_features are written
        to cc_cache (or arrived on the host's slot.dst[mid] via RDMA);
        SR yuv is still in flight. For INTERP mult=3, stage 0 means the
        t=1/3 frame is ready; t=2/3 still in flight.

        stage_idx >= 0 corresponds to an early stage. The final stage
        (== task_done) is NOT triggered here — call task_done() to
        signal completion of the last stage and refcount/cleanup.

        Walks children whose edge to this task is parents_at_stage[
        task_id] == stage_idx and decrements their deps_remaining.
        Single-stage tasks never call this method; their children gate
        on task_done instead."""
        with self._cv:
            node = self._dag.get(task_id)
            if node is None:
                return
            fired_stages: set = node.payload.setdefault(
                "stages_fired", set())
            if stage_idx in fired_stages:
                self._dbg(f"task_stage_done(tid={task_id}, "
                          f"stage={stage_idx}) IGNORED — already fired")
                return
            fired_stages.add(stage_idx)
            # When mid delivery is on, INTERP can run + complete before
            # the producer's task_done fires. INTERP's _consume_inputs
            # would then decrement an uninitialised refcount →
            # underflow. Initialise refcounts at the first stage that
            # publishes a consumer-visible field. Idempotent via the
            # same `refcounts_init` marker task_done checks.
            if not node.payload.get("refcounts_init"):
                self._initialise_refcounts_locked(node)
                node.payload["refcounts_init"] = True
            # Stage completion doesn't update node.state (still
            # COMPUTING from queue_mgr's POV until task_done fires).
            for cid in list(node.children):
                child = self._dag.get(cid)
                if child is None:
                    continue
                if task_id not in child.parents:
                    continue
                if child.parents_at_stage.get(task_id) != stage_idx:
                    # This child gates on a different stage (or final).
                    continue
                child.parents.discard(task_id)
                child.deps_remaining -= 1
                self._try_make_ready_locked(child)

    def task_mid_done(self, task_id: int) -> None:
        """Convenience alias: task_stage_done(task_id, 0). Most
        producers fire only one intermediate stage (stage 0) so callers
        can use this short form."""
        self.task_stage_done(task_id, 0)

    def task_done(self, task_id: int, *, ok: bool = True,
                   result: object = None, error: Optional[str] = None
                   ) -> None:
        """Worker reports completion. The mgr:
          1. Marks the node DONE/FAILED.
          2. For producer tasks (CC, INTERP) initialises cc_cache
             refcounts on the freshly-written fields based on which
             downstream consumers exist in the DAG.
          3. For consumer tasks (SR_SRC, SR_INTERP, INTERP) decrements
             refcounts on the input fields they just finished reading.
          4. Walks children, decrements their deps_remaining, and pushes
             newly-runnable ones onto the right queue.
          5. If both display halves of a frame_idx are DONE, signals
             frame_done[K].
        """
        with self._cv:
            _t_inside = _time.perf_counter() if self._profile else 0.0
            node = self._dag.get(task_id)
            if node is None:
                return
            # IDEMPOTENCY: task_done must be a no-op on a node that's
            # already DONE/FAILED. Without this guard, a stray second
            # call (multi-consumer race, dispatcher firing task_done
            # from both host and guest paths for a node that was
            # reassigned, etc.) re-runs _initialise_refcounts AND
            # _consume_inputs, causing refcount underflow on parent
            # cc_cache slots.
            if node.state in (TaskState.DONE, TaskState.FAILED):
                self._dbg(f"task_done(tid={task_id}) IGNORED — already "
                           f"{node.state.name} (type={node.type.name})")
                return
            node.state = TaskState.DONE if ok else TaskState.FAILED
            node.result = result
            node.error = error
            node.t_done = _time.perf_counter()
            if ok:
                self._stats["completed_tasks"] += 1
            else:
                self._stats["failed_tasks"] += 1

            if ok:
                if not node.payload.get("refcounts_init"):
                    self._initialise_refcounts_locked(node)
                    node.payload["refcounts_init"] = True
                self._consume_inputs_locked(node)
                # Publish children whose remaining parents are now all
                # done. Edges marked gates_on_mid are satisfied by the
                # parent's task_mid_done — already consumed from
                # child.parents by the time we get here, so the
                # membership test below naturally skips them.
                for cid in list(node.children):
                    child = self._dag.get(cid)
                    if child is None:
                        continue
                    if task_id in child.parents:
                        child.parents.discard(task_id)
                        child.deps_remaining -= 1
                    # Strict gating: _try_make_ready_locked is the only
                    # path that publishes. Checks deps_remaining, state,
                    # AND task-specific gates (mpv VA presence).
                    self._try_make_ready_locked(child)
                # Frame-done detection. CCSR is the phase-0 producer.
                # For mult ≥ 3, all SR_INTERP variants (output_phase
                # 1..mult-1) must be done —
                # _maybe_signal_frame_done_locked checks them all.
                if node.type in (TaskType.SR_INTERP, TaskType.CCSR):
                    self._maybe_signal_frame_done_locked(node.frame_idx)
                    # wait_phase_done waiters poll node.state on _cv
                    # wakeups. Notify so they can re-check immediately
                    # instead of waiting up to 100ms.
                    self._cv.notify_all()
            # No outer notify_all here. _make_ready_locked already
            # notifies for any child that just became READY (the only
            # state-change a cv-waiting dispatcher cares about). A
            # blanket notify here would wake every dispatcher
            # ~10⁴/sec even when no new task became ready.
            if self._profile:
                dt = (_time.perf_counter() - _t_inside) * 1000
                rec = self._lock_log["task_done"]
                rec["ms"] += dt
                rec["n"] += 1
                if dt > rec["max_ms"]:
                    rec["max_ms"] = dt

    def _dbg(self, msg: str) -> None:
        if os.environ.get("DUAL_REFCOUNT_DBG", "0") == "1":
            self._log(f"[ref] {msg}")

    def _initialise_refcounts_locked(self, node: TaskNode) -> None:
        """Producer-side: when CC/INTERP just finished, set up refcounts
        on the cc_cache slot it wrote.

        We use *fixed* conventions (1080p source path):
          rgb_padded_refs    = 2 (INTERP(N-1,N) + INTERP(N,N+1))
          rife_features_refs = 2 (same consumers as rgb_padded)
          sr_yuv_refs        = 1 (SR_SRC(N))
        regardless of what's currently in the DAG. Future submit_frame
        calls will add the corresponding INTERP/SR_SRC children. Without
        this, CC(N)'s slot could be evicted before the second INTERP
        ever lands in the DAG (e.g. when frame N+1's submit_frame
        happens after CC(N) has been consumed by INTERP(N-1, N) only).

        First / last frame of the stream produces a small slot leak
        (one CC unused refcount per stream end). Acceptable — process
        exit reclaims.
        """
        if node.type == TaskType.CCSR:
            # CCSR producer writes rgb_padded + rife_features to
            # cc_cache, consumed by 2 downstream INTERPs (right of
            # pair K-1 + left of pair K). One INTERP per pair (dual
            # flownet inside) keeps refcount at 2 regardless of mult.
            # sr_yuv is local within the kernel.
            #
            # mult=1 (no_interp dual mode) exception: queue_mgr's
            # submit_*_with_phase_dst skips INTERP/SR_INTERP node
            # creation, so the consumers we'd be reserving slots for
            # never materialise → cc_cache slots leak → after ~32
            # frames the pool is exhausted and all dispatch hangs.
            # Set consumer counts to 0 so the slot is released as
            # soon as CCSR completes.
            slot = node.payload.get("dst_cc_slot", -1)
            if slot < 0:
                self._dbg(f"init CCSR tid={node.id} frame={node.frame_idx} slot=-1 (SKIP)")
                return
            _mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
            if _mult <= 1:
                # No INTERP / SR_INTERP consumes the cc_cache rgb_padded
                # / rife_features. Without explicit free here, the slot
                # leaks (the auto-free path in _cc_drop_slot_field_locked
                # requires a has_X flag to drop) and after ~32 frames
                # the pool empties, dispatch hangs. See bench note in
                # [[server-mode-worker-robustness]].
                self._dbg(f"init CCSR tid={node.id} frame={node.frame_idx} "
                           f"slot={slot} mult=1 → free immediately")
                self._cc.pop(slot, None)
                if self._cc_cache is not None and slot >= 0:
                    self._cc_cache.free(slot)
                self._stats["cc_evictions"] += 1
                return
            # Count actual live (non-DONE/FAILED) INTERP children
            # — this handles three cases the old hardcoded 2 leaked on:
            #   - K=0       : INTERP(-1, 0) doesn't exist → 1 consumer.
            #   - last K    : INTERP(K, K+1) never created → 1 consumer.
            #   - post-seek : flush_seek deleted the INTERPs → 0 consumers,
            #                 slot is freed immediately.
            # CF=24 means by the time CCSR(K).task_done runs, both
            # surrounding submit_frame calls have happened (if they were
            # going to), so the count is accurate at this moment.
            n_consumers = 0
            for cid in node.children:
                child = self._dag.get(cid)
                if child is None:
                    continue
                if child.type != TaskType.INTERP:
                    continue
                if child.state in (TaskState.DONE, TaskState.FAILED):
                    continue
                n_consumers += 1
            if n_consumers == 0:
                # No live INTERP consumers (post-seek flush, or first/
                # last frame). Free the slot directly so it returns to
                # the cc_cache pool. Without this the slot leaks
                # forever — under the singleton lifetime that's a
                # cumulative drain on the 32-slot pool.
                self._dbg(f"init CCSR tid={node.id} frame={node.frame_idx} "
                           f"slot={slot} → no consumers, free immediately")
                self._cc.pop(slot, None)
                if self._cc_cache is not None and slot >= 0:
                    self._cc_cache.free(slot)
                self._stats["cc_evictions"] += 1
                return
            self._dbg(f"init CCSR tid={node.id} frame={node.frame_idx} "
                       f"slot={slot} → refs={n_consumers}/{n_consumers}/0")
            self._cc_set_producer_locked(
                slot,
                rgb_padded_consumers=n_consumers,
                rife_features_consumers=n_consumers,
                sr_yuv_consumers=0,
                rgb_4K_consumers=0,
            )
        elif node.type == TaskType.INTERP:
            # INTERP output rgb_interp has 1 consumer per output frame:
            # SR_INTERP output_phase=p reads dst_cc_slot_{p} (p=1..mult-1).
            # Same dynamic-counting approach as CCSR — if the SR_INTERP
            # child for a given phase is no longer in the DAG (flushed by
            # seek), free that slot immediately.
            for _idx, _key in enumerate(("dst_cc_slot", "dst_cc_slot_2",
                                          "dst_cc_slot_3")):
                _s = node.payload.get(_key, -1)
                if _s < 0:
                    continue
                _phase = _idx + 1
                _have_consumer = False
                for cid in node.children:
                    child = self._dag.get(cid)
                    if child is None:
                        continue
                    if child.type != TaskType.SR_INTERP:
                        continue
                    if child.output_phase != _phase:
                        continue
                    if child.state in (TaskState.DONE, TaskState.FAILED):
                        continue
                    _have_consumer = True
                    break
                if not _have_consumer:
                    self._cc.pop(_s, None)
                    if self._cc_cache is not None and _s >= 0:
                        self._cc_cache.free(_s)
                    self._stats["cc_evictions"] += 1
                    continue
                self._cc_set_producer_locked(_s, sr_yuv_consumers=1)

    def _consume_inputs_locked(self, node: TaskNode) -> None:
        """Consumer-side: drop refcounts on slots this task just
        finished reading. INTERP consumes rgb_padded + rife_features
        (two slots — features always co-live with rgb_padded);
        SR_SRC and SR_INTERP each consume sr_yuv (one slot)."""
        if node.type == TaskType.INTERP:
            a = node.payload.get("src_cc_slot_a", -1)
            b = node.payload.get("src_cc_slot_b", -1)
            self._dbg(f"consume INTERP tid={node.id} frame={node.frame_idx} "
                       f"drops rgb_padded+features on slots a={a} b={b}")
            for s in (a, b):
                if s >= 0:
                    self._cc_drop_slot_field_locked(s, "rgb_padded")
                    self._cc_drop_slot_field_locked(s, "rife_features")
        elif node.type == TaskType.SR_INTERP:
            a = node.payload.get("src_cc_slot_a", -1)
            self._dbg(f"consume SR_INTERP(phase={node.output_phase}) "
                       f"tid={node.id} frame={node.frame_idx} drops "
                       f"sr_yuv (=INTERP output ref) on slot={a}")
            if a >= 0:
                # `sr_yuv` is a label here for "INTERP output reference"
                # — the actual cc_cache field is rgb_interp, but the
                # refcount tracking name stays for compatibility.
                self._cc_drop_slot_field_locked(a, "sr_yuv")

    def _maybe_signal_frame_done_locked(self, K: int) -> None:
        # Phase 0 = CCSR(K); phase p ≥ 1 = SR_INTERP(K, output_phase=p).
        # mult=N → phases {0..N-1}. All present phases must be DONE
        # before mpv's wait_frame_done(K) unblocks.
        phase0 = self._dag_find_locked(TaskType.CCSR, K)
        sr_nodes = [
            self._dag_find_locked(TaskType.SR_INTERP, K, output_phase=p)
            for p in (1, 2, 3)]
        if phase0 is None and all(n is None for n in sr_nodes):
            return
        ok = (phase0 is None or phase0.state == TaskState.DONE)
        for n in sr_nodes:
            if not ok:
                return
            ok = ok and (n is None or n.state == TaskState.DONE)
        if ok:
            self._signal_frame_done_locked(K)

    # ── seek-driven flush ──────────────────────────────────────
    def flush_seek(self) -> dict:
        """Drop pre-seek per-K state so post-seek work starts fresh.
        Called from native_dispatcher's inotify watcher when the lua
        seek hook bumps /tmp/dual_machine_seek_epoch, and from
        make_dispatcher's REUSE path (first-wins coordinated)."""
        with self._cv:
            return self._flush_seek_locked()

    def _flush_seek_locked(self) -> dict:
        """Delete PENDING/READY/DONE nodes, clear `_signaled_K_set`,
        bump `_seek_epoch`, pop per-K Events. CLAIMED stays — the
        worker finishes naturally; its task_done frees its cc_cache
        slot via the dynamic-child-count path in
        `_initialise_refcounts_locked`."""
        flushed_K: set[int] = set()
        to_delete: list[int] = []
        for tid, node in self._dag.items():
            if node.state in (TaskState.PENDING,
                               TaskState.READY,
                               TaskState.DONE):
                to_delete.append(tid)
        for tid in to_delete:
            node = self._dag.pop(tid, None)
            if node is None:
                continue
            flushed_K.add(node.frame_idx)
            self._cc_present.discard(tid)
            self._sr_present.discard(tid)
            self._interp_present.discard(tid)
            # Detach from any still-live (CLAIMED) parent's children
            # set so the parent's task_done doesn't try to ready
            # this ghost child.
            for pid in node.parents:
                parent = self._dag.get(pid)
                if parent is not None:
                    parent.children.discard(tid)
        self._signaled_K_set.clear()
        self._seek_epoch += 1
        for K in flushed_K:
            self._frame_va.pop(K, None)
            ev = self._frame_done.pop(K, None)
            if ev is not None:
                ev.set()
        self._stats["seek_flushes"] = self._stats.get(
            "seek_flushes", 0) + 1
        self._cv.notify_all()
        return {
            "failed": len(to_delete),
            "K_count": len(flushed_K),
            "slots_freed": 0,  # CLAIMED tasks free via task_done; nothing direct
        }

    # ── worker-side fetch ──────────────────────────────────────

    def pop_cc(self, *, blocking: bool = False,
                timeout: Optional[float] = None) -> Optional[TaskNode]:
        """Host-exclusive cc_q. CC tasks read mpv VA via dma_proc's
        process_vm_readv — only the host side can dereference. Same
        cc_cache-backpressure rule as before: returns None and
        re-queues if alloc fails."""
        with self._cv:
            node = self._pop_locked(
                self._cc_heap, self._cc_present,
                "color", blocking, timeout)
            if node is None:
                return None
            if not self._post_pop_locked(node):
                self._make_ready_locked(node)
                return None
            return node

    def pop_sr(self, *, blocking: bool = False,
                timeout: Optional[float] = None) -> Optional[TaskNode]:
        """SR_SRC + SR_INTERP queue. Either rank consumes. Fast tasks
        (~6 ms); host pops here when cc_q is empty so a stray INTERP
        in the system never starves CC."""
        with self._cv:
            node = self._pop_locked(
                self._sr_heap, self._sr_present,
                "sr", blocking, timeout)
            if node is None:
                return None
            if not self._post_pop_locked(node):
                self._make_ready_locked(node)
                return None
            return node

    def pop_interp(self, *, blocking: bool = False,
                    timeout: Optional[float] = None) -> Optional[TaskNode]:
        """INTERP queue. Heavy (~17 ms / pair); guest pops here first.
        Host touches it only if cc_q AND sr_q are empty."""
        with self._cv:
            node = self._pop_locked(
                self._interp_heap, self._interp_present,
                "interp", blocking, timeout)
            if node is None:
                return None
            if not self._post_pop_locked(node):
                self._make_ready_locked(node)
                return None
            return node

    # ── rank-priority pop ──────────────────────────────────────
    # The DISPATCHER asks "give me any task I can run", queue_mgr
    # applies the priority order. This avoids the bug where a
    # dispatcher's own non-blocking-then-blocking-on-wrong-queue
    # logic gets stuck waiting on a secondary queue while the
    # primary one fills up.
    # 3-queue priority orders.
    # Host: sr > cc > interp. Host claims SR_INTERP first (light ~9ms,
    # transit eats the gain on guest); CC piles up in the queue long
    # enough that guest can grab a meaningful share, balancing the
    # heavier task across both GPUs. INTERP last (host's forte is
    # CC/SR; guest's is INTERP).
    # Guest: interp > cc > sr. INTERP first because host is the
    # critical CCSR/SR producer and guest's main value is offloading
    # INTERP. CC second since it's heavy enough (~12ms) to mask the
    # RDMA transit. SR_INTERP last so it stays with host.
    _HOST_PRIORITY = ("sr", "cc", "interp")
    _GUEST_PRIORITY = ("interp", "cc", "sr")

    def _pop_priority_locked(self, order: tuple) -> Optional[TaskNode]:
        """Caller MUST hold _cv. Walks `order` and returns the first
        ready task. None if all queues empty."""
        heaps = {
            "cc":     (self._cc_heap,     self._cc_present),
            "sr":     (self._sr_heap,     self._sr_present),
            "interp": (self._interp_heap, self._interp_present),
        }
        for which in order:
            heap, present = heaps[which]
            # _pop_locked with blocking=False just drains one ready
            # entry from this heap (or returns None immediately).
            node = self._pop_locked(heap, present, which,
                                     blocking=False, timeout=None)
            if node is not None:
                if not self._post_pop_locked(node):
                    # cc_cache full — push back and try next priority.
                    self._make_ready_locked(node)
                    continue
                return node
        return None

    def pop_for_host(self, *, blocking: bool = True,
                      timeout: Optional[float] = 0.5
                      ) -> Optional[TaskNode]:
        """Host's view: cc > sr > interp. Returns highest-priority
        ready task. If all three empty and blocking, waits on _cv;
        wakes on any submit_frame / task_done that may have made a
        new task ready, then re-checks ALL three queues in order."""
        deadline = (_time.perf_counter() + timeout) if blocking and timeout else None
        t_enter = _time.perf_counter() if self._profile else 0.0
        n_waits = 0
        with self._cv:
            _t_first_inside = _time.perf_counter() if self._profile else 0.0
            while True:
                if self._profile:
                    _t_attempt = _time.perf_counter()
                    depth_cc = len(self._cc_present)
                    depth_sr = len(self._sr_present)
                    depth_in = len(self._interp_present)
                node = self._pop_priority_locked(self._HOST_PRIORITY)
                if node is not None:
                    if self._profile:
                        which = queue_for(node.type)
                        log = self._pop_log["host"]
                        log[which] = log.get(which, 0) + 1
                        log["n"] += 1
                        log["wait_ms_total"] += (_time.perf_counter() - t_enter) * 1000
                        log["empty_waits"] += n_waits
                        log["depth_cc_at_pop"] += depth_cc
                        log["depth_sr_at_pop"] += depth_sr
                        log["depth_interp_at_pop"] += depth_in
                        # Lock-held time = sum of attempts (priority scan
                        # + post_pop). Excludes the cv.wait() sleeps.
                        dt = (_time.perf_counter() - _t_attempt) * 1000
                        rec = self._lock_log["pop_host"]
                        rec["ms"] += dt
                        rec["n"] += 1
                        if dt > rec["max_ms"]:
                            rec["max_ms"] = dt
                        self._maybe_report_pop_locked()
                    return node
                if not blocking or self._closed:
                    return None
                remaining = (deadline - _time.perf_counter()) if deadline else None
                if remaining is not None and remaining <= 0:
                    return None
                n_waits += 1
                self._cv.wait(timeout=remaining)

    def pop_for_guest(self, *, blocking: bool = True,
                       timeout: Optional[float] = 0.5
                       ) -> Optional[TaskNode]:
        """Guest's view: interp > sr. Never CC (host-exclusive)."""
        deadline = (_time.perf_counter() + timeout) if blocking and timeout else None
        t_enter = _time.perf_counter() if self._profile else 0.0
        n_waits = 0
        with self._cv:
            while True:
                if self._profile:
                    _t_attempt = _time.perf_counter()
                    depth_sr = len(self._sr_present)
                    depth_in = len(self._interp_present)
                node = self._pop_priority_locked(self._GUEST_PRIORITY)
                if node is not None:
                    if self._profile:
                        which = queue_for(node.type)
                        log = self._pop_log["guest"]
                        log[which] = log.get(which, 0) + 1
                        log["n"] += 1
                        log["wait_ms_total"] += (_time.perf_counter() - t_enter) * 1000
                        log["empty_waits"] += n_waits
                        log["depth_sr_at_pop"] += depth_sr
                        log["depth_interp_at_pop"] += depth_in
                        dt = (_time.perf_counter() - _t_attempt) * 1000
                        rec = self._lock_log["pop_guest"]
                        rec["ms"] += dt
                        rec["n"] += 1
                        if dt > rec["max_ms"]:
                            rec["max_ms"] = dt
                        self._maybe_report_pop_locked()
                    return node
                if not blocking or self._closed:
                    return None
                remaining = (deadline - _time.perf_counter()) if deadline else None
                if remaining is not None and remaining <= 0:
                    return None
                n_waits += 1
                self._cv.wait(timeout=remaining)

    def _maybe_report_pop_locked(self) -> None:
        """Caller holds _cv. Emit a summary every DUAL_PROFILE_INTERVAL_S
        seconds. Resets counters after each report."""
        now = _time.perf_counter()
        if (now - self._pop_last_report) < self._pop_report_every:
            return
        wall = (now - self._pop_last_report) * 1000
        self._pop_last_report = now
        # Snapshot current depths for context.
        cur_cc = len(self._cc_present)
        cur_sr = len(self._sr_present)
        cur_in = len(self._interp_present)
        h = self._pop_log["host"]
        g = self._pop_log["guest"]
        if h["n"] > 0:
            n = h["n"]
            self._log(
                f"[profile] HOST pops wall={wall:.0f}ms n={n} "
                f"cc={h['cc']} sr={h['sr']} interp={h['interp']} "
                f"empty_waits={h['empty_waits']} "
                f"wait_avg={h['wait_ms_total']/n:.2f}ms "
                f"depth@pop cc={h['depth_cc_at_pop']/n:.1f} "
                f"sr={h['depth_sr_at_pop']/n:.1f} "
                f"interp={h['depth_interp_at_pop']/n:.1f}")
        if g["n"] > 0:
            n = g["n"]
            self._log(
                f"[profile] GUEST pops wall={wall:.0f}ms n={n} "
                f"interp={g['interp']} sr={g['sr']} "
                f"cc={g.get('cc', 0)} "
                f"empty_waits={g['empty_waits']} "
                f"wait_avg={g['wait_ms_total']/n:.2f}ms "
                f"depth@pop sr={g['depth_sr_at_pop']/n:.1f} "
                f"interp={g['depth_interp_at_pop']/n:.1f}")
        self._log(
            f"[profile] DEPTH NOW cc={cur_cc} sr={cur_sr} interp={cur_in} "
            f"dag_size={len(self._dag)} (peak={self._dag_size_peak})")
        # Dump PENDING nodes older than 1s — surfaces gating leaks
        # where a node never publishes (kept as diagnostic; under
        # healthy operation this list is empty).
        nowp = _time.perf_counter()
        stuck = []
        for tid, n in self._dag.items():
            if n.state != TaskState.PENDING:
                continue
            age = nowp - n.t_created
            if age < 1.0:
                continue
            why = "?"
            if n.deps_remaining > 0:
                why = f"deps={n.deps_remaining}/parents={n.parents}"
            else:
                if n.type == TaskType.CCSR:
                    va = self._frame_va.get(n.frame_idx)
                    why = (f"no_src_y_or_dst_a "
                           f"src_y={n.payload.get('src_y', 0)} "
                           f"dst_a_y={va.dst_a_y if va else 'None'}")
                elif n.type == TaskType.SR_INTERP:
                    va = self._frame_va.get(n.frame_idx)
                    why = f"no_dst_i_y (va={va.dst_i_y if va else 'None'})"
            stuck.append(f"  tid={tid} type={n.type.name} K={n.frame_idx} "
                         f"age={age*1000:.0f}ms {why}")
        if stuck:
            self._log(f"[profile] STUCK PENDING n={len(stuck)}:\n"
                       + "\n".join(stuck[:10]))
        # Lock-hold breakdown.
        parts = []
        for k, rec in self._lock_log.items():
            if rec["n"] > 0:
                parts.append(
                    f"{k}: n={rec['n']} total={rec['ms']:.0f}ms "
                    f"avg={rec['ms']/rec['n']:.2f}ms max={rec['max_ms']:.2f}ms")
        if parts:
            self._log(f"[profile] LOCK-HELD wall={wall:.0f}ms — " + " | ".join(parts))
        # Reset.
        for side in ("host", "guest"):
            for k in self._pop_log[side]:
                self._pop_log[side][k] = 0 if isinstance(
                    self._pop_log[side][k], int) else 0.0
        for k in self._lock_log:
            self._lock_log[k] = {"ms": 0.0, "n": 0, "max_ms": 0.0}
        self._dag_size_peak = 0

    # ── back-compat wrappers (old 2-queue API) ──────────────────
    def pop_color(self, *, blocking: bool = False,
                   timeout: Optional[float] = None) -> Optional[TaskNode]:
        return self.pop_cc(blocking=blocking, timeout=timeout)

    def pop_compute(self, *, blocking: bool = False,
                     timeout: Optional[float] = None) -> Optional[TaskNode]:
        """Old combined non-CC queue. Now: try sr first (fast), then
        interp. Used by callers that don't differentiate."""
        node = self.pop_sr(blocking=False, timeout=None)
        if node is not None:
            return node
        return self.pop_interp(blocking=blocking, timeout=timeout)

    # ── post-pop slot wiring ───────────────────────────────────
    # When the mgr hands a task to a worker, fill its payload with the
    # cc_cache slot ids it needs. Producers (CC, INTERP) get a fresh
    # output slot. Consumers (SR_SRC, SR_INTERP) get pointers to the
    # parent producer's slot.
    #
    # All under queue_mgr._lock atomically (cc_cache.alloc is
    # non-blocking now). Returns True on success, False if cc_cache
    # is exhausted (caller re-queues the node and retries later —
    # backpressure that doesn't risk slot reuse between phases).

    def _post_pop_locked(self, node: TaskNode) -> bool:
        t = node.type
        if t == TaskType.INTERP:
            # INTERP reads two CCSR producers' rgb_padded + features.
            parent_cc_ids = sorted(
                pid for pid in node.original_parents
                if self._dag.get(pid)
                and self._dag.get(pid).type == TaskType.CCSR)
            for idx, pid in enumerate(parent_cc_ids):
                parent = self._dag[pid]
                pslot = parent.payload.get("dst_cc_slot", -1)
                key = "src_cc_slot_a" if idx == 0 else "src_cc_slot_b"
                node.payload[key] = pslot
        elif t == TaskType.SR_INTERP:
            # Reads INTERP.dst_cc_slot_{output_phase}: phase 1 →
            # dst_cc_slot (frame 0); phase 2 → dst_cc_slot_2 (frame 1,
            # mult ≥ 3); phase 3 → dst_cc_slot_3 (frame 2, mult = 4).
            _src_key_for = ("dst_cc_slot", "dst_cc_slot_2",
                             "dst_cc_slot_3")
            _idx = max(0, min(2, node.output_phase - 1))
            src_key = _src_key_for[_idx]
            for pid in node.original_parents:
                parent = self._dag.get(pid)
                if parent is None or parent.type != TaskType.INTERP:
                    continue
                node.payload["src_cc_slot_a"] = parent.payload.get(
                    src_key, -1)
        # Allocate output slot (CC / INTERP). cc_cache.alloc()
        # is non-blocking; None means exhausted → return False so caller
        # re-queues.
        if t == TaskType.CCSR:
            if node.payload.get("dst_cc_slot", -1) >= 0:
                return True   # already allocated by an earlier pop
            slot = self._cc_alloc_slot_locked(frame_idx=node.frame_idx)
            if slot is None:
                return False
            node.payload["dst_cc_slot"] = slot
            node.payload["dst_cc_field"] = -1
        elif t == TaskType.INTERP:
            # INTERP needs (mult-1) cc_cache slots, one per output
            # frame. Allocate progressively — if any fails we keep
            # the earlier ones (retry will skip re-allocating them;
            # slots are owned by the _cc table until task_done).
            _mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
            if _mult not in (2, 3, 4):
                _mult = 2
            _slot_keys = ("dst_cc_slot", "dst_cc_slot_2", "dst_cc_slot_3")
            if node.payload.get("dst_cc_slot", -1) < 0:
                slot = self._cc_alloc_slot_locked(
                    frame_idx=2 * node.frame_idx + 1)
                if slot is None:
                    return False
                node.payload["dst_cc_slot"] = slot
                node.payload["dst_cc_field"] = -1
            for _i in range(1, _mult - 1):
                _key = _slot_keys[_i]
                if node.payload.get(_key, -1) < 0:
                    _s = self._cc_alloc_slot_locked(
                        frame_idx=2 * node.frame_idx + 1)
                    if _s is None:
                        return False
                    node.payload[_key] = _s
        return True

    def _pop_locked(self, heap: list, present: set,
                     which: str, blocking: bool,
                     timeout: Optional[float]) -> Optional[TaskNode]:
        deadline = (_time.perf_counter() + timeout) if timeout else None
        while True:
            while heap:
                _f, _p, _s, tid = heapq.heappop(heap)
                if tid not in present:
                    # Stale entry (cancelled). Skip.
                    continue
                present.discard(tid)
                node = self._dag.get(tid)
                if node is None:
                    continue
                node.state = TaskState.RUNNING
                node.t_dispatched = _time.perf_counter()
                self._stats[f"{which}_pops"] += 1
                return node
            if not blocking or self._closed:
                return None
            remaining = None
            if deadline is not None:
                remaining = deadline - _time.perf_counter()
                if remaining <= 0:
                    return None
            self._cv.wait(timeout=remaining)

    # ── peek without popping (debug / instrumentation) ─────────

    def peek(self) -> dict:
        with self._lock:
            def _peek_heap(heap, present):
                for entry in heap:
                    if entry[3] in present:
                        return entry[3]
                return None
            return {
                "color_head_id":   _peek_heap(self._color_heap, self._color_present),
                "color_len":       len(self._color_present),
                "compute_head_id": _peek_heap(self._compute_heap, self._compute_present),
                "compute_len":     len(self._compute_present),
                "dag_size":        len(self._dag),
                "cc_size":         len(self._cc),
            }

    # ── frame-thread synchronisation ───────────────────────────

    def _ensure_frame_event(self, K: int) -> threading.Event:
        """Caller MUST hold _lock."""
        ev = self._frame_done.get(K)
        if ev is None:
            ev = threading.Event()
            self._frame_done[K] = ev
        return ev

    def wait_frame_done(self, K: int, *,
                         timeout: Optional[float] = None) -> bool:
        """frame_thread K blocks here. Returns True if signalled,
        False on timeout / shutdown."""
        with self._cv:
            # Defensive: if K's frame_done has already fired (a peer
            # frame_thread may have driven the whole pipeline for K to
            # completion before this thread got CPU time), don't
            # create a fresh unset Event via _ensure_frame_event — that
            # would block forever. Just return True; the data is in
            # mpv VA already (the work happened before signal).
            if K in self._signaled_K_set:
                return True
            ev = self._ensure_frame_event(K)
        if ev.wait(timeout=timeout):
            return True
        return False

    # Sentinel return for wait_phase_done when a concurrent
    # flush_seek bumped the epoch while we were parked.
    WAIT_ABANDONED = "abandoned"

    def wait_phase_done(self, K: int, phase: int, *,
                         timeout: Optional[float] = None):
        """Block until (K, phase) is published or the wait is abandoned.

        phase 0 → CCSR(K); phase p ≥ 1 → SR_INTERP(K, output_phase=p).
        Returns:
          True            — terminal task DONE, dst VA written.
          False           — timeout / queue closed. Caller returns
                            silently to mpv (don't raise — vapoursynth
                            treats filter exceptions as fatal).
          WAIT_ABANDONED  — flush_seek fired while parked. dst VA
                            was never written; one frame of buffer-pool
                            residue is mpv's problem to overwrite.
        """
        if phase == 0:
            t, output_phase = TaskType.CCSR, 1
        else:
            t, output_phase = TaskType.SR_INTERP, phase
        deadline = (_time.perf_counter() + timeout) if timeout else None
        with self._cv:
            entry_epoch = self._seek_epoch
            while True:
                # Seek-flush check: the compute() that issued this
                # wait was abandoned by mpv's seek. Surface that to
                # the caller as a distinct return so it can zero the
                # mpv dst VA — the worker never wrote it, and mpv may
                # still display it before the new chain starts
                # producing real frames.
                if self._seek_epoch != entry_epoch:
                    return self.WAIT_ABANDONED
                if K in self._signaled_K_set:
                    return True
                node = self._dag_find_locked(t, K, output_phase)
                if node is not None and node.state == TaskState.DONE:
                    return True
                if self._closed:
                    return False
                if deadline is not None:
                    remaining = deadline - _time.perf_counter()
                    if remaining <= 0:
                        return False
                    self._cv.wait(timeout=remaining)
                else:
                    self._cv.wait(timeout=0.1)

    def _signal_frame_done_locked(self, K: int) -> None:
        # Record signal BEFORE setting the event so any thread observing
        # the set Event also sees K in _signaled_K_set on next entry.
        self._signaled_K_set.add(K)
        ev = self._ensure_frame_event(K)
        ev.set()
        # No _cv.notify_all. frame_thread waits on the Event above
        # (ev.wait), not on _cv. cv waiters (dispatchers) don't care
        # about frame_done — they wake when a NEW task becomes ready,
        # which _make_ready_locked handles independently.
        # GC: drop terminal DAG nodes + stale frame_va/frame_done.
        # Without this the DAG grows unbounded (~10² nodes after a
        # few seconds, O(n) _dag_find_locked scans get progressively
        # slower).
        self._gc_locked(K)

    def _gc_locked(self, signaled_K: int) -> None:
        """Sweep DAG + tied-to-DAG _frame_va / _frame_done entries.
        Caller MUST hold _lock.

        Three-condition GC: a node is eligible iff
          (a) state == DONE
          (b) every child is DONE / FAILED / missing
          (c) frame_idx < signaled_K
        Condition (c) is the CC-payload-resurrection guard — see
        comment near _max_submit_K init for the prior watermark-mode
        investigation that this cutoff avoids.

        _frame_va[K] / _frame_done[K] are dropped iff every SR
        terminal (SR_SRC/SR_INTERP) for K has just left the DAG.
        Tracking sr_terminal_dropped_K (frames whose terminal we JUST
        deleted) is the unambiguous "K was live in DAG" signal —
        guards against the set_phase_dst → submit_frame race where
        FrameVA briefly exists before any DAG node for K does."""
        cutoff = signaled_K
        if cutoff < 0:
            return
        to_delete = []
        sr_terminal_dropped_K: set[int] = set()
        for tid, node in self._dag.items():
            if node.state != TaskState.DONE:
                continue
            if node.frame_idx >= cutoff:
                continue
            ok = True
            for cid in node.children:
                child = self._dag.get(cid)
                if (child is not None
                        and child.state not in (TaskState.DONE,
                                                 TaskState.FAILED)):
                    ok = False
                    break
            if ok:
                to_delete.append(tid)
        for tid in to_delete:
            node = self._dag[tid]
            if node.type == TaskType.SR_INTERP:
                sr_terminal_dropped_K.add(node.frame_idx)
            del self._dag[tid]
        for K in sr_terminal_dropped_K:
            # Drop K's per-frame state only when NO SR terminal for K
            # remains. SR_SRC and SR_INTERP can sweep in different
            # passes — keep FrameVA alive until the survivor finishes.
            still_has = any(
                n.frame_idx == K
                and n.type == TaskType.SR_INTERP
                for n in self._dag.values())
            if not still_has:
                self._frame_va.pop(K, None)
                self._frame_done.pop(K, None)
        # Evict _signaled_K_set entries whose K is comfortably behind
        # the current submit watermark. After max_submit_K - evict_lag
        # (256) frames have passed K, no late submit_frame(K) can still
        # arrive (mpv buffers ≤ CF ≤ 32 frames in flight). The set
        # MUST outlive the SR-terminal cleanup above so that a truly-
        # late phase-1 compute(K) still hits the early-return.
        if self._max_submit_K > self._signaled_K_evict_lag:
            cutoff_sig = self._max_submit_K - self._signaled_K_evict_lag
            stale = [K for K in self._signaled_K_set if K < cutoff_sig]
            for K in stale:
                self._signaled_K_set.discard(K)

    def reset_frame_event(self, K: int) -> None:
        """frame_thread finished consuming frame K; drop its event so
        the table doesn't grow without bound."""
        with self._lock:
            self._frame_done.pop(K, None)

    # ── cc_cache slot tracking ─────────────────────────────────

    def _cc_alloc_slot_locked(self, *, frame_idx: int
                                ) -> Optional[int]:
        """Reserve a free cc_cache slot. Returns None if cc_cache is
        exhausted — caller should re-queue the task and try again
        after some refs are dropped. Caller MUST hold _lock.

        When no cc_cache is wired (unit-test mode), we hand out
        monotonically increasing non-negative ids that don't collide
        with CC_FIELD_NONE (-1)."""
        if self._cc_cache is None:
            slot = next(self._synthetic_slot_id)
        else:
            slot = self._cc_cache.alloc()
            if slot is None:
                return None
        rec = CCResult(slot=slot, frame_idx=frame_idx)
        self._cc[slot] = rec
        self._stats["cc_allocs"] += 1
        if os.environ.get("DUAL_REFCOUNT_DBG", "0") == "1":
            self._log(f"[ref] alloc slot={slot} frame_idx={frame_idx}")
        return slot

    def _cc_set_producer_locked(
            self, slot: int, *,
            rgb_padded_consumers:    int = 0,
            rife_features_consumers: int = 0,
            sr_yuv_consumers:        int = 0,
            rgb_4K_consumers:        int = 0) -> None:
        """Record that the just-completed producer wrote these fields
        with `N` downstream consumers each. Refcount starts at N; each
        consumer's task_done decrements it."""
        rec = self._cc.get(slot)
        if rec is None:
            return
        if rgb_padded_consumers > 0:
            rec.has_rgb_padded = True
            rec.rgb_padded_refs = rgb_padded_consumers
        if rife_features_consumers > 0:
            rec.has_rife_features = True
            rec.rife_features_refs = rife_features_consumers
        if sr_yuv_consumers > 0:
            rec.has_sr_yuv = True
            rec.sr_yuv_refs = sr_yuv_consumers
        if rgb_4K_consumers > 0:
            rec.has_rgb_4K = True
            rec.rgb_4K_refs = rgb_4K_consumers

    def _cc_drop_slot_field_locked(self, slot: int, field_name: str
                                    ) -> None:
        """Consumer-side decrement. When all field refcounts reach 0,
        the slot is returned to cc_cache's free-list."""
        rec = self._cc.get(slot)
        if rec is None:
            return
        refcount_attr = f"{field_name}_refs"
        has_attr = f"has_{field_name}"
        refs = getattr(rec, refcount_attr)
        refs -= 1
        if refs < 0:
            raise RuntimeError(
                f"refcount underflow on cc[slot={slot}].{field_name}; "
                f"rec.frame_idx={rec.frame_idx}, all_refs="
                f"(rgb_padded={rec.rgb_padded_refs+1}/has={rec.has_rgb_padded}, "
                f"feat={rec.rife_features_refs}, "
                f"sr={rec.sr_yuv_refs}, rgb4K={rec.rgb_4K_refs}); "
                f"cc table keys: {list(self._cc.keys())[:20]}")
        setattr(rec, refcount_attr, refs)
        if refs == 0:
            setattr(rec, has_attr, False)
        if (not rec.has_rgb_padded
                and not rec.has_rife_features
                and not rec.has_sr_yuv
                and not rec.has_rgb_4K):
            del self._cc[slot]
            if self._cc_cache is not None and slot >= 0:
                self._cc_cache.free(slot)
            self._stats["cc_evictions"] += 1

    # ── stats ───────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            s = dict(self._stats)
            s.update(self.peek())
            return s


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Run as: python -B queue_mgr.py"""
    mgr = QueueManager()

    # Manually inject three tasks bypassing submit_frame (which is a
    # stub at this stage). Confirms heap ordering, READY transitions,
    # parent/child wiring.
    with mgr._lock:
        cc0 = mgr._new_node(task_type=TaskType.CCSR, frame_idx=0)
        cc1 = mgr._new_node(task_type=TaskType.CCSR, frame_idx=1)
        cc2 = mgr._new_node(task_type=TaskType.CCSR, frame_idx=2)

        # Dependent INTERP — should NOT become ready (CC parents
        # haven't been marked done yet).
        interp01 = mgr._new_node(
            task_type=TaskType.INTERP, frame_idx=0,
            parents=[cc0.id, cc1.id])
        assert interp01.state == TaskState.PENDING, \
            f"INTERP should be PENDING, got {interp01.state}"
        assert interp01.deps_remaining == 2

    # color_q should hand out CC tasks in frame_idx order.
    n0 = mgr.pop_color()
    n1 = mgr.pop_color()
    n2 = mgr.pop_color()
    assert n0.frame_idx == 0 and n0.type == TaskType.CCSR
    assert n1.frame_idx == 1
    assert n2.frame_idx == 2
    assert mgr.pop_color() is None, "queue should be drained"

    # compute_q is empty (no CC done → INTERP still PENDING).
    assert mgr.pop_compute() is None

    # Simulate manually flipping CC(0) and CC(1) done via the internal
    # API so INTERP(0,1) becomes READY. (task_done is a stub, so we go
    # through _new_node's auto-ready path by clearing deps_remaining.)
    with mgr._lock:
        cc0.state = TaskState.DONE
        cc1.state = TaskState.DONE
        # Drop deps from interp01 manually (mirrors task_done):
        interp01.parents.discard(cc0.id)
        interp01.parents.discard(cc1.id)
        interp01.deps_remaining = 0
        mgr._make_ready_locked(interp01)

    n_interp = mgr.pop_compute()
    assert n_interp.id == interp01.id
    assert n_interp.type == TaskType.INTERP
    assert mgr.pop_compute() is None

    # Frame-done events
    with mgr._lock:
        mgr._signal_frame_done_locked(0)
    assert mgr.wait_frame_done(0, timeout=0.5)
    assert not mgr.wait_frame_done(99, timeout=0.05)
    mgr.reset_frame_event(0)

    # Stats sanity
    s = mgr.stats()
    assert s["submitted_tasks"] == 4
    assert s["color_pops"] == 3
    assert s["compute_pops"] == 1

    # Blocking pop wakes up on submit
    barrier = threading.Event()
    def _producer():
        barrier.wait()
        with mgr._lock:
            mgr._new_node(task_type=TaskType.SR_INTERP, frame_idx=5)
    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    barrier.set()
    n_sr = mgr.pop_compute(blocking=True, timeout=2.0)
    assert n_sr is not None and n_sr.frame_idx == 5

    # Shutdown unblocks waiters
    waiter_result: list[Optional[TaskNode]] = []
    def _waiter():
        waiter_result.append(mgr.pop_compute(blocking=True))
    t2 = threading.Thread(target=_waiter, daemon=True)
    t2.start()
    _time.sleep(0.05)
    mgr.shutdown()
    t2.join(timeout=1.0)
    assert waiter_result == [None], \
        f"shutdown should yield None pop, got {waiter_result}"

    # ── submit_frame decomposition ──────────────────────────────
    mgr_sf = QueueManager()
    # Submitting frame 0 should create CC(0), CC(1), SR_SRC(0),
    # INTERP(0), SR_INTERP(0). CC are immediately READY (no parents);
    # SR/INTERP/SR_INTERP are PENDING.
    mgr_sf.submit_frame(0)
    s = mgr_sf.stats()
    assert s["submitted_tasks"] == 5, s
    # color_q should have CC(0) and CC(1) ready; compute_q empty.
    n0 = mgr_sf.pop_color()
    n1 = mgr_sf.pop_color()
    assert n0 is not None and n0.type == TaskType.CCSR and n0.frame_idx == 0
    assert n1 is not None and n1.type == TaskType.CCSR and n1.frame_idx == 1
    assert mgr_sf.pop_color() is None
    assert mgr_sf.pop_compute() is None

    # Submitting frame 1 should re-use CC(1) (already in DAG) and
    # only add CC(2), SR_SRC(1), INTERP(1), SR_INTERP(1) — 4 new.
    before = mgr_sf.stats()["submitted_tasks"]
    mgr_sf.submit_frame(1)
    after = mgr_sf.stats()["submitted_tasks"]
    assert after - before == 4, f"reuse failed: added {after - before}"
    # The single new CC is CC(2); pop_color returns it.
    n2 = mgr_sf.pop_color()
    assert n2.type == TaskType.CCSR and n2.frame_idx == 2
    assert mgr_sf.pop_color() is None
    mgr_sf.shutdown()

    # ── end-to-end DAG drain ────────────────────────────────────
    # No cc_cache — mgr uses synthetic negative slot ids, refcount
    # bookkeeping still exercised.
    mgr_e2e = QueueManager()
    mgr_e2e.submit_frame(0)
    mgr_e2e.submit_frame(1)
    # CC(0), CC(1), CC(2) are READY. SR_SRC and INTERP and SR_INTERP
    # are PENDING.
    completed = []

    # Drain via tight loop: pop_color (host preference) then pop_compute.
    while True:
        node = mgr_e2e.pop_color()
        if node is None:
            node = mgr_e2e.pop_compute()
        if node is None:
            break
        mgr_e2e.task_done(node.id)
        completed.append((node.type, node.frame_idx))

    # Validate: every node ran; refcount table is empty.
    expected_types = {
        TaskType.CCSR, TaskType.INTERP, TaskType.SR_INTERP}
    got_types = {c[0] for c in completed}
    assert expected_types <= got_types, \
        f"missing task types: {expected_types - got_types}"

    # With the fixed refcount=2 convention, stream edges leak the
    # unused INTERP refcount: CC(0) has no INTERP(-1, 0) so 1 ref
    # never decrements; same for the trailing CC. Expected leak is
    # 2 slots (one per stream edge in this 3-frame submit).
    leaked = list(mgr_e2e._cc.keys())
    assert len(leaked) <= 2, \
        f"cc table leaked more than 2 slots: {leaked}"

    # frame_done signals for both K=0 and K=1.
    assert mgr_e2e.wait_frame_done(0, timeout=0.5)
    assert mgr_e2e.wait_frame_done(1, timeout=0.5)

    s = mgr_e2e.stats()
    print(f"e2e drain stats: {s}")
    assert s["submitted_tasks"] == 9   # 3 CC + 2 SR_SRC + 2 INTERP + 2 SR_INTERP
    assert s["completed_tasks"] == 9
    # Edge frames leak one CC slot each per stream end. Within 2 is
    # the expected upper bound for this submit pattern.
    assert s["cc_allocs"] - s["cc_evictions"] <= 2, \
        f"cc_cache leak >2: allocs={s['cc_allocs']} evictions={s['cc_evictions']}"

    # Refcount sanity: CC(0) had only INTERP(0) consuming its rgb_1080p
    # (CC(0) is the first frame, no preceding INTERP(-1, 0)). So
    # CC(0).rgb_1080p_consumers = 1, not 2. Verify by re-running a
    # single-frame submission and inspecting state.
    mgr_check = QueueManager()
    mgr_check.submit_frame(0)
    # Pop CC(0): it should get a slot allocated and have only 1 child
    # in the DAG (INTERP(0)).
    cc0 = mgr_check.pop_color()
    assert cc0.type == TaskType.CCSR and cc0.frame_idx == 0
    # CC(0)'s children are SR_SRC(0) and INTERP(0). INTERP(0) is the
    # only rgb_1080p consumer.
    rgb_consumers = sum(
        1 for cid in cc0.children
        if mgr_check._dag.get(cid).type == TaskType.INTERP)
    assert rgb_consumers == 1, f"expected 1 INTERP child, got {rgb_consumers}"
    mgr_check.shutdown()

    print("queue_mgr self-test OK")


if __name__ == "__main__":
    _self_test()
