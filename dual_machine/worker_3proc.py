"""3-process worker: rdma_proc + buffer_mgr_proc + compute_proc.

The same module hosts both the worker-side run_worker_3proc_pipeline
(called from worker.py) and HostMP — the host-side wrapper that spawns
three children with the same shm layout, mirroring the worker on the
host side. The shared slot-ring uses page-aligned per-slot src + dst
bundles backed by anonymous shm, registered as a CUDA-pinned MR for
RDMA and exposed to the compute_proc as zero-copy GPU tensor views.

Compute_proc's HANDLERS dispatch on task_type to:
  CCSR       — merged CC + SR_SRC (raw YUV → rgb_padded + features +
                 4K SR output). Host writes cc_cache + slot.dst; guest
                 writes slot.dst extended (rgb_padded + features +
                 4K SR) which host's guest_mp CQ thread memmoves into
                 host cc_cache.
  INTERP     — RIFE flownet on two CCSR producers' rgb_padded +
                 features → sr_yuv. Host: reads cc_cache, writes
                 cc_cache. Guest: reads slot.src, writes slot.dst.
  SR_INTERP  — FSRCNNX on the INTERP output sr_yuv → 4K SR (slot.dst).
                 Reuses _run_sr_src kernel (same fp32 normalized
                 layout).
"""
import ctypes
import errno
import mmap
import os
import struct
import sys

# Pyverbs import is deferred — rdma_proc is the only consumer.

# ──────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────

# Per-slot metadata. All fields are read/written from BOTH processes
# via shm; use ctypes for explicit layout. 128 B (two cache lines)
# leaves room for split-task fields while still preventing
# false-sharing across adjacent slots.

# Shared 3-proc scaffolding (layout, shm, eventfd, MPPipelineBase).
from mp_pipeline import *  # noqa: F401,F403
# Underscore-prefixed helpers aren't picked up by `import *`; re-export
# explicitly so existing `from worker_3proc import _align_up, ...` calls
# (host_3proc, etc) keep working.
from mp_pipeline import (  # noqa: F401
    _align_up, _setup_logging, _parse_layout_from_env, _ef_from_env,
    _PAGE, _WR3_RECV_SRC, _WR3_SEND_DST, _WR3_SEND_MID,
)


def rdma_proc_main():
    """RDMA process: owns the QP, polls CQ in a hot loop. Owns no GPU
    work. Reads slot states from shm, drives state transitions via
    NIC completions, signals buffer_mgr on every transition.

    Two threads in this process:
      - Main: polls CQ in C (releases GIL); reacts to RECV / SEND
        completions.
      - send_handler: blocks on ef_mgr_to_rdma; when notified, scans
        shm for DST_READY slots and post_send's them.
    """
    import threading
    log = _setup_logging("rdma")
    layout = _parse_layout_from_env()
    shm = SlotRingShm(layout, name=os.environ["WMP_SHM_NAME"])
    shm.open_and_map()
    # Page-fault every page so MR registration sees resident pages.
    ctypes.memset(shm.addr, 0, layout.total_size)

    # RDMA setup. We import here so the parent's pyverbs context (if
    # any) isn't inherited.
    sys.path.insert(0, "/usr/lib/python3/dist-packages")
    from rdma_transport import RDMAContext, RDMAChannel
    import pyverbs.enums as e

    rdma_dev = os.environ.get("WMP_RDMA_DEV", "rocep1s0f0")
    rdma_port = int(os.environ.get("WMP_RDMA_PORT", "29900"))
    rdma_gid = int(os.environ.get("WMP_RDMA_GID", "3"))
    # Gating env for early-delivery wire. When set, multi-stage
    # producers emit a separate mid SEND followed by a FULL-range
    # SEND.
    _mid_delivery_enabled = os.environ.get(
        "DUAL_MID_DELIVERY", "0") == "1"
    log(f"opening RDMA ctx dev={rdma_dev} gid={rdma_gid}"
        f"{' [mid-delivery=on]' if _mid_delivery_enabled else ''}")
    ctx = RDMAContext(dev_name=rdma_dev, port=1, gid_index=rdma_gid,
                       max_cqe=max(64, 8 * layout.n_slots))

    access = (e.IBV_ACCESS_LOCAL_WRITE |
              e.IBV_ACCESS_REMOTE_WRITE |
              e.IBV_ACCESS_REMOTE_READ)
    mr = shm.register_rdma_mr(ctx.pd, access)
    log(f"MR registered: lkey={mr.lkey} rkey={mr.rkey}")

    # Listen for host. RDMAChannel(is_server=True) blocks on accept.
    log(f"listening on port {rdma_port} for host connection…")
    ch = RDMAChannel(ctx, is_server=True, peer_ip="", peer_port=rdma_port,
                      max_wr=max(64, 8 * layout.n_slots))
    log(f"connected. my_qpn={ch.my_qpn} peer_qpn={ch.peer_qpn}")

    # Parse the host's slot-info blob (extra_payload from the TCP
    # handshake). Format: u32 n_slots, then n_slots × (u64 addr, u32
    # rkey). These are the WRITE_WITH_IMM target addresses on the host
    # for the dst bundle of each slot.
    import struct as _struct
    host_dst_remote = []  # list of (addr, rkey)
    if ch.peer_extra:
        peer_n = _struct.unpack_from("!I", ch.peer_extra, 0)[0]
        off = 4
        for _s in range(peer_n):
            addr, rkey = _struct.unpack_from("!QI", ch.peer_extra, off)
            host_dst_remote.append((addr, rkey))
            off += 12
        log(f"received host_dst_remote: {peer_n} slots")
    # Host's guest_mp n_slots may differ from worker's layout.n_slots.
    # That's fine — host_slot in the request header always indexes
    # into the host's own table, so as long as host_dst_remote covers
    # every host_slot value, the worker can route the response
    # correctly.
    if not host_dst_remote:
        log("WARN: peer_extra empty; WRITE_WITH_IMM disabled")
        host_dst_remote = None

    # We intentionally don't consume ef_to_mgr / ef_from_mgr in the
    # hot path. rdma_proc polls shm directly for DST_READY slots
    # (compute_proc sets them), and compute_proc polls shm for
    # RECV_DONE slots (rdma_proc sets them). No eventfd RTT between
    # them. mgr is kept around for future scheduling/instrumentation
    # but is no longer in the critical path.
    ef_to_mgr = _ef_from_env("rdma_to_mgr")
    ef_from_mgr = _ef_from_env("mgr_to_rdma")

    # Per-slot lock to avoid races on state transitions (only matters
    # between the two threads in THIS process; cross-process slot
    # ownership is enforced by the state machine itself).
    state_lock = threading.Lock()

    # Pre-post recv on all slots.
    for i in range(layout.n_slots):
        addr = shm.slot_src_addr(i)
        ch.post_recv_at(addr, layout.src_size, mr.lkey,
                         wr_id=_WR3_RECV_SRC | i)
        with state_lock:
            meta = shm.slot_state(i)
            meta.state = ST_RECV_PENDING
            del meta
    log(f"pre-posted recv on all {layout.n_slots} slots; ready")

    stopping = threading.Event()

    # RDMA timing instrumentation:
    #   _post_send_ts[i]      = time post_send was issued for slot i
    #   _recv_done_ts[i]      = time of last RECV CQ completion for slot i
    #   _rdma_stats[task_type]= dict of accumulated timings per task type
    import time as _rdma_time
    _rdma_prof = os.environ.get("DUAL_RDMA_PROF", "0") == "1"
    _post_send_ts = [0.0] * layout.n_slots
    _recv_done_ts = [0.0] * layout.n_slots
    _send_task_type = [0] * layout.n_slots   # task_type at post_send time
    _rdma_stats: dict[int, dict] = {}
    _rdma_stats_lock = threading.Lock()
    _rdma_last_report = _rdma_time.perf_counter()
    _rdma_report_interval = float(os.environ.get(
        "DUAL_RDMA_PROF_INTERVAL_S", "2.0"))

    def _rdma_stats_add(tt: int, key: str, val_ms: float):
        with _rdma_stats_lock:
            d = _rdma_stats.setdefault(tt, {"send_ms": 0.0, "send_n": 0,
                                             "rtt_ms":  0.0, "rtt_n":  0})
            d[key] += val_ms
            d[key.replace("_ms", "_n")] += 1

    def _rdma_stats_maybe_report():
        nonlocal _rdma_last_report
        now = _rdma_time.perf_counter()
        if now - _rdma_last_report < _rdma_report_interval:
            return
        wall = (now - _rdma_last_report) * 1000
        _rdma_last_report = now
        with _rdma_stats_lock:
            parts = []
            for tt, d in sorted(_rdma_stats.items()):
                name = TT_NAMES.get(tt, str(tt))
                if d["send_n"] > 0:
                    s_avg = d["send_ms"] / d["send_n"]
                else:
                    s_avg = 0.0
                if d["rtt_n"] > 0:
                    r_avg = d["rtt_ms"] / d["rtt_n"]
                else:
                    r_avg = 0.0
                parts.append(
                    f"{name}: send={s_avg:.2f}ms(n={d['send_n']}) "
                    f"recv-rtt={r_avg:.2f}ms(n={d['rtt_n']})")
                # reset
                d["send_ms"] = 0.0; d["send_n"] = 0
                d["rtt_ms"]  = 0.0; d["rtt_n"]  = 0
            if parts:
                log(f"RDMA-PROF wall={wall:.0f}ms — " + " | ".join(parts))

    def _send_handler():
        """Hot busy-poll shm for {MID,DST}_READY → post_send. This thread
        is the critical-path equivalent of NCCL's "kernel-queued send"
        — every ms we delay here adds ms of host round-trip latency.
        We burn one core (Grace has 16) to minimize the gap. Main
        thread is in pyverbs C (cq.poll releases GIL) so it isn't
        blocked by our spin.

        Multi-stage producers (CCSR; INTERP mult ≥ 3) write a mid
        SEND covering slot.dst[mid_off:mid_off+mid_size] as soon as
        stage A is NIC-visible (ST_MID_READY), then a final SEND
        covering slot.dst[full_off:full_off+full_size] at ST_DST_READY.
        Single-stage tasks (mid_size==0) skip the mid SEND and emit
        one SEND covering the whole slot.dst.
        """
        # WRITE_WITH_IMM imm encoding (must match guest_mp._imm_encode):
        # bits 0-15 = slot, bits 16-19 = stage_code (4 bits).
        #   stage_code 0..14 = mid stage_idx
        #   stage_code 15    = final (terminal write, fires task_done)
        _STAGE_FINAL = 15
        def _imm(slot: int, stage_code: int) -> int:
            return (slot & 0xFFFF) | ((stage_code & 0xF) << 16)
        # Debug: count WRITEs per (slot, stage_code) to verify the
        # N-stage protocol on host-slot basis. Enable via
        # DUAL_STAGE_COUNT_DBG=1; dump on bench exit.
        _stage_count_dbg = os.environ.get("DUAL_STAGE_COUNT_DBG", "0") == "1"
        _stage_counts: dict[tuple[int, int], int] = {}
        try:
            while not stopping.is_set():
                for i in range(layout.n_slots):
                    with state_lock:
                        meta = shm.slot_state(i)
                        st = meta.state
                        if is_mid_ready_state(st):
                            # N-stage: stage_idx is encoded directly in
                            # the state value (st - ST_MID_READY) so
                            # the read is a single atomic uint32 load —
                            # no race with pending_stage_idx (which is
                            # written separately on ARM64 weak memory).
                            stage_idx = mid_ready_stage(st)
                            mid_off = int(meta.mid_off)
                            mid_size = int(meta.mid_size)
                            host_slot = int(meta.host_slot)
                            tt = int(meta.task_type)
                            meta.state = ST_MID_SENT
                            del meta
                            if tt == TT_INTERP and stage_idx > 0:
                                stage_off  = stage_idx * mid_size
                                stage_size = mid_size
                            else:
                                stage_off  = mid_off
                                stage_size = mid_size
                            if (stage_size > 0 and host_dst_remote is not None
                                    and host_slot < len(host_dst_remote)):
                                raddr, rkey = host_dst_remote[host_slot]
                                ch.post_write_with_imm(
                                    src_addr=shm.slot_dst_addr(i) + stage_off,
                                    src_lkey=mr.lkey,
                                    length=stage_size,
                                    remote_addr=raddr + stage_off,
                                    rkey=rkey,
                                    imm=_imm(host_slot, stage_idx),
                                    wr_id=_WR3_SEND_MID | i)
                                if _stage_count_dbg:
                                    k = (host_slot, stage_idx)
                                    _stage_counts[k] = _stage_counts.get(k, 0) + 1
                                    if sum(_stage_counts.values()) % 30 == 0:
                                        log(f"[stage_dbg] counts: {dict(sorted(_stage_counts.items()))}")
                            continue
                        if st != ST_DST_READY:
                            del meta
                            continue
                        meta.state = ST_SEND_PENDING
                        tt_for_send = int(meta.task_type)
                        mid_size_l = int(meta.mid_size)
                        full_off_l = int(meta.full_off)
                        full_size_l = int(meta.full_size)
                        host_slot = int(meta.host_slot)
                        del meta
                    if _rdma_prof:
                        _post_send_ts[i] = _rdma_time.perf_counter()
                        _send_task_type[i] = tt_for_send
                    if (host_dst_remote is None
                            or host_slot >= len(host_dst_remote)):
                        log(f"WRITE_WITH_IMM disabled or bad host_slot="
                            f"{host_slot}; worker slot {i} dropped")
                        continue
                    raddr, rkey = host_dst_remote[host_slot]
                    # N-stage: when at least one mid was sent (mid_size > 0
                    # AND mid delivery enabled), the final WRITE covers only
                    # full_off..full_off+full_size (the final-stage bytes).
                    # Otherwise write the whole slot.dst at offset 0.
                    if _mid_delivery_enabled and mid_size_l > 0:
                        ch.post_write_with_imm(
                            src_addr=shm.slot_dst_addr(i) + full_off_l,
                            src_lkey=mr.lkey,
                            length=full_size_l,
                            remote_addr=raddr + full_off_l,
                            rkey=rkey,
                            imm=_imm(host_slot, _STAGE_FINAL),
                            wr_id=_WR3_SEND_DST | i)
                        if _stage_count_dbg:
                            k = (host_slot, _STAGE_FINAL)
                            _stage_counts[k] = _stage_counts.get(k, 0) + 1
                            if sum(_stage_counts.values()) % 30 == 0:
                                log(f"[stage_dbg] counts: {dict(sorted(_stage_counts.items()))}")
                    else:
                        ch.post_write_with_imm(
                            src_addr=shm.slot_dst_addr(i),
                            src_lkey=mr.lkey,
                            length=layout.dst_size,
                            remote_addr=raddr,
                            rkey=rkey,
                            imm=_imm(host_slot, _STAGE_FINAL),
                            wr_id=_WR3_SEND_DST | i)
        except Exception as ex:
            log(f"send_handler fatal: {type(ex).__name__}: {ex}")
            import traceback; log(traceback.format_exc())

    send_t = threading.Thread(target=_send_handler, daemon=True,
                                name="rdma_proc-send")
    send_t.start()

    # Main loop: poll CQ.
    n_recv = 0
    n_send = 0
    try:
        while not stopping.is_set():
            try:
                wr_id = ch.poll_cq_blocking()
            except Exception:
                if stopping.is_set():
                    break
                raise
            op = wr_id & 0xFFFF0000
            s = wr_id & 0xFFFF
            if op == _WR3_RECV_SRC:
                # NIC delivered src into slot s. Read the header
                # eagerly so compute_proc has pair_k + task_type in
                # SlotMeta and doesn't need to re-read the (volatile)
                # src bundle header during dispatch.
                hdr_off = shm.layout.slot_src_off(s)
                # zc header is 16 int64 starting at off 0 of src bundle.
                # Indices match rdma_transport._HDR_*.
                hdr_ptr = (ctypes.c_int64 *
                            16).from_address(shm.addr + hdr_off)
                tt = ctypes.c_uint32(hdr_ptr[10]).value
                tid = ctypes.c_uint64(hdr_ptr[11]).value
                # _HDR_HOST_SLOT = 4 — host's guest_mp slot index for
                # this request, needed for WRITE_WITH_IMM response.
                host_slot = ctypes.c_uint32(hdr_ptr[4]).value
                # cc_cache wire fields. RDMA-READ back-fill of
                # slot.src from host's cc_cache region is not
                # implemented — we refuse any task that would require
                # it (cc_cache slot set on a non-guest-mode worker)
                # so a misrouted dispatch fails loudly instead of
                # silently consuming garbage.
                cc_slot_a = int(hdr_ptr[12])
                cc_slot_b = int(hdr_ptr[13])
                cc_field  = int(hdr_ptr[14])
                # Header field 15 = INTERP's second dst cc_cache slot
                # (mult ≥ 3 second frame). -1 or 0 means single-frame
                # INTERP (mult=2).
                _dst_cc_slot_2 = int(hdr_ptr[15])
                # The worker is guest-side by construction — cc_slot_*
                # fields in the wire header are HOST-side bookkeeping
                # for the mpv-side cc_cache; the worker doesn't read
                # them since the payload is already inline in slot.src.
                if _rdma_prof:
                    # RTT = previous send-CQ → this recv-CQ for THIS slot.
                    # Includes host's processing time + 2 wire hops.
                    # First cycle has no prior send → skip.
                    _now = _rdma_time.perf_counter()
                    if _recv_done_ts[s] != 0.0:
                        # _recv_done_ts[s] was overwritten in send CQ
                        # path to "post_send completion time". Now we
                        # are at "next recv-done". Difference = RTT.
                        rtt_ms = (_now - _recv_done_ts[s]) * 1000
                        _rdma_stats_add(tt, "rtt_ms", rtt_ms)
                    _recv_done_ts[s] = _now
                with state_lock:
                    meta = shm.slot_state(s)
                    meta.pair_k = hdr_ptr[0]
                    meta.src_a_idx = hdr_ptr[2]
                    meta.src_b_idx = hdr_ptr[3]
                    meta.task_type = tt
                    meta.task_id   = tid
                    meta.host_slot = host_slot
                    # cc_cache fields propagated into SlotMeta so
                    # compute_proc can find its cc_cache slots.
                    meta.src_cc_slot_a = cc_slot_a
                    meta.src_cc_slot_b = cc_slot_b
                    meta.src_cc_field  = cc_field
                    meta.dst_cc_slot   = -1   # filled by mgr-supplied wire field once it exists
                    meta.dst_cc_field  = -1
                    # Second cc_cache slot for INTERP mult ≥ 3's
                    # frame 2 (host-side address). Worker writes back
                    # via dst_cc_slot_2 in addition to dst_cc_slot.
                    meta.dst_cc_slot_2 = _dst_cc_slot_2
                    meta.state = ST_RECV_DONE
                    meta.n_recv += 1
                    del meta
                ef_to_mgr.notify()
                n_recv += 1
            elif op == _WR3_SEND_MID:
                # Mid SEND drained. Slot is still in flight for the
                # FULL SEND; no recv to repost yet. Account for n_send
                # so RDMA profiling stays accurate.
                n_send += 1
                continue
            elif op == _WR3_SEND_DST:
                # NIC drained dst from slot s; can repost recv.
                if _rdma_prof and _post_send_ts[s] != 0.0:
                    _send_ms = (_rdma_time.perf_counter() - _post_send_ts[s]) * 1000
                    _rdma_stats_add(_send_task_type[s], "send_ms", _send_ms)
                    # Reuse _recv_done_ts as "last send-done for slot s"
                    # for the RTT measurement above on the next recv.
                    _recv_done_ts[s] = _rdma_time.perf_counter()
                addr = shm.slot_src_addr(s)
                ch.post_recv_at(addr, layout.src_size, mr.lkey,
                                 wr_id=_WR3_RECV_SRC | s)
                with state_lock:
                    meta = shm.slot_state(s)
                    meta.state = ST_RECV_PENDING
                    meta.n_send += 1
                    del meta
                # Notify mgr so it can wake compute — it may have new
                # work pending that we couldn't get to during the
                # previous batch. Keeping this notify is cheap and
                # avoids stalls where compute_proc is sleeping despite
                # a freshly-arrived RECV_DONE in shm.
                ef_to_mgr.notify()
                n_send += 1
                if n_send % 60 == 0:
                    log(f"recv={n_recv} send={n_send} (steady)")
                if _rdma_prof:
                    _rdma_stats_maybe_report()
            else:
                log(f"unknown wr_id=0x{wr_id:x}")
    except Exception as ex:
        log(f"main poll fatal: {type(ex).__name__}: {ex}")
        import traceback; log(traceback.format_exc())
    finally:
        stopping.set()
        ef_from_mgr.notify()  # wake send_handler to exit
        log(f"exiting. recv={n_recv} send={n_send}")


def buffer_mgr_proc_main():
    """Buffer manager: select()s on ef_from_rdma and ef_from_compute,
    forwards events to the corresponding downstream. For v1 this is
    near-trivial — could be folded into rdma_proc later, but the
    separation makes the state machine easier to reason about and
    keeps a clean place to add scheduling later."""
    import select
    log = _setup_logging("mgr")
    ef_from_rdma = _ef_from_env("rdma_to_mgr")
    ef_from_compute = _ef_from_env("compute_to_mgr")
    ef_to_compute = _ef_from_env("mgr_to_compute")
    ef_to_rdma = _ef_from_env("mgr_to_rdma")
    log("started")
    n_rdma = 0
    n_compute = 0
    try:
        while True:
            rlist, _, _ = select.select(
                [ef_from_rdma.fd, ef_from_compute.fd], [], [])
            for fd in rlist:
                v = os.eventfd_read(fd)
                if fd == ef_from_rdma.fd:
                    # rdma transitioned a slot (RECV_DONE or post-SEND).
                    # In either case, compute *might* have new work —
                    # cheap to just poke it.
                    ef_to_compute.notify()
                    n_rdma += 1
                else:
                    # compute finished a slot. Tell rdma to post_send.
                    ef_to_rdma.notify()
                    n_compute += 1
                    if n_compute % 60 == 0:
                        log(f"rdma_evts={n_rdma} compute_evts={n_compute}")
    except Exception as ex:
        log(f"fatal: {type(ex).__name__}: {ex}")


def compute_proc_main():
    """Compute process: owns the GPU. Loads RIFE + FSRCNNX engines,
    pre-warms, signals 'ready' to the orchestrator, then enters the
    compute loop: wait on ef_from_mgr, claim a RECV_DONE slot, run
    compute_rife_interp + runner.forward, write result to slot's dst
    region, mark DST_READY, notify mgr."""
    import time as _time
    log = _setup_logging("compute")
    layout = _parse_layout_from_env()
    shm = SlotRingShm(layout, name=os.environ["WMP_SHM_NAME"])
    shm.open_and_map()
    # Register cuda for THIS process (independent of rdma_proc's
    # registration in its own ctx).
    shm.register_cuda_pinned(mapped=True)
    log(f"shm addr=0x{shm.addr:x} dev=0x{shm.cuda_device_ptr():x}")

    # Set up torch / cuda. Imports are deferred so they don't run
    # until compute_proc actually starts (engine load takes ~1-2s).
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    # Force cuDNN to pick deterministic algorithms — must match the
    # single-process worker's environment so output is bit-identical
    # across the two paths.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"cuda device set: {dev}")

    # Reconstitute the same paths that worker.py sets up so we can
    # import the engine builders. Use __file__ rather than a hard-coded
    # install path — works for both the secondary box (where install.sh
    # drops files under ~/.local/share/dgxspark-mpv/worker/) and the
    # host (where it's the project's dual_machine/ subdir). Without
    # insert(0,...) Python's `-c -B`
    # mode puts the parent's cwd at sys.path[0], which on the host
    # ends up shadowing dual_machine/vs_gpu_helpers.py with the stale
    # top-level copy at `<project>/vs_gpu_helpers.py`.
    # fsrcnnx-cudnn bundle (matches sr_keys_helper.py / worker.py).
    # `~/.config/mpv/fsrcnnx-cudnn/` is shipped by install.sh.
    from pathlib import Path as _PPath
    _mpv_home = _PPath(
        os.environ.get("MPV_HOME") or
        (os.environ.get("XDG_CONFIG_HOME") or
         os.path.expanduser("~/.config")) + "/mpv"
    )
    _fsrcnnx_bundle = _mpv_home / "fsrcnnx-cudnn"
    if _fsrcnnx_bundle.exists() and str(_fsrcnnx_bundle) not in sys.path:
        sys.path.insert(0, str(_fsrcnnx_bundle))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Read compute config from env (set by orchestrator from handshake).
    H = layout.H
    W = layout.W
    cH = layout.cH
    cW = layout.cW
    out_h = layout.oH
    out_w = layout.oW
    oc_h = layout.ocH
    oc_w = layout.ocW
    variant = os.environ["WMP_VARIANT"]
    rife_model = os.environ["WMP_RIFE_MODEL"]
    matrix_s = os.environ["WMP_MATRIX_S"]
    color_range = os.environ["WMP_COLOR_RANGE"]
    chroma_mode = os.environ["WMP_CHROMA_MODE"]
    bits = int(os.environ.get("WMP_BITS", "10"))
    downsample_pre = int(os.environ.get("WMP_DOWNSAMPLE_PRE", "1"))
    MAX = float((1 << bits) - 1)
    # Processing dimensions = source dimensions after the optional
    # pre-CC luma downsample. downsample_pre=1 -> identity (proc==src);
    # =2 -> 4K source path, RIFE/FSRCNNX runs at 1080p internal.
    proc_h = H // downsample_pre
    proc_w = W // downsample_pre
    proc_cH = cH // downsample_pre
    proc_cW = cW // downsample_pre

    interp_kw = dict(size=(oc_h, oc_w), mode=chroma_mode)
    if chroma_mode != "nearest":
        interp_kw["align_corners"] = False

    # Build FSRCNNX runner. Input dim = proc dims (source after optional
    # luma downsample). Output dim = proc * variant scale.
    # DUAL_NO_SR=1 (set by orchestrator from handshake) → skip runner
    # build entirely; _run_ccsr / _run_sr_interp take a no-SR branch
    # that copies raw Y to ov["yao"] instead.
    no_sr = os.environ.get("DUAL_NO_SR", "0") == "1"
    from fsrcnnx_cudnn.model.cudnn_runner import FSRCNNXcuDNNRunner
    # chroma_krig import retained for future Step 3 retry — the
    # precompile() in the warmup hook below makes the JIT compile a
    # one-shot cost so reactivation is cheap once the in-pipeline race
    # (krig output → 0 on ~10% of frames; see project memory) is fixed.
    from fsrcnnx_cudnn.chroma_krig import krig_bilateral_chroma  # noqa: F401
    WEIGHTS_DIR = os.environ.get(
        "WMP_WEIGHTS_DIR", "/home/spark/dual_machine/weights")
    if no_sr:
        log(f"DUAL_NO_SR=1 — skipping FSRCNNX runner build "
            f"(variant={variant} would have been {proc_w}x{proc_h})")
        runner = None
    else:
        log(f"building cuDNN runner: {variant} {proc_w}x{proc_h} "
            f"(src={W}x{H}, downsample_pre={downsample_pre}, "
            f"weights={WEIGHTS_DIR})")
        t0 = _time.time()
        runner = FSRCNNXcuDNNRunner(
            f"{WEIGHTS_DIR}/{variant}.npz", variant,
            H=proc_h, W=proc_w, device=dev)
        log(f"runner built in {(_time.time()-t0)*1000:.0f}ms")

    # Build RIFE engines at proc dims (1080p for the 4K-source path).
    import vs_gpu_helpers as vgh
    log(f"loading vsrife engines model={rife_model} {proc_w}x{proc_h}…")
    t0 = _time.time()
    rife_cfg = vgh._load_engines(rife_model, proc_h, proc_w, 1.0, True, dev)
    log(f"vsrife engines loaded in {(_time.time()-t0)*1000:.0f}ms")

    # Direct CUDA tensor views into shm via __cuda_array_interface__.
    # On Grace SoC, shm is cuda-pinned + mapped (via cudaHostRegister
    # in this proc's ctx); host_ptr == device_ptr, so the same bytes
    # are accessible from GPU at NVLink-C2C bandwidth. The NIC writes
    # the recv bundle into shm; compute kernels read the bundle right
    # out of shm — no HtoD cudaMemcpy. Same for the result direction:
    # compute kernels write into shm; the NIC reads it directly — no
    # d2h cudaMemcpy. compute_proc becomes pure compute; data shuffle
    # is the responsibility of rdma_proc (which already only manages
    # the NIC and shm state).
    dev_base = shm.cuda_device_ptr()  # == shm.addr on Grace

    _DTYPE_TYPESTR = {
        torch.int16: "<i2",
        torch.int64: "<i8",
        torch.uint8: "|u1",
        torch.float16: "<f2",
    }

    def _make_cuda_views(off_in_shm: int, layout_list):
        """Build a dict of CUDA torch tensors that view the slot's
        sub-regions of shm. Holds the CAI shim objects alive so the
        tensors stay valid for the process lifetime."""
        views = {}
        cai_refs = []
        for name, off, nbytes, shape, dtype in layout_list:
            addr = dev_base + off_in_shm + off
            esize = torch.empty(0, dtype=dtype).element_size()
            nelems = nbytes // esize
            cai = type("_CAI", (), {})()
            cai.__cuda_array_interface__ = {
                "shape": (nelems,),
                "typestr": _DTYPE_TYPESTR[dtype],
                "data": (addr, False),
                "strides": None,
                "version": 3,
            }
            cai_refs.append(cai)
            t = torch.as_tensor(cai, device=dev).view(*shape)
            views[name] = t
        return views, cai_refs

    slot_src_views_gpu = []
    slot_dst_views_gpu = []
    _cai_keepalive = []  # must outlive every tensor view above
    for i in range(layout.n_slots):
        sv, sr = _make_cuda_views(layout.slot_src_off(i), layout.src_layout)
        dv, dr = _make_cuda_views(layout.slot_dst_off(i), layout.dst_layout)
        slot_src_views_gpu.append(sv)
        slot_dst_views_gpu.append(dv)
        _cai_keepalive.extend(sr)
        _cai_keepalive.extend(dr)
    log(f"built {layout.n_slots} per-slot zero-copy CUDA views "
        f"(input + output both in shm, no HtoD/DtoH cudaMemcpy)")

    # ── cc_cache ────────────────────────────────────────────────
    # Optional: when HMP_CC_CACHE_NAME is set, open the host's
    # cc_cache shm region, cuda-pin it in this process, and build
    # per-slot per-field GPU tensor views so CC / SR_SRC / INTERP /
    # SR_INTERP kernels can write/read directly without going through
    # slot.dst memcpy.
    cc_shm_local = None
    cc_layout_local = None
    # cc_views[slot_id] = {
    #   "rgb_padded":    (1, 3, pH, pW) fp16,
    #   "rife_features": (1, enc_ch, pH, pW) fp16,
    #   "sr_yuv":        {"y": (1,1,H,W) fp16, "u": (1,1,cH,cW) fp16,
    #                     "v": (1,1,cH,cW) fp16},
    #   "rgb_4K":        (1, 3, oH, oW) int16,
    # }
    # cc_views[slot_id]['sr_yuv']['y'/'u'/'v'] are fp16 views;
    # RIFE inputs (rgb_padded, rife_features) are fp16 (engine native).
    cc_views: list[dict] = []

    # _cai_tensor builds a torch GPU view on a raw cuda device pointer
    # via the __cuda_array_interface__ protocol. Defined out here (not
    # inside `if cc_name_env`) because the split_src_views path below
    # also calls it in guest mode.
    def _cai_tensor(addr, n_elems, typestr, shape):
        cai = type("_CAI", (), {})()
        cai.__cuda_array_interface__ = {
            "shape": (n_elems,), "typestr": typestr,
            "data": (addr, False), "strides": None, "version": 3,
        }
        _cai_keepalive.append(cai)
        return torch.as_tensor(cai, device=dev).view(*shape)

    cc_name_env = os.environ.get("HMP_CC_CACHE_NAME")
    if cc_name_env:
        from cc_cache import (
            CCCacheLayout, CCCacheShm,
            FIELD_RGB_PADDED, FIELD_RIFE_FEATURES, FIELD_SR_YUV,
            FIELD_RGB_INTERP, FIELD_RGB_4K)
        cc_layout_local = CCCacheLayout(
            n_slots=int(os.environ["HMP_CC_CACHE_NSLOTS"]),
            H_src=int(os.environ["HMP_CC_CACHE_H_SRC"]),
            W_src=int(os.environ["HMP_CC_CACHE_W_SRC"]),
            H_out=int(os.environ["HMP_CC_CACHE_H_OUT"]),
            W_out=int(os.environ["HMP_CC_CACHE_W_OUT"]),
            pH=int(os.environ["HMP_CC_CACHE_PH"]),
            pW=int(os.environ["HMP_CC_CACHE_PW"]),
            enc_channels=int(os.environ["HMP_CC_CACHE_ENC_CH"]),
            sub_h=int(os.environ["HMP_CC_CACHE_SUB_H"]),
            sub_w=int(os.environ["HMP_CC_CACHE_SUB_W"]),
        )
        cc_shm_local = CCCacheShm(cc_layout_local, name=cc_name_env)
        cc_shm_local.open()
        cc_shm_local.register_cuda_pinned()
        log(f"cc_cache opened in compute_proc: "
            f"{cc_layout_local.describe()}")
        cc_dev_base = cc_shm_local.cuda_device_ptr(0, FIELD_RGB_PADDED) \
            - cc_layout_local.field_off_in_slot(FIELD_RGB_PADDED)

        H_src = cc_layout_local.H_src
        W_src = cc_layout_local.W_src
        pH    = cc_layout_local.pH
        pW    = cc_layout_local.pW
        enc_ch = cc_layout_local.enc_channels
        cH_src = H_src >> cc_layout_local.sub_h
        cW_src = W_src >> cc_layout_local.sub_w
        H_out_cc = cc_layout_local.H_out
        W_out_cc = cc_layout_local.W_out

        for sid in range(cc_layout_local.n_slots):
            slot_base = cc_dev_base + sid * cc_layout_local.slot_stride
            entry = {}

            # rgb_padded: fp16 NCHW (1, 3, pH, pW)
            entry["rgb_padded"] = _cai_tensor(
                slot_base + cc_layout_local.off_rgb_padded,
                3 * pH * pW, "<f2", (1, 3, pH, pW))

            # rife_features: fp16 (1, enc_ch, pH, pW)
            entry["rife_features"] = _cai_tensor(
                slot_base + cc_layout_local.off_rife_features,
                enc_ch * pH * pW, "<f2", (1, enc_ch, pH, pW))

            # sr_yuv: fp16 Y (1,1,H,W) + U,V (1,1,cH,cW). Packed; each
            # plane is exposed as its own view so kernels write
            # independently. fp16 stores (int16_value / MAX) in [0, 1]
            # — same width as int16 but ~1 LSB precision loss vs the
            # 10-bit source. The loss propagates through FSRCNNX into
            # ~33 dB SR drift vs single-machine, which is accepted.
            sr_off = cc_layout_local.off_sr_yuv
            y_bytes  = H_src * W_src * 2
            uv_bytes = cH_src * cW_src * 2
            sr = {
                "y": _cai_tensor(
                    slot_base + sr_off,
                    H_src * W_src, "<f2", (1, 1, H_src, W_src)),
                "u": _cai_tensor(
                    slot_base + sr_off + y_bytes,
                    cH_src * cW_src, "<f2", (1, 1, cH_src, cW_src)),
                "v": _cai_tensor(
                    slot_base + sr_off + y_bytes + uv_bytes,
                    cH_src * cW_src, "<f2", (1, 1, cH_src, cW_src)),
            }
            entry["sr_yuv"] = sr

            # rgb_4K: fp16 NCHW (1, 3, H_out, W_out) — 4K source path only.
            entry["rgb_4K"] = _cai_tensor(
                slot_base + cc_layout_local.off_rgb_4K,
                3 * H_out_cc * W_out_cc, "<f2",
                (1, 3, H_out_cc, W_out_cc))

            # rgb_interp: fp16 NCHW (1, 3, proc_h, proc_w) — INTERP
            # writes RGB at proc dims (post-RIFE, post-need_pad crop);
            # SR_INTERP reads and does matrix + scale-aware chroma resize.
            # cc_cache layout reserves rgb_interp_size = 3·H_src·W_src·2
            # bytes (oversized for downsample_pre>1) — view at proc dims
            # is the actually-used sub-region.
            entry["rgb_interp"] = _cai_tensor(
                slot_base + cc_layout_local.off_rgb_interp,
                3 * proc_h * proc_w, "<f2",
                (1, 3, proc_h, proc_w))

            cc_views.append(entry)
        log(f"cc_cache: built per-slot GPU views for "
            f"{cc_layout_local.n_slots} slots × 5 fields")

    # ── split-task views into slot.src ──────────────────────────
    # Guest-mode INTERP / SR_SRC / SR_INTERP read input from slot.src
    # (host packs cc_cache field data inline with the wire request).
    # We build per-slot GPU views that overlay the right offsets for
    # each task-type.
    # Conditional on cc_shm_local is None: split tasks on host always
    # have a real cc_cache so the cc_views path is preferred there.
    #
    # Layout WITHIN slot.src (skipping the 128 B header at off 0):
    #
    #   For INTERP (task_type == TT_INTERP):
    #     [   128 ..    +rgb)   rgb_padded_a    (fp16, 1×3×pH×pW)
    #     [  +rgb ..    +feat)  rife_features_a (fp16, 1×enc_ch×pH×pW)
    #     [ +rgb+feat .. +2*rgb+feat)   rgb_padded_b    (fp16)
    #     [+2*rgb+feat .. +2*(rgb+feat)) rife_features_b (fp16)
    #
    #   For SR_SRC / SR_INTERP (task_type == TT_SR_SRC / TT_SR_INTERP):
    #     [   128 .. +H*W*2)          sr_y (fp16, 1×1×H×W)
    #     [+H*W*2 .. +H*W*2+cH*cW*2)  sr_u (fp16, 1×1×cH×cW)
    #     [...]                       sr_v
    #
    # Both layouts coexist on the same shm bytes — at any moment the
    # slot is processing one task type and only the corresponding
    # views have meaningful data. Output for these tasks lives in
    # slot.dst (SR_*: existing yao/uao/vao; INTERP: small region at
    # the start of dst, mirrored to host cc_cache.sr_yuv on read-back).
    split_src_views = None
    if cc_shm_local is None:
        # Guest mode: pull pH/pW/enc_ch off the layout itself — they
        # came in via the handshake and are the canonical values for
        # this session.
        pH     = layout.pH
        pW     = layout.pW
        enc_ch = layout.enc_ch
        HDR_BYTES = 16 * 8  # _ZC_HEADER_INT64 * sizeof(int64)
        rgb_padded_bytes  = 1 * 3      * pH * pW * 2
        rife_features_bytes = 1 * enc_ch * pH * pW * 2
        H_l, W_l = layout.H, layout.W
        downsample_pre_l = int(os.environ.get("WMP_DOWNSAMPLE_PRE", "1"))
        proc_h_l = H_l // downsample_pre_l
        proc_w_l = W_l // downsample_pre_l
        split_src_views = []
        for i in range(layout.n_slots):
            base = shm.addr + layout.slot_src_off(i) + HDR_BYTES
            interp_a_off = 0
            interp_a_feat = interp_a_off + rgb_padded_bytes
            interp_b_off = interp_a_feat + rife_features_bytes
            interp_b_feat = interp_b_off + rgb_padded_bytes
            entry = {
                # INTERP layout overlay
                "rgb_padded_a": _cai_tensor(
                    base + interp_a_off,  3 * pH * pW,
                    "<f2", (1, 3, pH, pW)),
                "features_a":   _cai_tensor(
                    base + interp_a_feat, enc_ch * pH * pW,
                    "<f2", (1, enc_ch, pH, pW)),
                "rgb_padded_b": _cai_tensor(
                    base + interp_b_off,  3 * pH * pW,
                    "<f2", (1, 3, pH, pW)),
                "features_b":   _cai_tensor(
                    base + interp_b_feat, enc_ch * pH * pW,
                    "<f2", (1, enc_ch, pH, pW)),
                # SR_INTERP overlay (same base offset as INTERP — only
                # one task type active at a time per slot). rgb_interp
                # at proc dims (= H/downsample_pre, W/downsample_pre).
                "rgb_interp": _cai_tensor(
                    base + 0, 3 * proc_h_l * proc_w_l,
                    "<f2", (1, 3, proc_h_l, proc_w_l)),
            }
            split_src_views.append(entry)
        log(f"split-task views: built {layout.n_slots} per-slot overlays "
            f"in slot.src for guest mode "
            f"(pH={pH} pW={pW} enc_ch={enc_ch})")

    # Per-slot views into slot.dst where INTERP writes its rgb_interp
    # output (fp16 RGB at proc dims). Host's guest_mp memmoves these
    # back into cc_cache.rgb_interp on the host side via _cc_writeback.
    #
    # Dense-pack: frame k at offset k * frame_sz. We always
    # materialise rgb/rgb2/rgb3 views (unused ones cost just a CAI
    # tensor header). Decoupled from CCSR's named regions so future
    # CCSR field changes don't move INTERP frame anchors.
    split_dst_interp_views = None
    if cc_shm_local is None:
        downsample_pre_l = int(os.environ.get("WMP_DOWNSAMPLE_PRE", "1"))
        proc_h_l = layout.H // downsample_pre_l
        proc_w_l = layout.W // downsample_pre_l
        _frame_sz_l = 3 * proc_h_l * proc_w_l * 2   # fp16 NCHW
        split_dst_interp_views = []
        for i in range(layout.n_slots):
            base = shm.addr + layout.slot_dst_off(i)
            split_dst_interp_views.append({
                "rgb": _cai_tensor(
                    base + 0, 3 * proc_h_l * proc_w_l,
                    "<f2", (1, 3, proc_h_l, proc_w_l)),
                "rgb2": _cai_tensor(
                    base + _frame_sz_l, 3 * proc_h_l * proc_w_l,
                    "<f2", (1, 3, proc_h_l, proc_w_l)),
                "rgb3": _cai_tensor(
                    base + 2 * _frame_sz_l, 3 * proc_h_l * proc_w_l,
                    "<f2", (1, 3, proc_h_l, proc_w_l)),
            })

    # Pre-allocated fp32 staging for the int16→fp32 cast (SR / chroma
    # bilinear all want fp32 in [0,1]). Without these, each per-pair
    # `src["ya"].float() * (1/MAX)` allocates a fresh fp32 cuda tensor;
    # the caching allocator does fine but the explicit per-slot
    # buffers make ownership obvious and let us do the cast as a
    # single int16→fp32 copy + in-place mul (instead of two implicit
    # intermediates from `.float()` and `*` operators).
    #
    # The output side keeps its existing pipeline (clamp → *MAX+0.5
    # → cast int16 → write to dst view); a similar pre-alloc there
    # would also tidy things up but matters even less since SR's
    # output goes straight to int16 anyway.
    fp32_buf = [
        {
            "ya": torch.empty(1, 1, H, W,   dtype=torch.float32, device=dev),
            "ua": torch.empty(1, 1, cH, cW, dtype=torch.float32, device=dev),
            "va": torch.empty(1, 1, cH, cW, dtype=torch.float32, device=dev),
            "yi": torch.empty(1, 1, H, W,   dtype=torch.float32, device=dev),
            "ui": torch.empty(1, 1, cH, cW, dtype=torch.float32, device=dev),
            "vi": torch.empty(1, 1, cH, cW, dtype=torch.float32, device=dev),
        }
        for _ in range(layout.n_slots)
    ]
    log(f"pre-allocated {layout.n_slots} per-slot fp32 staging buffers "
        f"(6 planes × {2*(H*W + 2*cH*cW)*4/1e6:.1f}MB total)")

    # Pre-warm: build engine graphs at the right shape so the first
    # real task doesn't pay the cuDNN/TRT heuristic-pick latency.
    # Drives the same forward paths the compute kernels (CCSR / INTERP)
    # take: RIFE encode + flownet on a padded RGB pair, plus FSRCNNX
    # runner on a Y plane.
    log("pre-warming engines…")
    t0 = _time.time()
    flownet           = rife_cfg["flownet"]
    encode            = rife_cfg["encode"]
    ph_pw             = (rife_cfg["ph"], rife_cfg["pw"])
    tenFlow_div       = rife_cfg["tenFlow_div"]
    backwarp_tenGrid  = rife_cfg["backwarp_tenGrid"]
    stream_inf        = rife_cfg["stream_inf"]
    lock_inf          = rife_cfg["lock_inf"]
    with torch.inference_mode(), lock_inf, torch.cuda.stream(stream_inf):
        img0 = torch.zeros(1, 3, ph_pw[0], ph_pw[1],
                            dtype=torch.float16, device=dev)
        img1 = img0
        timestep_t = torch.full(
            (1, 1, ph_pw[0], ph_pw[1]), 0.5,
            dtype=torch.float16, device=dev)
        if encode is not None:
            f0 = encode(img0); f1 = encode(img1)
            flownet(img0, img1, timestep_t,
                    tenFlow_div, backwarp_tenGrid, f0, f1)
        else:
            flownet(img0, img1, timestep_t,
                    tenFlow_div, backwarp_tenGrid)
    torch.cuda.default_stream().wait_stream(stream_inf)
    if runner is not None:
        with torch.inference_mode():
            # FSRCNNX runner pre-warm at proc dims (= H/downsample_pre).
            y_f = torch.zeros(1, 1, proc_h, proc_w,
                              dtype=torch.float32, device=dev)
            runner.forward(y_f).clamp_(0, 1)
            torch.cuda.synchronize(device=dev)
    # Compile the krig CUDA extension up-front (~18 s cold; no-op when
    # the torch_extensions cache already has the .so).
    try:
        from fsrcnnx_cudnn.chroma_krig import precompile as _krig_precompile
        _krig_precompile()
    except Exception as _ke:
        log(f"krig precompile failed: {type(_ke).__name__}: {_ke}")
    log(f"pre-warm done in {(_time.time()-t0)*1000:.0f}ms")

    # Signal main worker.py "ready" so it can tell host via NCCL.
    ef_ready = _ef_from_env("compute_ready")
    ef_ready.notify()
    log("signaled compute_ready")

    ef_to_mgr = _ef_from_env("compute_to_mgr")
    import time as _t_for_sleep
    import threading as _th
    import queue as _qu

    # Per-slot cuda event marks "the kernel that produced this slot's
    # dst is done". sync_thread waits on it (releases GIL) and marks
    # DST_READY so rdma_proc may post_send.
    #
    # N-stage: each slot has 1..MAX_MID_STAGES events for intermediate
    # stage signals plus one for the final. MAX_MID_STAGES bounds the
    # per-pair flownet count (= mult - 1 for INTERP); 8 is ample for
    # any realistic temporal multiplier.
    MAX_MID_STAGES = 8
    done_events = [torch.cuda.Event() for _ in range(layout.n_slots)]
    done_events_mid = [
        [torch.cuda.Event() for _ in range(MAX_MID_STAGES)]
        for _ in range(layout.n_slots)]
    # sync_q items: (slot, stage_idx_or_None).
    #   stage_idx is an int (>= 0) → intermediate mid stage_idx;
    #   None → final (terminal), legacy "full" semantics.
    sync_q: "_qu.Queue[tuple[int,int|None]|None]" = _qu.Queue()

    def _sync_worker():
        while True:
            item = sync_q.get()
            if item is None:
                return
            slot_idx, stage_idx = item
            is_final = stage_idx is None
            ev = (done_events[slot_idx] if is_final
                  else done_events_mid[slot_idx][stage_idx])
            try:
                ev.synchronize()
            except Exception as ex:
                log(f"sync_worker: event.sync slot={slot_idx} "
                    f"stage={stage_idx} failed: {ex}")
                continue
            # State machine: COMPUTING → MID_READY+k → MID_SENT →
            # MID_READY+(k+1) → MID_SENT → ... → DST_READY → SEND_PENDING
            # stage_idx is encoded in the state value itself
            # (state = ST_MID_READY + stage_idx) so the entire MID
            # transition is ONE atomic uint32 write — no separate
            # pending_stage_idx field race on ARM64.
            #
            # Wait condition for stage k>0 or is_final after mids:
            # state must NOT be MID_READY (i.e., send_handler advanced
            # past previous mid). Without this wait we overwrite a
            # MID_READY state before send_handler observes it → that
            # stage's WRITE never posts → host recv_buf bytes for
            # that frame stay zero-initialised → 8-17 dB PSNR.
            # Tight busy-spin (no sleep): send_handler transitions in
            # microseconds. Host-mode (cc_shm_local != None) has no
            # send_handler so we skip the spin (_hmp_watcher reads
            # MID_READY without flipping it).
            if is_final:
                if cc_shm_local is None:
                    while True:
                        meta = shm.slot_state(slot_idx)
                        if not is_mid_ready_state(meta.state):
                            del meta
                            break
                        del meta
                meta = shm.slot_state(slot_idx)
                meta.state = ST_DST_READY
                meta.n_compute += 1
                del meta
            elif stage_idx == 0:
                meta = shm.slot_state(slot_idx)
                meta.state = ST_MID_READY + 0
                del meta
            else:
                # stage k>0 (mult=4 only)
                if cc_shm_local is None:
                    while True:
                        meta = shm.slot_state(slot_idx)
                        if not is_mid_ready_state(meta.state):
                            del meta
                            break
                        del meta
                meta = shm.slot_state(slot_idx)
                meta.state = ST_MID_READY + int(stage_idx)
                del meta
            ef_to_mgr.notify()

    _sync_t = _th.Thread(target=_sync_worker, daemon=True,
                          name="compute-sync")
    _sync_t.start()

    # Gating env. When enabled both worker and host post the early-
    # delivery mid SEND (see DUAL_MID_DELIVERY plumbing in guest_mp.py).
    _mid_delivery_enabled = os.environ.get(
        "DUAL_MID_DELIVERY", "0") == "1"

    # Per-slot helper for N-stage producers to mark an intermediate
    # stage done. CCSR uses stage_idx=0 at the CC+features/SR boundary;
    # INTERP mult ≥ 3 uses stage_idx=0..mult-3 between flownets.
    # Single-stage handlers (INTERP mult=2, SR_INTERP) never call it.
    # No-op when mid delivery is disabled (slot goes COMPUTING →
    # DST_READY).
    def _mark_stage_done(cur: int, stage_idx: int):
        if not _mid_delivery_enabled:
            return
        assert 0 <= stage_idx < MAX_MID_STAGES, (
            f"_mark_stage_done: stage_idx={stage_idx} out of range")
        done_events_mid[cur][stage_idx].record(torch.cuda.current_stream())
        sync_q.put((cur, stage_idx))

    # Back-compat alias used by CCSR / older callers (stage 0 only).
    def _mark_mid_done(cur: int):
        _mark_stage_done(cur, 0)

    # Compute loop. With zero-copy shm views:
    #   * No HtoD step — compute reads src views (which alias shm)
    #     directly through NVLink-C2C.
    #   * No d2h step — compute writes dst views (which also alias
    #     shm) directly; the NIC reads the same bytes.
    #   * Only synchronisation needed: cuda Event after the final
    #     write kernel, so sync_thread knows when the NIC may read.
    # INTERP frame_sz for dense-pack stage byte ranges. Read env
    # mult here (compute_proc reads on each task too via
    # _interp_mult_cache, this is just for task_dst_ranges below).
    _compute_interp_mult = int(os.environ.get("DUAL_INTERP_MULT", "2"))
    if _compute_interp_mult not in (1, 2, 3, 4):
        _compute_interp_mult = 2
    # mult=1 = no_interp: only CCSR per pair, no INTERP/SR_INTERP. CCSR
    # in this mode skips rgb_padded + rife_features writes (no consumer).
    _compute_no_interp = (_compute_interp_mult == 1)
    _compute_frame_sz = 3 * proc_h * proc_w * 2   # fp16 NCHW
    REPORT_EVERY = 60
    tsum = {"claim_wait": 0.0, "compute": 0.0, "post": 0.0}
    tcount = 0
    n_done = 0
    # GPU idle instrumentation: pair of cuda events records when
    # each kernel finishes; the next iteration computes elapsed_ms from
    # the PREVIOUS task's end to NOW (kernel-start time). That's the
    # GPU's idle gap between consecutive task launches on this stream.
    _gpu_dbg = os.environ.get("DUAL_GPU_IDLE_DBG", "0") == "1"
    _profile = os.environ.get("DUAL_PROFILE", "0") == "1"
    # Profile mode needs cuda events to break kernel time per-type.
    _need_events = _gpu_dbg or _profile
    _ev_prev_end = torch.cuda.Event(enable_timing=True) if _need_events else None
    _ev_this_start = torch.cuda.Event(enable_timing=True) if _need_events else None
    _ev_this_end = torch.cuda.Event(enable_timing=True) if _need_events else None
    _gpu_sum = {"kernel_ms": 0.0, "idle_ms": 0.0, "n": 0}
    # Per-task-type bucket: kernel_ms, idle_ms (gap BEFORE this task's
    # kernel), n. Idle is attributed to whichever task type ran NEXT —
    # so "INTERP idle=5ms" means the GPU was idle 5ms before the next
    # INTERP kernel began (regardless of what was running before).
    _by_type = {}  # task_type → {kernel_ms, idle_ms, n, claim_wait_ms, post_ms}
    # idle_before histogram per task type — bucket boundaries (ms):
    #   0=<0.5, 1=0.5-2, 2=2-5, 3=5-10, 4=10-20, 5=>20
    _idle_hist_edges = (0.5, 2.0, 5.0, 10.0, 20.0)
    _idle_hist: dict[str, list[int]] = {}  # type → [6-entry list]
    def _hist_bucket(ms: float) -> int:
        for b, edge in enumerate(_idle_hist_edges):
            if ms < edge:
                return b
        return len(_idle_hist_edges)
    _have_prev = False
    _prof_t_last_report = _time.perf_counter()
    # claim_wait per-type accumulators (CPU side: time spent in the
    # polling scan before this task showed up RECV_DONE)
    _ks = ["INTERP", "SR_INTERP", "CCSR", "UNK"]
    def _type_name(tt):
        if tt == TT_INTERP: return "INTERP"
        if tt == TT_SR_INTERP: return "SR_INTERP"
        if tt == TT_CCSR: return "CCSR"
        return "UNK"
    # ── task dispatch ──────────────────────────────────────────
    # Each handler runs the GPU kernels for one task type. Inputs:
    #   cur          — slot index
    #   src, ov      — zero-copy shm views for this slot
    #   src_a_idx, src_b_idx — frame indices (semantics vary per
    #                  task_type)
    #   task_id      — round-trip handle for queue mgr (unused here
    #                  but logged for split-task diagnostics)
    # All handlers must record done_events[cur] before returning so
    # _sync_worker can mark DST_READY.

    def _run_unimplemented(task_name):
        def _handler(cur, src, ov, src_a_idx, src_b_idx, task_id):
            log(f"task_type={task_name} task_id={task_id} slot={cur} "
                f"hit stub — kernel not yet implemented")
        return _handler

    # Read once on first INTERP, cached for the life of the
    # compute_proc. Worker decides mult=2 vs mult≥3 multi-flownet
    # based on this. _run_interp uses `nonlocal` to mutate.
    _interp_mult_cache: int | None = None

    def _run_interp(cur, src, ov, src_a_idx, src_b_idx, task_id):
        """INTERP: read padded RGB + cached RIFE encode features, run
        RIFE flownet, write fp16 RGB into slot.dst (guest mode) or
        cc_cache.rgb_interp (host mode).

        One INTERP task per pair. Worker reads DUAL_INTERP_MULT env
        at startup; for mult ≥ 3 runs flownet (mult-1) times at
        t=k/mult for k=1..mult-1 and writes EACH frame into separate
        destinations:
          - guest mode: split_dst_interp_views[cur]["rgb"/"rgb2"/"rgb3"]
                        (slot.dst dense pack at frame k * frame_sz)
          - host mode:  cc_views[dst_cc_slot{_2/_3}]["rgb_interp"]
        After each frame finishes, _mark_stage_done(cur, k-1) lets
        the host's SR_INTERP start in parallel with the next flownet.
        Wire input (rgb_padded + features for both src frames,
        ~56 MB) is paid ONCE per pair regardless of mult.
        """
        meta = shm.slot_state(cur)
        src_a = int(meta.src_cc_slot_a)
        src_b = int(meta.src_cc_slot_b)
        dst   = int(meta.dst_cc_slot)
        dst2  = int(meta.dst_cc_slot_2)
        dst3  = int(meta.dst_cc_slot_3)
        del meta
        guest_mode = (cc_shm_local is None)
        if not guest_mode and (src_a < 0 or src_b < 0 or dst < 0):
            log(f"INTERP slot={cur} task_id={task_id} cc_cache refs "
                f"missing (a={src_a} b={src_b} dst={dst}); discarded")
            return
        if guest_mode and split_src_views is None:
            log(f"INTERP slot={cur} task_id={task_id} — guest mode but "
                f"split_src_views not initialised; discarded")
            return
        flownet = rife_cfg["flownet"]
        encode = rife_cfg["encode"]
        pw, ph = rife_cfg["pw"], rife_cfg["ph"]
        need_pad = rife_cfg["need_pad"]
        tenFlow_div = rife_cfg["tenFlow_div"]
        backwarp_tenGrid = rife_cfg["backwarp_tenGrid"]
        stream_inf = rife_cfg["stream_inf"]
        lock_inf = rife_cfg["lock_inf"]

        if guest_mode:
            sv = split_src_views[cur]
            img0p = sv["rgb_padded_a"]
            img1p = sv["rgb_padded_b"]
        else:
            view_a = cc_views[src_a]
            view_b = cc_views[src_b]
            img0p = view_a["rgb_padded"]
            img1p = view_b["rgb_padded"]
        nonlocal _interp_mult_cache
        if _interp_mult_cache is None:
            _interp_mult_cache = int(os.environ.get("DUAL_INTERP_MULT", "2"))
            if _interp_mult_cache not in (2, 3, 4):
                _interp_mult_cache = 2
        mult = _interp_mult_cache
        if mult == 2:
            timesteps = [0.5]
        else:
            # mult=N → timesteps [1/N, 2/N, ..., (N-1)/N]. N-1 flownets.
            timesteps = [(k + 1) / mult for k in range(mult - 1)]
        # Dense-pack destinations: stage k → frame k.
        if guest_mode:
            stage_keys = ("rgb", "rgb2", "rgb3")
        cc_slot_for_stage = (dst, dst2, dst3)
        if os.environ.get("DUAL_SPLIT_DEBUG", "0") == "1":
            log(f"_run_interp slot={cur} task_id={task_id} "
                f"mult={mult} timesteps={timesteps} "
                f"dst={dst} dst2={dst2} dst3={dst3}")
        with torch.inference_mode():
            for tstep_idx, tstep in enumerate(timesteps):
                timestep_t = torch.full([1, 1, ph, pw], tstep,
                                         dtype=torch.float16, device=dev)
                with lock_inf, torch.cuda.stream(stream_inf):
                    if encode is not None:
                        if guest_mode:
                            f0 = split_src_views[cur]["features_a"]
                            f1 = split_src_views[cur]["features_b"]
                        else:
                            f0 = view_a["rife_features"]
                            f1 = view_b["rife_features"]
                        out = flownet(img0p, img1p, timestep_t, tenFlow_div,
                                       backwarp_tenGrid, f0, f1)
                    else:
                        out = flownet(img0p, img1p, timestep_t, tenFlow_div,
                                       backwarp_tenGrid)
                    if need_pad:
                        out = out[:, :, :proc_h, :proc_w]
                    if guest_mode:
                        target = split_dst_interp_views[cur][
                            stage_keys[tstep_idx]]
                    else:
                        target = cc_views[cc_slot_for_stage[tstep_idx]][
                            "rgb_interp"]
                    target.copy_(out, non_blocking=True)
                    # N-stage mid: each non-final frame fires its own
                    # stage_done so SR_INTERP(phase=k+1) can start while
                    # later flownets still run. Final frame
                    # (tstep_idx == mult - 2) uses the task_done flow.
                    # Must record the cuda event on stream_inf (the
                    # stream that ran target.copy_), not on default
                    # stream — otherwise event.sync returns before the
                    # copy is visible, and the downstream SR_INTERP on
                    # default stream reads stale cc_cache bytes (8 dB
                    # PSNR). Keep this INSIDE the
                    # `with torch.cuda.stream(stream_inf)` block.
                    if tstep_idx < mult - 2:
                        _mark_stage_done(cur, tstep_idx)
        torch.cuda.default_stream().wait_stream(stream_inf)

    def _run_sr_interp(cur, src, ov, src_a_idx, src_b_idx, task_id):
        """SR_INTERP: read fp16 RGB from cc_cache[src_cc_slot_a]
        .rgb_interp (host) or slot.src (guest), do RGB→YUV matrix at
        proc dims, FSRCNNX on Y → out dims, chroma resize from proc to
        oc dims (identity for scale=2, real resize for x3/x4), write
        out int16 YUV planes into slot.dst's yao/uao/vao region.
        """
        meta = shm.slot_state(cur)
        src_slot = int(meta.src_cc_slot_a)
        del meta
        guest_mode = (cc_shm_local is None)
        if not guest_mode and src_slot < 0:
            log(f"SR_INTERP slot={cur} task_id={task_id} — cc_cache or "
                f"src_cc_slot_a unavailable (src_slot={src_slot}); "
                f"output discarded")
            return
        if guest_mode and split_src_views is None:
            log(f"SR_INTERP slot={cur} task_id={task_id} — guest mode but "
                f"split_src_views not initialised; discarded")
            return
        with torch.inference_mode():
            if guest_mode:
                rgb_in = split_src_views[cur]["rgb_interp"]
            else:
                rgb_in = cc_views[src_slot]["rgb_interp"]
            # RGB(fp16) → YUV(fp32) via matrix at proc dims.
            rgb_f = rgb_in[0].permute(1, 2, 0).to(torch.float32)
            m = vgh._RGB2YUV_MATRIX[matrix_s].to(dev)
            yuv = torch.einsum("hwc,rc->hwr", rgb_f, m)
            y_full = yuv[..., 0]
            u_full = yuv[..., 1]
            v_full = yuv[..., 2]
            # Range-encode to match CCSR's FSRCNNX input convention
            # (both fed raw_yuv/MAX into the network — same value range).
            y_scale, y_offset, uv_scale, uv_offset = \
                vgh._RANGE_CONSTS_10[color_range]
            inv_max_local = 1.0 / MAX
            y_norm = ((y_full * y_scale  + y_offset ) * inv_max_local
                     ).clamp_(0.0, 1.0)
            u_norm = ((u_full * uv_scale + uv_offset) * inv_max_local
                     ).clamp_(0.0, 1.0)
            v_norm = ((v_full * uv_scale + uv_offset) * inv_max_local
                     ).clamp_(0.0, 1.0)
            y_sr_in = y_norm.view(1, 1, proc_h, proc_w)
            if runner is None:
                # no_sr path: proc_h == oc_h (rife.vpy rejects no_sr +
                # downsample_pre>1). y_sr_in already [0,1] fp32 at the
                # right dim — use as-is.
                y_sr = y_sr_in.clamp(0, 1)
            else:
                y_sr = runner.forward(y_sr_in).clamp_(0, 1)
            # Chroma: proc → oc. For scale=2 4:2:0 out, proc == oc →
            # identity quantize (no resize). For non-baseline (proc != oc,
            # i.e. 720p×3 / 480p×4 / 4K-DS) krig with y_sr as guide
            # replaces the legacy bicubic upsample.
            u_in = u_norm.view(1, 1, proc_h, proc_w)
            v_in = v_norm.view(1, 1, proc_h, proc_w)
            if (proc_h, proc_w) == (oc_h, oc_w):
                # Identity — no upsample needed. Quantize u_in/v_in
                # directly (already in [0, 1] fp32).
                ov["uao"].copy_(
                    (u_in * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
                ov["vao"].copy_(
                    (v_in * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
            elif os.environ.get("KRIG_DEBUG_BYPASS", "0") == "1":
                u_up = F.interpolate(u_in, **interp_kw).clamp_(0, 1)
                v_up = F.interpolate(v_in, **interp_kw).clamp_(0, 1)
                ov["uao"].copy_(
                    (u_up * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
                ov["vao"].copy_(
                    (v_up * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
            else:
                # Krig at output chroma dim using 4K SR'd luma as guide.
                # u_in / v_in are fp32 in [0, 1]; krig accepts fp16 only,
                # so cast at the boundary. Range-normalise the luma guide
                # so the bilateral similarity threshold matches the
                # source's ColorRange (see _run_ccsr Stage A note).
                y_sr_norm = (
                    (y_sr.to(torch.float32) * MAX - y_offset) / y_scale
                ).clamp_(0.0, 1.0).to(torch.float16)
                u_in_fp16 = u_in.to(torch.float16)
                v_in_fp16 = v_in.to(torch.float16)
                u_sr_out = torch.empty(
                    (1, 1, oc_h, oc_w), device=dev, dtype=torch.float16)
                v_sr_out = torch.empty_like(u_sr_out)
                krig_bilateral_chroma(
                    y_sr_norm, u_in_fp16, v_in_fp16,
                    u_out=u_sr_out, v_out=v_sr_out)
                u_sr_out.clamp_(0.0, 1.0)
                v_sr_out.clamp_(0.0, 1.0)
                ov["uao"].copy_(
                    (u_sr_out * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
                ov["vao"].copy_(
                    (v_sr_out * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
            ov["yao"].copy_(
                (y_sr * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                non_blocking=True)

    def _run_ccsr(cur, src, ov, src_a_idx, src_b_idx, task_id):
        """CCSR: merged CC + SR_SRC. Reads raw YUV420P10, writes
        rgb_padded + rife_features (for INTERP) and 4K SR output (for
        mpv VA writeback). sr_yuv stays local within this kernel.

        downsample_pre=1 (default): CC at source dims, then FSRCNNX
        on Y produces 4K SR output to dst_a.

        downsample_pre>1 (4K-source path): luma bicubic-downsample to
        proc dims, treat chroma (already at proc dims for 4K 4:2:0)
        as 4:4:4, CC at proc dims (yuv_p10_to_rgb hits the no-op
        upsample fast-path). dst_a passthrough: raw source YUV
        memcpy → ov (no FSRCNNX). The original 4K frame stays
        bit-exact while interpolated frames go through the down→RIFE→
        SR_INTERP up loop.

        Host mode (cc_shm_local set): mid-out → host cc_cache, 4K SR /
        passthrough → slot.dst.yao/uao/vao (host memmoves into mpv VA
        dst_a).

        Guest mode (cc_shm_local is None): mid-out → slot.dst extended
        region (ov["rgb_padded"], ov["rife_features"]); host's
        guest_mp CQ thread later memmoves it into host cc_cache.
        """
        meta = shm.slot_state(cur)
        dst_slot = int(meta.dst_cc_slot)
        del meta
        guest_mode = (cc_shm_local is None)
        if os.environ.get("DUAL_SPLIT_DEBUG", "0") == "1":
            log(f"_run_ccsr slot={cur} task_id={task_id} "
                f"guest={guest_mode} dst_slot={dst_slot} "
                f"downsample_pre={downsample_pre}")
        if not guest_mode and dst_slot < 0:
            log(f"CCSR slot={cur} task_id={task_id} — host mode but "
                f"dst_slot={dst_slot}; output discarded")
            return
        encode      = rife_cfg["encode"]
        padding     = rife_cfg["padding"]
        need_pad    = rife_cfg["need_pad"]
        stream_inf  = rife_cfg["stream_inf"]
        lock_inf    = rife_cfg["lock_inf"]
        inv_max     = 1.0 / MAX
        host_view = None if guest_mode else cc_views[dst_slot]

        if downsample_pre > 1:
            # 4K-source DS variant — see docstring.
            with torch.inference_mode(), lock_inf, torch.cuda.stream(stream_inf):
                y_full = src["ya"].view(1, 1, H, W)
                # GPU bicubic luma DS to proc dims. Round + clamp to
                # restore the 10-bit integer range; yuv_p10_to_rgb's
                # input-side normalize then maps back to fp.
                y_proc = F.interpolate(
                    y_full.to(torch.float32),
                    size=(proc_h, proc_w),
                    mode="bicubic", align_corners=False
                ).round_().clamp_(0, MAX).to(torch.int16)
                # Chroma path:
                #   • 4:2:0 src → chroma dims already at (proc_h, proc_w)
                #     for downsample_pre=2 4K. Fast path: just .view().
                #   • 4:2:2 / 4:4:4 src → chroma > proc dims. GPU bicubic
                #     DS to (proc_h, proc_w) then treat as 4:4:4 at proc
                #     dims (same as 4:2:0 fast path consumes it).
                if cH == proc_h and cW == proc_w:
                    u_proc = src["ua"].view(proc_h, proc_w)
                    v_proc = src["va"].view(proc_h, proc_w)
                else:
                    u_full = src["ua"].view(1, 1, cH, cW)
                    v_full = src["va"].view(1, 1, cH, cW)
                    u_proc = F.interpolate(
                        u_full.to(torch.float32),
                        size=(proc_h, proc_w),
                        mode="bicubic", align_corners=False,
                    ).round_().clamp_(0, MAX).to(torch.int16
                    ).view(proc_h, proc_w)
                    v_proc = F.interpolate(
                        v_full.to(torch.float32),
                        size=(proc_h, proc_w),
                        mode="bicubic", align_corners=False,
                    ).round_().clamp_(0, MAX).to(torch.int16
                    ).view(proc_h, proc_w)
                rgb = vgh.yuv_p10_to_rgb(
                    y_proc.view(proc_h, proc_w),
                    u_proc, v_proc, matrix_s,
                    color_range=color_range,
                    out_dtype=torch.float16,
                )
                rgb_padded = F.pad(rgb, padding) if need_pad else rgb
                if guest_mode:
                    ov["rgb_padded"].copy_(rgb_padded, non_blocking=True)
                    if encode is not None:
                        ov["rife_features"].copy_(
                            encode(rgb_padded), non_blocking=True)
                else:
                    host_view["rgb_padded"].copy_(rgb_padded, non_blocking=True)
                    if encode is not None:
                        host_view["rife_features"].copy_(
                            encode(rgb_padded), non_blocking=True)
                # dst_a passthrough: raw source YUV → ov. Source dims
                # equal output dims when downsample_pre×fsrcnnx_scale=1.
                ov["yao"].copy_(src["ya"], non_blocking=True)
                ov["uao"].copy_(src["ua"], non_blocking=True)
                ov["vao"].copy_(src["va"], non_blocking=True)
                # Mark stage A done — mid range (rgb_padded +
                # rife_features) is NIC-visible on stream_inf. Stage B
                # is no-op for the downsample-pre path (no FSRCNNX),
                # so the FULL stage is also done by here, but we mark
                # mid first so queue_mgr can unblock downstream INTERP
                # before the FULL event fires. Same-machine: the same
                # rgb_padded/features land in cc_cache via host_view,
                # already shm-visible — mid signal alone is sufficient.
                _mark_mid_done(cur)
            torch.cuda.default_stream().wait_stream(stream_inf)
            return

        # Default (no pre-downsample) — CC + inline SR_SRC at source dims.
        # Stage A on stream_inf (CC + RIFE encode + rgb_padded write),
        # Stage B on default stream (FSRCNNX runner + ov[*] writes).
        # The runner.forward TRT engine binds to default stream; moving
        # it to stream_inf produces wrong luma output. Strict serial
        # across frames is enforced by the compute_proc loop's frame-
        # boundary event sync (stream_inf.wait_event before claim).
        #
        # Unified single-krig design:
        #   Stage A (stream_inf):
        #     - krig src→YUV444 at src luma dim → u_o / v_o (VRAM fp16)
        #     - yuv_p10_to_rgb → rgb_padded → encode → ov rgb_padded /
        #       rife_features
        #     - x2 baseline (H == oc_h): quantize u_o → ov["uao"]/vao
        #       HERE (same stream as krig, no cross-stream handoff).
        #   Stage B (default stream):
        #     - FSRCNNX runner → y_sr → ov["yao"]
        #     - 4:4:4 src or x3/x4: krig with y_sr guide → ov["uao"]/vao
        # Krig kernel now properly runs on the current torch stream
        # (chroma_krig.cu uses at::cuda::getCurrentCUDAStream()) so all
        # the cross-stream and allocator-pool races are gone.
        with torch.inference_mode(), lock_inf, torch.cuda.stream(stream_inf):
            y = src["ya"].view(H, W)
            u = src["ua"].view(cH, cW)
            v = src["va"].view(cH, cW)
            if (cH, cW) == (H, W):
                # 4:4:4 src — no krig at ingest, no Stage A chroma upsample.
                u_for_rgb = u
                v_for_rgb = v
                u_o = v_o = None
            else:
                # Range-aware krig: feed luma + chroma in their
                # ColorRange-normalised form so single-machine
                # `yuv_p10_to_rgb` (which also range-normalises before
                # krig) sees the SAME inputs and therefore the SAME
                # weighted-average output — they only differ by the
                # round-trip int16 quantisation noise downstream.
                # Limited: Y stretches [64,940]→[0,1]; U/V centre at 0
                # in [-0.5, 0.5]. Full: identity scaling.
                # Krig is linear in chroma so the [-0.5, 0.5] range is
                # safe even though the docstring talks about [0, 1].
                y_scale_a, y_offset_a, uv_scale_a, uv_offset_a = \
                    vgh._RANGE_CONSTS_10[color_range]
                y_g = ((y.to(torch.float32) - y_offset_a) / y_scale_a
                       ).clamp_(0.0, 1.0).to(torch.float16
                       ).view(1, 1, H, W)
                u_i = ((u.to(torch.float32) - uv_offset_a) / uv_scale_a
                       ).to(torch.float16).view(1, 1, cH, cW)
                v_i = ((v.to(torch.float32) - uv_offset_a) / uv_scale_a
                       ).to(torch.float16).view(1, 1, cH, cW)
                u_o = torch.empty(
                    (1, 1, H, W), device=y.device, dtype=torch.float16)
                v_o = torch.empty_like(u_o)
                if os.environ.get("KRIG_DEBUG_BYPASS", "0") == "1":
                    u_o.copy_(F.interpolate(u_i, size=(H, W),
                                             mode="bilinear",
                                             align_corners=False))
                    v_o.copy_(F.interpolate(v_i, size=(H, W),
                                             mode="bilinear",
                                             align_corners=False))
                else:
                    krig_bilateral_chroma(
                        y_g, u_i, v_i, u_out=u_o, v_out=v_o)
                # u_o / v_o are range-normalised chroma in ~[-0.5, 0.5].
                # De-normalise back to raw int16 [0, MAX] for downstream
                # consumers (yuv_p10_to_rgb re-normalises per
                # color_range; ov["uao"]/vao SR_SRC output is raw int16
                # matching mpv frame planes).
                u_for_rgb = (u_o.to(torch.float32) * uv_scale_a + uv_offset_a
                             ).round_().clamp_(0, MAX).to(torch.int16
                             ).view(H, W)
                v_for_rgb = (v_o.to(torch.float32) * uv_scale_a + uv_offset_a
                             ).round_().clamp_(0, MAX).to(torch.int16
                             ).view(H, W)
                if (H, W) != (oc_h, oc_w):
                    # u_o / v_o will be read by Stage B's second krig
                    # on the default stream — mark lifetime accordingly.
                    u_o.record_stream(torch.cuda.default_stream())
                    v_o.record_stream(torch.cuda.default_stream())
            # Always compute rgb + rgb_padded (used to be gated on
            # _compute_no_interp but disabling that gate is needed to
            # avoid a stream-sync race that corrupts the bottom rows
            # of yao in mult=1 mode. The yuv_p10_to_rgb + encode pass
            # was implicitly serialising stream_inf long enough for
            # FSRCNNX's read of y_sr_in (also on stream_inf-allocated
            # storage) to settle before default_stream picked it up.
            # Need a proper sync fix instead of skipping this work;
            # revisit once that's understood.
            rgb = vgh.yuv_p10_to_rgb(
                y, u_for_rgb, v_for_rgb, matrix_s,
                color_range=color_range,
                out_dtype=torch.float16,
            )
            rgb_padded = F.pad(rgb, padding) if need_pad else rgb
            if guest_mode:
                ov["rgb_padded"].copy_(rgb_padded, non_blocking=True)
                if encode is not None:
                    ov["rife_features"].copy_(
                        encode(rgb_padded), non_blocking=True)
                y_sr_in = (y.view(1, 1, H, W).to(torch.float16) * inv_max)
                u_sr_in_raw = (u.view(1, 1, cH, cW).to(torch.float16)
                                * inv_max)
                v_sr_in_raw = (v.view(1, 1, cH, cW).to(torch.float16)
                                * inv_max)
                # See comment above: keep record_stream as a defensive
                # belt-and-suspenders even though the yuv_p10_to_rgb
                # serialisation also helps.
                y_sr_in.record_stream(torch.cuda.default_stream())
                u_sr_in_raw.record_stream(torch.cuda.default_stream())
                v_sr_in_raw.record_stream(torch.cuda.default_stream())
            else:
                host_view["rgb_padded"].copy_(rgb_padded, non_blocking=True)
                if encode is not None:
                    host_view["rife_features"].copy_(
                        encode(rgb_padded), non_blocking=True)
                sr = host_view["sr_yuv"]
                sr["y"].copy_(y.view(1, 1, H, W), non_blocking=True)
                sr["y"].mul_(inv_max)
                sr["u"].copy_(u.view(1, 1, cH, cW), non_blocking=True)
                sr["u"].mul_(inv_max)
                sr["v"].copy_(v.view(1, 1, cH, cW), non_blocking=True)
                sr["v"].mul_(inv_max)
                y_sr_in = sr["y"]
                u_sr_in_raw = sr["u"]
                v_sr_in_raw = sr["v"]
            # x2 baseline SR_SRC chroma OUTPUT: identity quantize on
            # stream_inf — no cross-stream handoff. krig kernel is now
            # correctly bound to current stream (chroma_krig.cu uses
            # at::cuda::getCurrentCUDAStream), so u_o is guaranteed to
            # be fully written by the time these ov writes execute.
            if u_o is not None and (H, W) == (oc_h, oc_w):
                ov["uao"].copy_(
                    u_for_rgb.view(1, 1, oc_h, oc_w), non_blocking=True)
                ov["vao"].copy_(
                    v_for_rgb.view(1, 1, oc_h, oc_w), non_blocking=True)
            # Mark stage A done. mid range (rgb_padded + features) is
            # NIC-visible on stream_inf. Stage B (FSRCNNX yao + 2nd
            # krig uao/vao) continues on default stream. Same-machine:
            # rgb_padded + features land in cc_cache via host_view,
            # already shm-visible — mid signal alone is sufficient.
            _mark_mid_done(cur)
        torch.cuda.default_stream().wait_stream(stream_inf)
        with torch.inference_mode():
            if runner is None:
                # no_sr path: skip FSRCNNX. With no_sr we know
                # H==oc_h==proc_h (rife.vpy rejects no_sr+downsample_pre>1)
                # so y_sr_in (= sr["y"] or inline equivalent at (1,1,H,W)
                # fp16/fp32 in [0,1]) is already at the output dim — just
                # quantize back to int16 → ov["yao"]. For chroma we keep
                # the source's subsampling end-to-end (dst_sub == src_sub
                # in native_dispatcher when no_sr), so ov["uao"]/vao are
                # sized at (cH, cW). Stage A's u_for_rgb / u_o are at
                # (H, W) — for the RGB→RIFE feed path, NOT compatible
                # with ov chroma when sub != 0. Just memcpy raw source
                # chroma into ov here regardless of u_o.
                ov["yao"].copy_(
                    (y_sr_in * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
                ov["uao"].copy_(
                    src["ua"].view(1, 1, cH, cW),
                    non_blocking=True)
                ov["vao"].copy_(
                    src["va"].view(1, 1, cH, cW),
                    non_blocking=True)
                return
            # FSRCNNX luma SR on default stream.
            y_sr = runner.forward(y_sr_in).clamp_(0, 1)
            ov["yao"].copy_(
                (y_sr * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                non_blocking=True)
            if u_o is None:
                # 4:4:4 src — no Stage A krig output. Fall back to
                # legacy bicubic upsample from raw src chroma.
                u_up = F.interpolate(u_sr_in_raw, **interp_kw).clamp_(0, 1)
                v_up = F.interpolate(v_sr_in_raw, **interp_kw).clamp_(0, 1)
                ov["uao"].copy_(
                    (u_up * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
                ov["vao"].copy_(
                    (v_up * MAX + 0.5).clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
            elif (H, W) != (oc_h, oc_w):
                # Scale > 2 (720p×3 / 480p×4 / 4K-DS): src luma dim !=
                # output chroma dim. Second krig with y_sr (4K SR'd
                # luma) as guide. Range-normalise the guide so the
                # bilateral similarity threshold matches the source's
                # ColorRange (see Stage A note).
                y_sr_norm = (
                    (y_sr.to(torch.float32) * MAX - y_offset_a) / y_scale_a
                ).clamp_(0.0, 1.0).to(torch.float16)
                u_sr_out = torch.empty(
                    (1, 1, oc_h, oc_w), device=y.device,
                    dtype=torch.float16)
                v_sr_out = torch.empty_like(u_sr_out)
                # u_o / v_o are range-normalised chroma in ~[-0.5, 0.5]
                # (Stage A change). Krig is linear in chroma; output
                # stays in the same range.
                krig_bilateral_chroma(
                    y_sr_norm, u_o, v_o,
                    u_out=u_sr_out, v_out=v_sr_out)
                # De-normalise range-normalised chroma back to raw int16.
                ov["uao"].copy_(
                    (u_sr_out.to(torch.float32) * uv_scale_a + uv_offset_a
                     ).round_().clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
                ov["vao"].copy_(
                    (v_sr_out.to(torch.float32) * uv_scale_a + uv_offset_a
                     ).round_().clamp_(0, MAX).to(torch.int16),
                    non_blocking=True)
            # else: x2 baseline — ov["uao"]/vao already written in Stage A
        return

    HANDLERS = {
        TT_INTERP:      _run_interp,
        TT_SR_INTERP:   _run_sr_interp,
        TT_CCSR:        _run_ccsr,
        # Secondary SR_INTERP (mult ≥ 3) uses the same TT_SR_INTERP
        # wire type — host-side dispatch picks the right cc_cache slot
        # (dst_cc_slot_2/3) + mpv VA target (dst_i2/dst_i3) via
        # TaskNode.output_phase before the slot is sent over.
    }

    log("compute loop started (zero-copy shm views — no HtoD/DtoH)")
    try:
        while True:
            _t0 = _time.perf_counter()
            # Poll shm directly. As soon as rdma_proc marks a slot
            # RECV_DONE we pick it up — no eventfd RTT. sleep(0)
            # when idle yields to the OS so we don't fight other
            # daemons / send_handler / sync_thread for cores.
            cur = -1
            task_type = TT_CCSR
            task_id = 0
            src_a_idx = 0
            src_b_idx = 0
            while cur < 0:
                for i in range(layout.n_slots):
                    meta = shm.slot_state(i)
                    if meta.state == ST_RECV_DONE:
                        cur = i
                        meta.state = ST_COMPUTING
                        src_a_idx = meta.src_a_idx
                        src_b_idx = meta.src_b_idx
                        task_type = int(meta.task_type)
                        task_id   = int(meta.task_id)
                        # Compute and stamp the per-task MID/FULL byte
                        # ranges so send_handler reads them straight
                        # from shm. mid_size==0 ⇒ single-stage (no MID
                        # SEND, no MID_READY transition). N-stage: only
                        # stage 0 + final are stored; stages 1..mult-3
                        # are derived from stage_idx * mid_size in
                        # send_handler.
                        mo, ms, fo, fs = layout.task_dst_ranges(
                            task_type,
                            interp_mult=_compute_interp_mult,
                            rgb_interp_size=_compute_frame_sz)
                        meta.mid_off   = mo
                        meta.mid_size  = ms
                        meta.full_off  = fo
                        meta.full_size = fs
                        # Read back state to verify the COMPUTING
                        # write persisted (debug).
                        if os.environ.get("DUAL_REFCOUNT_DBG", "0") == "1":
                            _readback = int(meta.state)
                            _dst_slot = int(meta.dst_cc_slot)
                            log(f"compute picked slot={i} tid={task_id} "
                                f"type={task_type} state_readback={_readback} "
                                f"dst_cc_slot={_dst_slot} "
                                f"mid=({mo},{ms}) full=({fo},{fs})")
                        del meta
                        break
                    del meta
                if cur < 0:
                    _t_for_sleep.sleep(0)
            _t1 = _time.perf_counter()
            src = slot_src_views_gpu[cur]
            ov  = slot_dst_views_gpu[cur]
            # Dispatch on task_type (CCSR / INTERP / SR_INTERP).
            handler = HANDLERS.get(task_type)
            if handler is None:
                log(f"unknown task_type={task_type} slot={cur} task_id={task_id}")
                handler = _run_unimplemented(f"UNK({task_type})")
            # Strict serial-across-frames: stream_inf waits for the
            # previous frame's default-stream work (its Stage B / ov[*]
            # writes) before this frame's Stage A starts. Without this,
            # consecutive frames' kernels can run concurrently on the
            # GPU (Stage A of frame N+1 on stream_inf, Stage B of frame
            # N on default), which violates the design's serial-compute
            # invariant. GPU-side event wait, no CPU block; RDMA recv
            # of frame N+2 still runs concurrently in rdma_proc.
            stream_inf.wait_stream(torch.cuda.default_stream())
            if _need_events:
                _ev_this_start.record(torch.cuda.current_stream())
            handler(cur, src, ov, src_a_idx, src_b_idx, task_id)
            _t2 = _time.perf_counter()
            # Record event after the final write kernel. Current stream
            # at this point is default (each handler waits stream_inf →
            # default at the Stage A→B boundary, so default has all
            # the writes). sync_thread waits on it then marks DST_READY
            # → rdma_proc post_sends.
            done_events[cur].record(torch.cuda.current_stream())
            sync_q.put((cur, None))
            if _need_events:
                _ev_this_end.record(torch.cuda.current_stream())
                # Measure idle gap from prev kernel END → this kernel START.
                if _have_prev:
                    try:
                        _ev_prev_end.synchronize()
                        _ev_this_start.synchronize()
                        idle = _ev_prev_end.elapsed_time(_ev_this_start)
                        _ev_this_start.synchronize()
                        _ev_this_end.synchronize()
                        kern = _ev_this_start.elapsed_time(_ev_this_end)
                        _gpu_sum["idle_ms"] += idle
                        _gpu_sum["kernel_ms"] += kern
                        _gpu_sum["n"] += 1
                        if _profile:
                            tn = _type_name(task_type)
                            b = _by_type.setdefault(tn, {
                                "kernel_ms": 0.0, "idle_ms": 0.0, "n": 0,
                                "claim_wait_ms": 0.0, "post_ms": 0.0})
                            b["kernel_ms"] += kern
                            b["idle_ms"] += idle
                            b["n"] += 1
                            b["claim_wait_ms"] += (_t1 - _t0) * 1000
                            b["post_ms"] += (_t2 - _t1) * 1000 - kern  # CPU overhead post-handler
                            hist = _idle_hist.setdefault(
                                tn, [0] * (len(_idle_hist_edges) + 1))
                            hist[_hist_bucket(idle)] += 1
                        if _gpu_sum["n"] >= REPORT_EVERY:
                            n = _gpu_sum["n"]
                            if _gpu_dbg:
                                log(f"GPU stream: kernel={_gpu_sum['kernel_ms']/n:.2f}ms "
                                    f"idle={_gpu_sum['idle_ms']/n:.2f}ms "
                                    f"util={_gpu_sum['kernel_ms']/(_gpu_sum['kernel_ms']+_gpu_sum['idle_ms'])*100:.0f}% "
                                    f"(n={n})")
                            if _profile and _by_type:
                                now = _time.perf_counter()
                                wall = (now - _prof_t_last_report) * 1000
                                _prof_t_last_report = now
                                tot_n = sum(b["n"] for b in _by_type.values())
                                total_kernel = sum(b["kernel_ms"] for b in _by_type.values())
                                total_idle = sum(b["idle_ms"] for b in _by_type.values())
                                log(f"PROFILE wall={wall:.0f}ms tasks={tot_n} "
                                    f"kernel_total={total_kernel:.0f}ms "
                                    f"idle_total={total_idle:.0f}ms "
                                    f"GPU_util={100*total_kernel/(total_kernel+total_idle):.0f}%")
                                for tn in sorted(_by_type.keys()):
                                    b = _by_type[tn]
                                    nn = b["n"]
                                    hist = _idle_hist.get(tn, [0]*6)
                                    log(f"PROFILE  {tn:>10s}: n={nn:3d} "
                                        f"kernel={b['kernel_ms']/nn:.2f}ms "
                                        f"idle_before={b['idle_ms']/nn:.2f}ms "
                                        f"claim_wait={b['claim_wait_ms']/nn:.2f}ms "
                                        f"idle_hist[<0.5|<2|<5|<10|<20|>=20]="
                                        f"{hist[0]}|{hist[1]}|{hist[2]}|{hist[3]}|{hist[4]}|{hist[5]}")
                                _by_type = {}
                                _idle_hist = {}
                            _gpu_sum["kernel_ms"] = 0.0
                            _gpu_sum["idle_ms"] = 0.0
                            _gpu_sum["n"] = 0
                    except Exception as exc:
                        log(f"gpu_idle_dbg err: {exc}")
                # Swap events for next iteration.
                _ev_prev_end, _ev_this_end = _ev_this_end, _ev_prev_end
                _have_prev = True
            _t3 = _time.perf_counter()
            tsum["claim_wait"] += _t1 - _t0
            tsum["compute"]    += _t2 - _t1
            tsum["post"]       += _t3 - _t2
            tcount += 1
            n_done += 1
            if tcount >= REPORT_EVERY:
                ms = {k: v / tcount * 1000 for k, v in tsum.items()}
                total = sum(ms.values())
                log(f"compute avg/pair (ms): "
                    f"wait={ms['claim_wait']:.1f} "
                    f"compute={ms['compute']:.1f} "
                    f"post={ms['post']:.1f} "
                    f"total={total:.1f} (n={n_done}, zero-copy shm views)")
                tsum = {k: 0.0 for k in tsum}
                tcount = 0
    except Exception as ex:
        log(f"compute loop fatal: {type(ex).__name__}: {ex}")
        import traceback; log(traceback.format_exc())




# ──────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────





class WorkerMP(MPPipelineBase):
    """Worker side: rdma_proc owns the QP."""

    EF_NAMES = MPPipelineBase._BASE_EF_NAMES + [
        # rdma_proc fires this once it has started listening on the RDMA
        # port (TCP accept on the QP handshake socket). Otherwise host's
        # HostPipelineHybridRDMA can race ahead and get ECONNREFUSED.
        "rdma_listening",
    ]

    def __init__(self, *, rdma_dev: str, rdma_port: int, rdma_gid: int,
                 **kw):
        super().__init__(**kw)
        self.rdma_dev = rdma_dev
        self.rdma_port = rdma_port
        self.rdma_gid = rdma_gid

    def _extra_child_config(self, cfg: dict, role: str) -> None:
        cfg["WMP_RDMA_DEV"] = self.rdma_dev
        cfg["WMP_RDMA_PORT"] = str(self.rdma_port)
        cfg["WMP_RDMA_GID"] = str(self.rdma_gid)

    def start(self):
        self._spawn("rdma",    "worker_3proc", "rdma_proc_main")
        self._spawn("mgr",     "worker_3proc", "buffer_mgr_proc_main")
        self._spawn("compute", "worker_3proc", "compute_proc_main")


# ──────────────────────────────────────────────────────────────────────
# Top-level entry — called from worker.py per accepted session.
# ──────────────────────────────────────────────────────────────────────


def run_worker_3proc_pipeline(*, H, W, scale, sub_w, sub_h, n_slots,
                                rdma_dev, rdma_port, rdma_gid,
                                variant, rife_model, matrix_s,
                                color_range, chroma_mode, bits,
                                signal_ready_cb, log,
                                dst_sub_w: int = 1, dst_sub_h: int = 1,
                                downsample_pre: int = 1,
                                pH: int = 1088, pW: int = 1920,
                                enc_ch: int = 4,
                                interp_mult: int = 2,
                                no_sr: int = 0,
                                wmp_out: dict | None = None):
    # Override env so compute_proc / buffer_mgr_proc / rdma_proc (which
    # all read DUAL_INTERP_MULT + DUAL_NO_SR via env) see the session's
    # handshake-derived values instead of whatever the systemd unit set.
    # This lets F8/F9 cycle on the host trigger a fresh handshake on each
    # vf reload and have the worker pick up the new values WITHOUT needing
    # a daemon restart (the persistent worker.py loop just re-handshakes).
    if interp_mult not in (1, 2, 3, 4):
        interp_mult = 2
    os.environ["DUAL_INTERP_MULT"] = str(interp_mult)
    os.environ["DUAL_NO_SR"] = "1" if no_sr else "0"
    """Spawns rdma_proc + buffer_mgr_proc + compute_proc, waits for
    compute to pre-warm, then calls signal_ready_cb (which sends the
    NCCL 'ready' signal to the host). Finally blocks on .join() until
    any child exits. Slot.src is sized internally from H/W/pH/pW/enc_ch
    so that both sides of the wire compute the same size from the
    handshake values."""
    import time as _t
    layout = SlotRingLayout(n_slots=n_slots, H=H, W=W,
                             scale=scale, sub_w=sub_w, sub_h=sub_h,
                             dst_sub_w=dst_sub_w, dst_sub_h=dst_sub_h,
                             pH=pH, pW=pW, enc_ch=enc_ch)
    shm_name = f"dgxspark_worker_3p_{os.getpid()}"
    wmp = WorkerMP(
        layout=layout, shm_name=shm_name,
        rdma_dev=rdma_dev, rdma_port=rdma_port, rdma_gid=rdma_gid,
        log=log,
        variant=variant, rife_model=rife_model,
        matrix_s=matrix_s, color_range=color_range,
        chroma_mode=chroma_mode, bits=bits,
        downsample_pre=downsample_pre,
    )
    wmp.setup()
    if wmp_out is not None:
        wmp_out["wmp"] = wmp
    try:
        wmp.start()
        t0 = _t.perf_counter()
        wmp.wait_compute_ready(timeout=120.0)
        log(f"compute_proc ready after {_t.perf_counter()-t0:.1f}s; "
            f"signaling host")
        signal_ready_cb()
        log("entering wait-loop for children")
        rc = wmp.join(timeout=None)
        log(f"3-proc worker exited with rc={rc}")
    finally:
        wmp.shutdown()
        if wmp_out is not None:
            wmp_out["wmp"] = None


# ──────────────────────────────────────────────────────────────────────
# Self-test (run directly to verify the layout + shm + cuda + MR plumbing)
# ──────────────────────────────────────────────────────────────────────

def _self_test():
    """Single-process sanity test of the shm layout. Does NOT spawn
    workers yet — that's stages 2+. Run with:
        python -B worker_3proc.py
    """
    print("=== SlotRingLayout ===")
    layout = SlotRingLayout(n_slots=4, H=1080, W=1920,
                            scale=2, sub_w=1, sub_h=1)
    print(layout.describe())
    print(f"  total = {layout.total_size:,} bytes")
    print(f"  slot 0 src @ +{layout.slot_src_off(0):,}, "
          f"dst @ +{layout.slot_dst_off(0):,}")
    print(f"  slot 1 src @ +{layout.slot_src_off(1):,}")

    print("\n=== shm allocate + map ===")
    shm = SlotRingShm(layout, name="dgxspark_worker_test")
    shm.create_truncate_and_map()
    print(f"  addr=0x{shm.addr:x} fd={shm.fd}")

    print("\n=== state table ===")
    for i in range(layout.n_slots):
        meta = shm.slot_state(i)
        meta.state = ST_FREE
        meta.generation = 0
        meta.pair_k = -1
    for i in range(layout.n_slots):
        meta = shm.slot_state(i)
        print(f"  slot {i}: state={ST_NAMES[meta.state]} gen={meta.generation}")

    print("\n=== cuda registration ===")
    try:
        shm.register_cuda_pinned(mapped=True)
        dev_ptr = shm.cuda_device_ptr()
        same = (dev_ptr == shm.addr)
        print(f"  registered. host=0x{shm.addr:x} dev=0x{dev_ptr:x} "
              f"{'(unified — Grace SoC)' if same else '(distinct)'}")
    except Exception as ex:
        print(f"  FAILED: {type(ex).__name__}: {ex}")

    print("\n=== RDMA MR registration ===")
    try:
        sys.path.insert(0, "/usr/lib/python3/dist-packages")
        from rdma_transport import RDMAContext
        import pyverbs.enums as e
        ctx = RDMAContext(dev_name="rocep1s0f0", port=1, gid_index=3)
        access = (e.IBV_ACCESS_LOCAL_WRITE |
                  e.IBV_ACCESS_REMOTE_WRITE |
                  e.IBV_ACCESS_REMOTE_READ)
        mr = shm.register_rdma_mr(ctx.pd, access)
        print(f"  registered. lkey={mr.lkey} rkey={mr.rkey}")
    except Exception as ex:
        print(f"  FAILED: {type(ex).__name__}: {ex}")

    print("\n=== spawn child + child re-maps shm ===")
    # Use subprocess (not fork) so the child doesn't inherit pyverbs /
    # cuda FDs from the parent — important for the real 3-proc design.
    import subprocess
    script = """
import sys; sys.path.insert(0, %r)
sys.path.insert(0, '/usr/lib/python3/dist-packages')
from worker_3proc import SlotRingLayout, SlotRingShm, ST_NAMES, ST_RECV_PENDING
layout = SlotRingLayout(n_slots=4, H=1080, W=1920, scale=2, sub_w=1, sub_h=1)
shm = SlotRingShm(layout, name=%r)
shm.open_and_map()
meta = shm.slot_state(0)
print(f'  [child pid={__import__("os").getpid()}] addr=0x{shm.addr:x} slot 0 state={ST_NAMES[meta.state]}')
meta.state = ST_RECV_PENDING
meta.pair_k = 99
del meta
shm.close()
""" % (os.path.dirname(os.path.abspath(__file__)), shm.name)
    r = subprocess.run([sys.executable, "-B", "-c", script],
                       capture_output=True, text=True, timeout=10)
    print(r.stdout.rstrip())
    if r.stderr:
        print(f"  [child stderr] {r.stderr.rstrip()}")
    meta = shm.slot_state(0)
    print(f"  [parent] after child write: slot 0 state="
          f"{ST_NAMES[meta.state]} pair_k={meta.pair_k}")
    del meta  # drop the ctypes view before close

    print("\n=== eventfd round-trip across processes ===")
    # Parent creates an eventfd, spawns a child that inherits the fd
    # via pass_fds, child notifies the parent, parent reads.
    ef = EventFdChannel("test_ef")
    import subprocess
    script = """
import os
fd = int(os.environ['EFD'])
os.eventfd_write(fd, 1)
os.eventfd_write(fd, 1)
"""
    t0 = __import__("time").perf_counter()
    p = subprocess.Popen([sys.executable, "-c", script],
                          env={**os.environ, "EFD": str(ef.fd)},
                          pass_fds=[ef.fd])
    n1 = ef.wait()
    n2 = ef.wait()
    p.wait(timeout=5)
    dt = (__import__("time").perf_counter() - t0) * 1000
    print(f"  parent read {n1} then {n2} from child-written eventfd "
          f"in {dt:.1f}ms")
    ef.close()

    print("\n=== cleanup ===")
    shm.close()
    shm.unlink()
    print("  ok")


if __name__ == "__main__":
    _self_test()
