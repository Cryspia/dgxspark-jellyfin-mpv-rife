"""Dual-machine worker daemon. Listens on a single TCP socket
(`DUAL_LIVENESS_PORT`, default 29905) for host sessions; per session
reads an 18-int64 handshake, spawns the 3-proc RDMA pipeline, sends a
ready byte, then parks watchdog-style on the same socket until the
host disconnects (FIN/RST → keepalive 1+1+2 ≈ 3 s detection).

There is NO NCCL/torch.distributed. The control plane is the TCP
socket; the data plane is RDMA (rdma_proc, port 29900). Removing NCCL
killed the 3 s per-collective deadline that used to gate the slow
worker prep path (cold TRT compile takes minutes).

Handshake wire format (`<18q`, 144 bytes, little-endian signed int64):
  [0] mode (== 2; only mode left)
  [1] H, [2] W            source dimensions
  [3] scale                fsrcnnx upscale factor (=2)
  [4] variant_idx          VARIANTS index
  [5] chroma_idx           CHROMA_MODES index
  [6] sub_w, [7] sub_h     source chroma subsampling shifts
  [8] matrix_idx           _Matrix property value (709/170m/2020ncl…)
  [9] color_range          0=full, 1=limited
  [10] rife_model_idx      0=4.26, 1=4.6
  [11] downsample_pre      pre-CC luma downsample factor (1/2)
  [12] dst_sub_w, [13] dst_sub_h
                           output chroma subsampling shifts
  [14] rife_pH, [15] rife_pW
                           host's RIFE-padded dims
  [16] interp_mult         temporal multiplier (2/3/4)
  [17] no_sr               1 = skip FSRCNNX SR (F8 OFF in dual)
"""
from __future__ import annotations
import os, sys, time, socket, struct, threading, traceback
from pathlib import Path

# Make worker_3proc / mp_pipeline / vs_gpu_helpers / etc. importable
# from wherever worker.py is — the install location (where install.sh
# put it) or the source tree (when running out of a git clone for dev).
# Avoid hardcoding any path here.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_PY_BIN = str(Path(sys.executable).parent)
if _PY_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PY_BIN + os.pathsep + os.environ.get("PATH", "")
_MPV_HOME = Path(
    os.environ.get("MPV_HOME") or
    (os.environ.get("XDG_CONFIG_HOME") or
     os.path.expanduser("~/.config")) + "/mpv"
)
_FSRCNNX_BUNDLE = _MPV_HOME / "fsrcnnx-cudnn"
if _FSRCNNX_BUNDLE.exists() and str(_FSRCNNX_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_FSRCNNX_BUNDLE))
_BUNDLE_WEIGHTS = _FSRCNNX_BUNDLE / "weights"
if _BUNDLE_WEIGHTS.exists() and "WMP_WEIGHTS_DIR" not in os.environ:
    os.environ["WMP_WEIGHTS_DIR"] = str(_BUNDLE_WEIGHTS)

import torch


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

VARIANTS = [
    "FSRCNNX_x2_8-0-4-1",
    "FSRCNNX_x2_16-0-4-1",
    "FSRCNNX_x3_16-0-4-1",
    "FSRCNNX_x4_16-0-4-1",
]
CHROMA_MODES = ["bilinear", "bicubic", "nearest"]
_MATRIX_BY_IDX = {1: "709", 5: "470bg", 6: "170m", 7: "240m",
                  9: "2020ncl", 10: "2020ncl"}

_LIVENESS_BACKLOG = 128
_DRAIN_INTERVAL_S = 1.0
_LIVENESS_ACCEPT_TIMEOUT = 30.0
_HANDSHAKE_FMT = "<18q"
_HANDSHAKE_BYTES = struct.calcsize(_HANDSHAKE_FMT)  # 144


def log(msg):
    sys.stderr.write(f"[worker {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# ──────────────────────────────────────────────────────────────────────
# Liveness listener (control plane)
# ──────────────────────────────────────────────────────────────────────

def _peer_alive(c: socket.socket) -> bool:
    """Non-destructive MSG_PEEK: True iff peer has NOT FIN'd."""
    c.setblocking(False)
    try:
        return c.recv(1, socket.MSG_PEEK) != b""
    except BlockingIOError:
        return True  # no data, no FIN — alive
    except OSError:
        return False
    finally:
        try: c.setblocking(True)
        except OSError: pass


class LivenessListener:
    """Persistent TCP listener for host control sessions.

    Each host opens a connection here, sends its 144-byte handshake,
    waits for the worker's 'R' ready byte, then holds the socket open
    for the session's lifetime. FIN/RST on the socket trips the
    per-session watchdog → pipeline teardown.

    Two drain paths keep the accept queue from filling:
      1. Background daemon thread sweeps every `_DRAIN_INTERVAL_S` and
         closes FIN'd connections.
      2. `accept_session()` (called at session start) re-drains and
         returns the newest live connection.

    Without (1), repeated init failures (worker briefly stopped, host
    crash loop) would eventually overflow the listen() backlog and the
    kernel would start RST'ing new probes.
    """

    def __init__(self, port: int):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("0.0.0.0", port))
        self.srv.listen(_LIVENESS_BACKLOG)
        self._live: list[tuple[socket.socket, tuple]] = []
        self._lock = threading.Lock()
        self._in_session = threading.Event()
        self._drainer = threading.Thread(
            target=self._drain_loop, daemon=True, name="liveness-drainer")
        self._drainer.start()
        log(f"liveness listener on 0.0.0.0:{port} (backlog={_LIVENESS_BACKLOG})")

    def accept_session(self) -> tuple[socket.socket | None, tuple | None]:
        self._in_session.set()
        try:
            with self._lock:
                self._sweep_locked()
                while self._live:
                    c, addr = self._live.pop()  # newest = last
                    if _peer_alive(c):
                        n_stale = len(self._live)
                        for cc, _ in self._live:
                            try: cc.close()
                            except OSError: pass
                        self._live.clear()
                        if n_stale:
                            log(f"drained {n_stale} stale liveness "
                                f"candidate(s); using newest from {addr}")
                        return c, addr
                    try: c.close()
                    except OSError: pass
            self.srv.settimeout(_LIVENESS_ACCEPT_TIMEOUT)
            try:
                return self.srv.accept()
            except OSError as e:
                log(f"liveness accept failed: {type(e).__name__}: {e}")
                return None, None
            finally:
                self.srv.settimeout(None)
        finally:
            self._in_session.clear()

    def _drain_loop(self):
        while True:
            if not self._in_session.is_set():
                try:
                    with self._lock:
                        self._sweep_locked()
                except Exception:
                    pass
            time.sleep(_DRAIN_INTERVAL_S)

    def _sweep_locked(self):
        self.srv.setblocking(False)
        try:
            while True:
                try:
                    c, addr = self.srv.accept()
                except (BlockingIOError, OSError):
                    break
                self._live.append((c, addr))
        finally:
            self.srv.setblocking(True)
        kept = [(c, a) for c, a in self._live if _peer_alive(c)]
        for c, _ in self._live:
            if (c, _) not in kept:
                try: c.close()
                except OSError: pass
        removed = len(self._live) - len(kept)
        self._live = kept
        if removed:
            log(f"liveness drainer: closed {removed} stale connection(s)")


def _setup_session_keepalive(conn: socket.socket):
    """1+1+2 = ~3 s detection of ungraceful host disconnect."""
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 2)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# Handshake (replaces the former NCCL int64[18] send)
# ──────────────────────────────────────────────────────────────────────

def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Block until exactly `n` bytes are received, or raise IOError on
    early EOF. TCP recv can return short reads; loop until full."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise IOError(f"liveness EOF during recv at {len(buf)}/{n}")
        buf.extend(chunk)
    return bytes(buf)


def _parse_handshake(buf: bytes) -> dict | None:
    """Decode 144-byte handshake into a dict, or None if mode != 2."""
    hs = struct.unpack(_HANDSHAKE_FMT, buf)
    if hs[0] != 2:
        return None
    return {
        "mode": hs[0],
        "H": hs[1], "W": hs[2], "scale": hs[3],
        "variant": VARIANTS[hs[4]],
        "chroma_mode": CHROMA_MODES[hs[5]],
        "sub_w": hs[6], "sub_h": hs[7],
        "matrix_s": _MATRIX_BY_IDX.get(hs[8] or 1, "709"),
        "color_range": "full" if hs[9] == 0 else "limited",
        "rife_model": "4.26" if hs[10] == 0 else "4.6",
        "downsample_pre": hs[11] or 1,
        "dst_sub_w": hs[12] if hs[12] > 0 else 1,
        "dst_sub_h": hs[13] if hs[13] > 0 else 1,
        "rife_pH": hs[14] if hs[14] > 0 else 1088,
        "rife_pW": hs[15] if hs[15] > 0 else 1920,
        "interp_mult": hs[16] if hs[16] in (1, 2, 3, 4) else 2,
        "no_sr": 1 if hs[17] == 1 else 0,
    }


# ──────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────

def serve_session(dev, conn: socket.socket, addr: tuple):
    """Handle one host session over the just-accepted liveness `conn`.

    Wire protocol:
        host → worker:  144-byte handshake (struct '<18q')
        worker → host:  1 byte 'R' after compute_proc engines are ready
        host → worker:  (none after handshake; FIN/RST signals end)

    Watchdog thread parks on `conn.recv(8)` — any return value (including
    0-byte = peer FIN) or exception tears down the pipeline.
    """
    _setup_session_keepalive(conn)
    log(f"liveness peer connected from {addr}")
    try:
        raw = _recv_exact(conn, _HANDSHAKE_BYTES)
    except IOError as e:
        log(f"handshake read failed: {e}")
        return
    cfg = _parse_handshake(raw)
    if cfg is None:
        log(f"unsupported mode={struct.unpack('<q', raw[:8])[0]} in handshake; "
            "only mode 2 (CCSR + 3-proc) is implemented")
        return
    log(f"session mode={cfg['mode']} {cfg['W']}x{cfg['H']}->x{cfg['scale']} "
        f"variant={cfg['variant']} chroma={cfg['chroma_mode']} "
        f"src_sub=({cfg['sub_w']},{cfg['sub_h']}) "
        f"dst_sub=({cfg['dst_sub_w']},{cfg['dst_sub_h']}) "
        f"matrix={cfg['matrix_s']} range={cfg['color_range']} "
        f"rife_model={cfg['rife_model']} "
        f"downsample_pre={cfg['downsample_pre']} "
        f"interp_mult={cfg['interp_mult']} no_sr={cfg['no_sr']}")

    rdma_port = int(os.environ.get("DUAL_RDMA_PORT", "29900"))
    rdma_dev = os.environ.get("DUAL_RDMA_DEV", "rocep1s0f0")
    rdma_gid = int(os.environ.get("DUAL_RDMA_GID", "3"))
    n_slots = 4  # must match host's _guest_slots in native_dispatcher
    log(f"mode-2 MP: peer_port={rdma_port} dev={rdma_dev} slots={n_slots}")

    wmp_box = {"wmp": None}

    def _watchdog():
        try:
            data = conn.recv(8)
            log(f"watchdog: liveness recv returned {len(data)} bytes")
        except Exception as e:
            log(f"watchdog: liveness socket {type(e).__name__}: {e}")
        finally:
            try: conn.close()
            except OSError: pass
        wmp = wmp_box.get("wmp")
        if wmp is not None:
            try: wmp.shutdown()
            except Exception as ex:
                log(f"watchdog: wmp.shutdown raised: {ex}")

    threading.Thread(target=_watchdog, daemon=True,
                     name="liveness-watchdog").start()

    def _signal_ready():
        """Sent via TCP (not NCCL) so cold TRT compile on compute_proc
        doesn't blow any collective deadline. Host has settimeout
        `DUAL_WORKER_READY_TIMEOUT` (default 300 s) on this recv."""
        try:
            conn.sendall(b"R")
            log("signaled 'ready' to host via liveness")
        except OSError as e:
            log(f"failed to send ready: {type(e).__name__}: {e}")

    import worker_3proc as _mp
    _mp.run_worker_3proc_pipeline(
        H=cfg["H"], W=cfg["W"], scale=cfg["scale"],
        sub_w=cfg["sub_w"], sub_h=cfg["sub_h"],
        dst_sub_w=cfg["dst_sub_w"], dst_sub_h=cfg["dst_sub_h"],
        n_slots=n_slots,
        rdma_dev=rdma_dev, rdma_port=rdma_port, rdma_gid=rdma_gid,
        variant=cfg["variant"], rife_model=cfg["rife_model"],
        matrix_s=cfg["matrix_s"], color_range=cfg["color_range"],
        chroma_mode=cfg["chroma_mode"], bits=10,
        signal_ready_cb=_signal_ready, log=log,
        downsample_pre=cfg["downsample_pre"],
        pH=cfg["rife_pH"], pW=cfg["rife_pW"],
        enc_ch=4,
        interp_mult=cfg["interp_mult"], no_sr=cfg["no_sr"],
        wmp_out=wmp_box,
    )


def _rotate_log_if_big(path, max_bytes=10 * 1024 * 1024,
                       keep_last=1 * 1024 * 1024):
    """Bound log size for long-running server-mode worker. On rotation,
    truncate the file to the last `keep_last` bytes plus a marker."""
    if not path or not os.path.exists(path):
        return
    try:
        sz = os.path.getsize(path)
    except OSError:
        return
    if sz < max_bytes:
        return
    try:
        with open(path, "rb") as f:
            f.seek(max(0, sz - keep_last))
            tail = f.read()
        with open(path, "wb") as f:
            f.write(
                b"[worker] log rotated (kept last " +
                str(len(tail)).encode() + b" bytes)\n")
            f.write(tail)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def _install_signal_handlers():
    """SIGTERM/SIGINT — release shm before exit (else /dev/shm leaks
    `dgxspark_worker_3p_*` across restarts). No NCCL group to destroy
    in the post-NCCL design."""
    import signal as _signal

    def _term(signo, _frame):
        log(f"received signal {signo}; shutting down")
        import glob
        for f in glob.glob(f"/dev/shm/dgxspark_worker_3p_{os.getpid()}.dat"):
            try: os.unlink(f)
            except OSError: pass
        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _term)
    _signal.signal(_signal.SIGINT, _term)


def main():
    _install_signal_handlers()
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    # Match worker_3proc.py's cuDNN setting so single-proc and 3-proc
    # paths pick the same conv algorithm.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    listener = LivenessListener(
        port=int(os.environ.get("DUAL_LIVENESS_PORT", "29905")))

    log_path = os.environ.get("DUAL_WORKER_LOG", "/tmp/dual_worker.log")
    while True:
        log("waiting for host (liveness accept)…")
        conn, addr = listener.accept_session()
        if conn is None:
            continue  # accept timeout — listener still alive, retry
        try:
            serve_session(dev, conn, addr)
            log("session ended cleanly")
        except Exception:
            log("session error:\n" + traceback.format_exc())
        finally:
            try: conn.close()
            except OSError: pass
        time.sleep(0.2)
        _rotate_log_if_big(log_path)


if __name__ == "__main__":
    main()
