"""Cuda-pinned shm region for CC (color-convert) outputs.

Four plane families per slot — every slot reserves space for all
four, even if a given source resolution uses only some. The waste
(~50 MB per unused 4K slot) is paid once and avoids slot pools per
field that would compete for free-list capacity.

Layout (one slot, 1080p source → 4K out):

  +---- slot_off -------------------------------+
  | rgb_padded    (fp16 NCHW, 1×3×pH×pW)        |   ≈ 11.95 MB
  | rife_features (fp16, 1×enc_channels×pH×pW)  |   ≈ 15.94 MB (enc=4)
  | sr_yuv        (fp16 Y at H,W + U,V at cH,cW)|   ≈  5.93 MB
  | rgb_4K        (int16 RGB at oH,oW)          |   ≈ 47.46 MB
  +---------------------------------------------+

Why rgb_padded + rife_features instead of plain rgb: RIFE's encode
head is what makes INTERP slow (~12 ms of the 24 ms per pair). By
caching the *padded* RGB and the encode features in CC — both shared
across the two INTERPs that consume frame N — we move the encode
work out of the INTERP hot path.

slot ≈ 81 MB. 32 slots ≈ 2.6 GB. Sits comfortably in the Grace SoC
128 GB LPDDR5x with the worker shm rings.

Slot lifecycle is driven by QueueManager:

  alloc()           — CC dispatch reserves a free slot.
  set_field()       — workers mark which fields actually got written
                      (avoids consumers reading uninitialised bytes).
  field_addr()      — INTERP / SR_SRC / display path looks up where
                      to read.
  free()            — mgr returns the slot to the free-list once all
                      field refcounts reach 0.

This module is *only* the storage. Refcount tracking and refcount→free
plumbing live in QueueManager — see queue_mgr.py CCResult.
"""
from __future__ import annotations

import ctypes
import mmap
import os
import sys
import threading
from dataclasses import dataclass


# Field index constants — exposed so callers can write `cc.FIELD_RGB_PADDED`
# instead of magic strings.
FIELD_RGB_PADDED    = "rgb_padded"     # fp16 (1, 3,           pH, pW)
FIELD_RIFE_FEATURES = "rife_features"  # fp16 (1, enc_channels, pH, pW)
FIELD_SR_YUV        = "sr_yuv"         # fp16 Y + U,V (CCSR's SR_SRC staging)
FIELD_RGB_INTERP    = "rgb_interp"     # fp16 (1, 3, H_src, W_src) — INTERP RGB
FIELD_RGB_4K        = "rgb_4K"         # int16 RGB at output res (4K source path)

_ALL_FIELDS = (FIELD_RGB_PADDED, FIELD_RIFE_FEATURES, FIELD_SR_YUV,
               FIELD_RGB_INTERP, FIELD_RGB_4K)


_PAGE = 4096


def _align_up(n: int, a: int = _PAGE) -> int:
    return (n + a - 1) // a * a


# ──────────────────────────────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CCCacheLayout:
    """Pure offset math — same args on host and (potentially) guest so
    they agree on the byte map.

    Construction:
      n_slots        — number of cache slots.
      H_src, W_src   — input frame resolution (e.g. 1080×1920).
      H_out, W_out   — output frame resolution (4K source path uses
                       this for rgb_4K).
      pH, pW         — RIFE-padded shape (multiple of modulo_base per
                       model; e.g. 1088×1920 for 1080p with model 4.26).
      enc_channels   — RIFE encode output channels (model-dependent;
                       4 for 4.26, 16 for 4.26.heavy, etc.).
      sub_h, sub_w   — chroma subsample (1 = YUV420 for mode-B).
    """

    n_slots: int
    H_src:   int
    W_src:   int
    H_out:   int
    W_out:   int
    pH:      int          # RIFE-padded height
    pW:      int          # RIFE-padded width
    enc_channels: int     # RIFE encode output channels
    sub_h:   int = 1
    sub_w:   int = 1

    # Computed in __post_init__
    rgb_padded_size:    int = 0
    rife_features_size: int = 0
    sr_yuv_size:        int = 0
    rgb_interp_size:    int = 0
    rgb_4K_size:        int = 0
    slot_size:          int = 0
    slot_stride:        int = 0
    total_size:         int = 0

    off_rgb_padded:    int = 0
    off_rife_features: int = 0
    off_sr_yuv:        int = 0
    off_rgb_interp:    int = 0
    off_rgb_4K:        int = 0

    def __post_init__(self):
        cH = self.H_src >> self.sub_h
        cW = self.W_src >> self.sub_w
        # fp16 throughout for RIFE inputs (engines compiled fp16) and
        # sr_yuv. The matrix einsum inside INTERP is still fp32 (kernel
        # casts internally), then quantizes to 10-bit and casts to fp16
        # for cc_cache storage — preserves all 10 bits exactly (fp16
        # represents integers up to 2048).
        self.rgb_padded_size    = 1 * 3 * self.pH * self.pW * 2
        self.rife_features_size = 1 * self.enc_channels * self.pH * self.pW * 2
        self.sr_yuv_size        = (self.H_src * self.W_src * 2
                                   + 2 * cH * cW * 2)
        # rgb_interp: INTERP writes RGB at proc dims (=H_src,W_src). SR_INTERP
        # reads it, does RGB→YUV matrix + per-scale chroma resize. Skipping
        # YUV conversion at INTERP keeps the kernel simple and uniform
        # across 4:2:0 / 4:4:4 inputs; the conversion work shifts to
        # SR_INTERP which is bandwidth-bound, not compute-bound.
        self.rgb_interp_size    = 1 * 3 * self.H_src * self.W_src * 2
        self.rgb_4K_size        = self.H_out * self.W_out * 3 * 2  # int16

        rgb_padded_a    = _align_up(self.rgb_padded_size)
        rife_features_a = _align_up(self.rife_features_size)
        sr_yuv_a        = _align_up(self.sr_yuv_size)
        rgb_interp_a    = _align_up(self.rgb_interp_size)
        rgb_4K_a        = _align_up(self.rgb_4K_size)

        self.off_rgb_padded    = 0
        self.off_rife_features = self.off_rgb_padded    + rgb_padded_a
        self.off_sr_yuv        = self.off_rife_features + rife_features_a
        self.off_rgb_interp    = self.off_sr_yuv        + sr_yuv_a
        self.off_rgb_4K        = self.off_rgb_interp    + rgb_interp_a
        self.slot_size         = (rgb_padded_a + rife_features_a
                                  + sr_yuv_a + rgb_interp_a + rgb_4K_a)
        self.slot_stride       = self.slot_size
        self.total_size        = _align_up(self.n_slots * self.slot_stride)

    def describe(self) -> str:
        return (
            f"CCCacheLayout(n_slots={self.n_slots} "
            f"src={self.H_src}x{self.W_src} pad={self.pH}x{self.pW} "
            f"enc_ch={self.enc_channels} out={self.H_out}x{self.W_out} "
            f"slot={self.slot_size/1e6:.1f}MB "
            f"rgb_padded={self.rgb_padded_size/1e6:.1f}MB "
            f"rife_features={self.rife_features_size/1e6:.1f}MB "
            f"sr_yuv={self.sr_yuv_size/1e6:.1f}MB "
            f"rgb_interp={self.rgb_interp_size/1e6:.1f}MB "
            f"rgb_4K={self.rgb_4K_size/1e6:.1f}MB "
            f"total={self.total_size/1e6:.1f}MB)")

    def field_size(self, field: str) -> int:
        return {
            FIELD_RGB_PADDED:    self.rgb_padded_size,
            FIELD_RIFE_FEATURES: self.rife_features_size,
            FIELD_SR_YUV:        self.sr_yuv_size,
            FIELD_RGB_INTERP:    self.rgb_interp_size,
            FIELD_RGB_4K:        self.rgb_4K_size,
        }[field]

    def field_off_in_slot(self, field: str) -> int:
        return {
            FIELD_RGB_PADDED:    self.off_rgb_padded,
            FIELD_RIFE_FEATURES: self.off_rife_features,
            FIELD_SR_YUV:        self.off_sr_yuv,
            FIELD_RGB_INTERP:    self.off_rgb_interp,
            FIELD_RGB_4K:        self.off_rgb_4K,
        }[field]


# ──────────────────────────────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────────────────────────────


class CCCacheShm:
    """Page-aligned anonymous shm sized for a CCCacheLayout, optionally
    cuda-pinned + mapped so the same address dereferences from GPU
    kernels via Grace SoC unified memory.

    Slot allocation API:
      .alloc()                 → slot_id (raises if exhausted)
      .free(slot_id)           → return to free-list
      .set_field(s, field)     → mark "field has valid data"
      .has_field(s, field)     → consumer-side check
      .clear_field(s, field)   → reset (used by free())
      .field_addr(s, field)    → host VA / GPU dev ptr (same on Grace)
      .field_view(s, field, *) → memoryview slice for ctypes / numpy

    All slot-state operations are guarded by a single threading.Lock so
    the QueueManager and dma_proc can both call in safely (the mgr
    allocates from the QueueManager thread; dma_proc reads field_addr
    from the dma listener thread — disjoint reads, but the alloc
    side needs locking).
    """

    def __init__(self, layout: CCCacheLayout, name: str):
        self.layout = layout
        self.name = name
        self.path = f"/dev/shm/{name}.dat"
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.addr: int | None = None
        self._cuda_registered = False
        self._libcudart = None

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._free_list: list[int] = []
        # per-slot field validity bitset (3 fields → 3 bits, but a dict
        # of bool per field is clearer and the overhead is negligible).
        self._valid: list[dict[str, bool]] = []

    # ── lifecycle ──────────────────────────────────────────────

    def create(self) -> None:
        """Creator (the mpv process). Removes any stale shm with the
        same name, truncates, maps, zeros, optionally cuda-registers.
        Sets self.addr to a valid raw pointer."""
        if os.path.exists(self.path):
            os.unlink(self.path)
        self.fd = os.open(self.path,
                           os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(self.fd, self.layout.total_size)
        self._map()
        ctypes.memset(self.addr, 0, self.layout.total_size)
        # Build free-list in slot-id order. We pop from the end (LIFO)
        # so the hot slots stay warm in cache.
        self._free_list = list(range(self.layout.n_slots))
        self._valid = [
            {f: False for f in _ALL_FIELDS}
            for _ in range(self.layout.n_slots)
        ]

    def open(self) -> None:
        """Attacher (worker child). Opens an existing shm and maps it
        but does NOT touch the free-list (the creator owns allocation
        state)."""
        self.fd = os.open(self.path, os.O_RDWR)
        st = os.fstat(self.fd)
        if st.st_size != self.layout.total_size:
            raise RuntimeError(
                f"shm {self.path} size {st.st_size} ≠ expected "
                f"{self.layout.total_size}")
        self._map()

    def _map(self) -> None:
        self.mm = mmap.mmap(self.fd, self.layout.total_size,
                             flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

    def register_cuda_pinned(self, *, mapped: bool = True) -> None:
        """Each process that wants GPU access to this region must call
        this in its own cuda context. On Grace SoC host_ptr ==
        device_ptr after registration, so callers can treat
        self.addr + field_off as a valid GPU address."""
        libcudart = ctypes.CDLL("libcudart.so")
        libcudart.cudaHostRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        libcudart.cudaHostRegister.restype = ctypes.c_int
        flags = 0x01 | (0x02 if mapped else 0)
        rc = libcudart.cudaHostRegister(
            ctypes.c_void_p(self.addr),
            ctypes.c_size_t(self.layout.total_size), flags)
        if rc not in (0, 712):  # cudaErrorHostMemoryAlreadyRegistered
            raise RuntimeError(
                f"cc_cache cudaHostRegister failed: cudaError={rc}")
        self._cuda_registered = True
        self._libcudart = libcudart

    def cuda_device_ptr(self, slot: int = 0, field: str | None = None
                         ) -> int:
        """Returns the GPU-side address for a slot/field. On Grace
        equals host address."""
        if not self._cuda_registered:
            raise RuntimeError("call register_cuda_pinned() first")
        base = ctypes.c_void_p(0)
        self._libcudart.cudaHostGetDevicePointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
        self._libcudart.cudaHostGetDevicePointer.restype = ctypes.c_int
        rc = self._libcudart.cudaHostGetDevicePointer(
            ctypes.byref(base), ctypes.c_void_p(self.addr), 0)
        if rc != 0:
            raise RuntimeError(
                f"cc_cache cudaHostGetDevicePointer failed: {rc}")
        gpu_base = base.value or 0
        return gpu_base + self._slot_field_offset(slot, field)

    def close(self) -> None:
        if self._cuda_registered and self._libcudart is not None:
            try:
                self._libcudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
                self._libcudart.cudaHostUnregister.restype = ctypes.c_int
                self._libcudart.cudaHostUnregister(
                    ctypes.c_void_p(self.addr))
            except Exception:
                pass
            self._cuda_registered = False
        if self.mm is not None:
            try:
                self.mm.close()
            except BufferError:
                pass  # outstanding ctypes views — fine, process exit reaps
            self.mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def unlink(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    # ── slot allocation ───────────────────────────────────────

    def alloc(self) -> int | None:
        """Reserve a free slot. Returns None when exhausted (caller
        backpressures upstream). NOT blocking — blocking here would
        deadlock with consumer threads that need queue_mgr's lock to
        free slots."""
        with self._lock:
            if not self._free_list:
                return None
            slot = self._free_list.pop()
            for f in _ALL_FIELDS:
                self._valid[slot][f] = False
            return slot

    def free(self, slot: int) -> None:
        with self._cv:
            if slot < 0 or slot >= self.layout.n_slots:
                raise IndexError(f"bad slot {slot}")
            for f in _ALL_FIELDS:
                self._valid[slot][f] = False
            self._free_list.append(slot)
            self._cv.notify()

    def free_count(self) -> int:
        with self._lock:
            return len(self._free_list)

    def set_field(self, slot: int, field: str) -> None:
        with self._lock:
            self._valid[slot][field] = True

    def clear_field(self, slot: int, field: str) -> None:
        with self._lock:
            self._valid[slot][field] = False

    def has_field(self, slot: int, field: str) -> bool:
        with self._lock:
            return self._valid[slot][field]

    # ── addresses ─────────────────────────────────────────────

    def _slot_field_offset(self, slot: int, field: str | None) -> int:
        if not (0 <= slot < self.layout.n_slots):
            raise IndexError(f"bad slot {slot}")
        slot_off = slot * self.layout.slot_stride
        if field is None:
            return slot_off
        return slot_off + self.layout.field_off_in_slot(field)

    def field_addr(self, slot: int, field: str) -> int:
        """Host (== GPU on Grace) address of a slot's field. Use this
        as a ctypes void pointer or to construct __cuda_array_interface__
        views."""
        if self.addr is None:
            raise RuntimeError("cc_cache shm not mapped")
        return self.addr + self._slot_field_offset(slot, field)

    def field_view(self, slot: int, field: str) -> memoryview:
        """memoryview slice of the field. Useful for ctypes.memmove
        sources / numpy.frombuffer."""
        off = self._slot_field_offset(slot, field)
        size = self.layout.field_size(field)
        return memoryview(self.mm)[off:off + size]


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────


def _self_test() -> None:
    # 1080p source, model 4.26: modulo_base=64, so pH=1088, pW=1920.
    # enc_channels for 4.26 is 4.
    layout = CCCacheLayout(
        n_slots=4, H_src=1080, W_src=1920, H_out=2160, W_out=3840,
        pH=1088, pW=1920, enc_channels=4)
    print(layout.describe())

    # sanity check sizes (fp16 throughout for rife & sr_yuv)
    assert layout.rgb_padded_size    == 1 * 3 * 1088 * 1920 * 2
    assert layout.rife_features_size == 1 * 4 * 1088 * 1920 * 2
    assert layout.sr_yuv_size        == 1920 * 1080 * 2 + 2 * 960 * 540 * 2
    assert layout.rgb_4K_size        == 3840 * 2160 * 3 * 2

    # offset monotonicity
    assert (layout.off_rgb_padded < layout.off_rife_features
            < layout.off_sr_yuv < layout.off_rgb_4K)
    assert layout.slot_size <= layout.slot_stride

    cache = CCCacheShm(layout, name=f"dgxspark_cc_cache_test_{os.getpid()}")
    cache.create()
    try:
        a = cache.alloc()
        assert a == 3, f"first alloc should be top of free-list (3), got {a}"

        view = cache.field_view(a, FIELD_RGB_PADDED)
        ctypes.memset(cache.field_addr(a, FIELD_RGB_PADDED), 0xAB, 8)
        assert bytes(view[:8]) == b"\xab" * 8

        cache.set_field(a, FIELD_RGB_PADDED)
        assert cache.has_field(a, FIELD_RGB_PADDED)
        assert not cache.has_field(a, FIELD_RIFE_FEATURES)
        cache.set_field(a, FIELD_RIFE_FEATURES)
        assert cache.has_field(a, FIELD_RIFE_FEATURES)

        # alloc exhaustion (3 more slots available)
        b = cache.alloc()
        c = cache.alloc()
        d = cache.alloc()
        assert cache.alloc() is None, "alloc should return None when exhausted"

        # free + re-alloc returns the same slot
        cache.free(b)
        b2 = cache.alloc()
        assert b == b2, f"free→alloc should recycle, got {b}→{b2}"

        # after free, has_field returns False
        cache.set_field(a, FIELD_SR_YUV)
        assert cache.has_field(a, FIELD_SR_YUV)
        cache.free(a)
        assert not cache.has_field(a, FIELD_SR_YUV)
        assert not cache.has_field(a, FIELD_RIFE_FEATURES)

        addr_0 = cache.field_addr(0, FIELD_RGB_PADDED)
        addr_1 = cache.field_addr(1, FIELD_RGB_PADDED)
        assert addr_1 - addr_0 == layout.slot_stride

        a_rgb = cache.field_addr(0, FIELD_RGB_PADDED)
        a_feat = cache.field_addr(0, FIELD_RIFE_FEATURES)
        a_sr   = cache.field_addr(0, FIELD_SR_YUV)
        assert a_feat - a_rgb == layout.off_rife_features
        assert a_sr   - a_rgb == layout.off_sr_yuv

        try:
            cache.register_cuda_pinned()
            gpu_a = cache.cuda_device_ptr(0, FIELD_RGB_PADDED)
            assert gpu_a == a_rgb, f"Grace UMA: gpu={hex(gpu_a)} cpu={hex(a_rgb)}"
            print("cuda_pinned: device==host (Grace UMA)")
        except OSError:
            print("cuda_pinned: skipped (no libcudart)")
        except RuntimeError as e:
            print(f"cuda_pinned: error {e}")

    finally:
        cache.close()
        cache.unlink()

    print("cc_cache self-test OK")


if __name__ == "__main__":
    _self_test()
