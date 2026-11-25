"""Verify direct transfers between real shared frame-cache slots on two nodes."""
import argparse
import ctypes
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from cc_cache import CCCacheLayout, CCCacheShm
from peer_link import FrameCachePeer, PeerLink


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("peer")
    parser.add_argument("--rails", required=True)
    parser.add_argument("--port", type=int, default=30220)
    args = parser.parse_args()
    rails = [(dev, int(gid)) for dev, gid in
             (item.rsplit(":", 1) for item in args.rails.split(","))]
    layout = CCCacheLayout(n_slots=4, H_src=540, W_src=960,
                           H_out=1080, W_out=1920, pH=544, pW=960,
                           enc_channels=4)
    cache = CCCacheShm(layout, name=f"codex_peer_probe_{os.getpid()}")
    cache.create()
    link = None
    peer = None
    try:
        link = PeerLink(rails, peer_ip=args.peer, port=args.port,
                        is_server=args.role == "server",
                        max_bytes=layout.off_sr_yuv)
        peer = FrameCachePeer(link, cache)
        for frame in range(4):
            slot = frame if args.role == "server" else 3 - frame
            addr = cache.addr + slot * layout.slot_stride
            payload = bytes((i + frame) % 251 for i in range(layout.off_sr_yuv))
            if args.role == "server":
                ctypes.memmove(addr, payload, len(payload))
                peer.send(9, frame, slot)
            else:
                # The field following the transferred data must stay intact.
                ctypes.memset(addr + layout.off_sr_yuv, 0xA5, 32)
                peer.receive(9, frame, slot)
                actual = ctypes.string_at(addr, len(payload))
                assert hashlib.sha256(actual).digest() == hashlib.sha256(payload).digest()
                assert ctypes.string_at(addr + layout.off_sr_yuv, 32) == b"\xa5" * 32
        print(f"Verified 4 frame slots across {len(rails)} rails.", flush=True)
    finally:
        if peer is not None:
            peer.close()
        if link is not None:
            link.close()
        cache.close()
        cache.unlink()


if __name__ == "__main__":
    main()
