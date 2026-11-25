"""Run on two nodes to verify peer RDMA framing and payload integrity.

Both endpoints need matching rail and port arguments. Start the server
before the client; this deliberately does not start a worker service.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from peer_link import PeerLink


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("peer")
    parser.add_argument("--rails", required=True,
                        help="comma-separated RDMA device:gid entries")
    parser.add_argument("--port", type=int, default=30140)
    args = parser.parse_args()
    rails = [(device, int(gid)) for device, gid in
             (entry.rsplit(":", 1) for entry in args.rails.split(","))]
    sizes = (0, 1, 101, 8193, 1024 * 1024)
    link = PeerLink(rails, peer_ip=args.peer, port=args.port,
                    is_server=args.role == "server", max_bytes=max(sizes))
    # A single MR covers several frame slots, as it does in the worker's
    # local CC cache. Use different source and destination slots.
    capacity = max(sizes)
    memory = ctypes.create_string_buffer(3 * capacity)
    buffer = link.register(ctypes.addressof(memory), len(memory))
    try:
        for frame, size in enumerate(sizes):
            payload = bytes((index + frame) % 251 for index in range(size))
            offset = capacity if args.role == "server" else 2 * capacity
            if args.role == "server":
                if frame % 2:
                    link.send(7, frame, payload)
                else:
                    ctypes.memmove(ctypes.addressof(memory) + offset, payload, size)
                    link.send_registered(7, frame, buffer, size, offset=offset)
            else:
                if frame % 2:
                    generation, received_frame, received = link.receive()
                else:
                    generation, received_frame, received_size = link.receive_registered(
                        buffer, offset=offset)
                    received = ctypes.string_at(ctypes.addressof(memory) + offset,
                                                received_size)
                if (generation, received_frame) != (7, frame):
                    raise AssertionError("peer frame metadata mismatch")
                if hashlib.sha256(received).digest() != hashlib.sha256(payload).digest():
                    raise AssertionError("peer frame payload mismatch")
        print(f"Verified {len(sizes)} payloads across {len(rails)} rails.")
    finally:
        buffer.close()
        link.close()


if __name__ == "__main__":
    main()
