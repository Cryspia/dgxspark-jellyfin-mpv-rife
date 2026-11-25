"""Verify striped RDMA READ against a registered buffer on another node.

Run the server before the client, with matching device, GID and port options.
This is a transport probe; it does not measure the playback pipeline.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from rdma_transport import MultiRailBuffer, RDMAChannel, RDMAContext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("peer")
    parser.add_argument("--rails", required=True,
                        help="comma-separated RDMA device:gid entries")
    parser.add_argument("--port", type=int, default=30470)
    parser.add_argument("--size", type=int, default=24_883_200)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    rails = [(device, int(gid)) for device, gid in
             (entry.rsplit(":", 1) for entry in args.rails.split(","))]
    if not rails or args.size < len(rails) or args.iterations < 1:
        parser.error("size and iteration count must cover all rails")
    if args.timeout <= 0 or not 1024 <= args.port <= 65535 - len(rails):
        parser.error("invalid timeout or port range")
    contexts = [RDMAContext(dev_name=device, gid_index=gid, max_cqe=128)
                for device, gid in rails]
    buf = MultiRailBuffer(contexts, args.size)
    pattern = bytes(range(251))
    payload = (pattern * ((args.size + len(pattern) - 1) // len(pattern)))[:args.size]
    if args.role == "server":
        ctypes.memmove(buf.addr, payload, args.size)
        extra = struct.pack(f"!Q{len(rails)}I", buf.addr, *buf.rkeys)
    else:
        extra = b""
    channels = [RDMAChannel(ctx, is_server=args.role == "server", peer_ip=args.peer,
                            peer_port=args.port + i,
                            extra_payload=extra if i == 0 else b"",
                            handshake_timeout=args.timeout)
                for i, ctx in enumerate(contexts)]
    try:
        if args.role == "server":
            print("READ source registered and ready", flush=True)
            time.sleep(args.timeout)
            return
        addr, *keys = struct.unpack(f"!Q{len(rails)}I", channels[0].peer_extra)
        start = time.monotonic()
        for _ in range(args.iterations):
            for rail, channel in enumerate(channels):
                offset = args.size * rail // len(rails)
                end = args.size * (rail + 1) // len(rails)
                channel.post_read(dst_addr=buf.addr + offset, dst_lkey=buf.lkeys[rail],
                                  length=end - offset, remote_addr=addr + offset,
                                  rkey=keys[rail], wr_id=1)
            for context in contexts:
                deadline = time.monotonic() + args.timeout
                while True:
                    count, completions = context.cq.poll(1)
                    if count:
                        if completions[0].status != 0:
                            raise RuntimeError(f"READ completion failed: {completions[0].status}")
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("RDMA READ completion timed out")
        elapsed = time.monotonic() - start
        actual = hashlib.sha256(ctypes.string_at(buf.addr, args.size)).digest()
        if actual != hashlib.sha256(payload).digest():
            raise RuntimeError("RDMA READ payload mismatch")
        rate = args.size * args.iterations * 8 / elapsed / 1e9
        print(f"Verified {args.iterations} reads across {len(rails)} rails: "
              f"{elapsed:.3f}s, {rate:.2f} Gbit/s", flush=True)
    finally:
        for channel in channels:
            channel.qp.close()
        for mr in buf.mrs:
            mr.close()
        for context in contexts:
            context.cq.close()
            context.pd.close()
            context.ctx.close()


if __name__ == "__main__":
    main()
