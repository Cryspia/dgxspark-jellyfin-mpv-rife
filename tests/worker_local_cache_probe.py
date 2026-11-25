"""Compare worker-local inputs with the existing inline RDMA path.

Run against an idle worker. This uses the experimental cache handshake;
it does not change the installed worker configuration.
"""
import argparse
import ctypes
import os
import socket
import struct
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dual_machine"))
from guest_mp import GuestMP, TT_CCSR, TT_INTERP, TT_SR_INTERP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker")
    parser.add_argument("--mult", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--port", type=int, default=29900)
    parser.add_argument("--fused", action="store_true")
    args = parser.parse_args()
    h, w, ph, pw = 1080, 1920, 1088, 1920
    os.environ.update(CLUSTER_LOCAL_FRAME_CACHE="1", CLUSTER_MID_DELIVERY="1",
                      CLUSTER_INTERP_MULT=str(args.mult), CLUSTER_DOWNSAMPLE_PRE="1",
                      CLUSTER_WORKER_FRAME_IO="1",
                      CLUSTER_WORKER_FUSED="1" if args.fused else "0")
    if args.fused and args.mult != 2:
        parser.error("fused probe requires --mult 2")
    sock = socket.create_connection((args.worker, args.port + 5), timeout=5)
    sock.settimeout(300)
    handshake = (7 if args.fused else 5, h, w, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
                 ph, pw, args.mult, 0, 4, 1)
    sock.sendall(struct.pack("<20q", *handshake))
    if sock.recv(1) != b"R":
        raise RuntimeError("worker did not become ready")
    completed = {}
    data = {}

    def save(slot, payload):
        if payload[0] == "ccsr":
            _, a, an, b, bn = payload
            data[slot] = ctypes.string_at(a, an) + ctypes.string_at(b, bn)
        elif payload[0] == "rgb_interp":
            _, addr, size = payload
            data[slot] = ctypes.string_at(addr, size)
        else:
            raise RuntimeError(f"unexpected payload {payload[0]}")

    guest = GuestMP(H=h, W=w, scale=2, pH=ph, pW=pw, enc_ch=4,
                    n_slots=2, peer_ip=args.worker,
                    peer_handshake_port=args.port,
                    cc_cache_writeback_fn=save,
                    task_done_fn=lambda task_id: completed[task_id].set(),
                    log=lambda msg: print(msg, flush=True))

    def run(task_id, task_type, payload, input_data=None, out=None):
        print(f"Submitting task {task_id} type {task_type}", flush=True)
        event = completed[task_id] = threading.Event()
        slot = guest.claim_slot(timeout=10)
        if slot is None:
            raise TimeoutError("no guest task slot")
        node = SimpleNamespace(id=task_id, type=task_type, frame_idx=0,
                               payload=payload)
        kwargs = {}
        if out is not None:
            kwargs = dict(dst_y_va=ctypes.addressof(out[0]),
                          dst_u_va=ctypes.addressof(out[1]),
                          dst_v_va=ctypes.addressof(out[2]),
                          dst_y_stride=2 * w * 2,
                          dst_uv_stride=w * 2)
        guest.fill_request(slot, node, **kwargs)
        if input_data is not None:
            ctypes.memmove(guest.get_send_buffer_addr(slot) + guest.get_send_payload_offset(),
                           input_data, len(input_data))
        guest.notify_submit(slot)
        if not event.wait(60):
            raise TimeoutError(f"task {task_id} did not complete")
        print(f"Completed task {task_id}", flush=True)

    try:
        rng = np.random.default_rng(713)
        for index in range(2):
            frame = rng.integers(64, 940, size=h * w + h * w // 2,
                                 dtype=np.uint16).tobytes()
            run(index + 1, TT_CCSR, {"dst_cc_slot": index}, frame)
        inline = data[0] + data[1]
        local_destinations = list(range(2, args.mult + 1))
        local_payload = {"worker_local_inputs": True,
                         "worker_src_cc_slot_a": 0,
                         "worker_src_cc_slot_b": 1,
                         "worker_dst_cc_slot": local_destinations[0],
                         "dst_cc_slot": 10}
        if args.mult >= 3:
            local_payload.update(worker_dst_cc_slot_2=local_destinations[1],
                                 dst_cc_slot_2=11)
        if args.mult >= 4:
            local_payload.update(worker_dst_cc_slot_3=local_destinations[2],
                                 dst_cc_slot_3=12)
        run(10, TT_INTERP, local_payload)
        inline_payload = {"dst_cc_slot": 20, "dst_cc_slot_2": 21,
                          "dst_cc_slot_3": 22}
        run(11, TT_INTERP, inline_payload, inline)
        for phase in range(args.mult - 1):
            if data[10 + phase] != data[20 + phase]:
                raise AssertionError(f"interpolation phase {phase + 1} differs")
            make_out = lambda: [ctypes.create_string_buffer(size) for size in
                                (h * w * 8, h * w * 2, h * w * 2)]
            local_out, inline_out = make_out(), make_out()
            run(20 + phase * 2, TT_SR_INTERP,
                {"worker_local_inputs": True,
                 "worker_src_cc_slot_a": local_destinations[phase]}, out=local_out)
            run(21 + phase * 2, TT_SR_INTERP, {}, data[20 + phase], inline_out)
            if any(a.raw != b.raw for a, b in zip(local_out, inline_out)):
                raise AssertionError(f"super resolution phase {phase + 1} differs")
        if args.fused:
            fused_out = make_out()
            run(40, TT_INTERP, {**local_payload, "worker_keep_data": True,
                                "worker_fused_sr": True}, out=fused_out)
            if any(a.raw != b.raw for a, b in zip(fused_out, local_out)):
                raise AssertionError("fused interpolation and SR differ")
        print(f"Worker-local interpolation and SR match inline path at x{args.mult}.")
    finally:
        guest.close()
        sock.close()


if __name__ == "__main__":
    main()
