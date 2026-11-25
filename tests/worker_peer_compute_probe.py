"""Exercise GPU frame preparation, direct peer transfer and interpolation."""
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
from peer_control import PeerControlClient
from remote_endpoint import RemoteEndpoint
from tictac_runtime import TicTacRuntime


class Worker:
    def __init__(self, host, port, *, write_peer=False):
        self.h, self.w = 1080, 1920
        self.results = {}
        self.events = {}
        self.stages = []
        self.stage_events = {}
        self.prefetch_events = {}
        self.writebacks = 0
        self.sock = socket.create_connection((host, port + 5), timeout=5)
        self.sock.settimeout(300)
        hs = (9 if write_peer else 6, self.h, self.w, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
              1088, 1920, 2, 0, 4, 1)
        self.sock.sendall(struct.pack("<20q", *hs))
        if self.sock.recv(1) != b"R":
            raise RuntimeError("worker did not become ready")
        self.sock.settimeout(None)
        self.control = PeerControlClient(self.sock)
        self.guest = GuestMP(H=self.h, W=self.w, scale=2, pH=1088, pW=1920,
                             enc_ch=4, n_slots=2, peer_ip=host,
                             peer_handshake_port=port,
                             cc_cache_writeback_fn=self._save,
                             task_done_fn=lambda task_id: self.events[task_id].set(),
                             task_stage_done_fn=self._stage_done,
                             task_prefetch_ready_fn=self._prefetch_ready,
                             log=lambda msg: print(msg, flush=True))

    def _save(self, slot, payload):
        self.writebacks += 1
        if payload[0] == "ccsr":
            _, a, an, b, bn = payload
            self.results[slot] = ctypes.string_at(a, an) + ctypes.string_at(b, bn)
        elif payload[0] == "rgb_interp":
            _, addr, size = payload
            self.results[slot] = ctypes.string_at(addr, size)
        else:
            raise ValueError("unexpected worker result")

    def _stage_done(self, task_id, stage):
        self.stages.append((task_id, stage))
        self.stage_events.setdefault((task_id, stage), threading.Event()).set()

    def _prefetch_ready(self, task_id):
        self.prefetch_events.setdefault(task_id, threading.Event()).set()

    def run(self, task_id, kind, payload, data=None, out=None, wait=True):
        self.events[task_id] = threading.Event()
        slot = self.guest.claim_slot(timeout=10)
        if slot is None:
            raise TimeoutError("no worker slot")
        kwargs = {}
        if out is not None:
            kwargs = dict(dst_y_va=ctypes.addressof(out[0]),
                          dst_u_va=ctypes.addressof(out[1]),
                          dst_v_va=ctypes.addressof(out[2]),
                          dst_y_stride=self.w * 4, dst_uv_stride=self.w * 2)
        self.guest.fill_request(slot, SimpleNamespace(
            id=task_id, type=kind, frame_idx=0, payload=payload), **kwargs)
        if data is not None:
            addr = self.guest.get_send_buffer_addr(slot) + self.guest.get_send_payload_offset()
            ctypes.memmove(addr, data, len(data))
        self.guest.notify_submit(slot)
        if wait and not self.events[task_id].wait(60):
            raise TimeoutError(f"worker task {task_id} timed out")
        return self.events[task_id]

    def close(self):
        self.guest.close()
        self.control.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--port", type=int, default=29900)
    parser.add_argument("--peer-port", type=int, default=30260)
    parser.add_argument("--write-peer", action="store_true")
    args = parser.parse_args()
    os.environ.update(CLUSTER_LOCAL_FRAME_CACHE="1", CLUSTER_MID_DELIVERY="1",
                      CLUSTER_INTERP_MULT="2", CLUSTER_DOWNSAMPLE_PRE="1",
                      CLUSTER_WORKER_FRAME_IO="1", CLUSTER_WORKER_PREFETCH="1")
    workers = []
    try:
        workers.append(Worker(args.first, args.port, write_peer=args.write_peer))
        workers.append(Worker(args.second, args.port, write_peer=args.write_peer))
        first, second = workers
        rng = np.random.default_rng(713)
        frames = []
        def output_buffers():
            return [ctypes.create_string_buffer(n) for n in
                    (1080 * 1920 * 8, 1080 * 1920 * 2, 1080 * 1920 * 2)]
        original_outputs = []
        for worker in workers:
            data = rng.integers(64, 940, size=1080 * 1920 * 3 // 2,
                                dtype=np.uint16).tobytes()
            frames.append(data)
            out = output_buffers()
            original_outputs.append(out)
            worker.run(1, TT_CCSR, {"dst_cc_slot": 0}, data, out=out)
        a = first.control.submit("connect", "pair", host=args.second,
                                 port=args.peer_port, server=True,
                                 transport="write" if args.write_peer else "send",
                                 token="0123456789abcdef0123456789abcdef")
        b = second.control.submit("connect", "pair", host=args.first,
                                  port=args.peer_port, server=False,
                                  transport="write" if args.write_peer else "send",
                                  token="0123456789abcdef0123456789abcdef")
        a.result(30)
        b.result(30)
        before = second.writebacks
        resident_output = output_buffers()
        stage = second.stage_events.setdefault((4, 0), threading.Event())
        prefetch = second.prefetch_events.setdefault(4, threading.Event())
        final = second.run(4, TT_CCSR, {"dst_cc_slot": 0,
                           "worker_keep_data": True, "worker_prefetch": True}, frames[1],
                           out=resident_output, wait=False)
        if not prefetch.wait(60):
            raise TimeoutError("worker did not receive the prefetched frame")
        if final.is_set() or stage.is_set():
            raise AssertionError("worker computed prefetched data before activation")
        second.control.submit("activate", "scheduler", task_id=4).result(10)
        if not stage.wait(60):
            raise TimeoutError("worker did not publish prepared features")
        if args.write_peer:
            second.control.submit("write", "pair", generation=3, frame=1,
                                  slot=0, destination_slot=1).result(30)
        else:
            receive = first.control.submit("receive", "pair", generation=3,
                                           frame=1, slot=1)
            send = second.control.submit("send", "pair", generation=3,
                                         frame=1, slot=0)
            send.result(30)
            receive.result(30)
        if not final.wait(60):
            raise TimeoutError("worker did not return the real frame")
        if second.writebacks != before:
            raise AssertionError("resident CCSR returned its features to the host")
        if any(a.raw != b.raw for a, b in zip(original_outputs[1], resident_output)):
            raise AssertionError("resident CCSR changed the final real frame")
        local = {"worker_local_inputs": True, "worker_src_cc_slot_a": 0,
                 "worker_src_cc_slot_b": 1, "worker_dst_cc_slot": 2,
                 "dst_cc_slot": 10}
        first.run(2, TT_INTERP, local)
        first.run(3, TT_INTERP, {"dst_cc_slot": 11},
                  first.results[0] + second.results[0])
        if first.results[10] != first.results[11]:
            raise AssertionError("peer transfer changed interpolation output")
        reference_interp = output_buffers()
        first.run(6, TT_SR_INTERP, {}, first.results[11], out=reference_interp)
        before = first.writebacks
        first.run(5, TT_INTERP, {**local, "worker_keep_data": True})
        if first.writebacks != before:
            raise AssertionError("resident interpolation returned intermediate data")
        endpoint = RemoteEndpoint(second.guest, second.control)
        ybytes = 1080 * 1920 * 2
        cbytes = ybytes // 4
        planes = (frames[1][:ybytes], frames[1][ybytes:ybytes + cbytes],
                  frames[1][ybytes + cbytes:])
        frame_task = endpoint.submit_frame(9, planes, 3, prefetch=True)
        frame_task.prefetched.result(30)
        if frame_task.done.done():
            raise AssertionError("prefetched task ran before activation")
        endpoint.activate(frame_task).result(10)
        frame_task.prepared.result(30)
        frame_output = frame_task.done.result(30)
        if any(a.raw != b.raw for a, b in zip(
                original_outputs[1], (frame_output.y, frame_output.u, frame_output.v))):
            raise AssertionError("remote endpoint changed the source output")
        endpoint.interpolate(9, 0, 3, (4,)).done.result(30)
        result = endpoint.super_resolve(9, 4).done.result(30)
        if not any(result.y.raw):
            raise AssertionError("remote endpoint produced an empty interpolated frame")
        # Use both real endpoints through the dynamic scheduler, including
        # a second directed link for the opposite ownership order.
        a = first.control.submit("connect", "reverse", host=args.second,
                                 port=args.peer_port + 10, server=True,
                                 transport="write" if args.write_peer else "send",
                                 token="fedcba9876543210fedcba9876543210")
        b = second.control.submit("connect", "reverse", host=args.first,
                                  port=args.peer_port + 10, server=False,
                                  transport="write" if args.write_peer else "send",
                                  token="fedcba9876543210fedcba9876543210")
        a.result(30)
        b.result(30)
        first_endpoint = RemoteEndpoint(first.guest, first.control)
        runtime = TicTacRuntime({"first": first_endpoint, "second": endpoint},
                                {("first", "second"): "reverse",
                                 ("second", "first"): "pair"},
                                peer_write=args.write_peer)
        def split_frame(data):
            return (data[:ybytes], data[ybytes:ybytes + cbytes], data[ybytes + cbytes:])
        packed = [split_frame(frame) for frame in frames]
        packed.extend(split_frame(rng.integers(64, 940, size=1080 * 1920 * 3 // 2,
                                               dtype=np.uint16).tobytes())
                      for _ in range(2))
        try:
            for index in range(3):
                runtime.submit_interval(index, packed[index], packed[index + 1])
            real = runtime.wait_output(0, 0, timeout=60)
            interpolated = runtime.wait_output(0, 1, timeout=60)
            if any(a.raw != b.raw for a, b in zip(
                    original_outputs[0], (real.y, real.u, real.v))):
                raise AssertionError("scheduler changed the real frame")
            if any(a.raw != b.raw for a, b in zip(
                    reference_interp, (interpolated.y, interpolated.u, interpolated.v))):
                raise AssertionError("scheduler changed the interpolated frame")
            for index in (1, 2):
                for phase in (0, 1):
                    runtime.wait_output(index, phase, timeout=60)
        finally:
            runtime.close()
        print("GPU interpolation after direct peer transfer matches inline path.")
    finally:
        for worker in reversed(workers):
            worker.close()


if __name__ == "__main__":
    main()
