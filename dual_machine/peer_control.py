"""Session-scoped commands for direct worker frame-cache transfers."""
from __future__ import annotations

import json
import socket
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor

_MAX_MESSAGE = 4096


def _read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("peer control connection closed")
        data.extend(chunk)
    return bytes(data)


def _read_message(sock):
    size = struct.unpack("!I", _read_exact(sock, 4))[0]
    if not 0 < size <= _MAX_MESSAGE:
        raise ValueError("invalid peer control message size")
    message = json.loads(_read_exact(sock, size))
    if not isinstance(message, dict):
        raise ValueError("invalid peer control message")
    return message


def _send_message(sock, message):
    data = json.dumps(message, separators=(",", ":")).encode()
    if len(data) > _MAX_MESSAGE:
        raise ValueError("peer control message too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


class WorkerPeerManager:
    """Own worker-to-worker QPs for one worker session.

    Shutdown waits for active operations and closes all QPs before the
    shared frame cache is deregistered or unmapped. Separate peer links
    may transfer concurrently.
    """

    def __init__(self, worker, rails):
        self.worker = worker
        self.rails = rails
        self.links = {}
        self.closed = False
        self._close_done = False
        self._connecting = set()
        self._active = 0
        self._cv = threading.Condition()
        self._activate_lock = threading.Lock()

    def execute(self, command):
        with self._cv:
            if self.closed:
                raise RuntimeError("worker peer manager is closed")
            self._active += 1
        try:
            self._execute(command)
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def _execute(self, command):
        from peer_link import FrameCachePeer, FrameCacheWritePeer, PeerLink
        operation = command.get("op")
        name = command.get("link")
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("invalid peer link name")
        if self.worker.local_cache is None:
            raise RuntimeError("worker peer manager is closed")
        if operation == "release_result":
            from mp_pipeline import ST_PULL_DONE, ST_SEND_PENDING
            slot, task_id = command.get("slot"), command.get("task_id")
            if type(slot) is not int or not 0 <= slot < self.worker.layout.n_slots:
                raise ValueError("invalid result slot")
            if type(task_id) is not int or task_id <= 0:
                raise ValueError("invalid result task id")
            with self._activate_lock:
                meta = self.worker.shm.slot_state(slot)
                if meta.state != ST_SEND_PENDING or meta.task_id != task_id:
                    raise ValueError("stale result release")
                meta.state = ST_PULL_DONE
            return
        if operation == "activate":
            from mp_pipeline import ST_PREFETCH_HELD, ST_RECV_DONE, TT_CCSR
            task_id = command.get("task_id")
            if type(task_id) is not int or task_id <= 0:
                raise ValueError("invalid prefetch task id")
            with self._activate_lock:
                for slot in range(self.worker.layout.n_slots):
                    meta = self.worker.shm.slot_state(slot)
                    if (meta.state == ST_PREFETCH_HELD and meta.task_id == task_id
                            and meta.task_type == TT_CCSR):
                        meta.state = ST_RECV_DONE
                        return
            raise ValueError("prefetch task is not held")
        if operation == "connect":
            port = command.get("port")
            server = command.get("server")
            host = command.get("host")
            token = command.get("token")
            transport = command.get("transport", "send")
            if (not isinstance(port, int) or not 1024 <= port <= 65534
                    or not isinstance(server, bool)
                    or not isinstance(host, str) or len(host) > 255
                    or not isinstance(token, str) or len(token) != 32
                    or transport not in ("send", "write")):
                raise ValueError("invalid peer endpoint")
            try:
                token_bytes = bytes.fromhex(token)
            except ValueError:
                raise ValueError("invalid peer link token") from None
            with self._cv:
                if name in self.links or name in self._connecting:
                    raise ValueError("peer link already exists")
                self._connecting.add(name)
            cache = self.worker.local_cache
            try:
                if transport == "write":
                    link = FrameCacheWritePeer(self.rails, cache=cache, peer_ip=host,
                                               port=port, is_server=server,
                                               timeout=10.0, token=token_bytes)
                    peer = link
                else:
                    link = PeerLink(self.rails, peer_ip=host, port=port,
                                    is_server=server, max_bytes=cache.layout.off_sr_yuv,
                                    timeout=10.0, token=token_bytes)
                    try:
                        peer = FrameCachePeer(link, cache)
                    except Exception:
                        link.close()
                        raise
                with self._cv:
                    self.links[name] = (link, peer)
            finally:
                with self._cv:
                    self._connecting.remove(name)
            return
        with self._cv:
            if name not in self.links:
                raise ValueError("unknown peer link")
            link, peer = self.links[name]
        if operation in ("send", "receive", "write"):
            generation, frame, slot = (command.get(k) for k in
                                       ("generation", "frame", "slot"))
            if any(type(value) is not int or value < 0
                   for value in (generation, frame, slot)):
                raise ValueError("invalid peer frame coordinates")
            if operation == "send":
                peer.send(generation, frame, slot)
            elif operation == "write":
                destination = command.get("destination_slot")
                if type(destination) is not int or destination < 0:
                    raise ValueError("invalid destination slot")
                peer.write(slot, destination)
            else:
                peer.receive(generation, frame, slot)
        elif operation == "close":
            peer.close()
            link.close()
            with self._cv:
                self.links.pop(name, None)
        else:
            raise ValueError("unknown peer control operation")

    def close(self):
        with self._cv:
            if self.closed:
                while not self._close_done:
                    self._cv.wait()
                return
            self.closed = True
            while self._active:
                self._cv.wait()
            links = tuple(self.links.values())
            self.links.clear()
        try:
            for link, peer in links:
                peer.close()
                link.close()
        finally:
            with self._cv:
                self._close_done = True
                self._cv.notify_all()


def serve_peer_commands(sock, worker_box, rails, log):
    """Read commands until disconnect, dispatching potentially blocking I/O."""
    write_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="peer-control")
    manager = None

    def run(request):
        nonlocal manager
        try:
            worker = worker_box.get("wmp")
            if worker is None:
                raise RuntimeError("worker is not ready")
            if manager is None:
                with worker._shutdown_lock:
                    if manager is None:
                        manager = WorkerPeerManager(worker, rails)
                        worker.peer_manager = manager
            manager.execute(request)
            reply = {"id": request["id"], "ok": True}
        except Exception as exc:
            reply = {"id": request.get("id"), "ok": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        try:
            with write_lock:
                _send_message(sock, reply)
        except OSError:
            pass

    try:
        while True:
            request = _read_message(sock)
            if type(request.get("id")) is not int:
                raise ValueError("invalid peer control request id")
            executor.submit(run, request)
    except EOFError:
        pass
    finally:
        # Fence jobs before the session can release the shared cache.
        executor.shutdown(wait=True, cancel_futures=True)
        if manager is not None:
            manager.close()
        log("peer control stopped")


class PeerControlClient:
    """Multiplex commands and completions on an established session socket."""

    def __init__(self, sock):
        self.sock = sock
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 1
        self._error = None
        self.on_disconnect = None
        self._reader = threading.Thread(target=self._read, daemon=True,
                                        name="peer-control-reader")
        self._reader.start()

    def submit(self, operation, link, **args):
        future = Future()
        with self._lock:
            if self._error is not None:
                future.set_exception(self._error)
                return future
            request_id = self._next_id
            self._next_id += 1
            self._pending[request_id] = future
            try:
                _send_message(self.sock, dict(id=request_id, op=operation,
                                              link=link, **args))
            except Exception as exc:
                del self._pending[request_id]
                future.set_exception(exc)
        return future

    def _read(self):
        try:
            while True:
                reply = _read_message(self.sock)
                with self._lock:
                    future = self._pending.pop(reply.get("id"), None)
                if future is None:
                    raise ValueError("unknown peer control response")
                if reply.get("ok") is True:
                    future.set_result(None)
                else:
                    future.set_exception(RuntimeError(reply.get("error", "peer command failed")))
        except Exception as exc:
            with self._lock:
                self._error = exc
                pending, self._pending = self._pending, {}
            for future in pending.values():
                future.set_exception(exc)
            callback = self.on_disconnect
            if callback is not None:
                callback(exc)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self._reader.join(timeout=2)
