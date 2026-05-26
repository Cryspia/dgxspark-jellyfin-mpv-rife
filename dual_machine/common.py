"""Shared helpers for the dual-machine pipeline. The whole point of
this module is that **no machine-specific addresses / usernames are
ever hard-coded** — they're discovered at runtime.

What the operator provides
==========================
- `DUAL_WORKER_HOST` env var: an IP or DNS name on a RoCE rail. The
  client reaches the worker through it; the worker's RoCE IPs are
  inferred from the same `/sys/class/infiniband/*` walk on the
  worker side.
- (optional) `DUAL_MASTER_PORT`: defaults to 29500.
- (optional) `DUAL_NCCL_DEBUG`: defaults to "WARN".

What we discover automatically
==============================
- The RoCE RDMA devices on this machine (`/sys/class/infiniband/*`).
- The kernel network interfaces backing each RDMA device.
- The IPv4 address bound to each of those interfaces.
- The local username (`$USER` / `getpass.getuser()`).

DGX Spark's CX7 looks like two PCIe-split devices that share one 200G
link — both rails need to be visible for the RDMA control plane to
discover GIDs on either.
"""
from __future__ import annotations
import os, getpass, json, socket, subprocess
from pathlib import Path


def _ib_devices() -> list[str]:
    """RDMA device names visible to the kernel, in stable order."""
    p = Path("/sys/class/infiniband")
    if not p.exists():
        return []
    return sorted(d.name for d in p.iterdir())


def _ib_netdev(rdma_dev: str) -> str | None:
    """Network interface name (e.g. enp1s0f0np0) for the given RDMA
    device's port 1. None if the port has no Ethernet view."""
    base = Path(f"/sys/class/infiniband/{rdma_dev}/device/net")
    if not base.exists():
        return None
    # there's usually exactly one entry; if multiple, pick the lexically
    # first one to be deterministic
    netdevs = sorted(p.name for p in base.iterdir())
    return netdevs[0] if netdevs else None


def _ip_for_iface(iface: str) -> str | None:
    """First IPv4 address bound to `iface`, or None."""
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", iface],
            check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        # ".. inet 10.200.128.1/24 .."
        parts = line.split()
        try:
            i = parts.index("inet")
            return parts[i + 1].split("/")[0]
        except (ValueError, IndexError):
            continue
    return None


def _carrier_up(iface: str) -> bool:
    try:
        return Path(f"/sys/class/net/{iface}/carrier").read_text().strip() == "1"
    except OSError:
        return False


def discover_roce() -> dict:
    """Walk /sys and return a dict describing the RoCE rails on this box.

    Result:
      {
        "rails": [
          {"rdma": "rocep1s0f0", "iface": "enp1s0f0np0", "ip": "10.200.128.1"},
          {"rdma": "roceP2p1s0f0", "iface": "enP2p1s0f0np0", "ip": "10.200.129.1"},
        ],
        "rdma_csv":  "rocep1s0f0,roceP2p1s0f0",
        "iface_csv": "enp1s0f0np0,enP2p1s0f0np0",
        "ip_csv":    "10.200.128.1,10.200.129.1",
      }
    Rails without an IP or with no link carrier are skipped — we only
    return the ones actually usable for transport right now.
    """
    rails = []
    for dev in _ib_devices():
        iface = _ib_netdev(dev)
        if iface is None or not _carrier_up(iface):
            continue
        ip = _ip_for_iface(iface)
        if ip is None:
            continue
        rails.append({"rdma": dev, "iface": iface, "ip": ip})
    # Sort by IPv4 so both hosts pick the same rail ordering. NCCL's
    # OOB bootstrap socket lives on rail-0; if host and worker disagree
    # on which device is "rail 0", bootstrap can advertise an address
    # the peer can't reach.
    def _ip_key(r):
        parts = r["ip"].split(".")
        return tuple(int(p) for p in parts)
    rails.sort(key=_ip_key)
    return {
        "rails": rails,
        "rdma_csv":  ",".join(r["rdma"]  for r in rails),
        "iface_csv": ",".join(r["iface"] for r in rails),
        "ip_csv":    ",".join(r["ip"]    for r in rails),
    }


def worker_host() -> str:
    """Worker hostname/IP. Required env: DUAL_WORKER_HOST."""
    v = os.environ.get("DUAL_WORKER_HOST", "").strip()
    if not v:
        raise RuntimeError(
            "DUAL_WORKER_HOST not set — point this at the worker's RoCE IP "
            "or any name that resolves to it (used both for SSH bootstrap "
            "and as NCCL MASTER_ADDR).")
    return v


def master_port() -> int:
    return int(os.environ.get("DUAL_MASTER_PORT", "29500"))


def local_user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def pretty(d: dict) -> str:
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    roce = discover_roce()
    print("local user:", local_user())
    print("hostname:  ", socket.gethostname())
    print("RoCE rails:")
    print(pretty(roce))
    try:
        print("worker host:", worker_host())
    except RuntimeError as e:
        print("worker host: (DUAL_WORKER_HOST unset —", e, ")")
