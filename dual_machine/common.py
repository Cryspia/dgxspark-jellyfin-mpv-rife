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
- The RoCEv2 GID index matching that address (what `DUAL_RDMA_GID`
  pins at install time).
- The local username (`$USER` / `getpass.getuser()`).

Fabric shape (see the spark-roce repo for the authoritative table).
A DGX Spark has ONE ConnectX-7 ASIC — all four functions report the
same phys_switch_id — with two physical ports, attached to the host
over TWO PCIe5 x4 root complexes. Each port is exposed on BOTH root
complexes (socket direct), so the four netdevs are 2 ports x 2 PCIe
paths, not four ports:

    netdev          function      devlink   PCIe path
    enp1s0f0np0     0000:01:00.0  port 0    A
    enP2p1s0f0np0   0002:01:00.0  port 0    B
    enp1s0f1np1     0000:01:00.1  port 1    A
    enP2p1s0f1np1   0002:01:00.1  port 1    B

Each PORT is 200G-capable (ethtool advertises 200000baseCR4/CR2).
Each PCIe PATH is x4 at 32 GT/s, about 128 Gb/s raw, so one path
cannot carry 200G. Hence the two root complexes.

So 200 Gb/s always means driving two PCIe paths in parallel — two
rdma devices. Cabling is a separate choice: one 200G cable in port 0
driven by enp1s0f0np0 + enP2p1s0f0np0 gets there, and so do two 100G
cables driven by one netdev per port on different paths. What is never
possible is 200 Gb/s through a single netdev, PCIe path, or QP.

This fabric uses the two-cable form (the switch and DACs are QSFP28
100G) and addresses exactly one netdev per port, on different paths.
Measured 2026-09-19 with ib_write_bw: 98.01 Gb/s per rail, 196.02
Gb/s with both in parallel. This project's transport opens ONE context
and one QP pair, so it uses one rail.
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
        # ".. inet <addr>/24 .."
        parts = line.split()
        try:
            i = parts.index("inet")
            return parts[i + 1].split("/")[0]
        except (ValueError, IndexError):
            continue
    return None


def roce_v2_gid_index(rdma_dev: str, ip: str) -> int | None:
    """Index of the RoCEv2 GID on `rdma_dev` port 1 that maps to IPv4
    `ip`, or None.

    The transport hard-codes 3 as a fallback. That holds only while the
    rail carries one IPv4 and nothing else. Entries are ordered by when
    the kernel added them, so one extra address — e.g. an IPv6 RA
    leaking in from a switch that bridges the fabric ports into a
    general-purpose LAN — shifts every index after it.
    """
    port = Path(f"/sys/class/infiniband/{rdma_dev}/ports/1")
    want = ":".join(f"{int(a):02x}{int(b):02x}"
                    for a, b in zip(ip.split(".")[::2], ip.split(".")[1::2]))
    try:
        indices = sorted(int(f.name) for f in (port / "gids").iterdir())
    except OSError:
        return None
    for i in indices:
        try:
            gid = (port / "gids" / str(i)).read_text().strip()
            typ = (port / "gid_attrs" / "types" / str(i)).read_text().strip()
        except OSError:
            continue
        if typ == "RoCE v2" and gid.endswith(f":ffff:{want}"):
            return i
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
          {"rdma": "rocep1s0f0",   "iface": "enp1s0f0np0",   "ip": "<rail0-ip>"},
          {"rdma": "roceP2p1s0f1", "iface": "enP2p1s0f1np1", "ip": "<rail1-ip>"},
        ],
        "rdma_csv":  "rocep1s0f0,roceP2p1s0f1",
        "iface_csv": "enp1s0f0np0,enP2p1s0f1np1",
        "ip_csv":    "<rail0-ip>,<rail1-ip>",
      }
    Rails without an IP or with no link carrier are skipped. That is
    what separates a real rail from the second PCIe view of the same
    wire: all four netdevs report carrier, only the two addressed ones
    are rails. Do not filter on name — the two schemes (enp1s0f* and
    enP<n>p1s0f*) carry no rule that survives a re-cable.
    """
    rails = []
    for dev in _ib_devices():
        iface = _ib_netdev(dev)
        if iface is None or not _carrier_up(iface):
            continue
        ip = _ip_for_iface(iface)
        if ip is None:
            continue
        rails.append({"rdma": dev, "iface": iface, "ip": ip,
                      "gid": roce_v2_gid_index(dev, ip)})
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
