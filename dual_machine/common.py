"""Shared cluster configuration, mode selection, and RDMA discovery.

Addresses are supplied by the operator or detected at runtime. A DGX Spark
can use two RDMA rails on separate PCIe paths; the transport stripes data
across the configured rails. See the multi-machine guide for network setup.
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
          {"rdma": "<rdma-0>", "iface": "<interface-0>", "ip": "<rail0-ip>"},
          {"rdma": "<rdma-1>", "iface": "<interface-1>", "ip": "<rail1-ip>"},
        ],
        "rdma_csv":  "<rdma-0>,<rdma-1>",
        "iface_csv": "<interface-0>,<interface-1>",
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
    # Sort by IPv4 so the configured rail ordering is deterministic.
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
    """Worker hostname/IP. Required env: CLUSTER_WORKER_HOST."""
    v = os.environ.get("CLUSTER_WORKER_HOST", "").strip()
    if not v:
        raise RuntimeError(
            "CLUSTER_WORKER_HOST not set — point this at the worker's RoCE IP "
            "or any name that resolves to it.")
    return v


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
        print("worker host: (CLUSTER_WORKER_HOST unset —", e, ")")


def use_tictac(worker_count: int, interp_mult: int, setting: str | None = None) -> bool:
    """Select frame-owned mode for two or more workers, unless overridden."""
    if worker_count < 2 or interp_mult not in (2, 3, 4):
        return False
    if setting is None or setting == "":
        return True
    return setting == "1"


def four_machine_policy() -> dict:
    """Load the playback thresholds shared with the mpv Lua sidecar."""
    path = Path(__file__).with_name("four_machine_policy.json")
    try:
        with path.open(encoding="utf-8") as source:
            policy = json.load(source)
        required = ("max_source_fps", "native_4k_max_source_fps",
                    "x4_1080_max_source_fps", "x3_max_source_fps",
                    "lowres_max_height", "lowres_max_width", "fourk_min_height", "fourk_min_width")
        if any(type(policy.get(key)) not in (int, float) or policy[key] <= 0
               for key in required):
            return {}
        return policy
    except (OSError, ValueError, TypeError):
        return {}


def default_interpolation_multiplier(fps: float, height: int, width: int,
                                     worker_count: int, setting: str | None = None,
                                     native_4k: str | None = None) -> int:
    """Choose a per-source multiplier; unknown rates use the conservative default."""
    import math
    if not math.isfinite(fps) or fps <= 0:
        return 2
    policy = four_machine_policy()
    if policy and worker_count >= 3 and use_tictac(worker_count, 2, setting):
        if fps > policy["max_source_fps"]:
            return 1
        fourk = height >= policy["fourk_min_height"] or width >= policy["fourk_min_width"]
        if (fourk and fps <= policy["native_4k_max_source_fps"]
                and native_4k != "0"):
            return 2
        if fps > policy["x3_max_source_fps"]:
            return 2
        if (height <= policy["lowres_max_height"] and width <= policy["lowres_max_width"]
                and not fourk):
            return 4
        if not fourk and fps <= policy["x4_1080_max_source_fps"]:
            return 4
        return 3
    if fps > 30:
        return 1
    max_mult = 4 if height <= 720 else 3
    return max_mult if fps <= 25 else max_mult - 1


def source_fps_from_sidecar() -> float:
    """Read only the rate written for the current mpv process and lifetime."""
    import math
    path = Path(f"/tmp/dual_machine_source_fps_{os.getpid()}")
    try:
        if path.stat().st_mtime < Path("/proc/self").stat().st_ctime:
            return 0.0
        rate = float(path.read_text(encoding="ascii").strip())
        return rate if math.isfinite(rate) and rate > 0 else 0.0
    except (OSError, ValueError):
        return 0.0
