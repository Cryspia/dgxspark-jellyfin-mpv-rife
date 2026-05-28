#!/usr/bin/env bash
# Build / install / uninstall the VK_LAYER_PRIORITY_BOOST Vulkan layer.
#
# Why system-global: the Vulkan loader treats ~/.local/share/vulkan/
# as untrusted when the consumer binary has file caps (AT_SECURE=1).
# mpv needs cap_sys_nice to request VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
# so the layer manifest + .so must live in a system path that the
# loader trusts under AT_SECURE — /usr/share/vulkan/implicit_layer.d
# and /usr/local/lib. That requires sudo (verified user-local was
# silently skipped by cap'd mpv on this system, May 2026).
#
# Usage:
#   bash build.sh                  # build only (no install)
#   bash build.sh install          # sudo install layer + setcap mpv
#   bash build.sh uninstall        # sudo remove layer + drop cap
#
# After install, opt into HIGH priority per mpv launch:
#   export VK_PRIORITY_BOOST_LEVEL=high
#   export VK_PRIORITY_BOOST_VERBOSE=1   # optional trace on stderr
# Default (env unset) is MEDIUM, which is the driver default = no-op.
# Global kill-switch: VK_PRIORITY_BOOST_DISABLE=1 — the layer skips
# itself entirely.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MPV_BIN="${MPV_BIN:-/home/spark/miniforge3/envs/vsmpv/bin/mpv}"
VK_INC="${VK_INC:-/home/spark/miniforge3/envs/vsmpv/include}"

USER_LIB_OUT="$HOME/.local/lib/vulkan/libVkLayer_priority_boost.so"
SYS_LIB="/usr/local/lib/libVkLayer_priority_boost.so"
SYS_MAN="/usr/share/vulkan/implicit_layer.d/VkLayer_priority_boost.json"

# Old user-local manifest path: silently ignored by cap'd mpv but
# present from earlier iterations. We sweep it in uninstall.
OLD_USER_MAN_EXPLICIT="$HOME/.local/share/vulkan/explicit_layer.d/VkLayer_priority_boost.json"
OLD_USER_MAN_IMPLICIT="$HOME/.local/share/vulkan/implicit_layer.d/VkLayer_priority_boost.json"

build() {
    mkdir -p "$(dirname "$USER_LIB_OUT")"
    gcc -O2 -fPIC -shared \
        -I"$VK_INC" \
        -o "$USER_LIB_OUT" \
        "$HERE/priority_boost.c"
    echo "Built: $USER_LIB_OUT"
}

manifest() {
    python3 - <<PY
import json
m = {
    "file_format_version": "1.2.0",
    "layer": {
        "name": "VK_LAYER_PRIORITY_BOOST",
        "type": "GLOBAL",
        "library_path": "$SYS_LIB",
        "api_version": "1.3.275",
        "implementation_version": "1",
        "description": "Injects VkDeviceQueueGlobalPriorityCreateInfoKHR "
                       "at vkCreateDevice. No-op unless "
                       "VK_PRIORITY_BOOST_LEVEL is set (low|medium|high|"
                       "realtime). HIGH/REALTIME require cap_sys_nice on "
                       "the calling binary.",
        "functions": {
            "vkNegotiateLoaderLayerInterfaceVersion":
                "vkNegotiateLoaderLayerInterfaceVersion"
        },
        "disable_environment": { "VK_PRIORITY_BOOST_DISABLE": "1" }
    }
}
import sys; json.dump(m, sys.stdout, indent=4)
PY
}

do_install() {
    build
    echo
    echo "=== installing to system paths (sudo) ==="
    sudo install -m 755 -D "$USER_LIB_OUT" "$SYS_LIB"
    manifest | sudo tee "$SYS_MAN" >/dev/null
    sudo chmod 644 "$SYS_MAN"
    echo
    echo "=== granting cap_sys_nice to $MPV_BIN (sudo) ==="
    sudo setcap cap_sys_nice+ep "$MPV_BIN"
    getcap "$MPV_BIN"
    echo
    echo "Layer: $SYS_LIB"
    echo "Manifest: $SYS_MAN"
    echo
    echo "Opt-in per mpv launch:"
    echo "  export VK_PRIORITY_BOOST_LEVEL=high"
}

do_uninstall() {
    echo "=== removing system-global layer artifacts (sudo) ==="
    sudo rm -fv "$SYS_LIB" "$SYS_MAN"
    echo
    echo "=== dropping cap from $MPV_BIN (sudo) ==="
    if getcap "$MPV_BIN" | grep -q .; then
        sudo setcap -r "$MPV_BIN"
    fi
    getcap "$MPV_BIN" || true
    echo
    echo "=== sweeping user-local artifacts (no sudo) ==="
    rm -fv "$USER_LIB_OUT" "$OLD_USER_MAN_EXPLICIT" "$OLD_USER_MAN_IMPLICIT"
}

case "${1:-build}" in
    build)     build ;;
    install)   do_install ;;
    uninstall) do_uninstall ;;
    *) echo "usage: $0 [build|install|uninstall]" >&2; exit 2 ;;
esac
