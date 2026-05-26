#!/usr/bin/env python3
"""Dual-machine secondary box system tray app.

Tiny GTK + AppIndicator app that shows up in the GNOME menu bar with
the mpv icon. The user sees that the worker is running and gets a
right-click menu with two items:

  - "Worker status: running / stopped" (live, polls systemd)
  - "Quit"  (stops the systemd unit and exits the tray)

That's all — the tray exists so a normal desktop user has a visible
sign that the box is acting as a dual-mode worker and a way to stop
it without opening a terminal.
"""
from __future__ import annotations

import os
import subprocess
import sys

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")
from gi.repository import Gtk, GLib, AyatanaAppIndicator3 as AppIndicator

SERVICE_NAME = "dgxspark-dual-worker.service"
APP_ID       = "dgxspark-dual-secondary"
ICON_NAME    = "mpv"      # fall through to system mpv icon


def _systemctl(*args: str) -> tuple[int, str]:
    p = subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True, text=True, check=False,
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def _is_running() -> bool:
    rc, _ = _systemctl("is-active", "--quiet", SERVICE_NAME)
    return rc == 0


class Tray:
    def __init__(self) -> None:
        self.ind = AppIndicator.Indicator.new(
            APP_ID, ICON_NAME,
            AppIndicator.IndicatorCategory.APPLICATION_STATUS,
        )
        self.ind.set_status(AppIndicator.IndicatorStatus.ACTIVE)

        self.menu = Gtk.Menu()
        self.status_item = Gtk.MenuItem(label="Worker status: ?")
        self.status_item.set_sensitive(False)
        self.menu.append(self.status_item)
        self.menu.append(Gtk.SeparatorMenuItem())
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self._on_quit)
        self.menu.append(quit_item)
        self.menu.show_all()
        self.ind.set_menu(self.menu)

        self._refresh()
        GLib.timeout_add_seconds(3, self._refresh)

    def _refresh(self) -> bool:
        if _is_running():
            self.status_item.set_label("Worker status: running")
        else:
            self.status_item.set_label("Worker status: stopped")
        return True   # keep timer alive

    def _on_quit(self, _widget) -> None:
        # Stop the worker service first, then exit the tray.
        _systemctl("stop", SERVICE_NAME)
        Gtk.main_quit()


def main() -> int:
    if os.geteuid() == 0:
        print("secondary_tray: refusing to run as root", file=sys.stderr)
        return 1
    Tray()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
