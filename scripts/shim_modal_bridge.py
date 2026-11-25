#!/usr/bin/env python3
"""Adapt Shim 3.3's playback HUD to the danmaku script's modal UI."""

import sys
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {text.count(old)}")
    return text.replace(old, new, 1)


def patch_renderer(text):
    if "mpvtk-external-modal" in text:
        return text
    for signature in ("local function phud_skip_bind()", "function phud_bind_summon()",
                      "function phud_summon(src)"):
        text = replace_once(text, signature + "\n", signature + "\n    if state.external_modal then return end\n")
    marker = "-- What Select or a direction means while the bar is HIDDEN. One"
    bridge = """-- Let external ASS modals use the display without the playback HUD.
mp.register_script_message('mpvtk-external-modal', function(on)
    state.external_modal = (on == 'yes' or on == 'true' or on == '1')
    if state.external_modal then
        if state.phud.shown then phud_hide() end
        phud_unbind_summon()
        phud_skip_hide()
    elseif state.phud.mode and not state.phud.shown then
        phud_bind_summon()
        phud_skip_bind()
    end
    request_render()
end)

"""
    return replace_once(text, marker, bridge + marker)


def patch_danmaku(text):
    if "mpvtk-external-modal" in text:
        return text
    marker = "local function modal_clear_bindings()"
    helper = '''local function shim_external_modal(active)
    mp.commandv("script-message", "mpvtk-external-modal", active and "yes" or "no")
end

local function modal_input_get(opts)
    shim_external_modal(true)
    local user_closed = opts.closed
    opts.closed = function(...)
        shim_external_modal(false)
        if user_closed then user_closed(...) end
    end
    local ok, err = pcall(input.get, opts)
    if not ok then
        shim_external_modal(false)
        error(err)
    end
end

'''
    text = replace_once(text, marker, helper + marker)
    text = replace_once(text, '    modal.active = nil\nend\n\n-- ============================================================================\n-- Settings panel', '    modal.active = nil\n    shim_external_modal(false)\nend\n\n-- ============================================================================\n-- Settings panel')
    text = replace_once(text, '    modal_close()  -- close any other modal\n    modal.active = "panel"', '    modal_close()  -- close any other modal\n    shim_external_modal(true)\n    modal.active = "panel"')
    text = replace_once(text, '        mp.osd_message("无结果", 1.5)\n        return\n    end\n\n    local cursor = 1', '        mp.osd_message("无结果", 1.5)\n        return\n    end\n    shim_external_modal(true)\n\n    local cursor = 1')
    if text.count('input.get({') != 2:
        raise RuntimeError('Unexpected danmaku input.get layout')
    text = text.replace('input.get({', 'modal_input_get({')
    for key, name, handler in (("Shift+F10", "danmaku-panel", "panel_open"), ("Ctrl+F10", "danmaku-search", "open_search_input")):
        old = f'mp.add_key_binding("{key}", "{name}", {handler})'
        text = replace_once(text, old, old.replace('add_key_binding', 'add_forced_key_binding'))
    return text


def main():
    renderer, danmaku = map(Path, sys.argv[1:])
    # Validate both changes before writing either file.
    updated = [(renderer, patch_renderer(renderer.read_text(encoding="utf-8"))),
               (danmaku, patch_danmaku(danmaku.read_text(encoding="utf-8")))]
    for path, content in updated:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
