-- dual_fps_override.lua — work around mpv's vsapi rate-propagation bug
-- by computing the dual interp_mult from ffprobe and writing it to
-- /tmp/dual_machine_mult_override, which rife.vpy already consults
-- as the F9 cycle override file.
--
-- Why: mpv hands vapoursynth a `clip` whose fps_num/fps_den are 0/0
-- for some containers (notably MKV/MP4 with non-integer rates like
-- 30000/1001). rife.vpy's in-vpy heuristic then sees src_fps=0 and
-- falls into the `≤ 24 → mult=3` branch — so a 30 fps source plays
-- at mult=3 (≈ 89 fps output) instead of the intended mult=2, which
-- on a 60 Hz vo just stutters. ffprobe over the file path is the
-- reliable source of truth; we run it once at `on_load` and stamp
-- the right mult before vapoursynth init.
--
-- DUAL_FPS_OVERRIDE_DISABLE=1 → no-op.

local mp  = require "mp"
local msg = require "mp.msg"

local DUAL_OFF_FLAG      = "/tmp/dual_machine_disabled"
local DUAL_MULT_OVERRIDE = "/tmp/dual_machine_mult_override"

local function file_exists(p)
    local f = io.open(p, "r")
    if f then f:close(); return true end
    return false
end

local function read_int_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local s = f:read("*l")
    f:close()
    if not s then return nil end
    return tonumber(s)
end

local function ffprobe_video(path)
    -- Returns (fps_num, fps_den, height) or nils on failure. Probe
    -- fields one at a time because ffprobe's csv emits fields in the
    -- container's declaration order, not in the order requested.
    local function one(field)
        local cmd = string.format(
            "ffprobe -v error -select_streams v:0 "
            .. "-show_entries stream=%s -of csv=p=0 %q "
            .. "2>/dev/null < /dev/null",
            field, path)
        local f = io.popen(cmd, "r")
        if not f then return nil end
        local line = f:read("*l")
        f:close()
        return line
    end
    local fr = one("r_frame_rate")
    local hs = one("height")
    if not fr or not hs then return nil end
    local num, den = fr:match("^(%d+)/(%d+)$")
    local h = tonumber(hs)
    if not num then return nil end
    return tonumber(num), tonumber(den), h
end

-- Per-file auto-pick. Does NOT honor the DUAL_MULT_OVERRIDE file
-- (F9 mid-playback toggle in sr_keys.lua); each new file starts
-- from the resolution/fps auto-default. Env var pin still wins
-- (bench scripts).
--
--   > 30 fps           → mult=1  (no temporal upscale)
--   ≤ 720p   + ≤ 25    → mult=4
--   ≤ 720p   + 26-30   → mult=3
--   1080p/4K + ≤ 25    → mult=3
--   1080p/4K + 26-30   → mult=2
local function pick_dual_mult(src_fps, src_h)
    local env = os.getenv("DUAL_INTERP_MULT")
    if env then
        local n = tonumber(env)
        if n and n >= 1 and n <= 4 then return n end
    end
    if src_fps > 30.0 then return 1 end
    local max_mult = (src_h <= 720) and 4 or 3
    if src_fps <= 25.0 then return max_mult end
    return max_mult - 1
end

local function on_load()
    if os.getenv("DUAL_FPS_OVERRIDE_DISABLE") == "1" then return end
    if not os.getenv("DUAL_WORKER_HOST") then return end
    if file_exists(DUAL_OFF_FLAG) then return end
    local path = mp.get_property("path")
    if not path then return end
    -- Plain file paths only — skip http://, etc.
    if path:match("^%a+://") then return end
    local num, den, h = ffprobe_video(path)
    if not num or not den or den == 0 then return end
    local src_fps = num / den
    if src_fps <= 0 then return end
    local mult = pick_dual_mult(src_fps, h or 0)
    local f = io.open(DUAL_MULT_OVERRIDE, "w")
    if f then f:write(tostring(mult)); f:close() end
    msg.info(string.format(
        "dual mult sidecar: src=%.3f h=%d → mult=%d (wrote %s)",
        src_fps, h or 0, mult, DUAL_MULT_OVERRIDE))
end

mp.add_hook("on_load", 50, on_load)
