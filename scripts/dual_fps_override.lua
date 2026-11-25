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
-- on a 60 Hz vo just stutters. ffprobe over the source path/URL is the
-- reliable source of truth; we run it once at `on_load` and stamp
-- the right mult before vapoursynth init. ffprobe reads local files and
-- http(s) URLs alike, so this also covers Jellyfin streams (the primary
-- use case — every Jellyfin source is an https stream URL).
--
-- CLUSTER_FPS_OVERRIDE_DISABLE=1 → no-op.

local function cluster_getenv(name)
    local value = os.getenv(name)
    if value == nil and name:sub(1, 8) == "CLUSTER_" then
        return os.getenv("DUAL_" .. name:sub(9))
    end
    return value
end

local mp  = require "mp"
local msg = require "mp.msg"
local utils = require "mp.utils"
local PINNED_MULT = cluster_getenv("CLUSTER_INTERP_MULT")
local pid = tostring(mp.get_property_native("pid") or "")
local SOURCE_FPS_FILE = pid:match("^%d+$") and ("/tmp/dual_machine_source_fps_" .. pid) or nil

local function load_policy()
    local path = mp.command_native({"expand-path", "~~/dual_machine/four_machine_policy.json"})
    local file = path and io.open(path, "r")
    if not file then return nil end
    local content = file:read("*a")
    file:close()
    local policy = utils.parse_json(content)
    local fields = {"max_source_fps", "native_4k_max_source_fps", "x4_1080_max_source_fps",
                    "x3_max_source_fps", "lowres_max_height", "lowres_max_width", "fourk_min_height", "fourk_min_width"}
    if type(policy) ~= "table" then return nil end
    for _, field in ipairs(fields) do
        if type(policy[field]) ~= "number" or policy[field] <= 0 then return nil end
    end
    return policy
end

local function is_four_machine_tictac()
    local setting = cluster_getenv("CLUSTER_TICTAC")
    if setting and setting ~= "" and setting ~= "1" then return false end
    local hosts = cluster_getenv("CLUSTER_WORKER_HOSTS") or ""
    if hosts:match("^%s*$") then hosts = cluster_getenv("CLUSTER_WORKER_HOST") or "" end
    local count = 0
    for host in hosts:gmatch("[^,]+") do
        if not host:match("^%s*$") then count = count + 1 end
    end
    return count >= 3
end

local CLUSTER_OFF_FLAG      = "/tmp/dual_machine_disabled"
local CLUSTER_MULT_OVERRIDE = "/tmp/dual_machine_mult_override"

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
    -- One probe returns all fields without repeatedly fetching remote headers.
    local result = mp.command_native({
        name = "subprocess", playback_only = false,
        args = {"timeout", "8", "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,height,width", "-of", "json", path},
        capture_stdout = true, capture_stderr = true,
    })
    if not result or result.status ~= 0 then return nil end
    local data = utils.parse_json(result.stdout or "")
    local video = data and data.streams and data.streams[1]
    if not video or type(video.r_frame_rate) ~= "string" then return nil end
    local num, den = video.r_frame_rate:match("^(%d+)/(%d+)$")
    local height, width = tonumber(video.height), tonumber(video.width)
    if not num or not height or height <= 0 or not width or width <= 0 then return nil end
    return tonumber(num), tonumber(den), height, width
end

local function pick_dual_mult(src_fps, src_h, src_w)
    if PINNED_MULT then
        local n = tonumber(PINNED_MULT)
        if n and n >= 1 and n <= 4 then return n end
    end
    local policy = load_policy()
    if policy and is_four_machine_tictac() then
        if src_fps > policy.max_source_fps then return 1 end
        local fourk = src_h >= policy.fourk_min_height or src_w >= policy.fourk_min_width
        if fourk and src_fps <= policy.native_4k_max_source_fps
                and cluster_getenv("CLUSTER_TICTAC_4K_NATIVE") ~= "0" then return 2 end
        if src_fps > policy.x3_max_source_fps then return 2 end
        if not fourk and src_h <= policy.lowres_max_height
                and src_w <= policy.lowres_max_width then return 4 end
        if not fourk and src_fps <= policy.x4_1080_max_source_fps then return 4 end
        return 3
    end
    if src_fps > 30.0 then return 1 end
    local max_mult = (src_h <= 720) and 4 or 3
    if src_fps <= 25.0 then return max_mult end
    return max_mult - 1
end

local function on_load()
    if SOURCE_FPS_FILE then os.remove(SOURCE_FPS_FILE) end
    if cluster_getenv("CLUSTER_FPS_OVERRIDE_DISABLE") == "1" then return end
    local hosts = cluster_getenv("CLUSTER_WORKER_HOSTS") or ""
    if hosts:match("^%s*$") then hosts = cluster_getenv("CLUSTER_WORKER_HOST") or "" end
    if hosts:match("^%s*$") then return end
    if file_exists(CLUSTER_OFF_FLAG) then return end
    local path = mp.get_property("path")
    if not path then return end
    -- ffprobe handles both local file paths and http(s) URLs. Jellyfin
    -- (the primary use case) always streams over https, so we must NOT
    -- skip network URLs — the stream URL carries its own auth token and
    -- ffprobe reads the container header directly. The `timeout` in
    -- ffprobe_video bounds the network probe.
    local num, den, h, w = ffprobe_video(path)
    if not num or not den or den == 0 or num <= 0 then
        local f = io.open(CLUSTER_MULT_OVERRIDE, "w")
        local pinned = tonumber(PINNED_MULT or "")
        if f then f:write(tostring(pinned and pinned >= 1 and pinned <= 4 and pinned or 2)); f:close() end
        msg.warn("dual mult sidecar: source rate unavailable; using conservative multiplier")
        return
    end
    local src_fps = num / den
    if SOURCE_FPS_FILE then
        local rate_file = io.open(SOURCE_FPS_FILE, "w")
        if rate_file then rate_file:write(string.format("%.12f", src_fps)); rate_file:close() end
    end
    local mult = pick_dual_mult(src_fps, h or 0, w or 0)
    local f = io.open(CLUSTER_MULT_OVERRIDE, "w")
    if f then f:write(tostring(mult)); f:close() end
    msg.info(string.format(
        "dual mult sidecar: src=%.3f h=%d → mult=%d (wrote %s)",
        src_fps, h or 0, mult, CLUSTER_MULT_OVERRIDE))
end

mp.add_hook("on_load", 50, on_load)
mp.register_event("shutdown", function()
    if SOURCE_FPS_FILE then os.remove(SOURCE_FPS_FILE) end
end)
