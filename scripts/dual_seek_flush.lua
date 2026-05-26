-- dual_seek_flush.lua — surface mpv seek + shutdown events to the
-- dual_machine Python dispatcher via /tmp marker files.
--
--   /tmp/dual_machine_seek_epoch     bumped on every seek
--   /tmp/dual_machine_shutdown       written on mpv exit
--
-- The dispatcher's inotify watcher reacts to both. Without this lua
-- the dispatcher has no way to learn about seeks (mpv's vapoursynth
-- filter API doesn't surface them) or mpv exits (the embedded
-- Python interp's atexit doesn't fire when mpv tears down).
--
-- DUAL_SEEK_FLUSH_DISABLE=1 → no-op.

local mp  = require "mp"
local msg = require "mp.msg"

local EPOCH_FILE = "/tmp/dual_machine_seek_epoch"
local SHUTDOWN_FILE = "/tmp/dual_machine_shutdown"
local last_bump_time = 0

local function bump_epoch()
    -- 200 ms debounce — `seeking` property and `seek` event both fire
    -- per seek; the second one would land after post-seek DAG nodes
    -- have been submitted, wiping them.
    local now = mp.get_time()
    if now - last_bump_time < 0.2 then return end
    last_bump_time = now
    local epoch = 0
    local f = io.open(EPOCH_FILE, "r")
    if f then
        epoch = tonumber(f:read("*l") or "0") or 0
        f:close()
    end
    epoch = epoch + 1
    local g, err = io.open(EPOCH_FILE, "w")
    if not g then
        msg.warn("dual seek-flush: cannot write " .. EPOCH_FILE
                 .. ": " .. tostring(err))
        return
    end
    g:write(tostring(epoch))
    g:close()
    msg.info(string.format("dual seek-flush epoch=%d", epoch))
end

local function on_seek_signal()
    if os.getenv("DUAL_SEEK_FLUSH_DISABLE") == "1" then return end
    bump_epoch()
end

-- `seeking` property fires before vf teardown; `seek` event fires
-- after. Both call bump_epoch (debounced), so the python side
-- unparks compute() at whichever signal arrives first.
mp.observe_property("seeking", "bool", function(_, v)
    if v == true then on_seek_signal() end
end)
mp.register_event("seek", on_seek_signal)

mp.register_event("shutdown", function()
    local f = io.open(SHUTDOWN_FILE, "w")
    if f then
        f:write(tostring(os.time()))
        f:close()
        msg.info("dual: shutdown marker written")
    end
end)
