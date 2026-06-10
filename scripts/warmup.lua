-- warmup.lua — force the vapoursynth chain to render ~1 s of frames
-- BEFORE playback starts, so the first second of actual playback is not
-- hitched by cold cuDNN graphs, cold krig nvcc cache, cold RIFE engine
-- load. Used by both single and dual paths (anything attached via
-- vf=vapoursynth=~~/rife.vpy).
--
-- Mechanism:
--   1) On file-loaded: pause, mute (saving both for restore), show
--      "warming up".
--   2) Issue `frame-step` once. mpv decodes + runs the vf chain on that
--      frame, then re-pauses.
--   3) `time-pos` increments → property observer fires → we know the
--      frame actually landed on the vo. Step the next one.
--   4) After N frames stepped (= ceil(source_fps * WARMUP_SECS)), seek
--      back to absolute 0, unpause, unmute. The vf scheduler's internal
--      `buffered-frames` (=12) and the warm cuDNN / RIFE / FSRCNNX
--      runners now sustain steady-state from frame 0.
--
-- Why frame-step instead of just sleeping while paused: paused mpv does
-- not pull frames through the filter (it idles), so a fixed sleep does
-- not warm anything. We need actual frame requests to flow through the
-- vf chain — that is what `frame-step` provides.
--
-- The OSD flashes the N stepped frames very briefly (mpv vo shows each
-- as it lands). The seek-back hides them — by the time the user looks
-- the vo is back at frame 0. On 1080p dual the whole sequence is
-- ~1.5–2 s total wall-clock (cold chain renders early frames at 0.5–1
-- fps; later frames at near-steady-state).
--
-- WARMUP_DISABLE=1 in env → no-op (debugging).

local mp = require "mp"
local msg = require "mp.msg"

local WARMUP_SECS = 1.0       -- target seconds of source frames to pre-render
local TIMEOUT_S   = 15.0      -- hard cap; cold first-load can hit ~10s
local STEP_BUDGET = 0.6       -- per-step wall-clock budget before we
                              -- give up on this frame and force-step
                              -- the next (prevents lock-up on stalled vf)
local WARM_HIT_FRAMES = 6     -- shrunk target when this mpv process has
                              -- already warmed the same chain shape:
                              -- engines/cuDNN graphs are hot, we only
                              -- need to prime the vf buffered-frames
                              -- pipeline (bf=12 fills from ~6 source
                              -- frames at factor 2)

-- vs_gpu_helpers._ensure_engines touches this while a TRT engine is
-- compiling (30-60 s, blocks the vf init thread with no feedback of
-- its own). We poll it to (a) surface an OSD note, (b) hold off the
-- warmup timeout so the compile doesn't get misread as a stall.
local TRT_COMPILING_FLAG = "/tmp/dgxspark_trt_compiling"

local function trt_compiling()
  local f = io.open(TRT_COMPILING_FLAG, "r")
  if f == nil then return false end
  f:close()
  return true
end

-- Chain shapes already warmed by this mpv process. Keyed by source
-- dims + fps band (which select the RIFE model + FSRCNNX runner
-- shapes) + the F8 variant override (which changes the runner). The
-- engine caches live in the embedded Python interpreter and survive
-- vf rebuilds / file changes, so a same-shape episode change needs
-- only a token warmup instead of the full ~2 s sequence.
local warmed = {}

local function chain_key()
  local w = mp.get_property_number("width") or 0
  local h = mp.get_property_number("height") or 0
  local fps = mp.get_property_number("container-fps") or 0
  local band = (fps > 0 and fps < 25) and "heavy" or "light"
  local variant = "auto"
  local f = io.open("/tmp/fsrcnnx_active_variant", "r")
  if f ~= nil then
    variant = (f:read("*l") or "auto")
    f:close()
  end
  return string.format("%dx%d/%s/%s", w, h, band, variant)
end

local state = {
  active        = false,
  target_frames = 0,
  stepped       = 0,
  last_pos      = nil,
  saved_pause   = nil,
  saved_mute    = nil,
  saved_pos     = nil,
  started_at    = 0,
  step_started  = 0,
  finished      = false,
}

local function finish(reason)
  if state.finished then return end
  state.finished = true
  state.active = false
  -- Record the warmed chain shape only on a clean finish — a timeout
  -- means the chain may still be cold (or wedged) and the next file
  -- with this shape should get the full warmup again.
  if reason ~= "timeout" and state.chain_key then
    warmed[state.chain_key] = true
  end
  -- Restore to the resume position. jellyfin-mpv-shim's play() does
  -- `loadfile <url>` (no start=) and then sets `playback_time=offset`
  -- once mpv reports `duration`. The property assignment fires after
  -- our `file-loaded` captured saved_pos at 0, so without the seek
  -- listener below we'd rewind to 0. on_seek() catches the shim's
  -- late seek during warmup and updates saved_pos in-flight; here we
  -- just restore to whatever the latest external seek targeted.
  mp.commandv("seek", tostring(state.saved_pos or 0), "absolute", "exact")
  -- Tiny delay so the seek's resulting frame request flushes before
  -- we unpause (otherwise the unpause race can show 1 stale frame).
  mp.add_timeout(0.05, function()
    if state.saved_mute ~= nil then
      mp.set_property_bool("mute", state.saved_mute)
    end
    if state.saved_pause ~= nil then
      mp.set_property_bool("pause", state.saved_pause)
    else
      mp.set_property_bool("pause", false)
    end
    local elapsed = mp.get_time() - state.started_at
    msg.info(string.format(
      "warmup done: %d frames in %.2fs (%s)",
      state.stepped, elapsed, reason))
    mp.osd_message(string.format(
      "warmup: %d frames / %.1fs", state.stepped, elapsed), 1.0)
  end)
end

local function step_one()
  if not state.active then return end
  if state.finished then return end
  -- Hard timeout — give up regardless of progress. A TRT compile in
  -- progress legitimately blocks the chain for up to a minute; keep
  -- pushing the clock forward (and the user informed) while it runs.
  if trt_compiling() then
    state.started_at = mp.get_time()
    mp.osd_message("compiling TensorRT engine for this resolution "
                   .. "(one-time, ~1 min)…", 2.0)
  end
  if mp.get_time() - state.started_at > TIMEOUT_S then
    finish("timeout")
    return
  end
  state.step_started = mp.get_time()
  state.last_pos = mp.get_property_number("time-pos")
  mp.commandv("frame-step")
end

local function on_time_pos(_, value)
  if not state.active then return end
  if state.finished then return end
  if value == nil then return end
  if state.last_pos ~= nil and value <= state.last_pos + 1e-6 then
    -- No advance yet — wait. The next observer fire will re-check.
    return
  end
  state.stepped = state.stepped + 1
  if state.stepped % 6 == 0 or state.stepped == state.target_frames then
    mp.osd_message(string.format(
      "warmup: %d / %d", state.stepped, state.target_frames), 2.0)
  end
  if state.stepped >= state.target_frames then
    finish("done")
    return
  end
  step_one()
end

-- Safety net: if frame-step somehow doesn't fire a time-pos update
-- (e.g. EOF on a too-short clip, or vf returned no frame), poll forward.
local function watchdog()
  if not state.active or state.finished then return end
  if trt_compiling() then
    -- Engine compile in flight — frames legitimately can't advance.
    -- Surface it and don't burn the step budget.
    state.step_started = mp.get_time()
    state.started_at = mp.get_time()
    mp.osd_message("compiling TensorRT engine for this resolution "
                   .. "(one-time, ~1 min)…", 2.0)
    mp.add_timeout(STEP_BUDGET, watchdog)
    return
  end
  if mp.get_time() - state.step_started > STEP_BUDGET then
    -- Stalled. Force-advance.
    state.stepped = state.stepped + 1
    if state.stepped >= state.target_frames then
      finish("watchdog-done")
      return
    end
    step_one()
  end
  mp.add_timeout(STEP_BUDGET, watchdog)
end

local function on_file_loaded()
  if os.getenv("WARMUP_DISABLE") == "1" then
    msg.info("warmup disabled via WARMUP_DISABLE=1")
    return
  end
  -- Reset state for this file.
  state.active        = true
  state.finished      = false
  state.stepped       = 0
  state.last_pos      = nil
  state.started_at    = mp.get_time()
  state.step_started  = state.started_at
  state.saved_pause   = mp.get_property_bool("pause")
  state.saved_mute    = mp.get_property_bool("mute")
  state.saved_pos     = mp.get_property_number("time-pos") or 0

  -- Pick frame count from source fps. container-fps is the demuxer's
  -- reported source rate (24/25/30); we want ~1s of source frames so
  -- the vf chain pulls roughly the same number of input pairs it
  -- will see during the first second of playback. Floor at 24 so
  -- container-fps=nil / 0 still warms enough.
  local src_fps = mp.get_property_number("container-fps") or 0
  if src_fps <= 0 then src_fps = 24 end
  state.target_frames = math.max(24, math.floor(src_fps * WARMUP_SECS + 0.5))

  -- Same chain shape already warmed in this mpv process (shim plays
  -- episodes back-to-back in one process): engines and cuDNN graphs
  -- are hot, so shrink to a token warmup that just refills the vf
  -- pipeline. ~2 s episode-change wait drops to a few hundred ms.
  state.chain_key = chain_key()
  if warmed[state.chain_key] then
    state.target_frames = math.min(state.target_frames, WARM_HIT_FRAMES)
    msg.info(string.format(
      "warmup: chain %s already warm — shrinking to %d frames",
      state.chain_key, state.target_frames))
  end

  mp.set_property_bool("pause", true)
  mp.set_property_bool("mute", true)
  mp.osd_message(string.format(
    "warming up (%d frames)…", state.target_frames), 5.0)
  msg.info(string.format(
    "warmup start: target=%d frames (src_fps=%.2f)",
    state.target_frames, src_fps))

  -- Kick the first step. observer fires when the first frame lands.
  mp.add_timeout(0.05, function()
    step_one()
    mp.add_timeout(STEP_BUDGET, watchdog)
  end)
end

-- Catch external seeks during warmup so we restore to the actual
-- resume target rather than the file-loaded 0. mpv's `seek` event
-- fires on user/script `seek` commands and on `playback-time=`
-- property assignment (= the shim's resume path); it does NOT fire
-- on frame-step, so our own warmup advances don't trigger this.
local function on_seek()
  if not state.active or state.finished then return end
  local p = mp.get_property_number("time-pos")
  if p == nil then return end
  -- Ignore tiny advances from frame-step that happen to bracket the
  -- event-loop tick (just in case): only treat as resume if the jump
  -- is larger than what a few frame-steps could account for.
  local prev = state.saved_pos or 0
  if math.abs(p - prev) > 0.5 then
    state.saved_pos = p
    msg.info(string.format(
      "warmup: external seek caught at %.2fs → updated resume target", p))
  end
end

mp.observe_property("time-pos", "number", on_time_pos)
mp.register_event("file-loaded", on_file_loaded)
mp.register_event("seek", on_seek)
