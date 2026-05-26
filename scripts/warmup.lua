-- warmup.lua — force the vapoursynth chain to render ~1 s of frames
-- BEFORE playback starts, so the first second of actual playback is not
-- hitched by cold cuDNN graphs, cold krig nvcc cache, cold RIFE engine
-- load. Used by both single and dual paths (anything attached via
-- vf=vapoursynth=~~/rife.vpy).
--
-- Mechanism:
--   1) On file-loaded: pause, mute, save audio-delay, show "warming up".
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
  -- Restore to the file-loaded position. Jellyfin resume hands mpv
  -- `loadfile … start=<pos>`, so time-pos at file-loaded is the resume
  -- point; defaulting to 0 here used to drag resumed playback back to
  -- the start of the file. exact seek so we land deterministically.
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
  -- Hard timeout — give up regardless of progress.
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

mp.observe_property("time-pos", "number", on_time_pos)
mp.register_event("file-loaded", on_file_loaded)
