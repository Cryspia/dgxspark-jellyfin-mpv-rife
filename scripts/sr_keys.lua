-- F8 / F9 — vapoursynth-chain keybinds.
--   F8: SR toggle.
--     * single mode: cycle FSRCNNX variant (x4_16 -> x3_16 -> x2_16 ->
--                    x2_8 -> off -> loop). Per-file (auto-reset on
--                    file-loaded).
--     * dual mode:   toggle SR on/off only (the bucket auto-picks the
--                    variant; there's no useful "x2_8" in dual since
--                    the worker only builds the bucket variant).
--   F9: interpolation toggle / multiplier.
--     * single mode: toggle RIFE on/off.
--     * dual mode:   cycle 4 -> 3 -> 2 -> off -> 4. "off" disables RIFE
--                    AND the dual chain (dual has no RIFE-less SR mode);
--                    rife.vpy falls back to single SR-only. Next press
--                    from off -> mult=4 (re-activates dual).
--   Shift+F8: toggle the krig-bilateral chroma GLSL shader (mpv's
--             display-time chroma upsample; orthogonal to the in-
--             vapoursynth krig). Instant — no vf reload.
--   Shift+F9: toggle dual-machine offload (no-op if DUAL_WORKER_HOST
--             isn't set).
--
-- F8 / F9 / Shift+F9 all force a vapoursynth filter reload (~1-3 s
-- freeze) — we rebuild the chain. Shift+F8 only flips an mpv property.
--
-- Communication with the .vpy chain (file-based; vf reload re-reads):
--   /tmp/fsrcnnx_variant            — single F8 override. Empty/missing
--                                     = auto. Otherwise variant name.
--                                     Cleared on file-loaded.
--   /tmp/fsrcnnx_active_variant     — what the single .vpy resolved to.
--   /tmp/rife_disabled              — touch-file. rife.vpy skips RIFE
--                                     when present. Used by single F9
--                                     and dual F9 "off".
--   /tmp/dual_machine_disabled      — touch-file. rife.vpy uses the
--                                     single chain even with the worker
--                                     configured. Toggled by Shift+F9.
--   /tmp/dual_machine_active        — "1" / "0" written by rife.vpy.
--                                     Read by F8/F9/Shift+F9 to pick
--                                     the right cycle behaviour.
--   /tmp/dual_machine_mult          — current effective dual mult
--                                     (2/3/4) written by rife.vpy.
--   /tmp/dual_machine_mult_override — F9 dual cycle writes "2"/"3"/"4"
--                                     /"off" here. rife.vpy mirrors
--                                     numeric values into
--                                     DUAL_INTERP_MULT and passes via
--                                     handshake.
--   /tmp/dual_machine_sr_disabled   — F8 dual writes (touch/remove).
--                                     rife.vpy reads -> passes no_sr=1
--                                     in handshake -> worker skips SR.
--   /tmp/dual_machine_sr_active     — "1" / "0" written by rife.vpy
--                                     reporting whether SR actually ran
--                                     in dual on the last load.

local mp = require "mp"

local OVERRIDE_FILE      = "/tmp/fsrcnnx_variant"
local ACTIVE_FILE        = "/tmp/fsrcnnx_active_variant"
local RIFE_OFF_FILE      = "/tmp/rife_disabled"
local DUAL_OFF_FILE      = "/tmp/dual_machine_disabled"
local DUAL_STATE_FILE    = "/tmp/dual_machine_active"
local DUAL_MULT_FILE     = "/tmp/dual_machine_mult"
local DUAL_MULT_OVERRIDE = "/tmp/dual_machine_mult_override"
local DUAL_SR_OFF_FILE   = "/tmp/dual_machine_sr_disabled"
local DUAL_SR_STATE_FILE = "/tmp/dual_machine_sr_active"

-- Fallback path used if the user starts mpv with glsl-shaders empty
-- and then presses Shift+F8. install.sh writes KrigBilateral.glsl under
-- this path; ~~/ is mpv's user-config-dir prefix.
local KRIG_GLSL_PATH = "~~/shaders/KrigBilateral.glsl"

local SINGLE_CYCLE    = { "x4_16", "x3_16", "x2_16", "x2_8", "off" }
-- Dual mult cycle: "1" = no_interp (SplitDual emits CCSR-only, INTERP
-- + SR_INTERP skipped end-to-end). Distinct from single-mode "off"
-- (rife_disabled) — F9 in dual never touches /tmp/rife_disabled.
local DUAL_MULT_CYCLE = { "4", "3", "2", "1" }

local function cycle_index(list, value)
  for i, v in ipairs(list) do
    if v == value then return i end
  end
  return nil
end

local function read_file(path)
  local fh = io.open(path, "r")
  if not fh then return nil end
  local s = fh:read("*all")
  fh:close()
  return s and s:match("^%s*(.-)%s*$") or nil
end

local function write_file(path, s)
  local fh = io.open(path, "w")
  if not fh then return false end
  fh:write(s); fh:close()
  return true
end

local function file_exists(path)
  local fh = io.open(path, "r")
  if fh then fh:close(); return true end
  return false
end

local function reload_vf()
  -- Clear-then-restore is the most reliable way to force a re-exec
  -- of the .vpy. `vf-command` doesn't reach inside vapoursynth.
  local current = mp.get_property("vf")
  if not current or current == "" then return end
  mp.set_property("vf", "")
  mp.set_property("vf", current)
end

local function is_dual_active()
  -- "dual is the active branch" = rife.vpy reported dual_active=1 on
  -- the last reload AND Shift+F9 hasn't toggled it off since. The
  -- disable-flag check guards the case where the user just hit
  -- Shift+F9 (DUAL_OFF_FILE present) but the reload hasn't happened
  -- yet (DUAL_STATE_FILE still says "1"), and then mashes F8/F9 —
  -- we'd otherwise cycle the dual files for an about-to-be-single
  -- session.
  return read_file(DUAL_STATE_FILE) == "1"
         and not file_exists(DUAL_OFF_FILE)
end

local function cycle_fsrcnnx_single()
  local active = read_file(ACTIVE_FILE)
  if not active or active == "" or active == "none" then
    -- Chain currently bypasses FSRCNNX (e.g. 4K -> 4K, ratio < 1.3).
    -- Start the cycle from the front rather than refusing.
    active = SINGLE_CYCLE[#SINGLE_CYCLE]
  end
  local idx = cycle_index(SINGLE_CYCLE, active)
  local next_idx = (idx and (idx % #SINGLE_CYCLE) + 1) or 1
  local next_variant = SINGLE_CYCLE[next_idx]

  write_file(OVERRIDE_FILE, next_variant)
  local label = (next_variant == "off") and "OFF" or next_variant
  mp.osd_message(string.format("FSRCNNX -> %s (reloading...)", label), 2)
  reload_vf()
end

local function toggle_fsrcnnx_dual()
  -- Dual only exposes on/off (the bucket determines variant). For 4K-DS
  -- the OFF state is inadmissible (RIFE-DS produces 1080p interp frames
  -- and needs SR to upscale back). rife.vpy enforces the gate downstream
  -- and re-publishes /tmp/dual_machine_sr_active to surface the actual
  -- post-reload state — we just toggle the request flag here.
  local now = read_file(DUAL_SR_STATE_FILE)
  if now == "1" then
    write_file(DUAL_SR_OFF_FILE, "")
    mp.osd_message("dual SR: OFF (reloading...)", 2)
  else
    os.remove(DUAL_SR_OFF_FILE)
    mp.osd_message("dual SR: ON (reloading...)", 2)
  end
  reload_vf()
end

local function cycle_fsrcnnx()
  if is_dual_active() then
    toggle_fsrcnnx_dual()
  else
    cycle_fsrcnnx_single()
  end
end

local function toggle_rife_single()
  if file_exists(RIFE_OFF_FILE) then
    os.remove(RIFE_OFF_FILE)
    mp.osd_message("RIFE: ON (reloading...)", 2)
  else
    write_file(RIFE_OFF_FILE, "")
    mp.osd_message("RIFE: OFF (reloading...)", 2)
  end
  reload_vf()
end

local function cycle_mult_dual()
  -- Current effective state: prefer the override file (most-recent
  -- press), fall back to rife.vpy's published-effective mult.
  local cur = read_file(DUAL_MULT_OVERRIDE)
  if cur == nil or cur == "" then
    cur = read_file(DUAL_MULT_FILE) or "2"
  end
  local idx = cycle_index(DUAL_MULT_CYCLE, cur)
  local nxt = DUAL_MULT_CYCLE[(idx and (idx % #DUAL_MULT_CYCLE) + 1) or 1]
  write_file(DUAL_MULT_OVERRIDE, nxt)
  if nxt == "1" then
    -- mult=1 == no_interp dual mode: SplitDual emits CCSR-only, no
    -- INTERP/SR_INTERP scheduled, worker CCSR skips rgb_padded +
    -- rife_features writes (no consumer).
    mp.osd_message("dual INTERP: OFF / SR only (reloading...)", 2)
  else
    mp.osd_message(string.format(
      "dual INTERP: x%s (reloading...)", nxt), 2)
  end
  reload_vf()
end

local function toggle_rife()
  if is_dual_active() then
    cycle_mult_dual()
  else
    toggle_rife_single()
  end
end

local function toggle_dual_machine()
  -- Hard gate: env not configured -> nothing to toggle. We don't even
  -- touch the disable flag, so a stale flag from a previous install
  -- (where the env was set) doesn't flip behaviour now.
  local env_host = os.getenv("DUAL_WORKER_HOST")
  if not env_host or env_host == "" then
    mp.osd_message(
      "dual-machine: DUAL_WORKER_HOST not set (toggle disabled)", 2)
    return
  end

  -- State-aware: rely on what .vpy reports it's actually doing, not
  -- the disable flag, so a failed-connect fallback still retries on
  -- the next press (rather than appearing to "double-disable").
  local active = read_file(DUAL_STATE_FILE)
  if active == "1" then
    -- Going dual -> single. Dual x3/x4 have no production
    -- single-mode equivalent (single x3/x4 exists for AB only, too
    -- slow to use). Warn that output rate drops back to 2x; rife.vpy
    -- builds single-mode at rife_factor=2 unless SINGLE_INTERP_MULT
    -- is set in the env.
    local mult = read_file(DUAL_MULT_FILE)
    if mult == "3" or mult == "4" then
      mp.osd_message(
        "dual-machine: OFF -- single mode drops to x2 "
        .. "(single x" .. mult .. " unsupported in production)", 3)
    else
      mp.osd_message("dual-machine: OFF (reloading...)", 2)
    end
    write_file(DUAL_OFF_FILE, "")
  else
    os.remove(DUAL_OFF_FILE)
    mp.osd_message("dual-machine: connecting... (reloading)", 2)
  end
  reload_vf()
end

-- Snapshot whatever glsl-shaders mpv.conf loaded at startup so the
-- toggle restores the user's actual configuration (not just the krig
-- default — if you've stacked extra shaders we don't want to clobber
-- them on the first ON cycle).
local _GLSL_SAVED = nil
local function _current_glsl()
  return mp.get_property("glsl-shaders") or ""
end

local function toggle_krig_glsl()
  local cur = _current_glsl()
  if cur ~= "" then
    _GLSL_SAVED = cur
    mp.set_property("glsl-shaders", "")
    mp.osd_message("chroma GLSL: OFF", 1.5)
  else
    local restore = _GLSL_SAVED
    if restore == nil or restore == "" then
      restore = KRIG_GLSL_PATH
    end
    mp.set_property("glsl-shaders", restore)
    mp.osd_message("chroma GLSL: ON (" .. restore .. ")", 1.5)
  end
end

local function reset_on_file_load()
  -- F8 single (FSRCNNX variant) resets per file — each video gets its
  -- own auto-pick starting point. F9 / Shift+F9 / Shift+F8 / dual
  -- mult override / dual SR disable all persist across files (global
  -- preferences rather than per-source tuning).
  os.remove(OVERRIDE_FILE)
  os.remove(ACTIVE_FILE)
end

mp.add_key_binding("F8",       "fsrcnnx-cycle",        cycle_fsrcnnx)
mp.add_key_binding("F9",       "rife-toggle",          toggle_rife)
mp.add_key_binding("Shift+F8", "krig-glsl-toggle",     toggle_krig_glsl)
mp.add_key_binding("Shift+F9", "dual-machine-toggle",  toggle_dual_machine)
mp.register_event("file-loaded", reset_on_file_load)

-- Worker-death watchdog. native_dispatcher's wait_phase_done timeout
-- touches /tmp/dual_machine_worker_dead when it gives up on the worker;
-- we detect that here, OSD-warn, and reload the vf so rife.vpy picks
-- the single chain. Without this auto-recovery a dead worker just
-- raises into mpv → playback halts.
local DUAL_WORKER_DEAD_FILE = "/tmp/dual_machine_worker_dead"
local function check_worker_death()
  if file_exists(DUAL_WORKER_DEAD_FILE) then
    os.remove(DUAL_WORKER_DEAD_FILE)
    mp.osd_message(
      "dual worker died — falling back to single (reloading...)", 4)
    -- The dispatcher already touched /tmp/dual_machine_disabled, so
    -- the vf reload's rife.vpy will see Shift+F9-off and take the
    -- single branch. reload_vf is the same primitive used by all
    -- the F8/F9/Shift+F9 keys.
    reload_vf()
  end
end
-- Poll every 2 s. Cheap (one os.stat per tick); only fires on real
-- worker death.
mp.add_periodic_timer(2.0, check_worker_death)
