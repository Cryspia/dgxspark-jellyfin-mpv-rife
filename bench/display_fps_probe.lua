-- display_fps_probe.lua — sample real-VO playback throughput on a timer.
--
-- Loaded only by bench/display_fps.sh (via --script). Every SAMPLE_S it
-- writes one clean line to stderr with the display-present rate and the
-- cumulative drop counters; on shutdown it prints a steady-state summary
-- over the window [WARMUP_S, end].
--
-- Metric choice: `estimated-vf-fps` just echoes the nominal RIFE output
-- framerate (72 for 1080p24 x3) because mpv derives it from output PTS,
-- so it is NOT a throughput signal — we log it but do not summarise it.
-- The real signals are `estimated-display-fps` (mpv's measured vsync
-- present rate — how many frames actually reach the screen per second)
-- and `frame-drop-count` (VO late-frame drops). A render pipeline that
-- can't finish inside the per-vsync budget shows up as display-fps below
-- the monitor's refresh and a steadily climbing drop count.

local mp = require "mp"
local msg = require "mp.msg"

local SAMPLE_S  = tonumber(os.getenv("DISPLAYPROBE_SAMPLE_S")) or 0.5
local WARMUP_S  = tonumber(os.getenv("DISPLAYPROBE_WARMUP_S")) or 6.0

local samples = {}      -- steady-state estimated-display-fps samples
local t_start = nil
local warm_t  = nil     -- wall time at first post-warmup sample
local warm_drop = nil   -- frame-drop-count at first post-warmup sample
local last_drop = nil
local last_t  = nil

local function num(prop)
  local v = mp.get_property_number(prop)
  return v
end

local function fmt(v, d)
  if v == nil then return "nil" end
  return string.format("%." .. (d or 2) .. "f", v)
end

local function tick()
  local now = mp.get_time()
  if t_start == nil then t_start = now end
  local elapsed = now - t_start

  local vf    = num("estimated-vf-fps")
  local disp  = num("estimated-display-fps")
  local tpos  = num("time-pos")
  local drop  = num("frame-drop-count")          -- VO drops (late frames)
  local decdr = num("decoder-frame-drop-count")  -- decoder drops

  io.stderr:write(string.format(
    "[displayprobe] t=%5.1f pos=%s vf=%s disp=%s drop=%s decdrop=%s\n",
    elapsed, fmt(tpos, 1), fmt(vf), fmt(disp), fmt(drop, 0), fmt(decdr, 0)))
  io.stderr:flush()

  -- Collect estimated-display-fps only after the warmup window so cuDNN /
  -- engine spin-up and mpv's vf-rebuild don't drag the average down.
  if elapsed >= WARMUP_S and disp ~= nil and disp > 1.0 then
    samples[#samples + 1] = disp
    if warm_t == nil then warm_t = now; warm_drop = drop end
    last_t = now
    last_drop = drop
  end
end

local function summary()
  local n = #samples
  if n == 0 then
    io.stderr:write("[displayprobe] SUMMARY: no steady-state samples\n")
    io.stderr:flush()
    return
  end
  table.sort(samples)
  local sum = 0
  for _, v in ipairs(samples) do sum = sum + v end
  local mean = sum / n
  local function pct(p)
    local i = math.max(1, math.min(n, math.floor(n * p / 100 + 0.5)))
    return samples[i]
  end
  local drop_rate = 0.0
  if warm_t ~= nil and last_t ~= nil and last_t > warm_t
     and warm_drop ~= nil and last_drop ~= nil then
    drop_rate = (last_drop - warm_drop) / (last_t - warm_t)
  end
  io.stderr:write(string.format(
    "[displayprobe] SUMMARY display-fps over %d samples (warmup>=%.1fs): "
    .. "mean=%.2f median=%.2f p10=%.2f min=%.2f max=%.2f | "
    .. "VO-drops/s=%.2f\n",
    n, WARMUP_S, mean, pct(50), pct(10), samples[1], samples[n], drop_rate))
  io.stderr:flush()
end

mp.add_periodic_timer(SAMPLE_S, tick)
mp.register_event("shutdown", summary)
