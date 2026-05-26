# Bench results — DGX Spark reference machine

Snapshot of the most recent run on the development primary box, kept
under version control so it doubles as a regression baseline.

- **Date:** 2026-05-23
- **Hardware:** 2 × DGX Spark (GB10, ARM64, CUDA 13), 200 G RoCE link
- **Software:** `dual-machine-ib-baseline` branch, post-lessons commit
- **Test clips:** `/tmp/sample-1080p-24-loop.mp4` (1080p HEVC, 24 fps,
  60 s; user-provided real video) for color, `bench/clips/sample-1080p-120.mp4`
  (synthesised by `bench/gen_clips.sh`) for fps.

Reproduce with:
```bash
CLIP_24=/tmp/sample-1080p-24-loop.mp4 bench/color.sh
bench/fps.sh
MODE=dual bench/timing.sh
```

## bench/color.sh — single vs dual PSNR

`N=20`, `CF=2`, comparing the FFV1-encoded outputs of each pair.

| pair | single bytes | dual bytes | Y avg | U avg | V avg | min Y | max Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| full        (RIFE + FSRCNNX) | 119,136,942 | 118,861,772 | 54.87 | 67.36 | 65.76 | 50.91 | inf |
| no\_sr      (RIFE only)       |  51,616,572 |  33,891,969 | 56.25 | 59.78 | 58.15 | 51.80 | inf |
| no\_interp  (FSRCNNX only)    | 115,059,539 | 120,394,538 | 69.15 | 59.14 | 56.37 | 60.98 | 62.41 |

`SR_SRC` (real-frame) outputs are byte-identical between single and
dual (`psnr_y=inf` every frame). `SR_INTERP` frames carry RIFE
cross-GPU non-determinism: 49–72 dB Y on full / no\_sr.

Determinism: three back-to-back runs produce byte-identical outputs
in every mode (file sizes above repeat exactly).

## bench/fps.sh — sustained throughput

`N=400`, `CF=24`, 1080p120 source → 4K output. `wall` includes ~3 s
of engine / cuDNN warm overhead; `dispatcher` is the worker's rolling
30-frame estimate (steady state).

| chain | mode | wall fps | dispatcher fps |
|---|---|---:|---:|
| single | single          | 35.2 | — |
| single | single\_no\_sr   | 51.1 | — |
| single | single\_no\_rife | 71.0 | — |
| dual   | dual            | 33.6 |  96.0 |
| dual   | dual\_no\_sr     | 43.4 | 148.8 |
| dual   | dual\_no\_interp | 37.0 | 113.7 |

Steady-state speedup vs single (full chain): **96 / 50 ≈ 1.95×**.

### dual at higher interp_mult

`N=200`, `CF=24`, 1080p120 source → 4K output. mult is forced by
env (`DUAL_INTERP_MULT=N`) for this comparison; production
auto-defaults are listed in dual_machine/README.md.

| `DUAL_INTERP_MULT` | output rate | dispatcher fps (steady) | mult × source budget |
|---|---|---:|---:|
| 2 | 2× source | 98.4 | 240 fps target | well within budget |
| 3 | 3× source | 95.1 | 360 fps target — INTERP fan-out adds 0–3 fps overhead |
| 4 | 4× source | 98.6 | 480 fps target — same envelope as mult=2 at 1080p |

mult=3 is slightly slower than mult=2/4 because the approach-A INTERP
runs single-task + dual-flownet (the secondary SR_INTERP fans out
from one INTERP task), so a partial extra dispatcher hop adds 0.5–3
fps overhead. mult=4's dispatcher fps matches mult=2 because the
INTERP task does 3 flownets in one call (single task overhead amortised).

For ≤720p source these numbers go higher (mult=4 at 720p hits ~140
fps dispatcher, the M8 design target).

## bench/timing.sh — per-task GPU + comm (mode=dual, N=600)

Worker side per task type (last `PROFILE` sample, GB10):

| task type | kernel (ms) | idle\_before (ms) |
|---|---:|---:|
| INTERP    | 5–7  | 0.3–0.8 |
| SR\_INTERP | 6–8  | 0.4–1.2 |
| CCSR      | 9–12 | 0.2–0.6 |

Worker GPU utilisation (rolling): **kernel ≈ 94 %**, idle ≈ 6 %.

Worker RDMA per task (`DUAL_RDMA_PROF=1`):

| task | send (ms) | recv-rtt (ms) |
|---|---:|---:|
| INTERP    | 5.7 | 16 |
| SR\_INTERP | 3.1 | 11 |
| CCSR      | 4.2 | 13 |

Host queue depth + lock-hold (`DUAL_PROFILE=1`):
- HOST pops: `cc` ≈ 30 / s, `sr` ≈ 60 / s, `interp` ≈ 0; `wait_avg < 0.1 ms`.
- GUEST pops: `interp` ≈ 30 / s, `cc` ≈ 0, `sr` ≈ 0; `wait_avg < 0.1 ms`.
- Pop lock-hold: ≈ 0.05 ms / call.

## Notes

- Dispatcher fps (`96`) is the trustworthy steady-state number; the
  wall figures for dual underrepresent because mpv has to drain the
  source pipeline before the dispatcher's steady-state kicks in.
- All numbers above are with `DUAL_INTERP_MULT=2` (the production
  default for 1080p / 24 fps source). For higher multipliers see
  the [`dual_machine/README.md`](../dual_machine/README.md#default-interp_mult-per-source).
