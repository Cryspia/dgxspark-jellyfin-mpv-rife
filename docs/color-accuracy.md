# Color accuracy

[简体中文](./color-accuracy.zh-CN.md)

PSNR measurements via `bench/color.sh`. All numbers are versus the
single-machine output as the user-confirmed reference baseline.

## Single-machine vs original 4K source

1080p source → 4K output, N=60 frames, krig chroma upsampling.

| metric | value |
|---|---|
| Y avg | 29.79 dB |
| U avg | 42.22 dB |
| V avg | 37.48 dB |
| min Y | 25.95 dB |
| bad U/V (PSNR < 25) | 0 / 0 |

(U / V > 40 dB threshold = visually indistinguishable.)

## Dual-machine vs single-machine

1080p source → 4K output, N=20 frames, CF=2. Three pair compares:

### full (RIFE + FSRCNNX)

|        | Y avg | U avg | V avg | min Y | max Y |
|---|---|---|---|---|---|
| dispatcher | 54.87 | 67.36 | 65.76 | 50.91 | inf |

Per-frame:
- **SR_SRC** frames (real, even output indices): byte-identical
  (`psnr_y=inf` every frame).
- **SR_INTERP** frames (interpolated, odd output indices): Y 49 – 65 dB
  (RIFE cross-GPU non-determinism on luma); U / V 56 – 66 dB (kriged
  from SR'd Y, smaller drift).

### no_sr (RIFE only, source-dim output)

|        | Y avg | U avg | V avg | min Y | max Y |
|---|---|---|---|---|---|
| dispatcher | 56.25 | 59.78 | 58.15 | 51.80 | inf |

- SR_SRC: byte-identical.
- SR_INTERP: Y 50 – 72 dB, U / V 44 – 58 dB.

### no_interp (FSRCNNX only, source-rate output)

|        | Y avg | U avg | V avg | min Y | max Y |
|---|---|---|---|---|---|
| dispatcher | 69.15 | 59.14 | 56.37 | 60.98 | 62.41 |

All frames SR_SRC: Y 69 dB consistent (cuDNN autotuner picks the same
algorithm both sides), U / V 55 – 59 dB.

## Determinism

Three back-to-back runs of `bench/color.sh` produce **byte-identical**
output files for every mode (single full, single no_sr, single no_rife,
dual full, dual no_sr, dual no_interp). The FSRCNNX cuDNN runner
thread-safety fix (fsrcnnx-cudnn upstream `e91adc7` + `e1344b8`) and
the `BlankClip(keep=False)` fix in our `vs_gpu_helpers.py rife_yuv`
removed the last sources of cross-run drift.

## 4K mixed mode

1080p-interp + 4K-real, N=240, vs full-flow 4.6 reference:

| comparison | sample-01 | sample-02 |
|---|---|---|
| real frames (passthrough) vs original 4K | 240 dB | 240 dB |
| interp frames vs full-flow 4.6           | 38.5 dB | 34.8 dB |
| half-flow interp vs full-flow            | 39.1 dB | 35.1 dB |
