# 颜色准度

[English](./color-accuracy.md)

PSNR 通过 `bench/color.sh` 测。 所有数都对照单机输出 (用户确认的
正确基准).

## 单机 vs 原始 4K 源

1080p 源 → 4K 输出，N=60 帧，krig chroma 升采样。

| metric | 值 |
|---|---|
| Y avg | 29.79 dB |
| U avg | 42.22 dB |
| V avg | 37.48 dB |
| min Y | 25.95 dB |
| bad U/V (PSNR < 25) | 0 / 0 |

（U / V > 40 dB 阈值 = 视觉上无区别。）

## 双机 vs 单机

1080p 源 → 4K 输出，N=20 帧，CF=2. 三组对比：

### full (RIFE + FSRCNNX)

|        | Y avg | U avg | V avg | min Y | max Y |
|---|---|---|---|---|---|
| dispatcher | 54.87 | 67.36 | 65.76 | 50.91 | inf |

逐帧：
- **SR_SRC** 帧 （真实帧，偶数输出 idx）： byte-identical (每帧
  `psnr_y=inf`).
- **SR_INTERP** 帧 （插帧，奇数输出 idx）： Y 49 – 65 dB (RIFE 跨 GPU
  非确定性影响 luma); U / V 56 – 66 dB (从超分后的 Y krig 出来，漂移
  更小).

### no_sr （只 RIFE, 源分辨率输出）

|        | Y avg | U avg | V avg | min Y | max Y |
|---|---|---|---|---|---|
| dispatcher | 56.25 | 59.78 | 58.15 | 51.80 | inf |

- SR_SRC: byte-identical.
- SR_INTERP: Y 50 – 72 dB, U / V 44 – 58 dB.

### no_interp （只 FSRCNNX, 源帧率输出）

|        | Y avg | U avg | V avg | min Y | max Y |
|---|---|---|---|---|---|
| dispatcher | 69.15 | 59.14 | 56.37 | 60.98 | 62.41 |

全 SR_SRC 帧：Y 69 dB 稳定 （cuDNN autotuner 两边选同一个算法），
U / V 55 – 59 dB.

## 确定性

`bench/color.sh` 连跑三次，每个模式的输出文件 **byte-identical**
(单机 full / 单机 no_sr / 单机 no_rife / 双机 full / 双机 no_sr /
双机 no_interp 全部). FSRCNNX cuDNN runner 的 thread-safety 修
（上游 `e91adc7` + `e1344b8`） 和我们 `vs_gpu_helpers.py rife_yuv` 的
`BlankClip(keep=False)` 修把跨 run 漂移的最后来源都去掉了。

## 4K mixed mode

1080p 插帧 + 4K 真帧，N=240, 对照 full-flow 4.6 基准：

| 对比 | sample-01 | sample-02 |
|---|---|---|
| 真实帧 (passthrough) vs 原始 4K | 240 dB | 240 dB |
| 插帧 vs full-flow 4.6           | 38.5 dB | 34.8 dB |
| half-flow 插帧 vs full-flow     | 39.1 dB | 35.1 dB |
