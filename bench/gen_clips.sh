#!/usr/bin/env bash
# Generate the synthetic test clips used by bench/*.sh.
# Runs ffmpeg once, writes two files into bench/clips/.
#
# CLIP-24:  1080p @ 24 fps, 10 s, slowly-zooming mandelbrot. Used
#           for color/PSNR (low rate, deterministic content).
# CLIP-120: 1080p @ 120 fps, 5 s, same content. Used for fps bench
#           (source rate ≥ output rate so playback isn't rate-capped).
#
# yuv420p10le matches the most common streaming case and exercises
# the 10-bit path through the chain.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/clips"
mkdir -p "$OUT"

C24="$OUT/sample-1080p-24.mp4"
C120="$OUT/sample-1080p-120.mp4"

if [[ ! -f $C24 ]]; then
  echo "[gen_clips] $C24"
  ffmpeg -hide_banner -loglevel warning -y \
    -f lavfi -i 'mandelbrot=size=1920x1080:rate=24' \
    -t 10 -c:v libx264 -pix_fmt yuv420p10le -preset fast -crf 18 \
    "$C24"
else
  echo "[gen_clips] $C24 (exists, skipped)"
fi

if [[ ! -f $C120 ]]; then
  echo "[gen_clips] $C120"
  ffmpeg -hide_banner -loglevel warning -y \
    -f lavfi -i 'mandelbrot=size=1920x1080:rate=120' \
    -t 5 -c:v libx264 -pix_fmt yuv420p10le -preset fast -crf 18 \
    "$C120"
else
  echo "[gen_clips] $C120 (exists, skipped)"
fi

ls -lh "$OUT"
