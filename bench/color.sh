#!/usr/bin/env bash
# bench/color.sh — single vs dual PSNR for the three production modes.
#
#   full       single (RIFE + SR) vs dual (RIFE + SR)
#   no_sr      single (RIFE only)  vs dual_no_sr (RIFE only)
#   no_interp  single (SR only)    vs dual_no_interp (SR only)
#
# CF=2 keeps the pipeline deterministic enough that any drift between
# single and dual surfaces as a low-PSNR frame. SR_SRC frames should
# be byte-identical (psnr_y=inf); SR_INTERP frames carry RIFE
# cross-GPU non-determinism on the luma plane (~40-65 dB is normal).

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_common.sh"

N="${N:-20}"
CF="${CF:-2}"

require_clips "$CLIP_24" || exit 1
needs_dual || exit 1

TMPDIR=$(mktemp -d /tmp/bench_color.XXXXXX)
flags_save_dir "$TMPDIR/saved_flags"
trap 'cleanup_all; flags_restore "$TMPDIR/saved_flags"; echo; echo "Artefacts: $TMPDIR"' EXIT

cleanup_all

render_pair() {
  local tag=$1 single_mode=$2 dual_mode=$3
  local out_s="$TMPDIR/${tag}_single.mkv"
  local out_d="$TMPDIR/${tag}_dual.mkv"
  local log_s="$TMPDIR/${tag}_single.log"
  local log_d="$TMPDIR/${tag}_dual.log"

  echo "  [$tag] rendering $single_mode + $dual_mode (N=$N)..."
  kill_local_mpv
  run_mpv "$out_s" "$single_mode" "$CLIP_24" "$N" "$CF" "$log_s"

  start_worker || return 1
  kill_local_mpv
  run_mpv "$out_d" "$dual_mode" "$CLIP_24" "$N" "$CF" "$log_d"

  local mult=2
  [[ $dual_mode = dual_no_interp ]] && mult=1
  printf "    %s_single: %d bytes\n" "$tag" "$(stat -c%s "$out_s")"
  printf "    %s_dual:   %d bytes (mult=%d)\n" "$tag" "$(stat -c%s "$out_d")" "$mult"

  local psnr_log="$TMPDIR/${tag}_psnr.log"
  local avg
  avg=$(ffmpeg -hide_banner -loglevel info \
    -i "$out_s" -i "$out_d" \
    -lavfi "[0:v]format=yuv420p10le[a];[1:v]format=yuv420p10le[b];[a][b]psnr=stats_file=$psnr_log" \
    -f null - 2>&1 | grep -oE 'PSNR.*' | tail -1)
  echo "    avg: ${avg:-(no PSNR line in ffmpeg output)}"

  echo "    per-frame:"
  awk -v mult=$mult '{
      n=-1; py=""; pu=""; pv="";
      for (i=1; i<=NF; i++) {
          if ($i ~ /^n:/)       n  = substr($i, 3);
          if ($i ~ /^psnr_y:/)  py = substr($i, 8);
          if ($i ~ /^psnr_u:/)  pu = substr($i, 8);
          if ($i ~ /^psnr_v:/)  pv = substr($i, 8);
      }
      if (n != -1) {
          oi = n - 1;
          if (mult == 1) kind = "SR_SRC   ";
          else           kind = (oi % mult == 0) ? "SR_SRC   " : "SR_INTERP";
          printf("      n=%-3d out=%-3d kind=%s  y=%-8s u=%-8s v=%s\n",
                 n, oi, kind, py, pu, pv);
      }
    }' "$psnr_log" | head -16
}

echo
echo "=========================================================="
echo "=== PSNR — three single-vs-dual pairs, CF=$CF, N=$N"
echo "===       src=$CLIP_24"
echo "=========================================================="

render_pair full      single          dual
echo
render_pair no_sr     single_no_sr    dual_no_sr
echo
render_pair no_interp single_no_rife  dual_no_interp

stop_worker
