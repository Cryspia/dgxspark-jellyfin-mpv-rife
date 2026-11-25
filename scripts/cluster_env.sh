#!/usr/bin/env bash
# Accept legacy environment names while preferring explicit cluster settings.
cluster_normalize_env() {
  local legacy canonical
  for legacy in ${!DUAL_@}; do
    canonical="CLUSTER_${legacy#DUAL_}"
    if [[ ! -v "$canonical" ]]; then
      printf -v "$canonical" '%s' "${!legacy}"
      export "$canonical"
    fi
  done
}
