#!/bin/bash
# Submit 7 pilot runs: K=50, N=64, seed=42.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
sbatch "${HERE}/binary_maxrl_K50_N64_seed42.sh"
sbatch "${HERE}/dg_K50_N64_seed42.sh"
sbatch "${HERE}/gos_K50_N64_seed42.sh"
sbatch "${HERE}/ordinal_ce_K50_N64_seed42.sh"
sbatch "${HERE}/plain_ce_K50_N64_seed42.sh"
sbatch "${HERE}/reinforce_K50_N64_seed42.sh"
sbatch "${HERE}/rloo_K50_N64_seed42.sh"
