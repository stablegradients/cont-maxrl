#!/bin/bash
# Submit gos_K50_N16384 across seeds 43/44/45 (168h wall limit).
set -euo pipefail

sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N16384_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N16384_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N16384_seed45.sh"
