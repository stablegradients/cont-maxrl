#!/bin/bash
# Submit Stage 4 (20 scripts).
set -euo pipefail

sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N4_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N4_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N4_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N16_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N16_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N16_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N256_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N256_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K50_N256_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N4_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N4_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N4_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N16_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N16_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N16_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N256_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N256_seed43.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K50_N256_seed44.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/gos_K2_N64_seed42.sh"
sbatch "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/binary_maxrl_K2_N64_seed42.sh"
