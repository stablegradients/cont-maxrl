#!/bin/bash
#SBATCH --job-name=bench_step
#SBATCH --partition=preempt
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/bench_step_%j.out

set -eo pipefail
cd /zfsauton/scratch/shrinivr/cont-maxrl
source ~/.bashrc
conda activate cont-maxrl

python -m experiment9_imagenet_localization.scripts.bench_step
