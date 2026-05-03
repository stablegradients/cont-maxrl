#!/bin/bash
# Submit large-N GOS runs as SINGLE-GPU jobs (bs=128, eff_batch=128).
#
# Why single-GPU: 53 BatchNorm2d modules in the ResNet backbone — DDP at
# bs=32/GPU computes BN running stats from per-rank mini-batches of 32,
# while the exp_iou_ord_ce baseline used bs=128 on a single GPU (BN over
# 128 images). That difference subtly shifts early training even when
# eff_batch is otherwise matched. Single-GPU bs=128 = exactly the baseline.
#
# Usage:
#   bash scripts/gen_singlegpu_largeN.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_singlegpu_largeN.sh # writes, prints submissions

set -euo pipefail

REPO_DIR="/zfsauton/scratch/shrinivr/cont-maxrl"
EXP_DIR="${REPO_DIR}/experiment9_imagenet_localization"
SCRIPTS_DIR="${EXP_DIR}/sweep_scripts"
RES_DIR="${EXP_DIR}/results"
LOGS_DIR="${SCRIPTS_DIR}/logs"
DATA_DIR="/zfsauton/datasets/imagenet"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)
GROUP_SUFFIX="${GROUP_SUFFIX:-singleGPU}"

mkdir -p "${LOGS_DIR}"

# (N, mem, partition, time)
# - N ≤ 16384 fits comfortably in 48h general
# - N ≥ 65536 needs preempt (7-day) — may be preempted; checkpointed every
#   10 epochs so we can resume.
declare -a CONFIGS=(
    "4096    96G   general    48:00:00"
    "16384   96G   general    48:00:00"
    "65536   128G  general    48:00:00"
)

write_script() {
    local N="$1" mem="$2" partition="$3" time="$4" seed="$5"
    local job="gos_K50_N${N}_seed${seed}_${GROUP_SUFFIX}"
    local f="${SCRIPTS_DIR}/${job}.sh"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --partition=${partition}
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --output=${LOGS_DIR}/${job}_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N${N}_${GROUP_SUFFIX}"

cd "${REPO_DIR}"
source ~/.bashrc
conda activate cont-maxrl

# Speedup stack: bf16 autocast (~1.4×, gradient cos verified ≈ 1.0 vs fp32)
# + torch.compile mode='default' (~1.3×) + channels_last layout (~1.5×,
# baked into model.to() inside run.py)
export BF16=1
# Compile is enabled by default; set DISABLE_COMPILE=1 to turn off.

python -m experiment9_imagenet_localization.run \\
    --method gos \\
    --K 50 \\
    --N ${N} \\
    --seed ${seed} \\
    --epochs 30 \\
    --batch_size 128 \\
    --lr 5e-4 \\
    --data_dir "${DATA_DIR}" \\
    --output_dir "${RES_DIR}/${job}" \\
    --num_workers 8 \\
    --wandb
EOF
    chmod +x "${f}"
    echo "${f}"
}

n_jobs=0
for cfg in "${CONFIGS[@]}"; do
    read -r N mem partition time <<< "${cfg}"
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${N}" "${mem}" "${partition}" "${time}" "${seed}")
        if [[ "${DRY_RUN}" = "1" ]]; then
            echo "DRY: sbatch ${f}  (N=${N}, partition=${partition})"
        else
            sbatch "${f}"
        fi
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total single-GPU jobs: ${n_jobs}"
