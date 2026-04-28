#!/bin/bash
# Submit large-N GOS runs as multi-GPU DDP jobs.
#
# QOS cap: 8 GPUs/user. Per-N GPU allocation:
#   N=4096:  4 GPUs/job → 2 jobs concurrent
#   N=16384: 4 GPUs/job → 2 jobs concurrent
#   N=65536: 8 GPUs/job → 1 job concurrent (uses entire QOS)
#
# Per-GPU batch_size kept conservative (memory-safe). Effective batch is
# nproc_per_node * batch_size; gradient direction unchanged in expectation.
#
# Usage:
#   bash scripts/gen_ddp_largeN.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_ddp_largeN.sh # writes, prints submissions

set -euo pipefail

REPO_DIR="/zfsauton/scratch/shrinivr/cont-maxrl"
EXP_DIR="${REPO_DIR}/experiment9_imagenet_localization"
SCRIPTS_DIR="${EXP_DIR}/sweep_scripts"
RES_DIR="${EXP_DIR}/results"
LOGS_DIR="${SCRIPTS_DIR}/logs"
DATA_DIR="/zfsauton/datasets/imagenet"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)
# Tag: original GOS (with *N) plus a survival-prob clamp on the empirical
# CDF (advantages.SURVIVAL_FLOOR_EPS=1e-4) — exact analog of the eps=1e-4
# clamp in localization_exp_iou_ord_ce_loss on P(IoU > τ).
GROUP_SUFFIX="${GROUP_SUFFIX:-survFloor}"

mkdir -p "${LOGS_DIR}"

# (N, n_gpus, per_gpu_batch_size, mem)
# Per-GPU batch is set so EFFECTIVE batch = n_gpus * per_gpu_batch = 128,
# matching the single-GPU exp_iou_ord_ce baseline (bs=128, eff=128).
# This keeps the lr schedule (steps/epoch, warmup, cosine) identical
# across all configurations — apples-to-apples comparison.
declare -a CONFIGS=(
    "4096    4   32    96G"
    "16384   4   32   128G"
    "65536   8   16   256G"
    "131072  8   16   384G"
    "262144  8   16   384G"
)

write_script() {
    local N="$1" ngpu="$2" batch_size="$3" mem="$4" seed="$5"
    local job="gos_K50_N${N}_seed${seed}_${GROUP_SUFFIX}"
    local f="${SCRIPTS_DIR}/${job}.sh"
    local cpus=$((ngpu * 8))
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --partition=general
# Accept either a6000 or a5000 — both have enough VRAM for these configs.
# Exclude smaller cards (rtx_2080_ti = 11GB, v100 = 16GB → too small).
#SBATCH --gres=gpu:${ngpu}
#SBATCH --exclude=gpu[11-16,18,20-21,23]
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=48:00:00
#SBATCH --output=${LOGS_DIR}/${job}_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="gos_K50_N${N}_${GROUP_SUFFIX}"

export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$((20000 + RANDOM % 20000))

cd "${REPO_DIR}"
source ~/.bashrc
conda activate cont-maxrl

torchrun \\
    --nproc_per_node=${ngpu} \\
    --master_addr=\${MASTER_ADDR} \\
    --master_port=\${MASTER_PORT} \\
    -m experiment9_imagenet_localization.run \\
        --method gos \\
        --K 50 \\
        --N ${N} \\
        --seed ${seed} \\
        --epochs 30 \\
        --batch_size ${batch_size} \\
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
    read -r N ngpu batch_size mem <<< "${cfg}"
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${N}" "${ngpu}" "${batch_size}" "${mem}" "${seed}")
        if [[ "${DRY_RUN}" = "1" ]]; then
            echo "DRY: sbatch ${f}  (N=${N}, gpus=${ngpu})"
        else
            sbatch "${f}"
        fi
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total DDP jobs: ${n_jobs}"
