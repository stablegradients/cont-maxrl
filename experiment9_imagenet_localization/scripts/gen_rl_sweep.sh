#!/bin/bash
# Generate + submit a clean RL sweep: gos at all N's, plus grpo/rloo/
# reinforce/dg/binary_maxrl at N=256, for seeds 43/44/45.
#
# Reflects the fix (no * N inside gos_advantage, .sum() over rollouts in
# the GOS branch of train.py) — both pieces are already in the codebase.
#
# Usage:
#   bash scripts/gen_rl_sweep.sh           # writes scripts + submits
#   DRY_RUN=1 bash scripts/gen_rl_sweep.sh # writes, prints submissions

set -euo pipefail

REPO_DIR="/zfsauton/scratch/shrinivr/cont-maxrl"
EXP_DIR="${REPO_DIR}/experiment9_imagenet_localization"
SCRIPTS_DIR="${EXP_DIR}/sweep_scripts"
RES_DIR="${EXP_DIR}/results"
LOGS_DIR="${SCRIPTS_DIR}/logs"
DATA_DIR="/zfsauton/datasets/imagenet"

DRY_RUN="${DRY_RUN:-0}"
SEEDS=(43 44 45)

# Tag this sweep so the W&B groups stay distinct from prior sweeps. The
# run displayName still matches the local output_dir so files line up,
# but W&B group is suffixed to keep histories separate.
GROUP_SUFFIX="${GROUP_SUFFIX:-fixedN}"

mkdir -p "${LOGS_DIR}"

write_script() {
    local method="$1" N="$2" seed="$3" batch_size="$4" mem="$5" time="$6"
    local job="${method}_K50_N${N}_seed${seed}_${GROUP_SUFFIX}"
    local f="${SCRIPTS_DIR}/${job}.sh"
    cat > "${f}" <<EOF
#!/bin/bash
#SBATCH --job-name=${job}
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${LOGS_DIR}/${job}_%j.out

set -eo pipefail
export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="${method}_K50_N${N}_${GROUP_SUFFIX}"

cd "${REPO_DIR}"
source ~/.bashrc
conda activate cont-maxrl

python -m experiment9_imagenet_localization.run \\
    --method ${method} \\
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

submit() {
    local f="$1"
    if [[ "${DRY_RUN}" = "1" ]]; then
        echo "DRY: sbatch ${f}"
    else
        sbatch "${f}"
    fi
}

# (method, N, batch_size, mem, time) — params chosen for stability:
#  - small N: default (bs=128, mem=32G)
#  - large N: smaller bs + bigger mem (rollout tensors get large)
declare -a CONFIGS=(
    "gos          16   128  32G  48:00:00"
    "gos          64   128  32G  48:00:00"
    "gos         256   128  32G  48:00:00"
    "gos        1024   128  32G  48:00:00"
    "gos        4096   128  32G  48:00:00"
    "gos       16384    64  64G  48:00:00"
    "gos       65536    16 128G  48:00:00"
    "grpo        256   128  32G  48:00:00"
    "rloo        256   128  32G  48:00:00"
    "reinforce   256   128  32G  48:00:00"
    "dg          256   128  32G  48:00:00"
    "binary_maxrl 256  128  32G  48:00:00"
)

n_jobs=0
for cfg in "${CONFIGS[@]}"; do
    read -r method N batch_size mem time <<< "${cfg}"
    for seed in "${SEEDS[@]}"; do
        f=$(write_script "${method}" "${N}" "${seed}" "${batch_size}" "${mem}" "${time}")
        submit "${f}"
        n_jobs=$((n_jobs+1))
    done
done

echo
echo "Total jobs: ${n_jobs}"
