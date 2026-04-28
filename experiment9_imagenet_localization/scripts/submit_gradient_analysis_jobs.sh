#!/bin/bash
# Submit gradient + cosine analysis as parallel SLURM jobs.
#
# QOS limit: 8 GPUs/user. We batch coarsely:
#   - one per_image job per seed: loops through every (method, N) combo with
#     checkpoints for that seed and runs all milestone epochs sequentially
#   - one cosine job per (ref_method, seed): loops through milestone epochs
#
# That keeps the total job count comfortably under the QOS cap.
#
# Usage:
#   bash scripts/submit_gradient_analysis_jobs.sh           # default seeds 43 44 45
#   SEEDS="43" bash scripts/submit_gradient_analysis_jobs.sh
#   DRY_RUN=1 bash scripts/submit_gradient_analysis_jobs.sh

set -euo pipefail

REPO_DIR="/zfsauton/scratch/shrinivr/cont-maxrl"
EXP_DIR="${REPO_DIR}/experiment9_imagenet_localization"
RESULTS_DIR="${EXP_DIR}/results"
LOGS_DIR="${EXP_DIR}/sweep_scripts/logs"
JOBS_DIR="${EXP_DIR}/sweep_scripts/grad_jobs"
DATA_DIR="/zfsauton/datasets/imagenet"

SEEDS="${SEEDS:-43 44 45}"
K="${K:-50}"
DRY_RUN="${DRY_RUN:-0}"

# (method, N) — N is the variant whose checkpoints are populated.
PER_IMAGE_METHOD_N=(
    "gos:64"
    "grpo:256"
    "reinforce:256"
    "rloo:256"
    "binary_maxrl:64"
    "dg:64"
    "exp_iou_ord_ce:64"
    "plain_ce:64"
)

# Cosine reference runs whose checkpoints are populated. (ordinal_ce / mse
# refs are blocked because ordinal_ce checkpoints don't exist.)
COSINE_REF_RUN_BASE=(
    "plain_ce:plain_ce_K${K}_N64"
)
COSINE_METHODS="gos,binary_maxrl,grpo,rloo,reinforce,dg"

mkdir -p "${LOGS_DIR}" "${JOBS_DIR}"

# Header writer — note: `set +u` is enabled here because conda's activation
# scripts touch unset shell vars (ADDR2LINE etc.) and we don't want to abort.
write_sbatch_header() {
    local job_name="$1"
    local out_path="$2"
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=${out_path}

set +u
cd "${REPO_DIR}"
source ~/.bashrc
conda activate cont-maxrl
set -u
set -eo pipefail
PYBIN="\${CONDA_PREFIX:-/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl}/bin/python"
EOF
}

submit() {
    local script="$1"
    chmod +x "${script}"
    if [[ "${DRY_RUN}" = "1" ]]; then
        echo "DRY: sbatch ${script}"
    else
        sbatch "${script}"
    fi
}

# ---------------------------------------------------------------------------
# 1. per_image — one job per seed.
# ---------------------------------------------------------------------------
echo "=== per_image jobs (1 per seed) ==="
n_per_image=0
for SEED in ${SEEDS}; do
    job_name="gpi_seed${SEED}"
    script="${JOBS_DIR}/${job_name}.sh"
    log_path="${LOGS_DIR}/${job_name}_%j.out"

    {
        write_sbatch_header "${job_name}" "${log_path}"
        echo
        echo "echo \"### per_image  seed=${SEED}  $(date) ###\""
        for entry in "${PER_IMAGE_METHOD_N[@]}"; do
            method="${entry%:*}"
            N="${entry#*:}"
            run_name="${method}_K${K}_N${N}_seed${SEED}"
            run_dir="${RESULTS_DIR}/${run_name}"
            ckpt_dir="${run_dir}/checkpoints"
            out_dir="${run_dir}/gradient_analysis"

            [[ ! -d "${ckpt_dir}" ]] && continue
            epochs=()
            for ep in 1 10 25 50; do
                [[ -f "${ckpt_dir}/epoch_${ep}.pt" ]] && epochs+=("${ep}")
            done
            [[ ${#epochs[@]} -eq 0 ]] && continue

            echo
            echo "mkdir -p \"${out_dir}\""
            for ep in "${epochs[@]}"; do
                cat <<EOF
echo "=== per_image  ${run_name}  epoch=${ep} ==="
"\${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis per_image \\
    --checkpoint "${ckpt_dir}/epoch_${ep}.pt" \\
    --data_dir "${DATA_DIR}" \\
    --method "${method}" \\
    --K ${K} \\
    --seed ${SEED} \\
    --n_images 500 \\
    --n_threshold_images 50 \\
    --output_dir "${out_dir}"
EOF
            done
        done
    } > "${script}"
    submit "${script}"
    n_per_image=$((n_per_image+1))
done

# ---------------------------------------------------------------------------
# 2. cosine — one job per (ref, seed).
# ---------------------------------------------------------------------------
echo
echo "=== cosine jobs (1 per ref × seed) ==="
n_cosine=0
for SEED in ${SEEDS}; do
    for entry in "${COSINE_REF_RUN_BASE[@]}"; do
        ref="${entry%:*}"
        run_base="${entry#*:}"
        run_name="${run_base}_seed${SEED}"
        run_dir="${RESULTS_DIR}/${run_name}"
        ckpt_dir="${run_dir}/checkpoints"
        out_dir="${run_dir}/gradient_analysis"

        [[ ! -d "${ckpt_dir}" ]] && continue
        epochs=()
        for ep in 1 10 25 50; do
            [[ -f "${ckpt_dir}/epoch_${ep}.pt" ]] && epochs+=("${ep}")
        done
        [[ ${#epochs[@]} -eq 0 ]] && continue

        job_name="gcos_${ref}_s${SEED}"
        script="${JOBS_DIR}/${job_name}.sh"
        log_path="${LOGS_DIR}/${job_name}_%j.out"

        {
            write_sbatch_header "${job_name}" "${log_path}"
            echo
            echo "mkdir -p \"${out_dir}\""
            for ep in "${epochs[@]}"; do
                cat <<EOF
echo "=== cosine  ref=${ref}  ${run_name}  epoch=${ep} ==="
"\${PYBIN}" -m experiment9_imagenet_localization.gradient_analysis cosine \\
    --checkpoint "${ckpt_dir}/epoch_${ep}.pt" \\
    --data_dir "${DATA_DIR}" \\
    --K ${K} \\
    --seed ${SEED} \\
    --ref_method "${ref}" \\
    --methods "${COSINE_METHODS}" \\
    --Gs "4,8,16,32,64,128,256,512,1024,4096" \\
    --n_images 200 \\
    --n_trials 10 \\
    --output_dir "${out_dir}"
EOF
            done
        } > "${script}"
        submit "${script}"
        n_cosine=$((n_cosine+1))
    done
done

echo
echo "Submitted: per_image=${n_per_image}  cosine=${n_cosine}  total=$((n_per_image+n_cosine))"
echo "Job scripts in: ${JOBS_DIR}"
echo "Logs in:        ${LOGS_DIR}"
