#!/usr/bin/env bash
#
# One-command setup. ALL packages installed inside conda env.
# No sudo. No system pip. No /tmp. No base env modification.
#
# Usage:  bash experiment4_llm/setup.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="cont-maxrl"

# All caches under repo root — no /tmp
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export RAY_TMPDIR="${REPO_ROOT}/.cache/ray"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"
export TMPDIR="${REPO_ROOT}/.cache/tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${HF_HOME}" \
         "${RAY_TMPDIR}" "${VLLM_CACHE_ROOT}" "${TMPDIR}"

echo "============================================"
echo "  Experiment 4: Environment Setup"
echo "  All installs inside conda env '${ENV_NAME}'"
echo "  No sudo. No system packages."
echo "============================================"

# ── Step 0: Preflight ──
if ! command -v conda &>/dev/null; then echo "ERROR: conda not in PATH"; exit 1; fi
if ! nvidia-smi &>/dev/null; then echo "ERROR: no GPU"; exit 1; fi
echo "conda: $(conda --version)"
echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c)"

# ── Step 1: Create conda env with Python + CUDA 12.4 toolkit ──
echo ""
echo "=== Step 1: conda env + CUDA 12.4 toolkit (inside env) ==="
conda create -n "${ENV_NAME}" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install CUDA 12.4 toolkit INSIDE the conda env.
# Use the labeled channel to prevent conda from pulling CUDA 13.x.
conda install -y -c "nvidia/label/cuda-12.4.1" cuda-toolkit cuda-nvcc

echo "  Python: $(python --version)"
echo "  nvcc:   $(nvcc --version | grep release)"
echo "  CUDA_HOME will be: ${CONDA_PREFIX}"

# ── Step 2: PyTorch (cu124) ──
echo ""
echo "=== Step 2: PyTorch 2.6.0+cu124 ==="
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')"

# ── Step 3: flash-attention (compiled with conda's CUDA 12.4) ──
echo ""
echo "=== Step 3: flash-attention 2.7.4.post1 ==="
pip install ninja packaging psutil

# CRITICAL: Point builds at conda's CUDA 12.4, NOT system CUDA 13.0
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

FLASH_DIR="${REPO_ROOT}/vendor/flash-attention"
[ -d "${FLASH_DIR}" ] || git clone https://github.com/Dao-AILab/flash-attention.git "${FLASH_DIR}"
cd "${FLASH_DIR}"
git checkout v2.7.4.post1

# Build only for A5000/A6000 (sm_86). No V100 (sm_70).
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=4 \
  CUDA_HOME="${CONDA_PREFIX}" \
  pip install -e . --no-build-isolation

cd "${REPO_ROOT}"
python -c "import flash_attn; print(f'  flash-attn {flash_attn.__version__}')"

# ── Step 4: vLLM ──
echo ""
echo "=== Step 4: vLLM 0.8.3 ==="
pip install vllm==0.8.3

# ── Step 5: verl (vendored with patches, --no-deps) ──
echo ""
echo "=== Step 5: verl (vendored) ==="
VERL_DIR="${REPO_ROOT}/vendor/verl"
pip install --no-deps -e "${VERL_DIR}"

# ── Step 6: Remaining packages (all inside env) ──
echo ""
echo "=== Step 6: Remaining dependencies ==="
pip install \
  ray==2.40.0 \
  transformers==4.48.0 \
  accelerate==1.2.0 \
  datasets==3.2.0 \
  peft==0.14.0 \
  tokenizers==0.21.0 \
  hydra-core==1.3.2 \
  omegaconf==2.3.0 \
  codetiming==1.4.0 \
  wandb==0.19.1 \
  numpy==1.26.4 \
  scipy==1.14.1 \
  matplotlib==3.9.4 \
  pyyaml==6.0.2 \
  pyarrow==18.1.0 \
  orjson==3.10.12 \
  tensordict<=0.6.2 \
  pytest==8.3.4 \
  pytest-timeout==2.3.1

# ── Step 7: Install this project ──
echo ""
echo "=== Step 7: cont-maxrl ==="
cd "${REPO_ROOT}" && pip install -e .

# ── Step 8: Verify ──
echo ""
echo "=== Step 8: Verification ==="
python experiment4_llm/scripts/check_env.py

echo ""
echo "============================================"
echo "  Done! Activate with: conda activate ${ENV_NAME}"
echo "============================================"
