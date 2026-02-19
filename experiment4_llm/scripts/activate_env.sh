#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="${1:-cont-maxrl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Explicitly prepend the activated env's bin dir to PATH so it takes
# precedence over any other conda env bin dirs that may have leaked in.
ENV_PREFIX="$(conda info --envs | awk -v env="$ENV_NAME" '$1==env{print $NF}')"
if [ -z "$ENV_PREFIX" ]; then
  # Fallback: ask conda directly
  ENV_PREFIX="$(conda run -n "$ENV_NAME" python -c "import sys,os; print(os.path.dirname(sys.executable))" 2>/dev/null || true)"
fi
if [ -n "$ENV_PREFIX" ] && [ -d "$ENV_PREFIX/bin" ]; then
  export PATH="$ENV_PREFIX/bin:$PATH"
fi

# Route all HuggingFace downloads (datasets + model weights) to scratch.
# HF_HOME controls both $HF_HOME/datasets and $HF_HOME/hub.
export HF_HOME="${HF_HOME:-/zfsauton/scratch/shrinivr/cont-maxrl/.cache/huggingface}"

echo "Activated conda env: $ENV_NAME"
echo "Python: $(python -V)"
echo "Python binary: $(command -v python)"
echo "HF_HOME: $HF_HOME"
