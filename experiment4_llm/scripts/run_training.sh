#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.yaml> [extra trainer args...]"
  exit 1
fi

CONFIG_PATH="$1"
shift

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Config not found: $CONFIG_PATH"
  exit 1
fi

# Route all HuggingFace downloads (datasets + model weights) to scratch.
export HF_HOME="${HF_HOME:-/zfsauton/scratch/shrinivr/cont-maxrl/.cache/huggingface}"

# Resolve the Python binary to avoid PATH pollution from other conda envs.
# Prefer the explicit binary for the cont-maxrl env if VERL_PYTHON is set;
# otherwise fall back to whatever 'python' resolves to.
_PYTHON="${VERL_PYTHON:-$(command -v python3 2>/dev/null || command -v python)}"
CMD="${VERL_TRAIN_CMD:-${_PYTHON} -m experiment4_llm.src.verl_main_ppo}"

echo "Running training with config: $CONFIG_PATH"

HYDRA_OVERRIDES=()
while [ $# -gt 0 ]; do
  arg="$1"
  shift

  if [[ "$arg" == --* ]]; then
    key="${arg#--}"
    if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
      HYDRA_OVERRIDES+=("${key}=$1")
      shift
    else
      HYDRA_OVERRIDES+=("${key}")
    fi
  else
    HYDRA_OVERRIDES+=("$arg")
  fi
done

echo "Command: $CMD --config $CONFIG_PATH ${HYDRA_OVERRIDES[*]}"

if [ -z "${VERL_TRAIN_CMD:-}" ]; then
  if ! "${_PYTHON}" -c "import verl" >/dev/null 2>&1; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    VERL_SUBMODULE="$REPO_ROOT/vendor/verl"
    if [ -f "$VERL_SUBMODULE/setup.py" ] || [ -f "$VERL_SUBMODULE/pyproject.toml" ]; then
      echo "verl not found — installing from submodule at $VERL_SUBMODULE"
      "${_PYTHON}" -m pip install -e "$VERL_SUBMODULE" --quiet
    else
      cat <<'EOF'
ERROR: Python module 'verl' is not installed and vendor/verl submodule is missing.
Run: git submodule update --init --recursive
Then: pip install -e vendor/verl
EOF
      exit 1
    fi
  fi
fi

# shellcheck disable=SC2086
$CMD --config "$CONFIG_PATH" "${HYDRA_OVERRIDES[@]}"
