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

CMD="${VERL_TRAIN_CMD:-python -m verl.trainer.main_ppo}"

echo "Running training with config: $CONFIG_PATH"
echo "Command: $CMD --config $CONFIG_PATH $*"

if [ -z "${VERL_TRAIN_CMD:-}" ]; then
  if ! python -c "import verl" >/dev/null 2>&1; then
    cat <<'EOF'
ERROR: Python module 'verl' is not installed in the current environment.
Install verl (and related training deps) or set VERL_TRAIN_CMD to your trainer entrypoint.

Example:
  VERL_TRAIN_CMD="python -m verl.trainer.main_ppo" experiment4_llm/scripts/run_training.sh <config.yaml>
EOF
    exit 1
  fi
fi

# shellcheck disable=SC2086
$CMD --config "$CONFIG_PATH" "$@"
