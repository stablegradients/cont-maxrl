#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SCRIPT="$ROOT_DIR/experiment4_llm/scripts/run_training.sh"

if [ ! -x "$TRAIN_SCRIPT" ]; then
  chmod +x "$TRAIN_SCRIPT"
fi

DRY_RUN="${DRY_RUN:-1}"

run_one() {
  local config="$1"
  local seed="$2"
  local group_size="$3"

  local cmd=("$TRAIN_SCRIPT" "$config" "--trainer.seed" "$seed" "--rollout.group_size" "$group_size")

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

methods_all=(grpo maxrl_unnorm maxrl_norm reinforce)
# 32B sweep intentionally excludes REINFORCE to match the Experiment 4 matrix.
methods_large=(grpo maxrl_unnorm maxrl_norm)
groups_small=(4 16 32)
groups_large=(16 32)
seeds_three=(42 43 44)
seeds_two=(42 43)

for method in "${methods_all[@]}"; do
  config="$ROOT_DIR/experiment4_llm/configs/qwen_1.5b_${method}.yaml"
  [ -f "$config" ] || continue
  for g in "${groups_small[@]}"; do
    for s in "${seeds_three[@]}"; do
      run_one "$config" "$s" "$g"
    done
  done
done

for method in "${methods_all[@]}"; do
  config="$ROOT_DIR/experiment4_llm/configs/qwen_7b_${method}.yaml"
  [ -f "$config" ] || continue
  for g in "${groups_small[@]}"; do
    for s in "${seeds_three[@]}"; do
      run_one "$config" "$s" "$g"
    done
  done
done

for method in "${methods_large[@]}"; do
  config="$ROOT_DIR/experiment4_llm/configs/qwen_32b_${method}.yaml"
  [ -f "$config" ] || continue
  for g in "${groups_large[@]}"; do
    for s in "${seeds_two[@]}"; do
      run_one "$config" "$s" "$g"
    done
  done
done
