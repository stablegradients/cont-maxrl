# Experiment 4 Setup

Use a repo-root variable instead of hard-coded paths:

```bash
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"
```

## Option A: Conda (recommended in this repo)

Create and activate:

```bash
conda env create -f experiment4_llm/environment.yml
source experiment4_llm/scripts/activate_env.sh cont-maxrl
```

Set local module path:

```bash
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
```

Run environment verification and smoke checks:

```bash
python experiment4_llm/scripts/check_env.py
bash experiment4_llm/scripts/smoke_run.sh results_experiment4/smoke
```

## Option B: Docker (for reproducibility)

```bash
docker pull verlai/verl:vllm011.latest
docker run --gpus all --rm -it \
  -v "$ROOT_DIR":"$ROOT_DIR" \
  -w "$ROOT_DIR" \
  verlai/verl:vllm011.latest bash
```

Inside container:

```bash
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
python experiment4_llm/scripts/check_env.py
bash experiment4_llm/scripts/smoke_run.sh results_experiment4/smoke
```

## Full Training Stack (verl + vLLM)

This repo vendors a patched fork of verl as a git submodule at `vendor/verl`.
After cloning, initialise the submodule first:

```bash
git submodule update --init --recursive
```

Then install training extras:

```bash
python -m pip install "vllm==0.12.0"
python -m pip install -e "$ROOT_DIR/vendor/verl"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
```

> **Note on flash-attn**: not required. `use_remove_padding` is set to
> `False` in our training config, so the flash-attn code path is never used.

> **Note on submodule URL**: `vendor/verl` currently points to a local path.
> If cloning on a different machine, update `.gitmodules` to point to your
> fork of `volcengine/verl` and run `git submodule sync && git submodule update --init`.

Run the check again:

```bash
python experiment4_llm/scripts/check_env.py
```

## Common commands

Training:

```bash
experiment4_llm/scripts/run_training.sh experiment4_llm/configs/qwen_1.5b_grpo.yaml
```

84-run sweep (dry-run first):

```bash
DRY_RUN=1 experiment4_llm/scripts/run_sweep.sh
DRY_RUN=0 experiment4_llm/scripts/run_sweep.sh
```

Binary reward ablation sweep:

```bash
DRY_RUN=1 experiment4_llm/scripts/run_sweep_binary.sh
```

Evaluation:

```bash
experiment4_llm/scripts/run_evaluation.sh <input_jsonl> <output_dir>
```
