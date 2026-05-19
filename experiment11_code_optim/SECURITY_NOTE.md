# SECURITY_NOTE.md — W&B API key leak (spec v9 §3.5)

**Status:** Action required from the human operator. The agent will NOT perform any of the remediation steps below (rotation/history-scrub/force-push are destructive and visible to others, per spec §0 hard rules and the default Claude Code "executing actions with care" guidance).

## Finding

The existing DeepCoder SLURM scripts in `experiment4_llm/scripts/` hardcode a real, valid Weights & Biases API key in plaintext. Confirmed by reading two of them at build time on 2026-05-19:

- `experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_maxrl_g16.sh:48`
- `experiment4_llm/scripts/qwen3_1.7b_base_deepcoder_grpo_g16.sh:48`

The literal value is on line 48 of each file:

```bash
export WANDB_API_KEY="wandb_v1_<REDACTED_BUT_PRESENT_IN_GIT>"
```

The key is committed into the repository's history (it appears in the latest commits on the `deltaAI` branch and is therefore reachable from the `pie-efficiency` branch as well).

There are likely more siblings affected. A quick grep across the script tree should be run by the human as part of the rotation work:

```bash
grep -rln 'WANDB_API_KEY=' experiment4_llm/scripts/
```

(The agent has not run this grep across the whole tree to avoid surfacing the secret further in tool outputs; the line:column references above are sufficient evidence of the leak.)

## PIE scripts do NOT copy this pattern

The PIE SLURM scripts in `experiment11_code_optim/scripts/qwen3_1.7b_base_pie_*.sh` instead require the key to be present in the submission environment:

```bash
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in submission env (spec §3.5 / SECURITY_NOTE.md)}"
```

This hard-fails before any training command runs if the variable is unset. The key never enters the script file or any committed artifact.

## Required human action (in order)

1. **Rotate the W&B key.** Revoke the leaked one at https://wandb.ai/authorize and generate a fresh one. Until rotation, the leaked key remains usable by anyone with read access to the repository (including its full git history).
2. **Scrub git history.** Use `git filter-repo --replace-text` (preferred) or BFG Repo-Cleaner to remove the literal key string from every commit. Force-push the cleaned history to the remote(s). Coordinate with collaborators because force-pushing rewrites SHAs.
3. **Move the new key into the submission environment**, not into the repo. Reasonable options:
   - `~/.bashrc` / `~/.bash_profile` on the login + compute nodes (DeltaAI shares home over GPFS)
   - A `.env` file outside the repo, sourced before submission
   - A SLURM submission wrapper that reads from a secret store
4. **Confirm `.gitignore`** excludes any `.env` or other local secret files (`.env` already appears as an untracked file in git status — verify it is in `.gitignore` before adding any secret to it).
5. **Optional but recommended:** add a pre-commit hook that greps for the literal `WANDB_API_KEY=` pattern and blocks the commit unless the right-hand side is an env-var reference (`${WANDB_API_KEY}` etc.) rather than a literal string.

## Scope and limits of this note

- Only the W&B credential leak is audited here.
- Other credentials in the repository (Hugging Face token, SLURM accounts, etc.) have not been audited; the agent recommends a broader sweep at the human's discretion.
- The PIE branch (`pie-efficiency`) inherits the leak from `deltaAI` until git history is scrubbed; remediation is a repo-wide action, not a branch-local one.
