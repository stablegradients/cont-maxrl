# Full-batch PIE gem5-speedup RL (GRPO vs GOS vs RLOO)

Full-dataset RL on the live gem5 **speedup** reward, comparing the three advantage
estimators under an otherwise-identical pure on-policy PG setup.

## Spec

| knob | value |
|---|---|
| model | Qwen3-1.7B (`Qwen/Qwen3-1.7B`) |
| data | `pie_gem5_train.parquet` — 16,452 src programs (bysrc), 1,301 problems |
| reward | gem5 **speedup ratio** `src_ticks / rollout_ticks`; correctness gated on **all** usable cases; rollout zeroed if any timed case exceeds **3×** src ticks |
| timed cases | **K=1**, **random** (`case_select=sample`), **group**-sampled (same case for all 16 rollouts of a prompt) |
| batch | **B=64**, **G=16** → **1024 rollouts/step** |
| optim | lr **1e-6**, pure on-policy PG: no KL / no entropy / no ref, clip inert (`clip_ratio=100`, old_log_prob recomputed → ratio≡1), `grad_clip=2.0`, 1 ppo epoch |
| estimators | `grpo`, `gos`, `rloo` — the **only** difference between the three runs is `algorithm.adv_estimator` |
| schedule | ~257 steps/epoch × 10 = **2570 steps** (`total_training_steps=2570`) |
| eval | **none during training** (`insitu_eval.eval_steps=[]`, `val_before_train=False`, `test_freq=-1`); post-hoc only |
| ckpts | keep-**all** actor (hf_model only, ~3.4GB) snapshots every **100** steps (`PIE_CARVEOUT_EVERY=100`) → `actor_ckpts/step_<N>/huggingface/`; rolling **1** full-state every `save_freq` (default 20) for resume |
| runtime | ~400–440 s/step (gen+update floor at 1024 rollouts × 4096 max-tok; gem5 reward is async-hidden under it) → 2570 steps ≈ **8–13 days/arm**; 3 arms run concurrently on separate nodes → ~8–13 days wall; ~4–7 resume segments/arm |
| node | full GH200×4 (`--gpus-per-node=4 --cpus-per-task=288 --mem=0`), FSDP shard, `ray_init.num_cpus=280`, `PIE_GEM5_REWARD_WORKERS=256` |
| storage | bgtv (`/work/nvme/bgtv/.../results/qwen3_1.7b_pie_gem5_fb_<adv>_g16_bs64_s0`) — bgmo /work is full |
| walltime | 47:50:00 per segment (cap 48h); resumes across segments automatically |

## Files

- **`run_fullbatch_gem5.sh <grpo|gos|rloo> [seed] [segments_left]`** — the self-managing launcher.
- **`spray_fullbatch.sh <grpo|gos|rloo|all> [seed] [segments]`** — fan out across the 7 accounts.
- **`status.sh [seed]`** — queue + elected survivor + step% + last reward line per estimator.
- **`stop.sh <grpo|gos|rloo|all> [seed]`** — cancel survivor + its queued resume child + clear lock.
- `_smoke_reward.py`, `_smoke_verl.sh` — offline reward + micro real-verl plumbing smokes.

## How the spray/reap/resume works (no babysitting daemon)

The launcher self-manages via a durable lock under `…/locks/<run_tag>.lock`:

1. **Election.** Spray submits the same job (same `--job-name`) to all 7 accounts.
   The **first copy to start running** `mkdir`-claims the lock and becomes the sole
   survivor; every other copy **self-cancels** on startup. A lock whose owner job is
   no longer in `squeue` is treated as **stale** and reclaimed — so an `afterany`
   resume child re-takes ownership from its dead parent.
2. **Reap.** The survivor `scancel`s the rest of its spray cohort (same job name) so
   no still-pending sibling can revive the run later.
3. **Resume.** The survivor submits **one** `afterany:self` resume child **upfront**
   (so it survives a walltime kill), gated on work-remaining and a segment budget.
   `resume_mode=auto` picks up the rolling full-state ckpt. The tail child that finds
   `step ≥ 2570` exits immediately.

Cross-estimator jobs have distinct names (`pie_fb_grpo_s0` vs `pie_fb_gos_s0` …) so
they never reap each other; all three arms run concurrently on separate nodes.

## Run

```bash
source .env                                    # WANDB_API_KEY + HF_TOKEN
experiment11_code_optim/scripts/fullbatch_gem5/spray_fullbatch.sh all 0 8
watch -n30 experiment11_code_optim/scripts/fullbatch_gem5/status.sh
```

## Post-hoc eval (after training)

Evaluate the kept actor snapshots on the held-out **test** set (878 rows, 40 problems)
— `max_reward@k` (k=1,4,16,64,256) at n=1024, gem5 reward. Built separately once the
arms are training (the carveout snapshots under `actor_ckpts/step_<N>` are the inputs).
