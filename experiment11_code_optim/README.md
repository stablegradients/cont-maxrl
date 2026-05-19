# experiment11_code_optim — PIE Instruction-Count Percentile Reward

Implements **SPEC v9 (FINAL): PIE Instruction-Count Percentile Reward — GOS vs GRPO/RLOO on Code Optimization**.

## Status

Built under the strict §14 gate-and-hand-back execution order.

- §14 step 1 (branch + skeleton): **done**
- §14 step 2 (parity transcription): **done**
- §14 step 2b onward: **awaiting human confirmation at GATE A**

See `PARITY_CONFIG.md` for the verified parity to DeepCoder and the 7 drift items
recorded for human review. See `SECURITY_NOTE.md` for the W&B API key leak
discovered in the sibling DeepCoder scripts.

The authoritative source of behavior is the **SPEC v9** prompt in the original
conversation, not this README.

## Layout (planned; non-empty in current chunk only where listed)

```
experiment11_code_optim/
├── README.md                  ← this file
├── train.py                   ← 2-line wrapper per spec §2 (current chunk)
├── PARITY_CONFIG.md           ← current chunk
├── SECURITY_NOTE.md           ← current chunk
├── scripts/                   ← SLURM scripts (current chunk)
├── reward/                    ← PIE reward fn (§8.2), built §14 step 7
├── measurement/               ← C++ core (§4), built §14 step 3
├── precompute/                ← reference table (§7), built §14 step 6
├── sft/                       ← cold-start SFT (§6), built §14 step 7
├── guards/                    ← isolation guard (§10), built §14 step 7
└── docs/                      ← PROBE_REPORT, REFERENCE_TABLE_REPORT,
                                  GATE_REPORT, FORMAT_PROBE, PROVENANCE,
                                  REQUIRED_DATA_MANIFEST.json — built per spec
```

## How to invoke

PIE training reuses the unchanged estimator core from `experiment4_llm/` via
`train.py`. Submission is via the per-arm SLURM scripts under `scripts/`. The
sole varying knob across arms is `algorithm.adv_estimator` (spec §3.3
estimator-only design).

Required environment variables at submission time (the scripts hard-fail
otherwise):

- `WANDB_API_KEY` — env-only per spec §3.5 (see `SECURITY_NOTE.md`)
- `PIE_STEPS_PER_EPOCH`, `PIE_TOTAL_STEPS`, `PIE_EVAL_STEPS` — computed from
  the staged PIE train parquet at §14 step 4 (formula in `PARITY_CONFIG.md`)
- `PIE_SFT_CKPT` — HF directory produced by the §14-step-7 SFT cold-start;
  if unset, the script falls back to the base model (spec §6
  `sft.mode=none`; expected to fail §9 GO/NO-GO gate)
