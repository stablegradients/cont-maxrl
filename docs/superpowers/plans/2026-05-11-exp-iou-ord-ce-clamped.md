# `exp_iou_ord_ce_clamped` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clamped variant of `localization_exp_iou_ord_ce_loss` (predicted box corners clipped to `[0,1]` before IoU) and submit 3 Adam SLURM jobs (seeds 42/43/44) under wandb group `exp_iou_ord_ce_clamped`.

**Architecture:** Single-arg extension to the existing loss (`clamp_pred: bool = False`). New supervised method string `exp_iou_ord_ce_clamped` registered in `run.py` / dispatched in `train.py`. Three new sweep scripts mirror the existing `exp_iou_ord_ce_K50_N64_seed*.sh` template. No new files outside `losses.py` test additions and scripts.

**Tech Stack:** PyTorch, pytest, SLURM (sbatch), wandb (read-only, just for group naming).

**Repository state caveat:** `run.py` and `train.py` already have uncommitted local changes (SGD support; online running-metrics). My edits land in *different regions* (line ~50 + ~269) — no conflict. **Do not commit during this plan.** The user will commit later alongside their own changes.

---

## File Structure

**Modified:**
- `experiment9_imagenet_localization/losses.py` — extend `localization_exp_iou_ord_ce_loss` signature + implementation.
- `experiment9_imagenet_localization/run.py:50` — add new method to `SUPERVISED_METHODS`.
- `experiment9_imagenet_localization/train.py:269` — extend the existing `elif` for the new method.
- `experiment9_imagenet_localization/tests/test_losses.py` — append two new tests.

**Created (sweep scripts):**
- `experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed42.sh`
- `experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed43.sh`
- `experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed44.sh`

---

### Task 1: Write failing tests for `clamp_pred` argument

**Files:**
- Test: `experiment9_imagenet_localization/tests/test_losses.py` (append at end of the `exp_iou_ord_ce` section, ~line 130)

- [ ] **Step 1: Append two tests**

Add these tests immediately after `test_exp_iou_ord_ce_handles_invalid_gt_rows` (around line 130):

```python
def test_exp_iou_ord_ce_clamp_pred_false_matches_baseline():
    """clamp_pred=False must produce *bit-identical* output to omitting the
    kwarg, so the existing exp_iou_ord_ce baseline is unaffected."""
    B, K = 2, 5
    logits = _random_logits(B, K, seed=7)
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.4, 0.4], [0.2, 0.3, 0.1, 0.2], [0, 0, 0, 0]],
        [[0.3, 0.7, 0.5, 0.3], [0, 0, 0, 0],          [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, True, False], [True, False, False]])
    base = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10,
    )
    explicit = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=False,
    )
    assert torch.equal(base, explicit), (
        f"clamp_pred=False must not change behavior; got {base} vs {explicit}"
    )


def test_exp_iou_ord_ce_clamp_pred_true_finite_grad_and_differs():
    """clamp_pred=True must return finite/positive loss with grad to all heads
    and must *differ* from clamp_pred=False on logits where the predicted box
    can extend outside [0,1]."""
    B, K = 2, 5
    # Fresh logits with grad enabled so we can check gradient flow.
    g = torch.Generator().manual_seed(11)
    logits = {h: torch.randn(B, K, generator=g, requires_grad=True)
              for h in HEAD_NAMES}
    gt_boxes = torch.tensor([
        [[0.5, 0.5, 0.6, 0.6], [0, 0, 0, 0]],
        [[0.2, 0.8, 0.5, 0.3], [0, 0, 0, 0]],
    ])
    gt_mask = torch.tensor([[True, False], [True, False]])

    loss_unclamped = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=False,
    )
    loss_clamped = localization_exp_iou_ord_ce_loss(
        logits, gt_boxes, gt_mask, K=K, n_thresholds=10, clamp_pred=True,
    )

    assert torch.isfinite(loss_clamped), f"clamped loss not finite: {loss_clamped}"
    assert loss_clamped.item() >= 0.0, f"clamped loss must be nonneg, got {loss_clamped}"
    # At K=5 the bin centers reach 0.9 and bin-width 0.9; for many (xc, w)
    # combos the unclamped corners fall outside [0,1], so the two losses
    # must differ on non-degenerate logits.
    assert not torch.allclose(loss_unclamped, loss_clamped), (
        f"clamp_pred=True did not change anything: "
        f"unclamped={loss_unclamped.item()} clamped={loss_clamped.item()}"
    )
    # Gradient flows to all heads under clamping.
    loss_clamped.backward()
    for h in HEAD_NAMES:
        g = logits[h].grad
        assert g is not None, f"no grad reached head {h!r}"
        assert torch.isfinite(g).all(), f"non-finite grad on head {h!r}: {g}"
        assert g.abs().sum().item() > 0.0, f"zero grad on head {h!r}"
```

- [ ] **Step 2: Run new tests, expect failure**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -m pytest \
  experiment9_imagenet_localization/tests/test_losses.py::test_exp_iou_ord_ce_clamp_pred_false_matches_baseline \
  experiment9_imagenet_localization/tests/test_losses.py::test_exp_iou_ord_ce_clamp_pred_true_finite_grad_and_differs \
  -v
```

Expected: both fail with `TypeError: localization_exp_iou_ord_ce_loss() got an unexpected keyword argument 'clamp_pred'`.

---

### Task 2: Implement `clamp_pred` in `localization_exp_iou_ord_ce_loss`

**Files:**
- Modify: `experiment9_imagenet_localization/losses.py:316-416` (signature + corner construction + `pred_area` + union)

- [ ] **Step 1: Update signature and docstring `Args:` entry**

Change the function signature at line ~316:

```python
def localization_exp_iou_ord_ce_loss(
    logits: dict[str, torch.Tensor],     # each (B, K)
    gt_boxes: torch.Tensor,              # (B, M, 4) xywh in [0, 1]
    gt_mask: torch.Tensor,               # (B, M) bool — valid GT rows
    K: int,
    n_thresholds: int = 100,
    eps: float = 1e-4,
    clamp_pred: bool = False,
) -> torch.Tensor:
```

Add to the docstring's `Args:` block (after `eps:`):

```
        clamp_pred:    if True, clamp predicted box corners to [0, 1] before
                       computing intersection/union, matching the RL path's
                       clamp_boxes_to_image. The predicted area becomes a
                       4-D tensor (K_xc, K_yc, K_w, K_h) because the clamped
                       width depends on xc (and clamped height on yc).
                       Default False (baseline behavior — bit-identical to
                       the pre-flag version).
```

- [ ] **Step 2: Branch on `clamp_pred` for corners + area**

Replace the block at `losses.py:376-383` (the unclamped corner construction and 2-D `pred_area`) with the conditional below. Keep the `# 3. Predicted corner intervals per (k1, k3) / (k2, k4)` comment.

```python
    # 3. Predicted corner intervals per (k1, k3) / (k2, k4)
    px1 = bc.view(K, 1) - bc.view(1, K) / 2    # (K, K)
    px2 = bc.view(K, 1) + bc.view(1, K) / 2
    py1 = bc.view(K, 1) - bc.view(1, K) / 2
    py2 = bc.view(K, 1) + bc.view(1, K) / 2

    if clamp_pred:
        # Mirror clamp_boxes_to_image in the RL path: corners pinned to [0, 1].
        # After clamping, the predicted width depends on (xc, w) jointly, so
        # pred_area becomes 4-D over (xc, yc, w, h).
        px1 = px1.clamp(0.0, 1.0)
        px2 = px2.clamp(0.0, 1.0)
        py1 = py1.clamp(0.0, 1.0)
        py2 = py2.clamp(0.0, 1.0)
        pred_w = (px2 - px1).clamp(min=0.0)                              # (K_xc, K_w)
        pred_h = (py2 - py1).clamp(min=0.0)                              # (K_yc, K_h)
        # axis order matches iou_m: (xc, yc, w, h)
        pred_area = pred_w[:, None, :, None] * pred_h[None, :, None, :]  # (K_xc, K_yc, K_w, K_h)
    else:
        # Predicted area (K_w, K_h) over (k3, k4) = w × h (full, unclamped)
        pred_area = bc[:, None] * bc[None, :]                            # (K_w, K_h)
```

- [ ] **Step 3: Branch on `clamp_pred` for the `union` broadcast inside the GT loop**

Replace lines `losses.py:411-415` (the `union = ( pred_area[None, None, None, :, :] + ... )` block) with:

```python
        if clamp_pred:
            # pred_area shape (K_xc, K_yc, K_w, K_h) → broadcast as (1, ...)
            union = (
                pred_area[None]
                + gt_area[:, None, None, None, None]
                - inter
            )
        else:
            # pred_area shape (K_w, K_h) → broadcast as (1, 1, 1, K_w, K_h)
            union = (
                pred_area[None, None, None, :, :]
                + gt_area[:, None, None, None, None]
                - inter
            )
```

- [ ] **Step 4: Run the two new tests, expect pass**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -m pytest \
  experiment9_imagenet_localization/tests/test_losses.py::test_exp_iou_ord_ce_clamp_pred_false_matches_baseline \
  experiment9_imagenet_localization/tests/test_losses.py::test_exp_iou_ord_ce_clamp_pred_true_finite_grad_and_differs \
  -v
```

Expected: both PASS.

- [ ] **Step 5: Run the full `test_losses.py` to confirm no regression**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -m pytest \
  experiment9_imagenet_localization/tests/test_losses.py -v
```

Expected: all tests PASS (baseline `exp_iou_ord_ce` tests must remain green — that's what guards the `clamp_pred=False` no-regression invariant).

---

### Task 3: Register new method in `run.py` and dispatch in `train.py`

**Files:**
- Modify: `experiment9_imagenet_localization/run.py:46-51`
- Modify: `experiment9_imagenet_localization/train.py:269-276`

- [ ] **Step 1: Add new method to `SUPERVISED_METHODS`**

In `run.py`, change the `SUPERVISED_METHODS` set (lines 46-51) from:

```python
SUPERVISED_METHODS = {
    "ordinal_ce", "plain_ce",
    "mse", "mse_iou_match", "mse_centroid_match",
    "l1_iou_match", "l1_centroid_match",
    "exp_iou_ord_ce",
}
```

to:

```python
SUPERVISED_METHODS = {
    "ordinal_ce", "plain_ce",
    "mse", "mse_iou_match", "mse_centroid_match",
    "l1_iou_match", "l1_centroid_match",
    "exp_iou_ord_ce", "exp_iou_ord_ce_clamped",
}
```

- [ ] **Step 2: Extend the dispatch in `train_supervised_epoch_localization`**

In `train.py`, change the branch starting at line 269 from:

```python
        elif method == "exp_iou_ord_ce":
            logits = model(images)                             # dict of (B, K)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_exp_iou_ord_ce_loss(
                logits, gt_boxes, gt_mask, K,
                n_thresholds=100,
            )
```

to:

```python
        elif method in ("exp_iou_ord_ce", "exp_iou_ord_ce_clamped"):
            logits = model(images)                             # dict of (B, K)
            gt_boxes = batch['gt_boxes'].to(device)             # (B, M, 4)
            gt_mask  = batch['gt_mask'].to(device)              # (B, M) bool
            loss = localization_exp_iou_ord_ce_loss(
                logits, gt_boxes, gt_mask, K,
                n_thresholds=100,
                clamp_pred=(method == "exp_iou_ord_ce_clamped"),
            )
```

- [ ] **Step 3: Smoke-check argparse accepts the new method**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -c "
from experiment9_imagenet_localization.run import ALL_METHODS, SUPERVISED_METHODS
assert 'exp_iou_ord_ce_clamped' in SUPERVISED_METHODS
assert 'exp_iou_ord_ce_clamped' in ALL_METHODS
print('OK:', sorted(SUPERVISED_METHODS))
"
```

Expected: prints `OK: [...]` including `'exp_iou_ord_ce_clamped'`. No error.

- [ ] **Step 4: Argparse `--method` round-trip**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -m experiment9_imagenet_localization.run --help 2>&1 | grep -A2 -- '--method'
```

Expected: the help output for `--method` lists `exp_iou_ord_ce_clamped` among the choices.

---

### Task 4: Create the three sweep scripts

**Files:**
- Create: `experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed42.sh`
- Create: `experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed43.sh`
- Create: `experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed44.sh`

- [ ] **Step 1: Write `..._seed42.sh`**

Write this content (copy verbatim, all lines including the final newline):

```bash
#!/bin/bash
#SBATCH --job-name=j_clamped_seed42
#SBATCH --partition=general
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs/j_clamped_seed42_%j.out

set -eo pipefail

export WANDB_DIR=/zfsauton/scratch/shrinivr/wandb
export WANDB_ENTITY=stablegradients
export WANDB_RUN_GROUP="exp_iou_ord_ce_clamped"

cd "/zfsauton/scratch/shrinivr/cont-maxrl"

source ~/.bashrc
conda activate cont-maxrl

python -m experiment9_imagenet_localization.run \
    --method exp_iou_ord_ce_clamped \
    --K 50 \
    --N 64 \
    --seed 42 \
    --epochs 30 \
    --batch_size 128 \
    --lr 5e-4 \
    --data_dir "/zfsauton/datasets/imagenet" \
    --output_dir "/zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/results/exp_iou_ord_ce_clamped_K50_N64_seed42" \
    --num_workers 8 \
    --wandb
```

- [ ] **Step 2: Write `..._seed43.sh`**

Same as above but replace **every** occurrence of `seed42` with `seed43` and `--seed 42` with `--seed 43`. (Affects: `#SBATCH --job-name`, `#SBATCH --output`, `--seed`, `--output_dir`.)

- [ ] **Step 3: Write `..._seed44.sh`**

Same as above but with `seed44` / `--seed 44`.

- [ ] **Step 4: Ensure log dir exists**

Run:
```
mkdir -p /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/logs
```

Expected: no output (silent success). If the dir already exists, `mkdir -p` is a no-op.

- [ ] **Step 5: Sanity-check scripts**

Run:
```
ls -la /zfsauton/scratch/shrinivr/cont-maxrl/experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_*.sh
```

Expected: 3 files listed. They do not need to be `chmod +x` because `sbatch` reads them as text.

---

### Task 5: Final pre-submit verification

- [ ] **Step 1: Run the full loss test file once more**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -m pytest \
  experiment9_imagenet_localization/tests/test_losses.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Verify data dir + python imports cleanly**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
ls -d /zfsauton/datasets/imagenet && \
/zfsauton/scratch/shrinivr/conda-envs/cont-maxrl/bin/python -c "
from experiment9_imagenet_localization.run import main
from experiment9_imagenet_localization.losses import localization_exp_iou_ord_ce_loss
import inspect
sig = inspect.signature(localization_exp_iou_ord_ce_loss)
assert 'clamp_pred' in sig.parameters
print('clamp_pred parameter present; default =', sig.parameters['clamp_pred'].default)
"
```

Expected: prints `/zfsauton/datasets/imagenet` then `clamp_pred parameter present; default = False`.

- [ ] **Step 3: Confirm sbatch + squeue are on `$PATH`**

Run:
```
which sbatch && which squeue
```

Expected: both resolve to paths.

---

### Task 6: Submit the 3 SLURM jobs

- [ ] **Step 1: Submit seed42**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
sbatch experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed42.sh
```

Expected: prints `Submitted batch job <JOBID>`. Record the JOBID.

- [ ] **Step 2: Submit seed43**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
sbatch experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed43.sh
```

Expected: `Submitted batch job <JOBID>`. Record.

- [ ] **Step 3: Submit seed44**

Run:
```
cd /zfsauton/scratch/shrinivr/cont-maxrl && \
sbatch experiment9_imagenet_localization/sweep_scripts/exp_iou_ord_ce_clamped_K50_N64_seed44.sh
```

Expected: `Submitted batch job <JOBID>`. Record.

- [ ] **Step 4: Verify all 3 jobs are queued/running**

Run:
```
squeue -u $USER --name=j_clamped_seed42,j_clamped_seed43,j_clamped_seed44
```

Expected: 3 rows, status `PD` (pending), `R` (running), or `CF` (configuring).

- [ ] **Step 5: Report job IDs to user**

Report the 3 JOBIDs in the assistant's next message, along with `squeue` output. No commit (per repo-state caveat in the header).

---

## Self-Review

**1. Spec coverage:**
- Loss change (clamp + 4-D pred_area + union broadcast) → Task 2 ✓
- Method registration in `run.py` → Task 3 Step 1 ✓
- Dispatch in `train.py` with `clamp_pred=(method==...)` → Task 3 Step 2 ✓
- Three sweep scripts (seeds 42/43/44, group `exp_iou_ord_ce_clamped`) → Task 4 ✓
- Adam, lr=5e-4, K=50, N=64 → Task 4 (verbatim from baseline) ✓
- Pre-submit verification (pytest + import + data path) → Task 5 ✓
- sbatch submission + squeue → Task 6 ✓
- No commits during this plan (per caveat) — handled by omitting `git commit` from every task.

**2. Placeholder scan:** no TBD/TODO; every code/script step shows the full content; all pytest/sbatch commands are exact.

**3. Type consistency:**
- `clamp_pred: bool = False` matches between signature (Task 2 Step 1), test calls (Task 1), and dispatch (Task 3 Step 2). ✓
- Method string `exp_iou_ord_ce_clamped` is identical across `SUPERVISED_METHODS` (Task 3 Step 1), dispatch (Task 3 Step 2), scripts (`--method`, Task 4), and wandb group (Task 4). ✓
- `pred_area` axis order: clamped path uses `(xc, yc, w, h)` and broadcasts as `pred_area[None]` to align with iou_m's `(B, xc, yc, w, h)`. Unclamped path uses `(w, h)` and broadcasts as `[None, None, None, :, :]`. ✓
