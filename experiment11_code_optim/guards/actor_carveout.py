"""
Per-epoch actor carve-out (spec v9 §8.3).

verl's training loop keeps `max_actor_ckpt_to_keep` checkpoints in its
rolling buffer (set to 1 in the PIE SLURM scripts to bound disk use).
That preserves resume integrity but loses per-epoch trajectory — the
post-hoc eval pipeline (best@K at K∈{1,4,16,64,256}, §8.5) needs every
saved step preserved.

This module installs a save-hook by monkey-patching verl's
`FSDPCheckpointManager.save_checkpoint`. After each verl save, rank 0
HARD-LINKS the saved files into `$PIE_ACTOR_CARVEOUT_DIR/step_<N>/`.
Hard-links are inode-level: zero extra disk space, instant, and they
survive verl's later `remove_previous_save_local_path` (the parent
inode stays alive while ANY link references it).

Call `install_actor_carveout_hook()` at training entry, AFTER importing
verl. `experiment11_code_optim/train.py` does this when
`PIE_ACTOR_CARVEOUT_DIR` is set.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


_INSTALLED = False


def _hardlink_tree(src: str, dst: str) -> None:
    """Mirror `src` to `dst` using os.link() on files (zero-copy, same FS
    only) and os.makedirs on directories. Falls back to copytree if the
    destination filesystem doesn't support hardlinks."""
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)

    try:
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            tgt_root = os.path.join(dst, rel) if rel != "." else dst
            os.makedirs(tgt_root, exist_ok=True)
            for fn in files:
                src_p = os.path.join(root, fn)
                tgt_p = os.path.join(tgt_root, fn)
                os.link(src_p, tgt_p)
    except OSError as e:
        # Cross-filesystem or permission issue — fall back to copy.
        print(
            f"[carveout] hardlink failed ({e}); falling back to copytree",
            file=sys.stderr,
        )
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)


def install_actor_carveout_hook() -> bool:
    """Install the save-hook. Returns True if installed, False if env not
    set or already installed. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return False

    carveout_dir = os.environ.get("PIE_ACTOR_CARVEOUT_DIR")
    if not carveout_dir:
        return False

    try:
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
    except ImportError as e:
        print(
            f"[carveout] WARN: could not import FSDPCheckpointManager ({e}); "
            f"hook NOT installed",
            file=sys.stderr,
        )
        return False

    os.makedirs(carveout_dir, exist_ok=True)
    original_save = FSDPCheckpointManager.save_checkpoint

    def save_with_carveout(self, local_path, hdfs_path=None, global_step=0,
                           max_ckpt_to_keep=None):
        original_save(
            self,
            local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        # Only rank 0 mirrors — the saved files are on a shared filesystem
        # so rank 0 sees every rank's shard.
        if getattr(self, "rank", 0) != 0:
            return
        if local_path is None:
            return
        target = os.path.join(carveout_dir, f"step_{global_step}")
        try:
            _hardlink_tree(local_path, target)
            print(
                f"[carveout] step={global_step} -> {target} (hardlinked)",
                file=sys.stderr,
            )
            # Write a sentinel so the post-hoc eval can tell which carve-outs
            # are complete (vs. mid-write at process crash).
            Path(target, ".carveout_complete").touch()
        except (OSError, RuntimeError) as e:
            print(
                f"[carveout] WARN: failed to mirror step={global_step}: {e}",
                file=sys.stderr,
            )

    FSDPCheckpointManager.save_checkpoint = save_with_carveout
    _INSTALLED = True
    print(
        f"[carveout] installed: per-save mirror to {carveout_dir}/step_<N>/",
        file=sys.stderr,
    )
    return True
