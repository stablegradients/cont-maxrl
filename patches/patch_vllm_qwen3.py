#!/usr/bin/env python3
"""Patch vLLM's qwen3.py to add missing sample()/get_sampler() methods.

Some vLLM installations (e.g., 0.8.3 source builds with a backported qwen3.py)
are missing the sample() method and sampler initialization that newer versions
include. This script patches the installed qwen3.py idempotently.

Usage:
    python patches/patch_vllm_qwen3.py
    python patches/patch_vllm_qwen3.py --check   # dry-run: just report status
    python patches/patch_vllm_qwen3.py --vllm-path /path/to/vllm  # explicit path
"""
import argparse
import re
import shutil
import sys
from pathlib import Path


def find_qwen3_py(vllm_path: str | None = None) -> Path:
    """Locate vllm's qwen3.py."""
    if vllm_path:
        p = Path(vllm_path) / "model_executor" / "models" / "qwen3.py"
        if p.exists():
            return p
        # Maybe they passed the models dir directly
        p = Path(vllm_path) / "qwen3.py"
        if p.exists():
            return p
        raise FileNotFoundError(f"qwen3.py not found at {vllm_path}")

    # Auto-detect from installed vllm
    try:
        import vllm
        vllm_dir = Path(vllm.__file__).parent
        p = vllm_dir / "model_executor" / "models" / "qwen3.py"
        if p.exists():
            return p
        raise FileNotFoundError(f"qwen3.py not found in {vllm_dir}")
    except ImportError:
        raise ImportError("vllm not installed and --vllm-path not provided")


def check_needs_patch(source: str) -> dict:
    """Check which patches are needed."""
    return {
        "import_get_sampler": "get_sampler" not in source,
        "sampler_init": "self.sampler = get_sampler()" not in source,
        "sample_method": "def sample(" not in source,
    }


def apply_patch(qwen3_path: Path, dry_run: bool = False) -> bool:
    """Patch qwen3.py. Returns True if changes were made."""
    source = qwen3_path.read_text()
    needs = check_needs_patch(source)

    if not any(needs.values()):
        print(f"OK: {qwen3_path} already has all required methods.")
        return False

    print(f"Patching: {qwen3_path}")
    for k, v in needs.items():
        print(f"  {'MISSING' if v else 'OK'}: {k}")

    if dry_run:
        print("  (dry run, no changes made)")
        return True

    # Backup
    backup = qwen3_path.with_suffix(".py.bak")
    if not backup.exists():
        shutil.copy2(qwen3_path, backup)
        print(f"  Backup: {backup}")

    patched = source

    # 1. Add get_sampler import if missing
    if needs["import_get_sampler"]:
        # Find the sampler import line and add get_sampler
        patched = patched.replace(
            "from vllm.model_executor.layers.sampler import SamplerOutput",
            "from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler",
        )
        # If SamplerOutput import doesn't exist either, add the full line
        if "get_sampler" not in patched:
            patched = patched.replace(
                "from vllm.model_executor.layers.logits_processor import LogitsProcessor",
                "from vllm.model_executor.layers.logits_processor import LogitsProcessor\n"
                "from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler",
            )
        # Also need SamplingMetadata import for the sample() signature
        if "SamplingMetadata" not in patched:
            patched = patched.replace(
                "from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler",
                "from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler\n"
                "from vllm.model_executor.sampling_metadata import SamplingMetadata",
            )

    # 2. Add self.sampler = get_sampler() in __init__ if missing
    if needs["sampler_init"]:
        # Insert after self.logits_processor = LogitsProcessor(...)
        patched = re.sub(
            r"(self\.logits_processor\s*=\s*LogitsProcessor\([^)]*\))",
            r"\1\n        self.sampler = get_sampler()",
            patched,
        )

    # 3. Add sample() method if missing
    if needs["sample_method"]:
        # Insert after compute_logits method
        sample_method = '''
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
'''
        # Find the end of compute_logits and insert after it
        # Look for the return statement in compute_logits
        patched = re.sub(
            r"(def compute_logits\(.*?\n(?:.*\n)*?        return logits\n)",
            r"\1" + sample_method,
            patched,
        )

    qwen3_path.write_text(patched)
    print("  Patched successfully.")

    # Verify
    verify = check_needs_patch(patched)
    if any(verify.values()):
        print("  WARNING: Some patches may not have applied correctly:")
        for k, v in verify.items():
            if v:
                print(f"    STILL MISSING: {k}")
        return True

    print("  Verified: all methods present.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Patch vLLM qwen3.py for sample() support")
    parser.add_argument("--check", action="store_true", help="Dry run: report status without modifying")
    parser.add_argument("--vllm-path", type=str, default=None, help="Path to vllm package directory")
    args = parser.parse_args()

    try:
        qwen3_path = find_qwen3_py(args.vllm_path)
    except (FileNotFoundError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found: {qwen3_path}")
    changed = apply_patch(qwen3_path, dry_run=args.check)
    sys.exit(0 if not args.check else (1 if changed else 0))


if __name__ == "__main__":
    main()
