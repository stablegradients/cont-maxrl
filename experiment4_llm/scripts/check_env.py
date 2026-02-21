"""
Environment verification script.
Checks that all dependencies are installed with correct versions and
that CUDA, GPU, and advantage functions work correctly.

Usage:
    python experiment4_llm/scripts/check_env.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys


def check():
    print("=== Environment Check ===\n")

    # 1. Python version
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}")
    assert v.major == 3 and v.minor == 10, \
        f"Expected Python 3.10, got {v.major}.{v.minor}"

    # 2. torch + CUDA
    import torch
    assert torch.cuda.is_available(), "CUDA not available to torch"
    gpu_name = torch.cuda.get_device_name(0)
    assert any(x in gpu_name for x in ["A5000", "A6000"]), \
        f"Expected Ampere GPU, got {gpu_name}"
    t = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
    assert t.dtype == torch.bfloat16, "bf16 not working"
    print(f"  torch: {torch.__version__}, GPU: {gpu_name}")

    # 3. flash-attn
    import flash_attn
    print(f"  flash-attn: {flash_attn.__version__}")

    # 4. vLLM
    import vllm
    print(f"  vllm: {vllm.__version__}")

    # 5. verl import
    import verl
    print(f"  verl: {verl.__version__}")

    # 6. Custom advantage estimators (directly integrated in ray_trainer.py)
    import experiment4_llm.src.verl_register as reg
    from verl.trainer.ppo.ray_trainer import AdvantageEstimator
    for name in ["maxrl_unnorm", "maxrl_norm", "reinforce_baseline"]:
        assert name in reg.CUSTOM_ESTIMATORS, f"{name} not in CUSTOM_ESTIMATORS!"
        assert hasattr(AdvantageEstimator, name.upper()), \
            f"AdvantageEstimator.{name.upper()} not found in ray_trainer.py!"
    print(f"  custom estimators: {list(reg.CUSTOM_ESTIMATORS.keys())}")
    print("  ray_trainer integration: OK")

    # 7. CUDA_HOME + PATH
    cuda_home = os.environ.get("CUDA_HOME", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if cuda_home and conda_prefix:
        assert cuda_home == conda_prefix, \
            f"CUDA_HOME ({cuda_home}) != CONDA_PREFIX ({conda_prefix})"
    path_entries = os.environ.get("PATH", "").split(":")
    conda_bin = os.path.join(conda_prefix, "bin") if conda_prefix else ""
    if conda_bin:
        cuda13_idx = [i for i, p in enumerate(path_entries) if "cuda-13" in p]
        conda_idx = [i for i, p in enumerate(path_entries) if p == conda_bin]
        if cuda13_idx and conda_idx:
            assert min(conda_idx) < min(cuda13_idx), \
                "System CUDA 13 before conda in PATH!"

    # 8. nvcc version (prefer conda env's nvcc over system)
    nvcc_candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        nvcc_candidates.append(os.path.join(conda_prefix, "bin", "nvcc"))
    # Also try the env that contains the running Python
    env_prefix = os.path.dirname(os.path.dirname(sys.executable))
    nvcc_candidates.append(os.path.join(env_prefix, "bin", "nvcc"))
    nvcc_candidates.append("nvcc")  # fallback to PATH

    nvcc_found = False
    for nvcc_path in nvcc_candidates:
        try:
            nvcc_out = subprocess.check_output([nvcc_path, "--version"], text=True)
            if "12.4" in nvcc_out:
                print(f"  nvcc: 12.4 OK ({nvcc_path})")
                nvcc_found = True
                break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    if not nvcc_found:
        print("  WARNING: nvcc 12.4 not found (system has 13.0, prebuilt flash-attn OK)")

    # 9. No /tmp in run scripts
    scripts_dir = os.path.join(
        os.path.dirname(__file__), "runs"
    )
    if os.path.isdir(scripts_dir):
        scripts = glob.glob(os.path.join(scripts_dir, "*.sh"))
        for s in scripts:
            with open(s) as f:
                for lineno, line in enumerate(f, 1):
                    # RAY_TMPDIR and TMPDIR use /tmp intentionally (short path for AF_UNIX)
                    if "/tmp" in line and "TMPDIR" not in line:
                        assert False, f"/tmp found in {s}:{lineno}: {line.strip()}"
        print(f"  run scripts: {len(scripts)} checked, no bad /tmp")
    else:
        print("  run scripts: directory not found (generate first)")

    # 10. Quick advantage function test
    from experiment4_llm.src.advantages import maxrl_unnormalized_advantage
    import torch
    rewards = torch.tensor([0.3, 0.9, 0.1, 0.6])
    adv = maxrl_unnormalized_advantage(rewards)
    expected = torch.tensor([-0.1333, 0.3167, -0.2000, 0.0167])
    assert torch.allclose(adv, expected, atol=1e-3), \
        f"Advantage mismatch: {adv}"
    print("  advantage self-test: OK")

    import ray
    print(f"  ray: {ray.__version__}")
    import transformers
    print(f"  transformers: {transformers.__version__}")

    print("\n  All checks passed!")


if __name__ == "__main__":
    check()
