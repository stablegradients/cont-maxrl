"""Launch post-eval v2 sweep across all runs, checkpoints, benchmarks."""
import os
import subprocess
import json
import time

REPO_ROOT = "/projects/bgmo/sramasubramanian/cont-maxrl"
JOB_SCRIPT = f"{REPO_ROOT}/experiment4_llm/scripts/post_evaluation/post_eval_v2_job.sh"
ACCOUNTS = ["bgmo-dtai-gh", "bgtu-dtai-gh", "bgts-dtai-gh", "bgtv-dtai-gh"]

BENCHMARKS = [
    ("humaneval_plus", "evalplus_humaneval_test.parquet"),
    ("mbpp_plus", "evalplus_mbpp_test.parquet"),
    ("livecodebench_v5", "livecodebench_v5_test.parquet"),
]

# Runs in scope (excluding latest checkpoint for training runs)
RUNS = [
    ("/work/nvme/bgmo/sramasubramanian/cont-maxrl/results/qwen3_1.7b_base_model_eval", [0]),
    ("/work/nvme/bgmo/sramasubramanian/cont-maxrl/results/qwen3_1.7b_base_deepcoder_binary_maxrl_g16_s0", [83, 166, 249, 332, 415]),
    ("/work/nvme/bgtu/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_grpo_g16_s0", [83, 166, 249, 332, 415, 498, 581]),
    ("/work/nvme/bgtu/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_maxrl_g16_s0", [83, 166, 249, 332, 415]),
    ("/work/nvme/bgts/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_grpo_g4_s0", [83, 166, 249, 332, 415, 498, 581, 664, 747, 830, 913, 996, 1000]),
    ("/work/nvme/bgts/sramasubramanian/cont-maxrl/results/qwen3_1.7b_deepcoder_maxrl_g4_s0", [83, 166, 249, 332, 415, 498, 581, 664, 747, 830, 913, 996, 1000]),
    ("/work/nvme/bgtv/sramasubramanian/cont-maxrl/results/qwen3_1.7b_base_deepcoder_rloo_g16_s0", [83, 166, 249, 332]),
]

jobs = []
for run_dir, steps in RUNS:
    run_name = os.path.basename(run_dir)
    for step in steps:
        for bench_name, data_file in BENCHMARKS:
            output_dir = os.path.join(run_dir, "post_eval_v2", bench_name, f"step_{step}")
            metrics_path = os.path.join(output_dir, "metrics.json")
            # Skip if already done
            if os.path.isfile(metrics_path):
                continue
            jobs.append({
                "run_dir": run_dir,
                "run_name": run_name,
                "step": step,
                "bench": bench_name,
                "data_file": data_file,
                "output_dir": output_dir,
            })

print(f"Total jobs to submit: {len(jobs)}")

# Round-robin across accounts
submitted = []
for i, job in enumerate(jobs):
    acct = ACCOUNTS[i % len(ACCOUNTS)]
    run_short = job["run_name"].replace("qwen3_1.7b_", "").replace("deepcoder_", "")[:20]
    job_name = f"v2_{run_short}_s{job['step']}_{job['bench'][:4]}"
    
    slurm_log_dir = os.path.join(job["run_dir"], "post_eval_v2", "slurm_logs")
    os.makedirs(slurm_log_dir, exist_ok=True)
    
    cmd = [
        "sbatch",
        f"--account={acct}",
        "--partition=ghx4-interactive",
        "--time=02:00:00",
        "--gpus-per-node=4",
        "--nodes=1",
        "--mem=0",
        "--cpus-per-task=288",
        f"--job-name={job_name}",
        f"--output={slurm_log_dir}/{job['bench']}_step{job['step']}_%j.out",
        f"--error={slurm_log_dir}/{job['bench']}_step{job['step']}_%j.err",
        JOB_SCRIPT,
        job["run_dir"],
        str(job["step"]),
        job["bench"],
        job["data_file"],
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode == 0:
        jid = result.stdout.strip().split()[-1]
        submitted.append((jid, job_name, acct))
        print(f"  [{acct}] {job_name} → {jid}")
    else:
        print(f"  FAILED: {job_name} — {result.stderr.strip()}")
    
    # Small delay to avoid hammering scheduler
    if (i + 1) % 20 == 0:
        time.sleep(1)

print(f"\nSubmitted {len(submitted)} jobs.")

# Save job manifest
manifest_path = os.path.join(REPO_ROOT, "post_eval_v2_manifest.json")
manifest = [{"job_id": jid, "job_name": name, "account": acct} for jid, name, acct in submitted]
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved to {manifest_path}")
