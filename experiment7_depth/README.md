# Experiment 7

Dense prediction task with K-sweep ablation across supervised and RL methods.

## Setup

```bash
# Download data
python -m experiment7_depth.download_nyu --data_dir ./data/nyu

# Run tests
pytest experiment7_depth/tests/ -v --ignore=experiment7_depth/tests/test_ddp_correctness.py

# DDP tests (requires 2+ GPUs)
torchrun --standalone --nproc_per_node=2 -m pytest experiment7_depth/tests/test_ddp_correctness.py -v

# Generate run scripts (165 jobs)
python -m experiment7_depth.generate_run_scripts --output_dir experiment7_depth/scripts/runs

# Memory check (4-GPU DDP, worst-case cells)
bash experiment7_depth/scripts/memory_check.sh

# Sanity check (3 quick single-GPU runs)
bash experiment7_depth/scripts/sanity_check.sh

# Full sweep (165 runs, ~4-5 days on 8 GPUs)
bash experiment7_depth/scripts/submit_all.sh

# Plot results
python -m experiment7_depth.plot \
    --results_dir ./experiment7_depth/results \
    --output_dir ./experiment7_depth/figures \
    --manifest ./experiment7_depth/scripts/runs/manifest.csv
```
