#!/usr/bin/env bash
# MICRO real-verl plumbing smoke for the full-batch gem5 config: 1 GPU, B=2,
# G=4, 2 steps, save_freq=1 + carveout_every=1, eval DISABLED. Confirms the
# gem5 reward runs INSIDE verl's batch loop, ckpt+carveout write, and a fresh
# start works. Run on a ghx4 node (e.g. via srun --overlap on an interactive job).
set +u                       # conda/profile init trips set -u under `bash -lc`
REPO_ROOT="/projects/bgmo/sramasubramanian/cont-maxrl"
PYTHON="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl/bin/python"
module load python/miniforge3_pytorch/2.10.0 >/dev/null 2>&1
eval "$(conda shell.bash hook)" >/dev/null 2>&1
conda activate /work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl >/dev/null 2>&1
if [ -f "${REPO_ROOT}/.env" ]; then set -a; source "${REPO_ROOT}/.env"; set +a; fi
export PYTHONNOUSERSITE=1

OUT="/work/nvme/bgtv/sramasubramanian/cont-maxrl/results/_SMOKE_fb_gem5"
rm -rf "${OUT}"; mkdir -p "${OUT}/actor_ckpts"
export PIE_ACTOR_CARVEOUT_DIR="${OUT}/actor_ckpts"; export PIE_CARVEOUT_EVERY=1

export CUDA_HOME="/work/nvme/bgmo/sramasubramanian/miniforge3/envs/cont-maxrl"
export PATH="${CUDA_HOME}/bin:${PATH}"; export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
unset NCCL_NET || true; unset NCCL_NET_PLUGIN || true; export NCCL_PLUGIN_P2P=0
export WANDB_MODE=offline
export HF_HOME="${REPO_ROOT}/.cache/huggingface"; export TORCH_HOME="${REPO_ROOT}/.cache/torch"
export VLLM_CACHE_ROOT="${REPO_ROOT}/.cache/vllm"

export PIE_MODE=1 PIE_REWARD_IMPL=gem5 PIE_GEM5_REWARD_KIND=ratio
export PIE_GEM5_REWARD_K=1 PIE_GEM5_CASE_SELECT=sample PIE_GEM5_REWARD_SAMPLE=group
export PIE_GEM5_REWARD_SEED=0 PIE_GEM5_BUDGET_MULT=3.0 PIE_GEM5_REWARD_WORKERS=48
export PIE_NAT_TIMEOUT=10 PIE_GEM5_REWARD_DEADLINE=1800
export PIE_TEST_CASE_DIR="/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"
export TMPDIR="/tmp/${USER}_smokeverl_$$"; export PIE_COMPILE_SCRATCH="${TMPDIR}/scratch"
export RAY_TMPDIR="/tmp/ray_${USER}_smv_$$"; mkdir -p "${TMPDIR}" "${PIE_COMPILE_SCRATCH}" "${RAY_TMPDIR}"
export VLLM_USE_V1=0 RAY_OBJECT_STORE_MEMORY=12000000000
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
TRAIN="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet/pie_gem5_train.parquet"

echo "=== MICRO verl smoke: B=2 G=4 steps=2 save_freq=1 carveout=1 eval=[] gem5 K=1 sample ==="
$PYTHON "${REPO_ROOT}/experiment11_code_optim/train.py" \
  +data.seed=0 algorithm.adv_estimator=grpo \
  algorithm.filter_overlong_responses=False algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.type=fixed algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${TRAIN}" data.val_files="${TRAIN}" \
  data.train_batch_size=2 data.max_prompt_length=2560 data.max_response_length=4096 \
  data.shuffle=True data.filter_overlong_prompts=True data.enable_thinking=False \
  data.reward_fn_key=data_source \
  actor_rollout_ref.model.path="Qwen/Qwen3-1.7B" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.strategy=fsdp actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 actor_rollout_ref.actor.clip_ratio=100 \
  actor_rollout_ref.actor.clip_ratio_high=100 actor_rollout_ref.actor.grad_clip=2.0 \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.max_model_len=6656 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward_model.enable=False reward_model.launch_reward_fn_async=True \
  custom_reward_function.path="${REPO_ROOT}/experiment11_code_optim/reward/gem5_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.total_training_steps=2 trainer.total_epochs=11 \
  trainer.project_name=cont-maxrl-pie trainer.experiment_name="_smoke_fb" \
  "trainer.logger=['console']" trainer.default_local_dir="${OUT}" \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=1 \
  "+actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  trainer.test_freq=-1 trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep=1 \
  trainer.val_before_train=False \
  "insitu_eval.eval_steps=[]" insitu_eval.data_file="${TRAIN}" \
  insitu_eval.output_dir="${OUT}/insitu_eval" insitu_eval.label="disabled" \
  ray_init.num_cpus=60
RC=$?
echo "=== verl exit=${RC} ==="
echo "--- ckpts written ---"; ls -1 "${OUT}" 2>/dev/null; echo "--- carveout snapshots ---"; ls -1 "${OUT}/actor_ckpts" 2>/dev/null
echo "--- latest_checkpointed_iteration ---"; cat "${OUT}/latest_checkpointed_iteration.txt" 2>/dev/null
exit ${RC}
