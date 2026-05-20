"""
PIE cold-start SFT (spec v9 §6).

Trains Qwen3-1.7B-Base briefly on PIE (src_code → tgt_code) pairs from
train_hq_only.jsonl so the §9 GO/NO-GO gate has a fighting chance — base
Qwen3-1.7B in zero-shot rarely emits a parseable ```cpp ... ``` block in
the §11 prompt template, and RL needs ≥20% of problems with ≥8/16 passing
to start.

Output: an FSDP full-state HuggingFace checkpoint at
`${PIE_SFT_CKPT_DIR}/final/` loadable directly by vLLM (the §11 RL launch
uses this path as `actor_rollout_ref.model.path`).

The prompt template is mirrored exactly in the RL rollout prompt assembly
(§11 prompt parity) so SFT and RL share the same conditioning. The
"### Optimized Version:" anchor lines up with `pie_reward_fn.extract_code`
so completions are parsed identically in SFT validation and RL gating.

Usage:
    PIE_SFT_CKPT_DIR=/work/nvme/.../sft \\
    python experiment11_code_optim/sft/cold_start_sft.py \\
        --train_jsonl /work/nvme/.../pie_data/raw/train_hq_only.jsonl \\
        --base_model Qwen/Qwen3-1.7B-Base \\
        --output_dir $PIE_SFT_CKPT_DIR \\
        --num_train_epochs 1 \\
        --per_device_train_batch_size 4 \\
        --gradient_accumulation_steps 16 \\
        --learning_rate 1e-5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# Prompt template — must match `experiment4_llm/src/insitu_eval.py`'s
# prompt assembly when PIE_MODE=1, AND must end with the
# "### Optimized Version:" anchor that pie_reward_fn.extract_code anchors on.
PROMPT_TEMPLATE = (
    "You are a C++ optimization expert. Below is a working C++ program. "
    "Rewrite it to run faster while producing identical output for all valid "
    "inputs. Keep the same input/output format and only change the algorithm/"
    "implementation.\n\n"
    "### Slow Version:\n"
    "```cpp\n{src_code}\n```\n\n"
    "### Optimized Version:\n"
)
COMPLETION_TEMPLATE = "```cpp\n{tgt_code}\n```"


def build_dataset(jsonl_path: str, tokenizer, max_length: int) -> Dataset:
    """Read (src_code, tgt_code) pairs and tokenize prompt+completion.

    Loss is computed only on the completion span — the prompt's loss is
    masked out via labels=-100. This is the standard "instruction-tuning"
    loss mask; without it the model wastes capacity learning to predict
    its own input (the slow program), which is the wrong objective.
    """
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("src_code")
            tgt = rec.get("tgt_code")
            if not src or not tgt:
                continue
            samples.append({"src_code": src, "tgt_code": tgt})

    def tokenize(ex):
        prompt = PROMPT_TEMPLATE.format(src_code=ex["src_code"])
        completion = COMPLETION_TEMPLATE.format(tgt_code=ex["tgt_code"]) + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        if len(ids) > max_length:
            # Right-truncate; this drops the END of the completion, which is
            # less harmful than dropping the prompt (the model still sees the
            # task spec).
            ids = ids[:max_length]
            labels = labels[:max_length]
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    ds = Dataset.from_list(samples)
    ds = ds.map(tokenize, remove_columns=["src_code", "tgt_code"], desc="tokenize")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_steps", type=int, default=0,
                    help="0 = save only at end (recommended for SFT cold-start)")
    args = ap.parse_args()

    print(f"[sft] base_model={args.base_model}")
    print(f"[sft] train_jsonl={args.train_jsonl}")
    print(f"[sft] output_dir={args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft] Building dataset (max_length={args.max_length})...")
    ds = build_dataset(args.train_jsonl, tokenizer, args.max_length)
    print(f"[sft] {len(ds)} training examples after tokenization")

    print(f"[sft] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()

    # FSDP full-state save: vLLM loads from a standard HF dir, so we set
    # save_strategy="epoch" with save_safetensors=True and full_state_dict
    # via accelerate's FSDP wrapper (configured by torchrun env in the SLURM
    # script). The final consolidated checkpoint is at output_dir/final/.
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        logging_steps=10,
        save_strategy="no" if args.save_steps == 0 else "steps",
        save_steps=args.save_steps if args.save_steps > 0 else 100,
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else ["none"],
        run_name=f"pie_sft_seed{args.seed}",
        gradient_checkpointing=True,
        # FSDP: configured via torchrun env in SLURM (accelerate config or
        # --fsdp / --fsdp_config args). Trainer respects FSDP_AUTO_WRAP_POLICY,
        # FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP via env.
        fsdp="full_shard auto_wrap",
        fsdp_transformer_layer_cls_to_wrap="Qwen3DecoderLayer",
    )

    # DataCollatorForSeq2Seq pads variable-length sequences: input_ids with
    # pad_token_id, attention_mask with 0, labels with -100 (ignore index).
    # DataCollatorForLanguageModeling assumes uniform-length input — wrong
    # for instruction-tuned SFT with masked-prompt labels.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100, return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print(f"[sft] Starting training...")
    trainer.train()

    # Consolidated full-state save (loadable by vLLM directly).
    final_dir = Path(args.output_dir) / "final"
    print(f"[sft] Saving consolidated checkpoint to {final_dir}...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[sft] Done. Wire as PIE_SFT_CKPT={final_dir} for the RL launch.")


if __name__ == "__main__":
    main()
