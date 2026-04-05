"""SFT training for SmolLM2-135M on maze navigation data."""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def tokenize_fn(examples, tokenizer, max_length=896):
    texts = [
        p + " " + c + tokenizer.eos_token
        for p, c in zip(examples["prompt"], examples["completion"])
    ]
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--train_file", default="data/maze_17x17/sft_train.jsonl")
    parser.add_argument("--val_file", default="data/maze_17x17/sft_val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/smollm2_135m_maze17x17_sft")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=896)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    train_data = load_jsonl(args.train_file)
    val_data = load_jsonl(args.val_file)
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        batched=True, remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        batched=True, remove_columns=val_ds.column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=3,
        bf16=True,
        report_to="wandb",
        run_name="smollm2_135m_maze17x17_sft",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"SFT model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
