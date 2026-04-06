"""Create the custom Qwen2 ~3.94M model and 32-token word-level tokenizer.

Matches the architecture from the MaxRL paper (Tajwar et al., arXiv 2602.02710):
  - Qwen2ForCausalLM, hidden=256, 4 layers, 4 heads, 2 KV heads
  - intermediate_size=1024, vocab_size=32, ~3.94M params
  - WordLevel tokenizer with 18 domain tokens + 14 reserved

Usage:
    PYTHONPATH=. python -m experiment6_maze.src.create_model \
        --output_dir checkpoints/maze17x17_qwen2_3m
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from tokenizers import Tokenizer, models as tok_models, pre_tokenizers, AddedToken
from transformers import AutoModelForCausalLM, Qwen2Config

# 18 meaningful tokens + 14 reserved = 32 total
MAZE_VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "GRID_START": 4,
    "GRID_END": 5,
    "PATH_START": 6,
    "DONE": 7,
    "PATH": 8,
    "WALL": 9,
    "GOAL": 10,
    "START": 11,
    "NEWLINE": 12,
    "UP": 13,
    "DOWN": 14,
    "LEFT": 15,
    "RIGHT": 16,
    "\n": 17,
}

NUM_RESERVED = 14
VOCAB_SIZE = len(MAZE_VOCAB) + NUM_RESERVED  # 32


def create_tokenizer(output_dir: str) -> None:
    """Create and save the 32-token word-level tokenizer."""
    reserved = {f"RESERVED_{i}": len(MAZE_VOCAB) + i for i in range(NUM_RESERVED)}
    full_vocab = {**MAZE_VOCAB, **reserved}

    tok = Tokenizer(tok_models.WordLevel(vocab=full_vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    for t in ["<pad>", "<bos>", "<eos>", "<unk>"]:
        tok.add_special_tokens([AddedToken(t, special=True)])

    tok.save(os.path.join(output_dir, "tokenizer.json"))

    tokenizer_config = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 512,
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    generation_config = {
        "bos_token_id": 1,
        "eos_token_id": [2, 7],  # <eos> and DONE both stop generation
        "pad_token_id": 0,
        "max_new_tokens": 128,
        "do_sample": True,
        "temperature": 1.0,
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)


def create_model(output_dir: str) -> None:
    """Create and save the Qwen2 ~3.94M parameter model."""
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=1024,
        max_position_embeddings=512,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=True,
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.save_pretrained(output_dir)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {num_params:,} params, vocab_size={VOCAB_SIZE}")
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create maze Qwen2 model + tokenizer")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/maze17x17_qwen2_3m_init",
        help="Output directory for model + tokenizer",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    create_model(args.output_dir)
    create_tokenizer(args.output_dir)

    # Verify
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)
    test = "<bos> GRID_START WALL PATH START NEWLINE GRID_END PATH_START"
    ids = tok.encode(test, add_special_tokens=False)
    print(f"Tokenizer test: {test!r}")
    print(f"  Token IDs: {ids}")
    print(f"  Decoded: {tok.decode(ids)}")

    msgs = [{"role": "user", "content": test}]
    chat_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print(f"  Chat template IDs: {chat_ids}")


if __name__ == "__main__":
    main()
