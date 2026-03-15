"""
Smoke test: vLLM Qwen3-4B generation with thinking mode + end-to-end reward.
"""
import json
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiment4_llm.src.reward_function import compute_score, extract_code


def main():
    print("=" * 70)
    print("TEST SUITE 5: vLLM Qwen3-4B Generation + End-to-End Reward")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

    t0 = time.time()
    llm = LLM(
        "Qwen/Qwen3-4B",
        trust_remote_code=True,
        gpu_memory_utilization=0.5,
        dtype="bfloat16",
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    problems = [
        {
            "prompt": "Write a Python function that checks if a number is prime. Read an integer from stdin and print True or False.",
            "gt": json.dumps({
                "inputs": ["2\n", "3\n", "4\n", "17\n", "1\n"],
                "outputs": ["True\n", "True\n", "False\n", "True\n", "False\n"],
                "fn_name": None,
            }),
            "difficulty": "easy",
        },
        {
            "prompt": "Write a Python function that finds the longest common subsequence length of two strings. Read two strings from stdin, one per line, and print the LCS length.",
            "gt": json.dumps({
                "inputs": ["abcde\nace\n", "abc\nabc\n", "abc\ndef\n"],
                "outputs": ["3\n", "3\n", "0\n"],
                "fn_name": None,
            }),
            "difficulty": "medium",
        },
        {
            "prompt": "Write a Python function to solve: given n and a list of n integers, find the maximum subarray sum (Kadane's algorithm). Read n from first line, then the array. Print the answer.",
            "gt": json.dumps({
                "inputs": ["5\n-2 1 -3 4 -1\n", "1\n-5\n", "3\n1 2 3\n"],
                "outputs": ["4\n", "-5\n", "6\n"],
                "fn_name": None,
            }),
            "difficulty": "medium",
        },
    ]

    sp = SamplingParams(n=4, temperature=0.6, top_p=0.95, max_tokens=4096)

    total_think_lens = []
    total_code_lens = []
    total_lens = []

    for pi, prob in enumerate(problems):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prob["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        t0 = time.time()
        out = llm.generate([prompt], sp)[0]
        gen_time = time.time() - t0

        print(f'Problem {pi + 1} [{prob["difficulty"]}]: {prob["prompt"][:55]}...')
        print(f"  Generation: {gen_time:.1f}s, {len(out.outputs)} samples")

        for i, o in enumerate(out.outputs):
            text = o.text
            think_end = text.find("</think>")
            total_ids = tok.encode(text, add_special_tokens=False)
            total_len = len(total_ids)

            if think_end != -1:
                think_text = text[:think_end]
                think_len = len(tok.encode(think_text, add_special_tokens=False))
                code_len = total_len - think_len
            else:
                think_len = 0
                code_len = total_len

            total_think_lens.append(think_len)
            total_code_lens.append(code_len)
            total_lens.append(total_len)

            # Test extract_code
            extracted = extract_code(text)
            has_code = len(extracted.strip()) > 0

            # Test end-to-end reward with special tokens (simulating skip_special_tokens=False)
            text_with_special = text + "<|im_end|>"
            reward = compute_score(text_with_special, prob["gt"], timeout_per_test=2)

            think_pct = think_len / max(total_len, 1) * 100
            print(
                f"  Sample {i}: total={total_len:4d} tok | "
                f"think={think_len:4d} ({think_pct:2.0f}%) | "
                f"code={code_len:3d} | "
                f"score={reward['score']:.2f} acc={reward['acc']:.0f} | "
                f"extract_ok={has_code}"
            )

    print()
    print("=" * 70)
    print("RESPONSE LENGTH SUMMARY (thinking enabled)")
    print("=" * 70)
    n = len(total_lens)
    print(f"  Samples: {n}")
    print(f"  Total tokens:    min={min(total_lens):4d}  max={max(total_lens):4d}  mean={sum(total_lens) / n:.0f}")
    print(f"  Thinking tokens: min={min(total_think_lens):4d}  max={max(total_think_lens):4d}  mean={sum(total_think_lens) / n:.0f}")
    print(f"  Code tokens:     min={min(total_code_lens):4d}  max={max(total_code_lens):4d}  mean={sum(total_code_lens) / n:.0f}")
    print(f"  Think/Total:     {sum(total_think_lens) / sum(total_lens) * 100:.1f}%")
    print()

    truncated = sum(1 for t in total_lens if t >= 4090)
    print(f"  Truncated (>=4090 tokens): {truncated}/{n}")
    if truncated > 0:
        print("  WARNING: some responses hit max_tokens limit")

    print()
    print("vLLM generation test PASSED")


if __name__ == "__main__":
    main()
