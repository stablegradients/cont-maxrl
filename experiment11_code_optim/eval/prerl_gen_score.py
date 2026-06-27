"""Generate N rollouts for the frozen 32-problem set under ONE (model, prompt,
think-mode) config, gem5-score every rollout, decompose each output into
think / visible-reasoning / answer token lengths, flag truncation + repetition +
source-copy, and store everything. One JSON per config: gen_<config_name>.json.

No system prompt; each model's own chat template; the thinking model keeps its
native <think> block (auto-injected by the template, never prefilled/forced).

The gem5 reward config is read from env (PIE_REWARD_IMPL=gem5 + PIE_GEM5_*),
identical to the n=1024 best_reward@k eval, so the stored speedups are directly
comparable to those headline numbers.

  python -m experiment11_code_optim.eval.prerl_gen_score \
    --config_name inst2507_P1 --model Qwen/Qwen3-4B-Instruct-2507 \
    --prompt P1 --think_mode off --problems_json .../problems32.json \
    --n 32 --max_tokens 16384 --out_dir .../prerl_analysis
"""
from __future__ import annotations
import argparse, difflib, json, os, re, sys, time
from pathlib import Path

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
from experiment11_code_optim.eval.post_hoc_eval import score_completions, best_at_k  # noqa: E402

# --- the three prompts; {src} is the slow program. Identical body across models. ---
PROMPTS = {
    "P1": ("You are given a working C++ program. Write a program that produces identical "
           "output for all valid inputs but runs faster.\n\n### Slow Version:\n```cpp\n{src}\n```\n\n"
           "Give your final program under a line that reads exactly `### Optimized Version:`, "
           "as a single ```cpp code block."),
    "P2": ("You are a C++ optimization expert. Below is a working C++ program; rewrite it to run "
           "faster while producing identical output for every valid input. Correctness comes first "
           "— a program that fails any test is worthless. Returning the original unchanged gives "
           "no speedup, so look for a lower-complexity algorithm or a better data structure, and keep "
           "only the work that's necessary for correctness. If after genuine effort no faster version "
           "exists, return the original.\n\n### Slow Version:\n```cpp\n{src}\n```\n\n"
           "Give your final program under a line that reads exactly `### Optimized Version:`, "
           "as a single ```cpp code block."),
    "P3": ("You are a C++ optimization expert. Below is a working C++ program. Think step by step "
           "about how to make it run faster — its time complexity, the main bottlenecks, the data "
           "structures and I/O — and then write the optimized program. It must produce identical "
           "output for every valid input.\n\n### Slow Version:\n```cpp\n{src}\n```\n\n"
           "After your reasoning, give your final program under a line that reads exactly "
           "`### Optimized Version:`, as a single ```cpp code block."),
}
ANCHOR = "### Optimized Version:"


def split_parts(text: str):
    """-> (think, reasoning, answer) substrings.
    think     = native <think> block content (empty for instruct).
    reasoning = visible prose before the LAST anchor.
    answer    = anchor + everything after (the fenced program).
    The thinking model's template injects the opening <think> into the PROMPT, so
    generated text is 'reasoning... </think> ...### Optimized Version: ```cpp```'."""
    t = text
    think = ""
    if "</think>" in t:
        think, t = t.split("</think>", 1)
        think = think.split("<think>", 1)[-1]
    elif t.lstrip().startswith("<think>"):
        return t.split("<think>", 1)[-1], "", ""      # opened, never closed (truncated)
    if ANCHOR in t:
        i = t.rfind(ANCHOR)
        return think, t[:i], t[i:]
    return think, t, ""


def norm_code(c: str | None) -> str:
    if not c:
        return ""
    c = re.sub(r"//[^\n]*", "", c)
    c = re.sub(r"/\*.*?\*/", "", c, flags=re.S)
    return re.sub(r"\s+", "", c)


def rep_ratio(text: str, n: int = 4) -> float:
    """word-level n-gram repetition: 1 - distinct/total (0 = none, ->1 = heavy loop)."""
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_name", required=True)        # e.g. inst2507_P1
    ap.add_argument("--model", required=True)              # HF id or local path
    ap.add_argument("--prompt", required=True, choices=list(PROMPTS))
    ap.add_argument("--think_mode", default="off", choices=["off", "native", "on"],
                    help="off=plain instruct (no enable_thinking); native=thinking-2507 "
                         "template auto-injects <think>; on=hybrid enable_thinking=True")
    ap.add_argument("--problems_json", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=32768)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_num_seqs", type=int, default=0,
                    help="cap concurrent vLLM sequences (0=default); set for long-context "
                         "(32k) runs so 1024 samples don't oversubscribe KV cache and thrash")
    ap.add_argument("--limit", type=int, default=0, help="only first K problems (smoke test)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_case_dir", default=os.environ.get(
        "PIE_TEST_CASE_DIR", "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"))
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    probs = json.loads(Path(a.problems_json).read_text())
    if a.limit:
        probs = probs[:a.limit]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    def render(src: str) -> str:
        body = PROMPTS[a.prompt].format(src=src)
        msgs = [{"role": "user", "content": body}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if a.think_mode == "on":
            kw["enable_thinking"] = True
        elif a.think_mode == "off":
            try:
                return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
            except TypeError:
                pass
        return tok.apply_chat_template(msgs, **kw)

    user_bodies = [PROMPTS[a.prompt].format(src=p["src_code"]) for p in probs]
    prompts = [render(p["src_code"]) for p in probs]
    gts = [p["ground_truth"] for p in probs]
    n_prompt_tok = [len(tok(pr, add_special_tokens=False)["input_ids"]) for pr in prompts]
    max_prompt = max(n_prompt_tok)
    print(f"[{a.config_name}] {len(prompts)} prompts x n={a.n} | think={a.think_mode} | "
          f"max_prompt_tok={max_prompt} | max_tokens={a.max_tokens} | model={a.model}", flush=True)

    # ---- generate (own vLLM loop so we capture finish_reason + token counts) ----
    from vllm import LLM, SamplingParams
    _llm_kw = dict(model=a.model, tokenizer=a.model, trust_remote_code=True,
                   tensor_parallel_size=a.tensor_parallel_size, gpu_memory_utilization=0.95,
                   max_model_len=max_prompt + a.max_tokens + 256, enforce_eager=True)
    if a.max_num_seqs > 0:
        _llm_kw["max_num_seqs"] = a.max_num_seqs
    llm = LLM(**_llm_kw)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens)
    t0 = time.monotonic()
    outs = llm.generate(prompts, sp)
    print(f"[{a.config_name}] gen {time.monotonic()-t0:.0f}s", flush=True)

    texts = [[o.text for o in r.outputs] for r in outs]
    finish = [[o.finish_reason for o in r.outputs] for r in outs]
    ntok = [[len(o.token_ids) for o in r.outputs] for r in outs]

    # ---- gem5 score (same env-driven reward as the eval) ----
    t0 = time.monotonic()
    per = score_completions(texts, gts, a.test_case_dir, reference_table_dir=None)
    print(f"[{a.config_name}] gem5 score {time.monotonic()-t0:.0f}s", flush=True)

    from experiment11_code_optim.reward.gem5_reward import extract_code  # noqa: E402

    recs = []
    for pi, p in enumerate(probs):
        src_norm = norm_code(p["src_code"])
        rollouts = []
        for j in range(len(texts[pi])):
            txt = texts[pi][j]; r = per[pi][j]
            think, reason, answer = split_parts(txt)
            code = extract_code(txt)
            cn = norm_code(code)
            sim = difflib.SequenceMatcher(None, cn, src_norm).ratio() if (cn and src_norm) else 0.0
            rollouts.append({
                "i": j,
                "score": float(r.get("score", 0.0)), "acc": float(r.get("acc", 0.0)),
                "speedup": float(r.get("speedup", 0.0)), "reason": r.get("reason") or "",
                "finish": finish[pi][j], "truncated": finish[pi][j] == "length",
                "n_tok_total": ntok[pi][j],
                "n_tok_think": len(tok(think, add_special_tokens=False)["input_ids"]) if think.strip() else 0,
                "n_tok_reason": len(tok(reason, add_special_tokens=False)["input_ids"]) if reason.strip() else 0,
                "n_tok_answer": len(tok(answer, add_special_tokens=False)["input_ids"]) if answer.strip() else 0,
                "think_empty": len(think.strip()) < 3,
                "rep_ratio": round(rep_ratio(txt), 4),
                "sim_to_src": round(sim, 4),
                "exact_copy": bool(cn and cn == src_norm),
                "has_code": bool(code),
                "code": code, "text": txt,
            })
        sc = [x["score"] for x in rollouts]; ac = [x["acc"] for x in rollouts]
        rec = {**{k: p[k] for k in ("problem_id", "src_id", "ceiling_speedup",
                                    "our_speedup_usable", "our_fastest_speedup", "n_usable")},
               "config_name": a.config_name, "model": a.model, "prompt": a.prompt,
               "think_mode": a.think_mode,
               "user_prompt": user_bodies[pi], "prompt_rendered": prompts[pi],
               "n": len(sc),
               "n_pass": int(sum(x == 1.0 for x in ac)),
               "best_at_n": best_at_k(sc, len(sc)) if sc else 0.0,
               "mean_score": (sum(sc) / len(sc)) if sc else 0.0,
               "max_speedup": max((x["speedup"] for x in rollouts), default=0.0),
               "max_correct_speedup": max((x["speedup"] for x in rollouts if x["acc"] == 1.0), default=0.0),
               "rollouts": rollouts}
        recs.append(rec)
        print(f"  {p['problem_id']}/{p['src_id']}: pass {rec['n_pass']}/{rec['n']} "
              f"best@{rec['n']}={rec['best_at_n']:.2f}x ceil={p['ceiling_speedup']:.1f}x", flush=True)

    pth = out / f"gen_{a.config_name}.json"
    pth.write_text(json.dumps(recs, indent=2))  # pretty-printed for text-editor viewing
    print(f"[{a.config_name}] wrote {pth} ({pth.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
