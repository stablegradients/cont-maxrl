"""Generate + gem5-score N rollouts for a FEW hand-picked (problem,src) test
prompts under ONE actor snapshot, and store every rollout (text + extracted code
+ speedup + correctness + failure reason). Run once per arm (grpo/gos/rloo); the
companion aggregator (compare_rollouts_aggregate.py) diffs the arms.

Same generate+score path and gem5 reward config as the n=1024 best_reward@k eval
(case_select=biggest, K=1, ratio reward, correctness on ALL usable cases) so the
stored speedups are directly comparable to the headline numbers.

Usage (driven by run_rollout_compare.sh):
  python -m experiment11_code_optim.eval.compare_rollouts \
    --arm gos --step 300 --hf_dir .../step_300/huggingface \
    --val_parquet pie_gem5_test.parquet \
    --problems p03275:s242733854,p03161:s106426206,p03730:s784784078 \
    --n 32 --out_dir .../rollout_compare
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
import pandas as pd

sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
from experiment11_code_optim.eval.post_hoc_eval import (   # noqa: E402
    generate_completions, score_completions, best_at_k,
)


def _get_extract():
    """Code extractor that MATCHES the reward (fall back to a fenced-block regex)."""
    try:
        from experiment11_code_optim.reward.gem5_reward import extract_code
        if callable(extract_code):
            return extract_code
    except Exception:
        pass
    fence = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    def _fb(text: str):
        m = list(fence.finditer(text or ""))
        return m[-1].group(1).strip() if m else None
    return _fb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--problems", required=True,
                    help="comma list of problem_id:src_id (the exact bysrc test prompts)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_case_dir",
                    default=os.environ.get("PIE_TEST_CASE_DIR",
                        "/work/nvme/bgtv/sramasubramanian/pie_merged_tests/merged_test_cases"))
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sel = [tuple(x.split(":")) for x in args.problems.split(",") if x]
    df = pd.read_parquet(args.val_parquet).reset_index(drop=True)

    # index the bysrc test rows by (problem_id, src_id)
    by_key = {}
    for _, row in df.iterrows():
        gt = json.loads(row["reward_model"]["ground_truth"])
        by_key[(gt["problem_id"], gt["src_id"])] = (row, gt)

    prompts, gts, meta = [], [], []
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    def render(c):
        return tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    for pid, sid in sel:
        if (pid, sid) not in by_key:
            print(f"WARN: {pid}:{sid} not in {args.val_parquet}", flush=True); continue
        row, gt = by_key[(pid, sid)]
        uc = next((m["content"] for m in row["prompt"] if m.get("role") == "user"), None)
        if uc is None:
            print(f"WARN: {pid}:{sid} no user content", flush=True); continue
        prompts.append(render(uc)); gts.append(row["reward_model"]["ground_truth"])
        meta.append({"problem_id": pid, "src_id": sid, "user_prompt": uc,
                     "n_usable": len(gt["usable_case_ids"])})

    if not prompts:
        print("FATAL: no prompts selected"); return 2
    print(f"[{args.arm} step{args.step}] {len(prompts)} prompts x n={args.n} "
          f"(temp={args.temperature} case={os.environ.get('PIE_GEM5_CASE_SELECT')} "
          f"K={os.environ.get('PIE_GEM5_REWARD_K')}) on {args.hf_dir}", flush=True)

    t0 = time.monotonic()
    comps = generate_completions(args.hf_dir, prompts, args.n,
                                 max_tokens=args.max_tokens, temperature=args.temperature,
                                 top_p=args.top_p, tensor_parallel_size=args.tensor_parallel_size)
    print(f"[{args.arm}] gen {time.monotonic()-t0:.0f}s", flush=True)
    t0 = time.monotonic()
    per = score_completions(comps, gts, args.test_case_dir, reference_table_dir=None)
    print(f"[{args.arm}] gem5 score {time.monotonic()-t0:.0f}s", flush=True)

    extract = _get_extract()
    recs = []
    for m, results, texts in zip(meta, per, comps):
        rollouts = []
        for j, (t, r) in enumerate(zip(texts, results)):
            rollouts.append({
                "i": j,
                "speedup": float(r.get("speedup", 0.0)),
                "score": float(r.get("score", 0.0)),
                "acc": float(r.get("acc", 0.0)),
                "reason": r.get("reason") or "",
                "code": extract(t),
                "text": t,
            })
        scores = [ro["score"] for ro in rollouts]
        accs = [ro["acc"] for ro in rollouts]
        # rank rollouts by speedup desc for easy reading
        rollouts.sort(key=lambda ro: ro["score"], reverse=True)
        rec = {**m, "arm": args.arm, "step": args.step, "n": len(scores),
               "n_pass": int(sum(a == 1.0 for a in accs)),
               "best_at_n": best_at_k(scores, len(scores)),
               "mean_score": (sum(scores) / len(scores)) if scores else 0.0,
               "max_speedup": max((ro["speedup"] for ro in rollouts), default=0.0),
               "max_correct_speedup": max((ro["speedup"] for ro in rollouts if ro["acc"] == 1.0),
                                          default=0.0),
               "rollouts": rollouts}
        recs.append(rec)
        print(f"  {m['problem_id']}/{m['src_id']}: pass {rec['n_pass']}/{rec['n']}  "
              f"best@{rec['n']}={rec['best_at_n']:.2f}x  max_correct={rec['max_correct_speedup']:.2f}x",
              flush=True)

    p = out / f"rollouts_{args.arm}_step{args.step}.json"
    p.write_text(json.dumps(recs, indent=2))
    print(f"[{args.arm}] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
