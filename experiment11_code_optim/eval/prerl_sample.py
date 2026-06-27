"""Freeze the 32-problem pre-RL analysis set (seed 0), reused across every config
so configs are directly comparable.

Samples N distinct problem_ids from the gem5 RL TRAIN parquet (the distribution
training optimizes), one src per problem, and joins src_code + headroom (ceiling
= src/tgt gem5 ratio) from the pairs parquet. The frozen ground_truth string is the
exact one training scored against (problem_id, usable_case_ids, gem5_src_per_tc_ticks).

  python -m experiment11_code_optim.eval.prerl_sample --out .../problems32.json
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gem5_parquet",
                    default="/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet/pie_gem5_train.parquet")
    ap.add_argument("--pairs_parquet",
                    default="/work/nvme/bgtv/sramasubramanian/pie_rl_dataset/pairs_train.parquet")
    ap.add_argument("--n_problems", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g = pd.read_parquet(a.gem5_parquet)
    bypid: dict[str, list] = {}
    for _, r in g.iterrows():
        gt = r["reward_model"]["ground_truth"]
        j = json.loads(gt) if isinstance(gt, (str, bytes)) else gt
        bypid.setdefault(j["problem_id"], []).append((j.get("src_id"), gt if isinstance(gt, str) else json.dumps(gt)))

    rng = random.Random(a.seed)
    pids = sorted(bypid.keys())
    rng.shuffle(pids)
    chosen = pids[:a.n_problems]

    pa = pd.read_parquet(a.pairs_parquet, columns=["problem_id", "src_id", "src_code",
                                                   "ceiling_speedup", "our_speedup_usable",
                                                   "our_fastest_speedup"])
    pa_idx = {(r.problem_id, r.src_id): r for r in pa.itertuples()}

    out = []
    for pid in chosen:
        srcs = sorted(bypid[pid])                       # deterministic order
        sid, gt = rng.choice(srcs)                      # one random src per problem
        j = json.loads(gt)
        pr = pa_idx.get((pid, sid))
        out.append({
            "problem_id": pid, "src_id": sid,
            "ground_truth": gt,
            "src_code": (pr.src_code if pr is not None else None),
            "ceiling_speedup": (float(pr.ceiling_speedup) if pr is not None else None),
            "our_speedup_usable": (float(pr.our_speedup_usable) if pr is not None else None),
            "our_fastest_speedup": (float(pr.our_fastest_speedup) if pr is not None else None),
            "n_usable": len(j.get("usable_case_ids") or []),
        })

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}: {len(out)} problems (seed {a.seed})")
    miss = [o["problem_id"] for o in out if o["src_code"] is None]
    if miss:
        print("WARN missing src_code:", miss)
    cl = sorted(o["ceiling_speedup"] for o in out if o["ceiling_speedup"])
    if cl:
        print(f"ceiling speedup: min {cl[0]:.2f}x  med {cl[len(cl)//2]:.2f}x  max {cl[-1]:.2f}x")
        print(f"  flat (ceiling<1.1x): {sum(c < 1.1 for c in cl)}/{len(cl)} ; "
              f">=2x: {sum(c >= 2 for c in cl)} ; >=5x: {sum(c >= 5 for c in cl)}")
    sl = sorted(len(o["src_code"] or "") for o in out)
    print(f"src chars: min {sl[0]} med {sl[len(sl)//2]} max {sl[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
