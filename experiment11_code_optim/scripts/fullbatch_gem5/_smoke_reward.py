"""Offline gem5-reward smoke for the FULL-BATCH launcher config.
Validates the exact path the launcher uses: K=1, case_select=sample (RANDOM),
reward_kind=ratio, against (a) the engine golden on bysrc rows and (b) the real
verl TRAIN parquet ground_truth format. Run on a gem5-capable compute node.
"""
import os, sys, json, time
sys.path.insert(0, "/projects/bgmo/sramasubramanian/cont-maxrl")
import pandas as pd
from experiment11_code_optim.reward import gem5_reward as G

BYSRC = "/work/nvme/bgtv/sramasubramanian/pie_rl_dataset/bysrc_train.parquet"
TRAIN = "/work/nvme/bgtv/sramasubramanian/pie_gem5_parquet/pie_gem5_train.parquet"
N = int(os.environ.get("SMOKE_N", "3"))

print("=" * 72)
print("PART A — engine golden @ K=1, case_select=sample (src->~1.0, tgt->dataset speedup)")
print("=" * 72)
df = pd.read_parquet(BYSRC)
df = df[df.n_usable >= 2].head(N)
jobs, expect = [], []
for _, row in df.iterrows():
    gt = {"problem_id": row.problem_id, "src_id": row.src_id,
          "usable_case_ids": json.loads(row.usable_case_ids) if isinstance(row.usable_case_ids, str) else list(row.usable_case_ids),
          "gem5_src_per_tc_ticks": row.gem5_src_per_tc_ticks}
    for tag, code in (("src", row.src_code), ("tgt", row.oracle_tgt_code)):
        jobs.append(G._job_from_ground_truth(len(jobs), f"```cpp\n{code}\n```", json.dumps(gt)))
        expect.append((row.problem_id, tag, 1.0 if tag == "src" else float(row.our_speedup_usable)))
# EXACT launcher reward config
cfg = G.Gem5RewardConfig(reward_kind="ratio", k_cases=1, sample_mode="group", case_select="sample")
print(f"cfg: kind={cfg.reward_kind} k={cfg.k_cases} sample={cfg.sample_mode} case_select={cfg.case_select} budget={cfg.budget_mult}x nat_to={cfg.native_timeout_s}s")
t0 = time.time()
res = G.compute_rewards_batch(jobs, cfg)
okA = True
for (pid, tag, exp), r in zip(expect, res):
    flag = ""
    if tag == "src" and not (r["acc"] == 1.0 and 0.80 <= r["score"] <= 1.20):
        flag = "  <-- FAIL src-as-rollout should be ~1.0"; okA = False
    if tag == "tgt" and r["acc"] == 1.0 and r["score"] < 1.0:
        flag = "  <-- note: tgt scored <1 (case-dependent at K=1)"
    print(f"  {pid} {tag:3s} score={r['score']:.3f} acc={r['acc']:.0f} "
          f"phase={r.get('phase')} reason={r.get('reason') or '-'} (dataset ref ~{exp:.2f}){flag}")
print(f"PART A wall {time.time()-t0:.1f}s -> {'PASS' if okA else 'FAIL'}")

print()
print("=" * 72)
print("PART B — real verl TRAIN parquet ground_truth -> compute_score (K=1, sample)")
print("=" * 72)
os.environ["PIE_GEM5_REWARD_K"] = "1"
os.environ["PIE_GEM5_CASE_SELECT"] = "sample"
os.environ["PIE_GEM5_REWARD_SAMPLE"] = "group"
os.environ["PIE_GEM5_REWARD_KIND"] = "ratio"
tdf = pd.read_parquet(TRAIN).head(N)
# join to bysrc to recover the src code (the train parquet only carries prompt+gt)
bdf = pd.read_parquet(BYSRC)[["problem_id", "src_id", "src_code"]]
okB = True
t0 = time.time()
for _, tr in tdf.iterrows():
    gt_str = tr["reward_model"]["ground_truth"]
    gt = json.loads(gt_str)
    m = bdf[(bdf.problem_id == gt["problem_id"]) & (bdf.src_id == gt["src_id"])]
    if m.empty:
        print(f"  {gt['problem_id']}/{gt['src_id']}: no bysrc src_code (skip)"); continue
    src_code = m.iloc[0].src_code
    out = G.compute_score(f"```cpp\n{src_code}\n```", gt_str)  # reads env config
    flag = "" if (out["acc"] == 1.0 and 0.80 <= out["score"] <= 1.20) else "  <-- FAIL (src should be ~1.0)"
    if flag:
        okB = False
    print(f"  {gt['problem_id']}/{gt['src_id']}  score={out['score']:.3f} acc={out['acc']:.0f} "
          f"reason={out.get('reason') or '-'}{flag}")
print(f"PART B wall {time.time()-t0:.1f}s -> {'PASS' if okB else 'FAIL'}")
print()
print("OVERALL:", "PASS" if (okA and okB) else "FAIL")
sys.exit(0 if (okA and okB) else 1)
