"""Pre-RL analysis over the per-config rollout dumps (prerl_gen_score.py).

Reads every gen_<config>.json under --out_dir (skips *_smoke), computes the
group A-G metrics from the brief, writes per-difficulty sub-tables, the named
plots, and a plain-English REPORT.md that the three decisions read straight off:
  1. thinking go/no-go  (p90 of GENUINE reasoning length, excl. repetition loops)
  2. liveliest reward signal + whether copy-collapse shows across all prompts
  3. whether instruct + step-by-step recovers the real speedups.

  python -m experiment11_code_optim.eval.prerl_analyze --out_dir .../prerl_analysis
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CONFIG_ORDER = ["inst17b_P1", "inst17b_P2", "inst17b_P3",
                "inst2507_P1", "inst2507_P2", "inst2507_P3", "think2507_P1", "think2507_P2"]
THIRD = 1.0 / 3.0
REP_LOOP = 0.5          # rep_ratio >= this == degenerate repetition loop
COPY_SIM = 0.98         # normalized-code similarity >= this == source copy
BUDGETS = [8192, 16384] # projected-truncation budgets


# --------------------------------------------------------------------------- io
def load_configs(out_dir: Path) -> dict[str, list]:
    data = {}
    for f in sorted(glob.glob(str(out_dir / "gen_*.json"))):
        name = Path(f).stem[len("gen_"):]
        if name.endswith("_smoke"):
            continue
        data[name] = json.loads(Path(f).read_text())
    return data


def flat(recs: list) -> list:
    return [r for rec in recs for r in rec["rollouts"]]


def is_think(cfg: str) -> bool:
    return cfg.startswith("think")


def diff_bin(ceiling: float) -> str:
    if ceiling is None:
        return "unk"
    if ceiling < 6:
        return "mod(<6x)"
    if ceiling < 10:
        return "high(6-10x)"
    return "vhigh(>=10x)"


# --------------------------------------------------------------- estimators
def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def frac_atleast1_atk(n: int, c: int, k: int) -> float:
    """P(>=1 of the c 'good' rollouts appears in a random size-k draw) — unbiased."""
    if c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def best_reward_atk(scores: list[float], k: int) -> float:
    """Unbiased E[max over a random size-k subset] (U-statistic)."""
    s = sorted(scores)
    n = len(s)
    if k >= n:
        return float(s[-1])
    denom = comb(n, k)
    tot = 0.0
    for i, v in enumerate(s):          # v is the max iff the other k-1 come from the i below it
        tot += v * comb(i, k - 1)
    return tot / denom


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def fmt(x, suf="", nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}{suf}"


# ----------------------------------------------------------- reason -> category
def fail_subtype(r: dict) -> str:
    """Failure sub-type for an acc==0 rollout. Reason strings (verbatim from
    gem5_reward._grade_one): 'native compile: …', 'case N: mismatch|timeout|exit_R|
    os_error:…|read_error:…', 'native phase over …s …', 'missing test files …'."""
    if not r.get("has_code"):
        return "no_code"
    rl = (r.get("reason") or "").lower()
    if "compile" in rl:
        return "compile_fail"
    if "mismatch" in rl:
        return "wrong_answer"
    if "timeout" in rl:
        return "native_timeout"
    if "exit_" in rl or "os_error" in rl or "read_error" in rl:
        return "runtime_crash"
    if "native phase over" in rl:
        return "too_slow_native"
    if "missing test" in rl or "infra" in rl:
        return "infra"
    return "other"


def edit_class(r: dict) -> str:
    """copy / cosmetic / real_speedup / correct_slower / timeout / fail.
    timeout = passed every correctness case but reward 0 (>3× src budget / sim fail);
    such rollouts carry acc==1, score==0 (base.acc overrides _zero in the reward)."""
    acc, sp, sc = r["acc"], r["speedup"], r["score"]
    if acc == 1.0 and sc > 0:
        if r.get("exact_copy") or r.get("sim_to_src", 0) >= COPY_SIM:
            return "copy"
        if abs(sp - 1.0) < 0.05:
            return "cosmetic"
        if sp > 1.05:
            return "real_speedup"
        return "correct_slower"
    if acc == 1.0 and sc == 0:
        return "timeout"
    return "fail"


# =====================================================================  GROUPS
def group_A(data, plotdir, md):
    md.append("## A. Thinking go/no-go — response length\n")
    md.append("Length in *generated* tokens. p90 of **genuine** length excludes repetition "
              "loops (rep_ratio≥%.1f). Projected-truncation@B = fraction of genuine rollouts "
              "with total tokens > B.\n" % REP_LOOP)
    md.append("| config | think? | p50 | p90 | p99 | max | mean think/reason/answer | "
              "trunc@cap | rep-loop% | empty-think% | genuine p90 | >8k | >16k |")
    md.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    A = {}
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        rs = flat(data[cfg])
        tot = np.array([r["n_tok_total"] for r in rs], float)
        th = np.array([r["n_tok_think"] for r in rs], float)
        re_ = np.array([r["n_tok_reason"] for r in rs], float)
        an = np.array([r["n_tok_answer"] for r in rs], float)
        trunc = np.array([r["truncated"] for r in rs], bool)
        reploop = np.array([r["rep_ratio"] >= REP_LOOP for r in rs], bool)
        genuine = tot[~reploop]
        emptyt = np.array([r["think_empty"] for r in rs], bool)
        A[cfg] = dict(tot=tot, think=th, reason=re_, answer=an, trunc=trunc,
                      reploop=reploop, genuine=genuine,
                      empty_think=float(emptyt.mean()) if is_think(cfg) else float("nan"))
        md.append("| %s | %s | %d | %d | %d | %d | %d/%d/%d | %s | %s | %s | %d | %s | %s |" % (
            cfg, "Y" if is_think(cfg) else "n",
            pct(tot, 50), pct(tot, 90), pct(tot, 99), tot.max(),
            th.mean(), re_.mean(), an.mean(),
            fmt(100 * trunc.mean(), "%", 1), fmt(100 * reploop.mean(), "%", 1),
            fmt(100 * A[cfg]["empty_think"], "%", 1) if is_think(cfg) else "–",
            pct(genuine, 90),
            fmt(100 * (genuine > BUDGETS[0]).mean(), "%", 1),
            fmt(100 * (genuine > BUDGETS[1]).mean(), "%", 1)))
    md.append("")
    _plot_length_cdf(A, plotdir)
    _plot_length_stack(A, plotdir)
    return A


def group_B(data, md):
    md.append("## B. Correctness & where the reward-0 mass comes from\n")
    md.append("0-reward mass splits cleanly by correctness: **fail** (acc=0) vs "
              "**correct-but-too-slow** (acc=1 yet reward 0 — passed every test but >3× src "
              "gem5, the over-budget bucket). Failure sub-types are from the reward's reason "
              "strings on the acc=0 rollouts.\n")
    md.append("| config | parse% | compile% | pass(acc=1)% | reward>0% | 0-mass | …fail | "
              "…too-slow | compile-fail | wrong-ans | crash | native-timeout | no-code |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        rs = flat(data[cfg]); n = len(rs)
        parse = np.mean([r["has_code"] for r in rs])
        passed = np.mean([r["acc"] == 1.0 for r in rs])
        rewpos = np.mean([r["score"] > 0 for r in rs])
        zero = np.mean([r["score"] == 0 for r in rs])
        too_slow = np.mean([(r["score"] == 0 and r["acc"] == 1.0) for r in rs])
        fail_real = np.mean([(r["score"] == 0 and r["acc"] == 0.0) for r in rs])
        sub = [fail_subtype(r) for r in rs if r["acc"] == 0.0]
        cnt = sub.count
        compile_ok = 1 - (cnt("compile_fail") + cnt("no_code")) / n
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %d | %d | %d | %d | %d |" % (
            cfg, fmt(100 * parse, "%", 1), fmt(100 * compile_ok, "%", 1), fmt(100 * passed, "%", 1),
            fmt(100 * rewpos, "%", 1), fmt(100 * zero, "%", 1), fmt(100 * fail_real, "%", 1),
            fmt(100 * too_slow, "%", 1), cnt("compile_fail"), cnt("wrong_answer"),
            cnt("runtime_crash"), cnt("native_timeout"), cnt("no_code")))
    md.append("")


def group_C(data, plotdir, md):
    md.append("## C. Is the reward signal alive & continuous?\n")
    md.append("Reward = gem5 src/cand on the biggest usable case, gated (0 if any test fails or "
              ">3× src). Correct rollouts are floored at 1/3; the interval (0, 1/3) is empty by "
              "design. 'dead problem' = all rollouts of a problem share one reward.\n")
    md.append("| config | mass@0 | ≈1.0 no-impr | exact1.0 echo | correct-slower | reward>1 | "
              "sp>1 med | sp>1 p90 | sp>1 max | %>1.1x | %>2x | mean per-prob CV | dead-prob% |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        rs = flat(data[cfg])
        sc = np.array([r["score"] for r in rs], float)
        sp = np.array([r["speedup"] for r in rs], float)
        corr = np.array([r["acc"] == 1.0 for r in rs], bool)
        spc = sp[corr & (sp > 1.02)]
        at0 = (sc == 0).mean()
        near1 = ((np.abs(sc - 1.0) < 0.02) & (sc > 0)).mean()    # no meaningful improvement
        exact1 = (np.abs(sc - 1.0) < 1e-6).mean()                # byte-identical echo
        slower = ((sc >= THIRD) & (sc < 0.98)).mean()
        above = (sc > 1.02).mean()
        cvs = []
        for rec in data[cfg]:
            s = np.array([x["score"] for x in rec["rollouts"]], float)
            cvs.append(s.std() / s.mean() if s.mean() > 0 else 0.0)
        dead = np.mean([len(set(round(x["score"], 6) for x in rec["rollouts"])) == 1 for rec in data[cfg]])
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            cfg, fmt(100 * at0, "%", 1), fmt(100 * near1, "%", 1), fmt(100 * exact1, "%", 1),
            fmt(100 * slower, "%", 1), fmt(100 * above, "%", 1),
            fmt(np.median(spc), "x") if len(spc) else "–",
            fmt(pct(spc, 90), "x") if len(spc) else "–",
            fmt(spc.max(), "x") if len(spc) else "–",
            fmt(100 * (corr & (sp > 1.1)).mean(), "%", 1),
            fmt(100 * (corr & (sp > 2.0)).mean(), "%", 1),
            fmt(np.mean(cvs), "", 2), fmt(100 * dead, "%", 1)))
    md.append("")
    # per-difficulty mass>1 + dead-prob
    md.append("**C by difficulty (ceiling band):** fraction of rollouts with reward>1 / dead-problem%\n")
    md.append("| config | " + " | ".join(["mod(<6x)", "high(6-10x)", "vhigh(>=10x)"]) + " |")
    md.append("|---|---|---|---|")
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        cells = []
        for band in ["mod(<6x)", "high(6-10x)", "vhigh(>=10x)"]:
            recb = [rec for rec in data[cfg] if diff_bin(rec.get("ceiling_speedup")) == band]
            if not recb:
                cells.append("–"); continue
            rsb = flat(recb)
            above = np.mean([r["score"] > 1 + 1e-6 for r in rsb])
            dead = np.mean([len(set(round(x["score"], 6) for x in rec["rollouts"])) == 1 for rec in recb])
            cells.append(f"{100*above:.0f}% / {100*dead:.0f}%")
        md.append("| %s | %s |" % (cfg, " | ".join(cells)))
    md.append("")
    _plot_reward_hist(data, plotdir)
    _plot_reward_heatmap(data, plotdir)


def group_D(data, plotdir, md):
    md.append("## D. Is the copy-collapse real (before any training)?\n")
    md.append("Copy = correct rollout with normalized-code similarity ≥ %.2f to source "
              "(the spike at reward 1.0). Diversity = distinct normalized programs ÷ n.\n" % COPY_SIM)
    md.append("| config | copy% (of correct) | exact-copy% (of all) | real-speedup% (of all) | "
              "mean sim-to-src | mean diversity/prob |")
    md.append("|---|--:|--:|--:|--:|--:|")
    classes = ["copy", "cosmetic", "real_speedup", "correct_slower", "timeout", "fail"]
    EC = {}
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        rs = flat(data[cfg]); n = len(rs)
        corr = [r for r in rs if r["acc"] == 1.0]
        copy_of_corr = np.mean([(r.get("exact_copy") or r.get("sim_to_src", 0) >= COPY_SIM) for r in corr]) if corr else float("nan")
        exact_all = np.mean([r.get("exact_copy", False) for r in rs])
        real_all = np.mean([edit_class(r) == "real_speedup" for r in rs])
        sim = np.mean([r.get("sim_to_src", 0) for r in rs])
        divs = []
        for rec in data[cfg]:
            hs = set()
            for r in rec["rollouts"]:
                c = r.get("code") or ""
                hs.add("".join(c.split()))
            divs.append(len(hs) / len(rec["rollouts"]))
        ec = [edit_class(r) for r in rs]
        EC[cfg] = {k: ec.count(k) / n for k in classes}
        md.append("| %s | %s | %s | %s | %s | %s |" % (
            cfg, fmt(100 * copy_of_corr, "%", 1), fmt(100 * exact_all, "%", 1),
            fmt(100 * real_all, "%", 1), fmt(sim, "", 3), fmt(np.mean(divs), "", 2)))
    md.append("")
    md.append("**Edit-class breakdown (% of all rollouts):**\n")
    md.append("| config | " + " | ".join(classes) + " |")
    md.append("|---|" + "|".join(["--:"] * len(classes)) + "|")
    for cfg in EC:
        md.append("| %s | %s |" % (cfg, " | ".join(f"{100*EC[cfg][k]:.0f}%" for k in classes)))
    md.append("")
    _plot_editclass(EC, classes, plotdir)
    _plot_sim_hist(data, plotdir)


def group_E(data, plotdir, md):
    md.append("## E. Headroom & exploitability (best-of-k)\n")
    md.append("Ceiling = src/tgt (human-fast) gem5 ratio. best@32 averaged over problems; "
              "ceiling-capture = best-correct-speedup ÷ ceiling. Reward-gap = best@32 − 1.0.\n")
    md.append("| config | mean ceiling | best@32 (avg) | mean ceiling-capture | reward-gap | "
              "%prob ≥1 correct | %prob ≥1 speedup>1 |")
    md.append("|---|--:|--:|--:|--:|--:|--:|")
    EK = {}
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        ceil = np.array([rec.get("ceiling_speedup") or float("nan") for rec in data[cfg]], float)
        best = np.array([rec["best_at_n"] for rec in data[cfg]], float)
        capt = []
        for rec in data[cfg]:
            c = rec.get("ceiling_speedup")
            if c and rec["max_correct_speedup"] > 0:
                capt.append(min(1.0, rec["max_correct_speedup"] / c))
            else:
                capt.append(0.0)
        any_corr = np.mean([rec["n_pass"] > 0 for rec in data[cfg]])
        any_fast = np.mean([any(r["acc"] == 1 and r["speedup"] > 1 for r in rec["rollouts"]) for rec in data[cfg]])
        # speedup-at-k curve
        ks = list(range(1, 33))
        curve_corr, curve_fast = [], []
        for k in ks:
            cc, ff = [], []
            for rec in data[cfg]:
                n = rec["n"]
                cc.append(frac_atleast1_atk(n, rec["n_pass"], k))
                cfast = sum(1 for r in rec["rollouts"] if r["acc"] == 1 and r["speedup"] > 1)
                ff.append(frac_atleast1_atk(n, cfast, k))
            curve_corr.append(np.mean(cc)); curve_fast.append(np.mean(ff))
        EK[cfg] = dict(ks=ks, corr=curve_corr, fast=curve_fast, capt=np.mean(capt))
        md.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            cfg, fmt(np.nanmean(ceil), "x"), fmt(np.mean(best), "x"),
            fmt(100 * np.mean(capt), "%", 1), fmt(np.mean(best) - 1.0, "x"),
            fmt(100 * any_corr, "%", 0), fmt(100 * any_fast, "%", 0)))
    md.append("")
    _plot_speedup_atk(EK, plotdir)


def group_F(data, plotdir, md):
    md.append("## F. Gradient liveness — GOS-vs-GRPO mechanism\n")
    md.append("Each problem's 32 rollouts are chunked into groups of G. A 'dead group' has all "
              "rewards identical → GRPO advantage ≡ 0 (no gradient). GOS can still move any group "
              "with ≥1 positive-advantage rollout.\n")
    md.append("| config | G | dead-group% | …all-0 | …all-1.0 | …all-other | mean within-grp std | "
              "%grp w/ signal |")
    md.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    F = {}
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        F[cfg] = {}
        for G in (8, 16):
            dead = a0 = a1 = aoth = live = 0; ngrp = 0; stds = []
            for rec in data[cfg]:
                s = [x["score"] for x in rec["rollouts"]]
                for g0 in range(0, len(s) - G + 1, G):
                    grp = s[g0:g0 + G]; ngrp += 1
                    stds.append(float(np.std(grp)))
                    if max(grp) - min(grp) < 1e-9:
                        dead += 1
                        if abs(grp[0]) < 1e-9:
                            a0 += 1
                        elif abs(grp[0] - 1.0) < 1e-6:
                            a1 += 1
                        else:
                            aoth += 1
                    else:
                        live += 1
            if ngrp == 0:
                F[cfg][G] = dict(dead=0.0, a0=0.0, a1=0.0, aoth=0.0, std=0.0, live=0.0)
                continue
            F[cfg][G] = dict(dead=dead / ngrp, a0=a0 / ngrp, a1=a1 / ngrp, aoth=aoth / ngrp,
                             std=float(np.mean(stds)), live=live / ngrp)
            md.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
                cfg, G, fmt(100 * F[cfg][G]["dead"], "%", 1), fmt(100 * F[cfg][G]["a0"], "%", 1),
                fmt(100 * F[cfg][G]["a1"], "%", 1), fmt(100 * F[cfg][G]["aoth"], "%", 1),
                fmt(F[cfg][G]["std"], "", 3), fmt(100 * F[cfg][G]["live"], "%", 1)))
    md.append("")
    # dead-group by difficulty at G=16
    md.append("**Dead-group% at G=16 by difficulty band:**\n")
    md.append("| config | mod(<6x) | high(6-10x) | vhigh(>=10x) |")
    md.append("|---|--:|--:|--:|")
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        cells = []
        for band in ["mod(<6x)", "high(6-10x)", "vhigh(>=10x)"]:
            recb = [rec for rec in data[cfg] if diff_bin(rec.get("ceiling_speedup")) == band]
            if not recb:
                cells.append("–"); continue
            dead = ngrp = 0
            for rec in recb:
                s = [x["score"] for x in rec["rollouts"]]
                for g0 in range(0, len(s) - 15, 16):
                    grp = s[g0:g0 + 16]; ngrp += 1
                    if max(grp) - min(grp) < 1e-9:
                        dead += 1
            cells.append(f"{100*dead/ngrp:.0f}%" if ngrp else "–")
        md.append("| %s | %s |" % (cfg, " | ".join(cells)))
    md.append("")
    _plot_deadgroup(F, plotdir)


def group_G(data, plotdir, md):
    md.append("## G. Does reasoning actually help?\n")
    md.append("Reward vs reasoning length (think tokens for thinking configs, visible-reason "
              "tokens for instruct). Instruct+P3 vs thinking-native on real-speedup rate, mean "
              "reward, and reasoning-token cost.\n")
    md.append("| config | reason-token field | mean reason-tok (correct) | real-speedup% | "
              "mean reward | mean reward (corr) |")
    md.append("|---|---|--:|--:|--:|--:|")
    series = {}
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        rs = flat(data[cfg])
        field = "n_tok_think" if is_think(cfg) else "n_tok_reason"
        rlen = np.array([r[field] for r in rs], float)
        rew = np.array([r["score"] for r in rs], float)
        corr = np.array([r["acc"] == 1.0 for r in rs], bool)
        real = np.mean([edit_class(r) == "real_speedup" for r in rs])
        series[cfg] = (rlen, rew, field)
        md.append("| %s | %s | %s | %s | %s | %s |" % (
            cfg, field, fmt(rlen[corr].mean() if corr.any() else float("nan"), "", 0),
            fmt(100 * real, "%", 1), fmt(rew.mean(), "", 3),
            fmt(rew[corr].mean() if corr.any() else float("nan"), "", 3)))
    md.append("")
    _plot_reward_vs_reason(series, plotdir)


def group_H(data, plotdir, md):
    md.append("## H. Copy-collapse mechanism — the GRPO/RLOO estimator pathology\n")
    md.append("Why mean-baseline estimators (GRPO/RLOO) reinforce the source-copy. With a low pass "
              "rate the fails (reward 0) drag the per-prompt group mean below 1, so the copy's "
              "reward of ≈1.0 sits **above the baseline** and earns a positive advantage — it gets "
              "reinforced. The reward is effectively trimodal {0 fail, ≈1.0 copy, >1 win} (the "
              "correct-but-slower band is near-empty), so a group is mostly fails + copies + a few wins.\n")
    md.append("`copy>baseline` = problems where a reward-1 copy beats the group mean (collapse "
              "pressure). `best≤1.02` = problems where even the *best* rollout is no-improvement "
              "(the model cannot beat source — copying is reward-optimal and **no** estimator, GOS "
              "included, can help). `GOS-moat` = copy above baseline **and** a real >1.5× win exists "
              "in the group — the slice GOS uniquely harvests and GRPO/RLOO waste.\n")
    md.append("| config | pass% | mean | median grp fail/slow/≈1.0/>1 | copy>baseline | "
              "best≤1.02 (copy ceiling) | GOS-moat | med(best−mean) |")
    md.append("|---|--:|--:|:--:|--:|--:|--:|--:|")
    H = {}
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        rs = flat(data[cfg]); npr = len(data[cfg])
        passr = np.mean([r["acc"] == 1 for r in rs])
        meanr = np.mean([r["score"] for r in rs])
        fails = []; slows = []; copies = []; wins = []; gaps = []; per = []
        copy_above = trap = moat = 0
        for rec in data[cfg]:
            ro = rec["rollouts"]
            fails.append(sum(r["score"] == 0 for r in ro))
            slows.append(sum(r["acc"] == 1 and r["speedup"] < 0.98 for r in ro))
            copies.append(sum(r["acc"] == 1 and abs(r["speedup"] - 1) < 0.05 for r in ro))
            wins.append(sum(r["acc"] == 1 and r["speedup"] > 1.02 for r in ro))
            m = float(np.mean([r["score"] for r in ro])); b = rec["best_at_n"]
            gaps.append(b - m); per.append((m, b))
            if m < 1.0: copy_above += 1
            if b <= 1.02: trap += 1
            if m < 1.0 and b > 1.5: moat += 1
        H[cfg] = dict(per=per)
        md.append("| %s | %s | %s | %d/%d/%d/%d | %d/%d | %d/%d | %d/%d | %s |" % (
            cfg, fmt(100 * passr, "%", 0), fmt(meanr, "", 2),
            int(np.median(fails)), int(np.median(slows)), int(np.median(copies)), int(np.median(wins)),
            copy_above, npr, trap, npr, moat, npr, fmt(np.median(gaps), "x")))
    md.append("")
    md.append("**Reading it.** The 'copy' outcome has two distinct causes needing different fixes: "
              "**(1) the model can't beat source** (`best≤1.02`) — copying is genuinely reward-"
              "optimal, no estimator helps; **(2) the estimator pathology proper** (`GOS-moat`) — a "
              "better rollout is right there in the group but the mean baseline still favours the safe "
              "copy. Cause (1) is fixed by competence (bigger model / a reasoning prompt that lifts "
              "best>1); cause (2) is fixed by the *estimator* (GOS chases the max, not the mean). High "
              "`copy>baseline` **and** high `best≤1.02` = deep collapse (1.7B, 4B-P1/P2); high "
              "`copy>baseline` with LOW `best≤1.02` and a fat `GOS-moat` = the regime where GOS "
              "out-earns GRPO/RLOO without either collapsing (4B-P3).\n")
    md.append("**By difficulty (ceiling band) — best≤1.02 ; GOS-moat (of n):**\n")
    md.append("| config | mod(<6x) | high(6-10x) | vhigh(>=10x) |")
    md.append("|---|--:|--:|--:|")
    for cfg in [c for c in CONFIG_ORDER if c in data]:
        cells = []
        for band in ["mod(<6x)", "high(6-10x)", "vhigh(>=10x)"]:
            recb = [rec for rec in data[cfg] if diff_bin(rec.get("ceiling_speedup")) == band]
            if not recb:
                cells.append("–"); continue
            t = sum(1 for rec in recb if rec["best_at_n"] <= 1.02)
            mo = sum(1 for rec in recb if float(np.mean([x["score"] for x in rec["rollouts"]])) < 1
                     and rec["best_at_n"] > 1.5)
            cells.append(f"{t} ; {mo} (/{len(recb)})")
        md.append("| %s | %s |" % (cfg, " | ".join(cells)))
    md.append("")
    _plot_mechanism(H, plotdir)


# =====================================================================  PLOTS
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_length_cdf(A, plotdir):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, d in A.items():
        x = np.sort(d["tot"]); y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=cfg, lw=2)
    for b in BUDGETS:
        ax.axvline(b, color="gray", ls="--", lw=1); ax.text(b, 0.02, f"{b//1024}k", fontsize=8)
    ax.set_xlabel("generated tokens"); ax.set_ylabel("CDF"); ax.set_title("A1 — response-length CDF")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plotdir / "A1_length_cdf.png", dpi=130); plt.close(fig)


def _plot_length_stack(A, plotdir):
    plt = _mpl()
    cfgs = list(A); x = np.arange(len(cfgs))
    th = [A[c]["think"].mean() for c in cfgs]
    re_ = [A[c]["reason"].mean() for c in cfgs]
    an = [A[c]["answer"].mean() for c in cfgs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, th, label="think", color="#9467bd")
    ax.bar(x, re_, bottom=th, label="reason", color="#1f77b4")
    ax.bar(x, an, bottom=np.array(th) + np.array(re_), label="answer", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean tokens"); ax.set_title("A2 — length decomposition"); ax.legend()
    fig.tight_layout(); fig.savefig(plotdir / "A2_length_stack.png", dpi=130); plt.close(fig)


def _plot_reward_hist(data, plotdir):
    plt = _mpl()
    cfgs = [c for c in CONFIG_ORDER if c in data]
    fig, axes = plt.subplots(2, len(cfgs), figsize=(3.2 * len(cfgs), 6), squeeze=False)
    for j, cfg in enumerate(cfgs):
        sc = np.array([r["score"] for r in flat(data[cfg])], float)
        ax = axes[0][j]
        ax.hist(sc, bins=40, color="#1f77b4")
        ax.axvline(THIRD, color="orange", ls="--", lw=1); ax.axvline(1.0, color="red", ls=":", lw=1)
        ax.set_title(cfg, fontsize=9); ax.set_yscale("log")
        if j == 0:
            ax.set_ylabel("count (log) — linear x")
        axl = axes[1][j]
        pos = sc[sc > 0]
        if len(pos):
            axl.hist(pos, bins=np.logspace(np.log10(max(THIRD, pos.min())), np.log10(pos.max() + 1e-9), 30), color="#2ca02c")
            axl.set_xscale("log")
        axl.axvline(1.0, color="red", ls=":", lw=1)
        if j == 0:
            axl.set_ylabel("count — log x (reward>0)")
    fig.suptitle("C1 — reward histogram (orange=1/3 floor, red=1.0 copy)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(plotdir / "C1_reward_hist.png", dpi=130); plt.close(fig)


def _plot_reward_heatmap(data, plotdir):
    plt = _mpl()
    cfgs = [c for c in CONFIG_ORDER if c in data]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(3.0 * len(cfgs), 6), squeeze=False)
    for j, cfg in enumerate(cfgs):
        recs = sorted(data[cfg], key=lambda r: -(r.get("ceiling_speedup") or 0))
        M = np.array([sorted([x["score"] for x in r["rollouts"]], reverse=True) for r in recs])
        ax = axes[0][j]
        im = ax.imshow(np.clip(M, 0, 5), aspect="auto", cmap="viridis")
        ax.set_title(cfg, fontsize=9); ax.set_xlabel("rollout (sorted)")
        if j == 0:
            ax.set_ylabel("problem (sorted by ceiling)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label="reward (clip 5)")
    fig.suptitle("C2 — per-problem reward spread", fontweight="bold")
    fig.savefig(plotdir / "C2_reward_heatmap.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def _plot_editclass(EC, classes, plotdir):
    plt = _mpl()
    cfgs = list(EC); x = np.arange(len(cfgs)); bottom = np.zeros(len(cfgs))
    colors = {"copy": "#d62728", "cosmetic": "#ff9896", "real_speedup": "#2ca02c",
              "correct_slower": "#98df8a", "timeout": "#ffbb78", "fail": "#7f7f7f"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in classes:
        vals = np.array([EC[c][k] for c in cfgs])
        ax.bar(x, vals, bottom=bottom, label=k, color=colors.get(k))
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("fraction of rollouts"); ax.set_title("D1 — edit-class breakdown")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(plotdir / "D1_editclass.png", dpi=130); plt.close(fig)


def _plot_sim_hist(data, plotdir):
    plt = _mpl()
    cfgs = [c for c in CONFIG_ORDER if c in data]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in cfgs:
        sim = [r.get("sim_to_src", 0) for r in flat(data[cfg]) if r["has_code"]]
        ax.hist(sim, bins=30, histtype="step", lw=2, label=cfg)
    ax.axvline(COPY_SIM, color="red", ls="--", lw=1, label=f"copy≥{COPY_SIM}")
    ax.set_xlabel("normalized-code similarity to source"); ax.set_ylabel("count")
    ax.set_title("D2 — similarity-to-source"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plotdir / "D2_sim_hist.png", dpi=130); plt.close(fig)


def _plot_speedup_atk(EK, plotdir):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for cfg, d in EK.items():
        ax[0].plot(d["ks"], d["corr"], label=cfg, lw=2)
        ax[1].plot(d["ks"], d["fast"], label=cfg, lw=2)
    ax[0].set_title("E1a — P(≥1 correct) @ k"); ax[1].set_title("E1b — P(≥1 speedup>1) @ k")
    for a in ax:
        a.set_xlabel("k (rollouts)"); a.set_ylabel("fraction of problems"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plotdir / "E1_speedup_atk.png", dpi=130); plt.close(fig)


def _plot_deadgroup(F, plotdir):
    plt = _mpl()
    cfgs = list(F); x = np.arange(len(cfgs)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, [100 * F[c][8]["dead"] for c in cfgs], w, label="G=8", color="#1f77b4")
    ax.bar(x + w / 2, [100 * F[c][16]["dead"] for c in cfgs], w, label="G=16", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(cfgs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("dead-group %"); ax.set_title("F1 — dead-group rate (GRPO gets no gradient)")
    ax.legend()
    fig.tight_layout(); fig.savefig(plotdir / "F1_deadgroup.png", dpi=130); plt.close(fig)


def _plot_reward_vs_reason(series, plotdir):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg, (rlen, rew, field) in series.items():
        if rlen.max() <= 0:
            continue
        bins = np.linspace(0, np.percentile(rlen, 95) + 1, 9)
        idx = np.digitize(rlen, bins)
        xs, ys = [], []
        for b in range(1, len(bins)):
            m = idx == b
            if m.sum() >= 3:
                xs.append(rlen[m].mean()); ys.append(rew[m].mean())
        ax.plot(xs, ys, "-o", label=cfg, lw=2, ms=4)
    ax.set_xlabel("reasoning tokens (think or visible)"); ax.set_ylabel("mean reward")
    ax.set_title("G1 — reward vs reasoning length (binned)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plotdir / "G1_reward_vs_reason.png", dpi=130); plt.close(fig)


def _plot_mechanism(H, plotdir):
    """Per-problem (mean reward, best@32) scatter — the four collapse regimes."""
    plt = _mpl()
    cfgs = list(H)
    ncol = min(len(cfgs), 4); nrow = (len(cfgs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.4 * nrow), squeeze=False)
    for i, cfg in enumerate(cfgs):
        ax = axes[i // ncol][i % ncol]
        per = H[cfg]["per"]
        xs = [m for m, b in per]; ys = [max(b, 1e-2) for m, b in per]
        cols = ["#d62728" if (m < 1 and b > 1.5) else ("#7f7f7f" if b <= 1.02 else "#1f77b4")
                for m, b in per]
        ax.scatter(xs, ys, s=26, c=cols, alpha=0.85, edgecolors="none")
        ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)     # copy-above-baseline boundary
        ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.6)   # copy-is-ceiling boundary
        ax.axhline(1.5, color="#d62728", ls=":", lw=0.8, alpha=0.4)
        ax.set_yscale("log"); ax.set_title(cfg, fontsize=9)
        ax.set_xlabel("per-problem mean reward")
        if i % ncol == 0:
            ax.set_ylabel("best@32 (log)")
        ax.grid(alpha=0.25)
    for j in range(len(cfgs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("H1 — collapse mechanism: per-problem (mean, best@32). red=GOS-moat (mean<1 & "
                 "best>1.5), grey=copy-ceiling (best≤1), blue=safe/recoverable", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(plotdir / "H1_mechanism.png", dpi=130); plt.close(fig)


# =====================================================================  MAIN
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    data = load_configs(out)
    if not data:
        print(f"no gen_*.json under {out}"); return 2
    plotdir = out / "plots"; plotdir.mkdir(exist_ok=True)
    print("configs:", list(data))

    md = ["# PIE pre-RL analysis — Qwen3 1.7B vs 4B (Instruct-2507 / Thinking-2507)\n",
          f"Configs: {', '.join(data)}. 32 train problems (seed 0, ceiling 4.9–49.7×, no flat "
          "problems — the RL train set is headroom-filtered), 32 rollouts each, temp 1.0/top_p 1.0. "
          "Reward = live gem5 src/cand speedup (K=1 biggest case, all-cases correctness gate, 3× "
          "budget), identical to the n=1024 best_reward@k eval.\n"]

    group_A(data, plotdir, md)
    group_B(data, md)
    group_C(data, plotdir, md)
    group_D(data, plotdir, md)
    group_E(data, plotdir, md)
    group_F(data, plotdir, md)
    group_G(data, plotdir, md)
    group_H(data, plotdir, md)

    md.append("## Plots\n")
    for p in ["A1_length_cdf", "A2_length_stack", "C1_reward_hist", "C2_reward_heatmap",
              "D1_editclass", "D2_sim_hist", "E1_speedup_atk", "F1_deadgroup", "G1_reward_vs_reason",
              "H1_mechanism"]:
        md.append(f"![{p}](plots/{p}.png)")
    md.append("")

    (out / "ANALYSIS.md").write_text("\n".join(md))
    print(f"[report] -> {out/'ANALYSIS.md'}")
    print(f"[plots]  -> {plotdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
