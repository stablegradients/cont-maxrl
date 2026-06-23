"""Build the RL-ready PIE dataset (pairs + bysrc) with ALL drop filters applied.

Phases (see docs/gem5_reward_fixes.md):
  2. per-program verdicts from pie_screen_v2 (compile_ok, n_fail, fails)
     bad_apple = not compile_ok OR n_fail > max(5, floor(0.05*ncases))
     kept[pid] = all_cases - union(fails of GOOD programs)   (drop case if >=1 good prog fails it)
  3. join gem5 timing (per_program parquet): per_tc_ticks, excluded, status
  4. strict-PIE wall: timed_wall = {c: ticks[c] <= 1.4e10}; K_eff = kept or (all cases if kept empty)
     usable(pair) = K_eff & src.timed_wall & tgt.timed_wall ; usable(src) = K_eff & src.timed_wall
  5. pairs: DROP if src/tgt bad_apple | src/tgt gem5 hard-excl | degenerate pid |
            |src.fails U tgt.fails| > thr | n_usable==0
  6. bysrc: good srcs only; oracle = good+timed tgt with max our_speedup_usable
  7. write parquets + drop ledger + README

REWARD-AGNOSTIC: full per-tc ticks carried; runtime subsamples K (-1=all) from usable_case_ids.
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd

OFF = "/work/nvme/bgtv/sramasubramanian/pie_official/PIE_Dataset"
TIMES = "/work/nvme/bgtv/sramasubramanian/pie_gem5_times"
SCREEN = "/work/nvme/bgtv/sramasubramanian/pie_screen_v2"
OUT = os.environ.get("PIE_RL_DATASET_OUT", "/work/nvme/bgtv/sramasubramanian/pie_rl_dataset")
WALL = float(os.environ.get("PIE_WALL_TICKS", "1.4e10"))   # strict PIE 2-min cutoff
FLOOR = 192e6
LOW_HEADROOM = 1.5
DEGENERATE = {"p00000", "p02068", "p00754"}
HARD_EXCL = {"startup_fail", "native_all_excluded"}
SPLITS = ["train", "val", "test"]

# PIE jsonl key -> our column (verbatim carry; dict/list values JSON-encoded)
PIE_COMMON = {
    "src_agg_runtime": "pie_src_agg_runtime", "tgt_agg_runtime": "pie_tgt_agg_runtime",
    "speedup": "pie_speedup", "fastest_agg_runtime": "pie_fastest_agg_runtime",
    "src_status": "pie_src_verdict", "tgt_status": "pie_tgt_verdict",
}
PIE_TRVAL = {
    "fastest_agg_runtime_updated": "pie_fastest_agg_runtime_updated",
    "src_reward_updated": "pie_src_reward_updated",
    "target_reward_updated": "pie_target_reward_updated",
    "src_reward_updated_pct_bin": "pie_src_reward_updated_pct_bin",
    "target_reward_updated_pct_bin": "pie_target_reward_updated_pct_bin",
}
PIE_TEST = {
    "n_tests": "pie_n_tests", "test_agg_runtime": "pie_test_agg_runtime",
    "test_accuracy": "pie_test_accuracy", "test_compilation": "pie_test_compilation",
    "src_code_accuracy": "pie_src_code_accuracy", "tgt_code_accuracy": "pie_tgt_code_accuracy",
    "src_code_compilation": "pie_src_code_compilation", "tgt_code_compilation": "pie_tgt_code_compilation",
    "src_code_runtime": "pie_src_code_runtime", "tgt_code_runtime": "pie_tgt_code_runtime",
    "fastest_code": "pie_fastest_code", "fastest_code_len": "pie_fastest_code_len",
    "fastest_code_runtime": "pie_fastest_code_runtime",
    "fastest_code_accuracy": "pie_fastest_code_accuracy",
    "fastest_code_compilation": "pie_fastest_code_compilation",
    "src_code_tc2time": "pie_src_code_tc2time", "tgt_code_tc2time": "pie_tgt_code_tc2time",
    "fastest_code_tc2time": "pie_fastest_code_tc2time",
}
_JSONIFY = {"src_code_tc2time", "tgt_code_tc2time", "fastest_code_tc2time"}


def _pie_fields(x, prefix_map):
    out = {}
    for k, col in prefix_map.items():
        v = x.get(k)
        out[col] = json.dumps(v) if (k in _JSONIFY and v is not None) else v
    return out


def load_screen(split):
    """pid -> {ncases, progs:{sid:{compile_ok,n_fail,fails(set)}}, kept(set), thr, good(set)}"""
    P = {}
    for f in sorted(glob.glob(f"{SCREEN}/{split}/shard_*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            pid, nc = r["pid"], r["ncases"]
            progs = {sid: {"compile_ok": d["compile_ok"], "n_fail": d["n_fail"],
                           "fails": set(d["fails"])} for sid, d in r["progs"].items()}
            thr = max(5, int(0.05 * nc))
            good = {sid for sid, d in progs.items()
                    if d["compile_ok"] and d["n_fail"] <= thr}
            allc = set(range(nc)) if False else None  # case ids are explicit; build kept from fails over good
            # kept = cases passed by EVERY good program = (union of all case idx seen) - union(good fails)
            # case universe from merged is 0..nc-1 by construction of the screen
            universe = set(range(nc))
            badcases = set()
            for sid in good:
                badcases |= progs[sid]["fails"]
            kept = universe - badcases
            P[pid] = {"ncases": nc, "progs": progs, "thr": thr, "good": good, "kept": kept}
    return P


def load_ticks(split):
    pp = pd.read_parquet(f"{TIMES}/per_program_{split}.parquet")
    per, status, excl = {}, {}, {}
    for r in pp.itertuples():
        d = json.loads(r.per_tc_ticks) if r.per_tc_ticks else {}
        per[(r.problem_id, r.submission_id)] = {int(k): int(v) for k, v in d.items()}
        status[(r.problem_id, r.submission_id)] = r.status
        excl[(r.problem_id, r.submission_id)] = r.excluded
    return per, status, excl


def timed_wall(d):
    return {c for c, t in d.items() if t <= WALL}


def build(split, led):
    S = load_screen(split)
    per, status, excl = load_ticks(split)

    def prog_ok(pid, sid):
        """good (compile+non-bad-apple) AND has gem5 timing not hard-excluded."""
        sp = S.get(pid)
        if not sp or sid not in sp["good"]:
            return False
        if status.get((pid, sid)) in HARD_EXCL:
            return False
        return True

    def keff(pid):
        kept = S[pid]["kept"] if pid in S else set()
        return kept if kept else set(range(S[pid]["ncases"])) if pid in S else set()

    pair_rows, by = [], {}
    for line in open(f"{OFF}/{split}.jsonl"):
        x = json.loads(line)
        pid, sid, tid = x["problem_id"], x.get("src_id"), x.get("tgt_id")
        led["pairs_seen"] += 1
        if pid in DEGENERATE:
            led["drop_degenerate"] += 1
            continue
        if pid not in S or sid not in S[pid]["progs"] or tid not in S[pid]["progs"]:
            led["drop_missing_screen"] += 1
            continue
        s_bad = sid not in S[pid]["good"]; t_bad = tid not in S[pid]["good"]
        if s_bad or t_bad:
            led["drop_badapple"] += 1
            continue
        if status.get((pid, sid)) in HARD_EXCL or status.get((pid, tid)) in HARD_EXCL:
            led["drop_hard_excl"] += 1
            continue
        thr = S[pid]["thr"]
        pair_fail = S[pid]["progs"][sid]["fails"] | S[pid]["progs"][tid]["fails"]
        if len(pair_fail) > thr:
            led["drop_pair_criterion"] += 1
            continue
        s_d, t_d = per.get((pid, sid), {}), per.get((pid, tid), {})
        K = keff(pid)
        usable = sorted(K & timed_wall(s_d) & timed_wall(t_d))
        if not usable:
            led["drop_zero_usable"] += 1
            continue
        led["pairs_kept"] += 1

        s_sum = sum(s_d[c] for c in usable); t_sum = sum(t_d[c] for c in usable)
        our_sp_u = (s_sum / t_sum) if t_sum else None
        common_all = sorted(set(s_d) & set(t_d))
        cs = sum(s_d[c] for c in common_all); ct = sum(t_d[c] for c in common_all)
        our_sp_all = (cs / ct) if ct else None
        meanpc = np.mean([s_d[c] for c in timed_wall(s_d)]) if timed_wall(s_d) else float("nan")
        ceil = meanpc / FLOOR if meanpc == meanpc else float("nan")

        row = {
            "problem_id": pid, "src_id": sid, "tgt_id": tid, "split": split,
            "language": x.get("language", "cpp"),
            "src_code": x.get("src_code"), "tgt_code": x.get("tgt_code"),
            "gem5_src_per_tc_ticks": json.dumps(s_d), "gem5_tgt_per_tc_ticks": json.dumps(t_d),
            "gem5_src_excluded": excl.get((pid, sid)), "gem5_tgt_excluded": excl.get((pid, tid)),
            "gem5_src_status": status.get((pid, sid)), "gem5_tgt_status": status.get((pid, tid)),
            "gem5_src_sum_ticks_all": cs, "gem5_tgt_sum_ticks_all": ct,
            "common_timed_case_ids": json.dumps(common_all),
            "correctness_kept_case_ids": json.dumps(sorted(S[pid]["kept"])),
            "usable_case_ids": json.dumps(usable),
            "n_common_timed": len(common_all), "n_kept": len(S[pid]["kept"]),
            "n_usable": len(usable), "n_cases_available": S[pid]["ncases"],
            "src_n_fail": S[pid]["progs"][sid]["n_fail"],
            "tgt_n_fail": S[pid]["progs"][tid]["n_fail"], "pair_n_fail": len(pair_fail),
            "our_speedup_usable": our_sp_u, "our_speedup_all": our_sp_all,
            "ceiling_speedup": ceil,
            "flag_low_headroom": bool(ceil < LOW_HEADROOM) if ceil == ceil else False,
            "flag_single_case": (S[pid]["ncases"] == 1),
            "flag_empty_kept": (len(S[pid]["kept"]) == 0),
        }
        row.update(_pie_fields(x, PIE_COMMON))
        row.update(_pie_fields(x, PIE_TRVAL if split != "test" else PIE_TEST))
        if split == "test" and x.get("tests") is not None:
            row["pie_tests_case_ids"] = json.dumps([int(i) for i in x["tests"]])
        pair_rows.append(row)

        # bysrc accumulate: oracle = good+timed tgt with max usable speedup
        key = (pid, sid)
        cand = our_sp_u if our_sp_u is not None else -1
        cur = by.get(key)
        if cur is None or cand > cur["_best"]:
            b = {
                "problem_id": pid, "src_id": sid, "split": split,
                "language": x.get("language", "cpp"), "src_code": x.get("src_code"),
                "gem5_src_per_tc_ticks": json.dumps(s_d), "gem5_src_excluded": excl.get((pid, sid)),
                "gem5_src_status": status.get((pid, sid)), "gem5_src_sum_ticks_all": cs,
                "oracle_tgt_id": tid, "oracle_tgt_code": x.get("tgt_code"),
                "gem5_oracle_tgt_per_tc_ticks": json.dumps(t_d),
                "gem5_oracle_tgt_status": status.get((pid, tid)),
                "our_speedup_usable": our_sp_u, "our_speedup_all": our_sp_all,
                "timed_case_ids": json.dumps(sorted(timed_wall(s_d))),
                "correctness_kept_case_ids": json.dumps(sorted(S[pid]["kept"])),
                "usable_case_ids": json.dumps(sorted(keff(pid) & timed_wall(s_d))),
                "n_timed": len(timed_wall(s_d)), "n_kept": len(S[pid]["kept"]),
                "n_usable": len(keff(pid) & timed_wall(s_d)), "n_cases_available": S[pid]["ncases"],
                "src_n_fail": S[pid]["progs"][sid]["n_fail"],
                "ceiling_speedup": ceil,
                "flag_low_headroom": bool(ceil < LOW_HEADROOM) if ceil == ceil else False,
                "flag_single_case": (S[pid]["ncases"] == 1),
                "flag_empty_kept": (len(S[pid]["kept"]) == 0),
                "pie_speedup": x.get("speedup"),
                "pie_src_agg_runtime": x.get("src_agg_runtime"),
                "pie_oracle_tgt_agg_runtime": x.get("tgt_agg_runtime"),
                "pie_fastest_agg_runtime": x.get("fastest_agg_runtime"),
                "pie_src_verdict": x.get("src_status"), "pie_oracle_tgt_verdict": x.get("tgt_status"),
                "_best": cand,
            }
            if split != "test":
                b["pie_fastest_agg_runtime_updated"] = x.get("fastest_agg_runtime_updated")
                b["pie_src_reward_updated"] = x.get("src_reward_updated")
                b["pie_src_reward_updated_pct_bin"] = x.get("src_reward_updated_pct_bin")
            else:
                b["pie_n_tests"] = x.get("n_tests"); b["pie_test_accuracy"] = x.get("test_accuracy")
                b["pie_src_code_accuracy"] = x.get("src_code_accuracy")
                b["pie_src_code_compilation"] = x.get("src_code_compilation")
                b["pie_src_code_tc2time"] = json.dumps(x.get("src_code_tc2time")) if x.get("src_code_tc2time") else None
                b["pie_oracle_tgt_code_tc2time"] = json.dumps(x.get("tgt_code_tc2time")) if x.get("tgt_code_tc2time") else None
                b["pie_fastest_code"] = x.get("fastest_code")
                b["pie_fastest_code_len"] = x.get("fastest_code_len")
                b["pie_fastest_code_runtime"] = x.get("fastest_code_runtime")
                b["pie_fastest_code_tc2time"] = json.dumps(x.get("fastest_code_tc2time")) if x.get("fastest_code_tc2time") else None
                b["pie_tests_case_ids"] = json.dumps([int(i) for i in x["tests"]]) if x.get("tests") is not None else None
            by[key] = b

    # ---- per-problem FASTEST program (pool = programs in THIS dataset) ----
    # Selection metric = mean per-case ticks over (kept & own timed-wall) = the
    # per-problem usable basis, normalized so candidates with different timed-case
    # sets compare apples-to-apples (per-case time is near-uniform within a program).
    def pick_fastest(pid, cand_ids):
        kept = S[pid]["kept"] if pid in S else set()
        best, best_mean = None, None
        for cid in cand_ids:
            d = per.get((pid, cid), {})
            tw = timed_wall(d)
            basis = (kept & tw) or tw
            if not basis:
                continue
            mean = sum(d[c] for c in basis) / len(basis)
            if best_mean is None or mean < best_mean:
                best, best_mean = cid, mean
        return best

    def fastest_fields(pid, s_d, usable, fid):
        if fid is None:
            return {"gem5_fastest_id": None, "gem5_fastest_per_tc_ticks": None,
                    "gem5_fastest_usable_case_ids": json.dumps([]),
                    "gem5_fastest_sum_ticks_usable": None, "our_fastest_speedup": None}
        fd = per.get((pid, fid), {})
        U = sorted(c for c in usable if c in fd and fd[c] <= WALL)
        fsum = sum(fd[c] for c in U)
        ssum = sum(s_d[c] for c in U if c in s_d)
        return {"gem5_fastest_id": fid, "gem5_fastest_per_tc_ticks": json.dumps(fd),
                "gem5_fastest_usable_case_ids": json.dumps(U),
                "gem5_fastest_sum_ticks_usable": fsum,
                "our_fastest_speedup": (ssum / fsum) if fsum else None}

    pair_cands, bysrc_cands = {}, {}
    for r in pair_rows:
        pair_cands.setdefault(r["problem_id"], set()).update([r["src_id"], r["tgt_id"]])
    for b in by.values():
        bysrc_cands.setdefault(b["problem_id"], set()).update([b["src_id"], b["oracle_tgt_id"]])
    fp = {pid: pick_fastest(pid, c) for pid, c in pair_cands.items()}
    fb = {pid: pick_fastest(pid, c) for pid, c in bysrc_cands.items()}
    for r in pair_rows:
        pid = r["problem_id"]
        r.update(fastest_fields(pid, per.get((pid, r["src_id"]), {}),
                                json.loads(r["usable_case_ids"]), fp.get(pid)))
    for b in by.values():
        pid = b["problem_id"]
        b.update(fastest_fields(pid, per.get((pid, b["src_id"]), {}),
                                json.loads(b["usable_case_ids"]), fb.get(pid)))
    nfp = sum(1 for r in pair_rows if r["gem5_fastest_id"] is not None)
    print(f"  fastest resolved: pairs {nfp}/{len(pair_rows)}, "
          f"bysrc {sum(1 for b in by.values() if b['gem5_fastest_id'] is not None)}/{len(by)}")

    os.makedirs(OUT, exist_ok=True)
    pdf = pd.DataFrame(pair_rows)
    pdf.to_parquet(f"{OUT}/pairs_{split}.parquet", index=False)
    bdf = pd.DataFrame([{k: v for k, v in r.items() if k != "_best"} for r in by.values()])
    bdf.to_parquet(f"{OUT}/bysrc_{split}.parquet", index=False)
    return pdf, bdf


def main():
    rows = []
    for sp in SPLITS:
        led = {k: 0 for k in ("pairs_seen", "pairs_kept", "drop_degenerate", "drop_missing_screen",
                              "drop_badapple", "drop_hard_excl", "drop_pair_criterion", "drop_zero_usable")}
        pdf, bdf = build(sp, led)
        print(f"\n==== {sp} drop ledger ====")
        for k, v in led.items():
            print(f"  {k:22s}: {v}")
        print(f"  -> pairs_{sp}={len(pdf)} rows, bysrc_{sp}={len(bdf)} rows")
        if len(pdf):
            r = pdf.dropna(subset=["our_speedup_usable", "pie_speedup"])
            if len(r) > 2:
                c = np.corrcoef(np.log(r.our_speedup_usable.clip(lower=1e-6)),
                                np.log(r.pie_speedup.clip(lower=1e-6)))[0, 1]
                print(f"  our_speedup_usable vs pie_speedup  log-corr={c:.4f}  median ratio="
                      f"{np.median(r.our_speedup_usable / r.pie_speedup):.4f}")
        rows.append((sp, len(pdf), len(bdf)))
    with open(f"{OUT}/README.txt", "w") as f:
        f.write("PIE RL dataset v2 (filtered, reward-agnostic) 2026-06-12.\n"
                "pairs_<split>: 1 row/surviving (src,tgt). bysrc_<split>: 1 row/unique (problem,src), oracle=best good tgt.\n"
                f"DROPS: compile-fail, bad-apple(n_fail>max(5,5%)), gem5 hard-excl{sorted(HARD_EXCL)}, "
                f"degenerate{sorted(DEGENERATE)}, pair-criterion(|src.fails U tgt.fails|>thr), zero-usable.\n"
                f"WALL={WALL:.3g} ticks (strict PIE). usable=kept(or all if empty) & both-timed & ticks<=WALL.\n"
                "*_per_tc_ticks JSON{idx:ticks ps}; subsample K(-1=all) from usable_case_ids at runtime.\n"
                "pie_* = carried verbatim from PIE jsonl. flags low_headroom/single_case/empty_kept are NON-dropping.\n")
    print(f"\nfinal: {rows}\nwrote {OUT}/README.txt")


if __name__ == "__main__":
    main()
