"""Harmonize schema across splits and upload the two standalone datasets to HF.

Builds a SELF-CONTAINED dataset card (README.md) with a full field dictionary so a
downstream agent that only has the HF repo (no chat history) can understand every column.
Token from env HF_TOKEN (never written to the repo). Public repos under stablegradients/.
  HF_TOKEN=... python upload_hf.py
"""
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

SRC = "/work/nvme/bgtv/sramasubramanian/pie_rl_dataset"
STAGE = f"{SRC}/hf_staging"
USER = "stablegradients"
TOKEN = os.environ["HF_TOKEN"]
SPLITS = {"train": "train", "val": "validation", "test": "test"}
REPOS = {"pairs": "pie-gem5-pairs", "bysrc": "pie-gem5-bysrc"}

# ---------------------------------------------------------------------------
# Field dictionary. One line per column. Used to render the card's field table.
# `gem5_*` / `our_*` / `*_case_ids` / counts / flags are OURS (this re-timing).
# `pie_*` are carried verbatim from the official PIE jsonl.
# Columns absent from a given dataset are simply skipped when the table is rendered.
# ---------------------------------------------------------------------------
FIELD_DOCS = {
    # --- identity ---
    "problem_id":   "PIE/online-judge problem id. Groups every submission for one problem; the unit the correctness-intersection and `tests` benchmark are defined over.",
    "src_id":       "Submission id of the SOURCE (slower, to-be-optimized) program.",
    "tgt_id":       "Submission id of the TARGET (faster) program in this pair. [pairs only]",
    "oracle_tgt_id":"Submission id of the chosen ORACLE target = the surviving target with the highest `our_speedup_usable` for this source. [bysrc only]",
    "split":        "Dataset split: train / validation / test.",
    "language":     "Programming language. Always `cpp` (C++).",
    # --- code ---
    "src_code":         "Full C++ source of the source program.",
    "tgt_code":         "Full C++ source of the target program. [pairs only]",
    "oracle_tgt_code":  "Full C++ source of the oracle target program. [bysrc only]",
    # --- OUR gem5 timing (the headline) ---
    "gem5_src_per_tc_ticks":        "JSON {case_idx: ticks}. gem5 picoseconds PER TEST CASE for the SRC program. Seconds = ticks * 1e-12. Holds EVERY case gem5 timed (incl. cases later discarded by correctness/wall filters) — select cases via `usable_case_ids`.",
    "gem5_tgt_per_tc_ticks":        "JSON {case_idx: ticks} for the TGT program (same convention). [pairs only]",
    "gem5_oracle_tgt_per_tc_ticks": "JSON {case_idx: ticks} for the oracle target (same convention). [bysrc only]",
    "gem5_fastest_per_tc_ticks":    "JSON {case_idx: ticks} for the FASTEST program of this problem (see `gem5_fastest_id`); full per-case gem5 timing, same convention.",
    "gem5_src_excluded":    "JSON list of case indices gem5 could NOT time for the SRC program (hang / crash / over compute budget during the sweep).",
    "gem5_tgt_excluded":    "JSON list of case indices gem5 could NOT time for the TGT program. [pairs only]",
    "gem5_src_status":      "gem5 sweep status for the src program: `ok` (all cases timed) or `forkloop_partial` (some cases excluded). Hard-fail statuses (startup_fail / native_all_excluded) are already dropped.",
    "gem5_tgt_status":      "gem5 sweep status for the tgt program. [pairs only]",
    "gem5_oracle_tgt_status":"gem5 sweep status for the oracle target. [bysrc only]",
    "gem5_fastest_id":      "Submission id of the FASTEST surviving program for this problem, chosen among programs that appear as a src/tgt in THIS dataset, ranked by summed gem5 ticks over the per-problem common usable case basis. The practical oracle ceiling.",
    "gem5_src_sum_ticks_all":       "Sum of SRC ticks over `common_timed_case_ids` (cases timed on both sides). Numerator of `our_speedup_all`.",
    "gem5_tgt_sum_ticks_all":       "Sum of TGT ticks over `common_timed_case_ids`. Denominator of `our_speedup_all`. [pairs only]",
    "gem5_fastest_sum_ticks_usable":"Sum of FASTEST-program ticks over `usable_case_ids`. Denominator of `our_fastest_speedup`.",
    # --- case-id sets (JSON int lists) ---
    "common_timed_case_ids":    "Case indices gem5 timed on BOTH src and tgt. [pairs only]",
    "timed_case_ids":           "Case indices gem5 timed for the src (within the 2-min wall). [bysrc only]",
    "correctness_kept_case_ids":"Case indices passed by EVERY good (non-bad-apple, compile-ok) program of the problem = the correctness intersection. Problem-level, identical across rows of a problem.",
    "usable_case_ids":          "THE case set to use for the reward = correctness-kept ∩ both-sides-timed ∩ ticks≤1.4e10 (pairs) / kept ∩ src-timed ∩ wall (bysrc). Falls back to all-timed when the kept set is empty (single-case problems).",
    # --- counts ---
    "n_common_timed":   "len(common_timed_case_ids). [pairs only]",
    "n_timed":          "len(timed_case_ids). [bysrc only]",
    "n_kept":           "len(correctness_kept_case_ids).",
    "n_usable":         "len(usable_case_ids). The effective case count behind the speedup.",
    "n_cases_available":"Total test cases that exist for the problem in the merged test-case set.",
    # --- correctness provenance ---
    "src_n_fail":   "# cases the SRC program failed in the native correctness screen (line-wise + 1e-3 float comparator). 0 = passed all.",
    "tgt_n_fail":   "# cases the TGT program failed in the native correctness screen. [pairs only]",
    "pair_n_fail":  "Size of (src.fails ∪ tgt.fails) — # cases failed by at least one side of the pair. [pairs only]",
    # --- derived speedups / headroom ---
    "our_speedup_usable":   "**Reference speedup** = sum(src ticks) / sum(tgt ticks) over `usable_case_ids`. Case-set-invariant; this is what training should reproduce.",
    "our_speedup_all":      "sum(src)/sum(tgt) over `common_timed_case_ids` (no correctness/wall filtering). Provenance/debug.",
    "our_fastest_speedup":  "sum(src ticks)/sum(fastest ticks) over `usable_case_ids`. Speedup of the src against the FASTEST program of the problem = the practical oracle ceiling for this source.",
    "ceiling_speedup":      "Mean per-case src ticks / startup floor (192M ticks). Theoretical max speedup attainable for this src given the gem5 startup floor. Drives `flag_low_headroom`.",
    # --- non-dropping flags ---
    "flag_low_headroom":"True if ceiling_speedup < 1.5 — little room to optimize (kept, not dropped).",
    "flag_single_case": "True if the problem has only one test case.",
    "flag_empty_kept":  "True if the correctness intersection was empty (single-case false-alarm problems; usable then falls back to all-timed).",
    # --- PIE carried verbatim (null where PIE didn't release it for that split) ---
    "pie_src_agg_runtime":          "PIE's published aggregate wall-clock runtime (seconds) of the src program (their gem5 measurement).",
    "pie_tgt_agg_runtime":          "PIE's published aggregate runtime (s) of the tgt program. [pairs only]",
    "pie_oracle_tgt_agg_runtime":   "PIE's published aggregate runtime (s) of the oracle tgt. [bysrc only]",
    "pie_speedup":                  "PIE's published src/tgt speedup. [train/val only]",
    "pie_fastest_agg_runtime":      "PIE's published aggregate runtime (s) of the problem's fastest solution.",
    "pie_src_verdict":              "Online-judge verdict for the src (e.g. Accepted). [train/val only]",
    "pie_tgt_verdict":              "Online-judge verdict for the tgt. [pairs, train/val only]",
    "pie_oracle_tgt_verdict":       "Online-judge verdict for the oracle tgt. [bysrc, train/val only]",
    "pie_fastest_agg_runtime_updated":"PIE's re-measured fastest-solution agg runtime. [train/val only]",
    "pie_src_reward_updated":       "PIE's annotated reward for the src. [train/val only]",
    "pie_target_reward_updated":    "PIE's annotated reward for the tgt. [pairs, train/val only]",
    "pie_src_reward_updated_pct_bin":"PIE's difficulty/percentile bin for the src reward. [train/val only]",
    "pie_target_reward_updated_pct_bin":"PIE's difficulty/percentile bin for the tgt reward. [pairs, train/val only]",
    "pie_n_tests":                  "# benchmark test cases PIE used for this problem. [test only]",
    "pie_test_agg_runtime":         "PIE's agg runtime (s) over its benchmark `tests` cases. [test only]",
    "pie_test_accuracy":            "PIE's reported accuracy over the benchmark cases. [test only]",
    "pie_test_compilation":         "PIE's reported compilation success over the benchmark. [test only]",
    "pie_src_code_accuracy":        "PIE's src accuracy over the benchmark cases. [test only]",
    "pie_tgt_code_accuracy":        "PIE's tgt accuracy. [pairs, test only]",
    "pie_src_code_compilation":     "PIE's src compilation flag. [test only]",
    "pie_tgt_code_compilation":     "PIE's tgt compilation flag. [pairs, test only]",
    "pie_src_code_runtime":         "PIE's src agg runtime (s) on the benchmark. [pairs, test only]",
    "pie_tgt_code_runtime":         "PIE's tgt agg runtime (s) on the benchmark. [pairs, test only]",
    "pie_fastest_code":             "Full C++ source of the fastest solution PIE identified for the problem. [test only]",
    "pie_fastest_code_len":         "Token/char length of pie_fastest_code as PIE recorded it. [test only]",
    "pie_fastest_code_runtime":     "PIE's agg runtime (s) of pie_fastest_code. [test only]",
    "pie_fastest_code_accuracy":    "PIE's accuracy of pie_fastest_code. [pairs, test only]",
    "pie_fastest_code_compilation": "PIE's compilation flag of pie_fastest_code. [pairs, test only]",
    "pie_src_code_tc2time":         "PIE's OWN per-test-case times for the src, JSON {idx: seconds}. [test only]",
    "pie_tgt_code_tc2time":         "PIE's OWN per-test-case times for the tgt, JSON {idx: seconds}. [pairs, test only]",
    "pie_oracle_tgt_code_tc2time":  "PIE's OWN per-test-case times for the oracle tgt, JSON {idx: seconds}. [bysrc, test only]",
    "pie_fastest_code_tc2time":     "PIE's OWN per-test-case times for the fastest solution, JSON {idx: seconds}. [test only]",
    "pie_tests_case_ids":           "JSON list of the benchmark case indices PIE released for this test problem (subset of 0..n_cases_available-1). [test only]",
    "gem5_fastest_usable_case_ids": "Usable case set for the fastest program = kept ∩ fastest-timed ∩ wall.",
}

# section grouping + ordering for the rendered table
GROUPS = [
    ("Identity", ["problem_id", "src_id", "tgt_id", "oracle_tgt_id", "split", "language"]),
    ("Code", ["src_code", "tgt_code", "oracle_tgt_code"]),
    ("Our gem5 per-test-case timing", [
        "gem5_src_per_tc_ticks", "gem5_tgt_per_tc_ticks", "gem5_oracle_tgt_per_tc_ticks",
        "gem5_fastest_id", "gem5_fastest_per_tc_ticks", "gem5_fastest_usable_case_ids",
        "gem5_fastest_sum_ticks_usable",
        "gem5_src_excluded", "gem5_tgt_excluded",
        "gem5_src_status", "gem5_tgt_status", "gem5_oracle_tgt_status",
        "gem5_src_sum_ticks_all", "gem5_tgt_sum_ticks_all"]),
    ("Case-id sets (JSON int lists) + counts", [
        "common_timed_case_ids", "timed_case_ids", "correctness_kept_case_ids", "usable_case_ids",
        "n_common_timed", "n_timed", "n_kept", "n_usable", "n_cases_available",
        "src_n_fail", "tgt_n_fail", "pair_n_fail"]),
    ("Derived speedups + flags", [
        "our_speedup_usable", "our_speedup_all", "our_fastest_speedup", "ceiling_speedup",
        "flag_low_headroom", "flag_single_case", "flag_empty_kept"]),
    ("Carried verbatim from PIE (`pie_*`)", [
        "pie_src_agg_runtime", "pie_tgt_agg_runtime", "pie_oracle_tgt_agg_runtime", "pie_speedup",
        "pie_fastest_agg_runtime", "pie_src_verdict", "pie_tgt_verdict", "pie_oracle_tgt_verdict",
        "pie_fastest_agg_runtime_updated", "pie_src_reward_updated", "pie_target_reward_updated",
        "pie_src_reward_updated_pct_bin", "pie_target_reward_updated_pct_bin",
        "pie_n_tests", "pie_test_agg_runtime", "pie_test_accuracy", "pie_test_compilation",
        "pie_src_code_accuracy", "pie_tgt_code_accuracy", "pie_src_code_compilation",
        "pie_tgt_code_compilation", "pie_src_code_runtime", "pie_tgt_code_runtime",
        "pie_fastest_code", "pie_fastest_code_len", "pie_fastest_code_runtime",
        "pie_fastest_code_accuracy", "pie_fastest_code_compilation",
        "pie_src_code_tc2time", "pie_tgt_code_tc2time", "pie_oracle_tgt_code_tc2time",
        "pie_fastest_code_tc2time", "pie_tests_case_ids"]),
]

CARD_HEAD = """---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- code-optimization
- code-performance
- gem5
- pie
pretty_name: {pretty}
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: validation
    path: validation.parquet
  - split: test
    path: test.parquet
---

# {pretty}

C++ program-optimization data derived from the [PIE](https://pie4perf.com/) dataset
(["Learning Performance-Improving Code Edits"](https://arxiv.org/abs/2302.07867)),
**re-timed end-to-end with gem5** (x86 Skylake, syscall-emulation mode) at
**per-test-case granularity**. {unit_line}

This dataset is **reward-agnostic**: it ships the full per-test-case reference timings and
case manifests so a downstream RL / eval pipeline decides at runtime how many cases to use
(`K`) and which reward transform to apply. Nothing is baked in.

> **For an automated agent reading this card:** every column is documented in
> [§ Field dictionary](#field-dictionary) below, including its type, encoding, and which
> splits populate it. Start with [§ How to compute the reward](#how-to-compute-the-reward).

## TL;DR units & encoding
- **Ticks are gem5 picoseconds.** `seconds = ticks * 1e-12`.
- Columns named `*_per_tc_ticks`, `*_tc2time`, `*_case_ids`, `*_excluded` are **JSON strings**
  (a dict `{{case_idx: value}}` or a list `[case_idx, ...]`) — `json.loads` them.
  Case indices are integers `0 .. n_cases_available-1`.
- `gem5_*` / `our_*` / `*_case_ids` / `n_*` / `flag_*` columns are produced by **this re-timing**.
  `pie_*` columns are **carried verbatim** from the official PIE release.

## How to compute the reward
The reward is a **speedup ratio** over a common set of test cases, robust to which cases you
sample:
```python
import json, random
src = json.loads(row["gem5_src_per_tc_ticks"])     # {{idx: ticks}}
tgt = json.loads(row["gem5_tgt_per_tc_ticks"])      # ("oracle" variant in the bysrc dataset)
usable = json.loads(row["usable_case_ids"])         # correctness-kept ∩ both-timed ∩ ticks<=wall
K = 5                                                # runtime choice; -1 / len(usable) = use all
cases = usable if K < 0 else random.sample(usable, min(K, len(usable)))
speedup = sum(src[str(c)] for c in cases) / sum(tgt[str(c)] for c in cases)
```
`usable_case_ids` is already the correct, fully-filtered set — **do not** re-derive correctness
or apply the wall yourself. Always clamp `K` to `len(usable)`; never pad, skip, or zero-reward a
pair that has fewer than `K` usable cases. K=5–10 reproduces the full ~100-case aggregate within
~1–2% (per-case time is near-uniform within a program).

`gem5_fastest_id` / `gem5_fastest_per_tc_ticks` give the **fastest surviving program of the
problem** (chosen among programs present in this dataset, ranked over the per-problem common
usable basis) as an oracle ceiling; `our_fastest_speedup` is src-vs-fastest over `usable_case_ids`.

## Provenance & validation
- Timing engine: **gem5.fast**, X86 SE mode, with fork-from-live startup amortization (one
  startup simulation, COW-forked per case; per-case ticks equal a fresh run, validated r=1.0).
- Validated against PIE's published numbers: per-test-case r=1.0 on the test split (aligned on
  PIE's released `tests` indices); speedup log-correlation ~0.97–0.98, median ratio ~0.99.
- **Caveat — pin your toolchain.** The gem5 startup floor depends on the compiler/glibc/link
  mode (≈13M ticks static ↔ ≈195M ticks dynamic). These reference ticks were produced with one
  fixed toolchain; if you re-time rollouts, use the *same* toolchain or regenerate references,
  otherwise speedups are not comparable.

## Filtering (rows are pre-filtered; bad records are DROPPED, not flagged)
- **Compile-fail** programs dropped (`g++ -O2 -std=gnu++17`).
- **Bad-apple** programs dropped: a submission failing > max(5, 5% of cases) is treated as
  wrong/mislabeled (line-wise + 1e-3 float comparator).
- **gem5 hard-excludes** (startup_fail / all-cases-hang-or-crash) dropped.
- **Case kept** iff every surviving good program of the problem passes it (correctness intersection).
- **Strict 2-min wall**: cases above 1.4e10 ticks dropped from `usable`.
- {pair_rule}
- 3 degenerate problems (empty input+output: p00000, p02068, p00754) dropped.
- Retained NON-dropping flags: `flag_low_headroom`, `flag_single_case`, `flag_empty_kept`.

## Splits & why some columns are null
| split | rows | notes |
|---|---|---|
| train | {ntr} | PIE released reward/difficulty annotations but no benchmark `tests` here. |
| validation | {nva} | same as train. |
| test | {nte} | PIE released benchmark `tests` indices + per-tc reference times here. |

PIE ships **different columns for test vs train/val**, so each row only fills the `pie_*`
columns that apply to its split; the rest are null **by design** (the schema is the union across
splits). Specifically: `pie_*_code_*`, `pie_fastest_code*`, `pie_*_tc2time`, `pie_n_tests`,
`pie_test_*`, `pie_tests_case_ids` are **test-only**; `pie_speedup`, `pie_*_verdict`,
`pie_*_reward_updated*`, `pie_fastest_agg_runtime_updated` are **train/val-only**. None of the
`gem5_*` / `our_*` / `*_case_ids` / `n_*` columns are ever null.

## Field dictionary
Availability: **all** = populated in every split; **test-only** / **train/val-only** = null elsewhere.

"""

CARD_TAIL = """
## Companion dataset
This is one of a pair built from the same timing:
- **pie-gem5-pairs** — one row per official (src, tgt) pair.
- **pie-gem5-bysrc** — one row per unique (problem, src); best surviving target kept as oracle.

Built 2026-06-12. Timing: gem5.fast with fork-from-live startup amortization.
"""

PRETTY = {"pairs": "PIE gem5-timed code optimization (src,tgt pairs)",
          "bysrc": "PIE gem5-timed code optimization (per source program)"}
UNIT = {"pairs": "One row per official (source, target) program pair.",
        "bysrc": "One row per unique (problem, source) program; the best surviving target is kept as an oracle ceiling."}
PAIR_RULE = {"pairs": "**Pair criterion**: drop a pair if |src.fails ∪ tgt.fails| > max(5, 5%).",
             "bysrc": "Source dropped if it fails > max(5, 5%); oracle target must also pass the same bar."}


def avail_label(col, present_all, present_test, present_trval):
    in_te = col in present_test
    in_tr = col in present_trval
    if in_te and in_tr:
        return "all"
    if in_te:
        return "test-only"
    if in_tr:
        return "train/val-only"
    return "—"


def render_field_table(cols_present, present_test, present_trval):
    """Markdown grouped field table for the columns actually present in this dataset."""
    seen = set()
    out = []
    for title, cols in GROUPS:
        rows = []
        for c in cols:
            if c not in cols_present or c in seen:
                continue
            seen.add(c)
            lab = avail_label(c, cols_present, present_test, present_trval)
            doc = FIELD_DOCS.get(c, "(undocumented)")
            rows.append(f"| `{c}` | {lab} | {doc} |")
        if rows:
            out.append(f"### {title}\n\n| field | splits | meaning |\n|---|---|---|\n" + "\n".join(rows))
    # any present column not covered by GROUPS
    leftover = [c for c in cols_present if c not in seen]
    if leftover:
        rows = [f"| `{c}` | {avail_label(c, cols_present, present_test, present_trval)} | "
                f"{FIELD_DOCS.get(c, '(undocumented)')} |" for c in leftover]
        out.append("### Other\n\n| field | splits | meaning |\n|---|---|---|\n" + "\n".join(rows))
    return "\n\n".join(out)


def harmonize(ds):
    """concat all splits -> uniform schema -> write per-split parquet to STAGE/ds/.
    Returns (out_dir, counts, present_test_cols, present_trval_cols)."""
    dfs = {}
    for raw, _ in SPLITS.items():
        dfs[raw] = pd.read_parquet(f"{SRC}/{ds}_{raw}.parquet")
    allcols = []
    for raw in ("train", "val", "test"):
        for c in dfs[raw].columns:
            if c not in allcols:
                allcols.append(c)
    big = pd.concat([dfs[r].reindex(columns=allcols) for r in ("train", "val", "test")],
                    ignore_index=True)
    schema = pa.Table.from_pandas(big, preserve_index=False).schema
    out = f"{STAGE}/{ds}"
    os.makedirs(out, exist_ok=True)
    counts = {}
    for raw, hf in SPLITS.items():
        part = big[big["split"] == raw].reset_index(drop=True)
        tbl = pa.Table.from_pandas(part, schema=schema, preserve_index=False)
        pq.write_table(tbl, f"{out}/{hf}.parquet")
        counts[hf] = len(part)
    # columns genuinely populated (not all-null) per split-group, for the card's availability labels
    test_df = big[big["split"] == "test"]
    trval_df = big[big["split"].isin(["train", "val"])]
    present_test = {c for c in allcols if not test_df[c].isna().all()}
    present_trval = {c for c in allcols if not trval_df[c].isna().all()}
    return out, counts, allcols, present_test, present_trval


def main():
    api = HfApi(token=TOKEN)
    for ds, name in REPOS.items():
        out, counts, cols, present_test, present_trval = harmonize(ds)
        table = render_field_table(cols, present_test, present_trval)
        card = (CARD_HEAD.format(pretty=PRETTY[ds], unit_line=UNIT[ds], pair_rule=PAIR_RULE[ds],
                                 ntr=counts["train"], nva=counts["validation"], nte=counts["test"])
                + table + "\n" + CARD_TAIL)
        open(f"{out}/README.md", "w").write(card)
        repo = f"{USER}/{name}"
        api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
        for fn in ("README.md", "train.parquet", "validation.parquet", "test.parquet"):
            api.upload_file(path_or_fileobj=f"{out}/{fn}", path_in_repo=fn,
                            repo_id=repo, repo_type="dataset")
        print(f"UPLOADED https://huggingface.co/datasets/{repo}  splits={counts}")


if __name__ == "__main__":
    main()
