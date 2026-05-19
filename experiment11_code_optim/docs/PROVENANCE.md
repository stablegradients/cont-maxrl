# PROVENANCE.md — PIE data sources, fingerprints, and build steps

**Date staged:** 2026-05-19 (raw JSONLs were already in place from a prior session; archive downloads + unified-tests build were performed at §14 step 2b on this date).

**Stage host:** DeltaAI compute node `gh152.hsn.cm.delta.internal.ncsa.edu` (ARM Neoverse-V2, 288 cores, 4× GH200).

**Stage root:** `/work/nvme/bgmo/sramasubramanian/pie_data/`

## 1. Raw JSONLs — LearningOpt/pie + madaan/pie-perf schema

Already-staged JSONLs under `raw/`. Schema heterogeneity is intentional and documented:

| File | Schema | Records | Unique problem_ids | SHA-256 |
|------|--------|---------|--------------------|---------|
| `raw/train.jsonl` | LearningOpt/pie (`src_code`/`tgt_code`/`problem_id`, no `tests`) | 77,967 | 1,474 | `114d595edfc4186b8151da983289bd3ad1082931544e14c18bbe121a3d6c810e` |
| `raw/train_hq_only.jsonl` | LearningOpt/pie | 4,085 | 1,474 (same set as `train`) | `5befaa190928d67aa5b5988e986d948fc6a11ac54ce5afa31179a9f2b1fd187c` |
| `raw/train_hq_and_synthetic.jsonl` | LearningOpt/pie | 5,570 | (not used) | `19016968d16244f9bb4ff344ae0eff3aea2e5cf2ef3f314c7c592b1d8b2486df` |
| `raw/train_with_synthetic.jsonl` | LearningOpt/pie | (large) | (not used) | `a5d26ab13438cdaee32e0b0b4add66e75919bf5d800fd5ba813114a4471f0182` |
| `raw/val.jsonl` | LearningOpt/pie | 2,544 | 77 | `6608a303c5c6ab61e7936599f3aa8cbde67bc36007e1a9739d7868665c531dab` |
| `raw/test.jsonl` | madaan/pie-perf (richer; includes `tests` column, `n_tests`, `test_accuracy`, etc.) | 978 | 41 | `e0b1ef26a22c0c60022acd469cffffac91fd55304a8161b333ae5aa51005d10b` |

**Cross-split discipline (empirically verified):**
- `train ∩ val = train ∩ test = val ∩ test = ∅` (problem_id overlap)
- `train_hq_only ⊂ train` (full 1474-problem overlap; `train_hq_only` is a per-problem subset of `train` with fewer slow→fast pairs)

**Roles (locked at GATE A on 2026-05-19):**
- RL training source: **`train.jsonl`** (77,967 records / 1,474 problems → `steps_per_epoch = 305`, `total_training_steps = 3,050`)
- SFT cold-start source: **`train_hq_only.jsonl`** (4,085 records, same 1,474 problems)
- In-training `insitu_eval` source: **`val.jsonl`** (200-problem subset, sampled deterministically at §14 step 4 data prep)
- Post-hoc §8.4 eval source: **`val.jsonl`** (full or sized subset; cost-gated at §13 step 5 probe)

## 2. Project_CodeNet (IBM Cloud Object Storage)

| Attribute | Value |
|-----------|-------|
| Download URL | `https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz` |
| Date downloaded | 2026-05-19 |
| Original size | 8,343,137,473 bytes (7.77 GB) |
| Server `Last-Modified` | Wed, 05 May 2021 21:13:40 GMT |
| Local tarball SHA-256 | not recorded — tarball was deleted post-extraction (user-authorised) to reclaim disk |
| What was extracted | `Project_CodeNet/derived/input_output/` only (selective `tar xzf` of that subtree) |
| Local extracted location | `/work/nvme/bgmo/sramasubramanian/pie_data/Project_CodeNet/derived/input_output/` |
| Extracted contents | `data/{problem_id}/input.txt` + `output.txt` (singular naming per CodeNet README) for 3,867 problems; plus `README.md`, `no_solutions.txt`, `unverified_accepted_solutions.txt` |
| Tests per problem | **exactly 1** ("Only the first sample input and sample output that were found... are included" — CodeNet README) |
| Role in our pipeline | fallback test-case source in `build_unified_test_cases.py` (precedence: Generated → Public → CodeNet). At build time on 2026-05-19, 0 problems needed CodeNet fallback — Generated+Public covered all 1,592 unique problem_ids across the four JSONLs. CodeNet remains on disk as future-proofing. |

## 3. pie-perf "Public test cases" (Google Drive)

| Attribute | Value |
|-----------|-------|
| Download URL | `https://drive.google.com/file/d/1RcUpZMOR8L2xYYWDZx7I0tHFzFgg7COO/view` (resolved via `gdown` 6.0.0) |
| Date downloaded | 2026-05-19 |
| Source attribution | madaan/pie-perf repo README (`raw.githubusercontent.com/madaan/pie-perf/main/README.md`) |
| Local zip | `pie_public_test_cases.zip` (5,923,327 bytes / 5.7 MB) |
| Local zip SHA-256 | `8dbeebaa02b3d7cb462cd848caf8f9a0b9f4674427f6a78c3b8ce8b5d6123135` |
| Drive `Last-Modified` | Tue, 17 Jan 2023 |
| Extracted location | `pie_perf_tests/codenet/public_test_cases/{problem_id}/` |
| Layout | `input.<i>.txt` + `output.<i>.txt` per problem, i = 0..N-1 |
| Coverage | 1,141 problems; mean 2.9 tests/problem (sample 200: min 1, max 7) |
| Role in our pipeline | second precedence in the union build — covers 1,414 of the 1,592 PIE problems |

## 4. pie-perf "Generated test cases" (Google Drive)

| Attribute | Value |
|-----------|-------|
| Download URL | `https://drive.google.com/file/d/1migwX4wpED0gDDxn7gS6q55vWeXIDgId/view` (resolved via `gdown` 6.0.0) |
| Date downloaded | 2026-05-19 |
| Source attribution | madaan/pie-perf README; ultimately derived from `google-deepmind/code_contests` (alphacode) |
| Local zip | `pie_generated_test_cases.zip` (172,528,316 bytes / 165 MB) |
| Local zip SHA-256 | `f856895cf2bf55d732c4756c5c147ad67c2e149da135c21616706f251d99aa39` |
| Drive `Last-Modified` | Tue, 27 Dec 2022 |
| Extracted location | `pie_perf_generated/codenet/generated_test_cases/{problem_id}/` |
| Layout | `input.<i>.txt` + `output.<i>.txt` per problem |
| Coverage | 247 problems; ~100 tests/problem (verified `p02676` has 100 inputs; consistent with `test.jsonl[0]` `n_tests=103` within noise) |
| Role in our pipeline | top precedence in the union build — covers 178 of the 1,592 PIE problems; supplies the bulk of `total_tests=21,758` |

## 5. Unified test-case archive (built by us)

| Attribute | Value |
|-----------|-------|
| Build script | `experiment11_code_optim/precompute/build_unified_test_cases.py` |
| Run date | 2026-05-19 |
| Output root | `/work/nvme/bgmo/sramasubramanian/pie_data/tests/` |
| Layout | `{problem_id}/input.<i>.txt` + `output.<i>.txt` with contiguous `i = 0..n_tests-1`; per-problem source recorded in `tests/MANIFEST.json` |
| Source precedence | Generated → Public → CodeNet (richest first) |
| Total problems covered | 1,592 (=`train ∪ train_hq_only ∪ val ∪ test`, 100% coverage; `missing=0`) |
| Total test cases | 21,758 (mean 13.7 / problem) |
| By source | Generated: 178 problems; Public: 1,414 problems; CodeNet: 0 problems (fallback not needed) |
| Per-split mean tests/problem | train: 13.5 · train_hq_only: 13.5 · val: 13.2 · test: 22.2 |

## 6. End-to-end pipeline validation

The C++ measurement core (`experiment11_code_optim/measurement/core.py`) was validated against the unified archive in `experiment11_code_optim/tests/test_measurement_integration.py`. The test compiles `val.jsonl[0].tgt_code` (a real Atcoder-Accepted submission for `p02587`) and grades it against all 5 Public test cases for that problem under g++ 14.2 -O3 + `perf stat -e instructions`. The strict all-pass gate passed and the perf-measured retired-instruction count was non-zero. All 10 unit + integration tests are green (3.6s total).

## 7. What is NOT recorded here

- The deleted tarball's SHA-256. The HTTP HEAD `Content-Length` (8343137473 bytes) was confirmed pre-download and the byte count of the downloaded file matched; the IBM COS server does not publish a checksum header, so we have no third-party fingerprint to cite. If you need to re-download, the IBM COS URL above is still live (verified 2026-05-19).
- Hashes of every extracted file inside `Project_CodeNet/derived/input_output/`, `pie_perf_tests/`, `pie_perf_generated/`, or `tests/`. The zip / tarball SHA-256s plus the deterministic `build_unified_test_cases.py` script are sufficient to reproduce the unified tree byte-for-byte from the recorded sources.
