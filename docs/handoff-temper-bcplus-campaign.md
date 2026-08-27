# Handoff — TEMPER + the BrowseComp-Plus campaign

Written 2026-08-24 for whoever picks up the optimizer work. Ledger of a
two-day campaign: the literature pass, the TEMPER mechanisms, the
BrowseComp-Plus rig, and two single-seed suites (gpt-oss-120b, toast-1)
against `gepa` 0.1.4. Everything referenced is **uncommitted working state on
`v1-final-sprint`**. Companion docs, in reading order:
[handoff-ratchet-campaign.md](handoff-ratchet-campaign.md) (the prior
HotpotQA campaign this builds on) → [temper-design.md](temper-design.md)
(literature synthesis, algorithm, pre-registered predictions, both results
tables) → this file (the operational handoff).

## 1. What happened, in one paragraph

The HotpotQA campaign ended at "equal mean, dominant robustness" because the
task was saturated. A four-agent literature sweep (through mid-2026) mapped
GEPA's documented weaknesses to mechanisms with evidence; four of them became
**TEMPER** — default-off knobs on the RATCHET engine (`optimizer/ratchet.rs`)
— and a new rig, **`benchmarks/bcplus/`**, was built on BrowseComp-Plus
(fixed 100K-doc corpus, agentic search, real headroom). Outcome of the
single-seed suites: **on gpt-oss-120b, RATCHET beat reference GEPA by
+14.2pp holdout [+0.067, +0.225], McNemar p = .012 — the first raw-mean
scalp in either campaign — while GEPA's claimed val gain (.28→.415) was
exposed as 100% overfit (19 verbatim train answers, 12.7K chars, +0 on
holdout)**. On toast-1 the ranking flipped: TEMPER was the only arm above
the seed (+5pp vs GEPA, n.s.) and its variant-tempered champion race went
**2-for-2 across students on correct ship/no-ship decisions**. The eval rig
itself is the other deliverable: it separates methods at n=60 single-seed
where hotpot separated nothing, and its paraphrase audit measures
"overfits the given examples" directly.

## 2. The code (all in `crates/dspy-rs/`)

### TEMPER — knobs on `src/optimizer/ratchet.rs` (all default OFF; RATCHET v2 byte-identical)

| Knob | TEMPER preset (`temper(variant_pairs)`, set `prompt_model` after) | Attacks |
|---|---|---|
| `parent_selection: Lexicase` + `lexicase_epsilon` (.05) | ε-lexicase over discriminative val columns, tie broken ∝ 1/(1+children) | Pareto sole-solver collapse (gepa PR #432) behind the measured anti-scaling |
| `batch_weighting: Learnability` | minibatches ∝ p(1−p) over the pool (Efraimidis–Spirakis, least-used-first tiers) | reflection wasted on never-solvable columns |
| `variant_pairs: Vec<(src, variant)>` | paraphrase twins as trainset rows, excluded from every batch/race, voting only in the endgame champion race | phrasing/example memorization the verbatim leak guard can't see |
| `length_cap: Some(1500)` | one shortening retry, then `TrialOutcome::LengthSkipped` | bloat↔overfit correlation (ours + Decagon) |

New `RatchetResult` fields: `length_retries`, `length_skipped`,
`variant_selection_cols` (serde-defaulted). New unit tests cover lexicase
preference/novelty, learnability weighting/epoch coverage, and the preset.
`cargo test -p dspy-rs`: all green, no API needed.

### The rig binary — `examples/31-bcplus.rs`

One binary, two modes. `BCPLUS_MODE=optimize` runs an arm
(`BCPLUS_OPT=temper|ratchet|gepa`); `BCPLUS_MODE=serve` is a JSONL
stdin/stdout rollout server that **backs both the Python reference arm and
every eval script** — all arms share byte-identical agent, retrieval, and
judging (tighter parity than hotpot, where Python re-implemented the
student). Agent = one tooled `Predict` (`search`, `open_doc`, optional
`final_answer` stop tool), max turns env-set. Metric = official BrowseComp
judge template (fixed model), shaped for optimizers: correct → 1.0 else
0.3 × gold-doc recall; evals report binary accuracy only. Students:
`openrouter:*` or `mxbai:*` (OpenAI-compatible base_url). Env knobs are
listed in the file header.

## 3. The rig data + scripts (`benchmarks/bcplus/`, own `.venv` via uv)

- **Data**: `fetch_data.py` decrypts Tevatron/browsecomp-plus (SHA256-XOR,
  public canary; corpus is plaintext) → seed-0 shuffle → **`holdout100`
  (FROZEN — never touched to date)**, `devholdout` 60, `train` 150, `val` 30,
  `val12`, `reserve` 490. `data/` is gitignored (canary contract).
- **Search**: `build_index.py` (bm25s, 747MB) + `search_server.py` :8123 —
  `/search` top-5 × 512-word snippets, `/doc`. Deterministic, zero API cost.
- **Variants**: `make_variants.py` — teacher-paraphrased, clue-preserving
  twins: `variants.jsonl` (24, TEMPER T3) and `devholdout_variants.jsonl`
  (60, the paraphrase audit).
- **Reference arm**: `run_python_gepa.py` (gepa 0.1.4 + litellm in the venv)
  over `common.py`'s `ServeClient`.
- **Eval**: `eval_holdout.py` — all arms one fresh session, paired
  per-question bootstrap + exact McNemar, train-answer leak audit, winner
  length, tools/q, gold recall, paraphrase gap; `--ref <prefix>` pools
  X_sN vs ref_sN across seeds. Raw rows always land in
  `eval_*_results.jsonl`; every printed number is recomputable.

## 4. Results ledger (both single-seed; see temper-design.md §6 for full tables)

| | gpt-oss-120b (devholdout 60×2) | toast-1 (same) |
|---|---|---|
| seed | .308 | .375 |
| `gepa` 0.1.4 | .317 — **19 leaked train answers, 12.7K chars, val gain was 100% overfit** | .358 — shipped its own seed (14 candidates, none beat val) |
| RATCHET | **.458 — +14.2pp vs GEPA [+.067,+.225], McNemar p=.012** | .358 — child ≈ seed (val .433 = curse) |
| TEMPER | .308 — shipped seed; demoted argmax confirmed +0.0 (**correct refusal**) | **.408 — only arm above seed** (+5pp vs GEPA, 7–2, n.s.); demoted argmax .325 |
| paraphrase gap | RATCHET +.125 *sig*; others ≈ 0 | GEPA/RATCHET ≈ +.08; TEMPER-argmax −.03 |

Interpretation that survives both tables: the **rig works** (separated
methods where hotpot couldn't; byte-identical arms ≈ noise floor .00–.017);
**RATCHET's search is the strongest but phrasing-sensitive**; **TEMPER's
variant-champion selection is 2-for-2 correct** while its search is
over-constrained (the length cap twice `LengthSkipped` the best lineage's
rewrites on gpt-oss); reference **GEPA never produced a real gain on either
student**. All method deltas are single-seed: direction, not claims.
Novelty status: components have scattered prior art (CAPO racing, REMO
memory, OPTS operator bandits, open gepa PRs #308/#432); the assembled
system + leak guard + fresh-column champion + variant tempering appear
unpublished as of mid-2026 (4-agent sweep, not a formal prior-art search).

## 5. Engineering gotchas (each cost a debugging cycle)

1. **Laptop sleep froze a 12h suite** — logs looked like a crawl; timestamp
   gaps (07→10, 14→16 UTC) told the truth. All long runs are now launched
   under `caffeinate -i`. Check gaps before diagnosing slowness.
2. **Killing a waiter shell does not kill its spawned child** — a duplicate
   eval ran ~275 redundant rollouts. Kill the python PID, not just the zsh.
3. `source .env` fails silently into exit 127 when the shell cwd drifted to
   `benchmarks/bcplus` — always `cd` absolute first.
4. **mxbai API**: `submit_answer` is a reserved hosted-tool name (422);
   long tool conversations (~46 messages) return 500 `internal_error`
   (context overflow, should be 400) — both worth filing. toast-1 emits
   **bare-string stop-tool args**, which the typed coercion rejects; the
   example's raw-text parse fallback handles it but **loses tool metadata**,
   so tools/q + gold-recall under-report for toast-1 (accuracy unaffected).
5. toast-1 must be given a stop tool (it never emits a final text turn) and
   hates prescriptive prompts (hand-written "strong" collapsed it 47→20%).
6. gepa 0.1.4 buffers stdout hard; judge progress by `runs/*/run_log.txt`
   mtime, and launch with `PYTHONUNBUFFERED=1`.
7. Exact aggregate ties between different arms are real (43/120 vs 43/120
   with 18/60 per-question disagreements) — check the raw rows before
   suspecting the eval.

## 6. Spend ledger

≈ **$100–140 total** (judge-dominated; teacher second). Includes ~$30–45 of
avoidable burn: the duplicate eval, a Phase-C launch on a misread approval
(killed on correction, no winners), and the sleep-frozen toast suite's
partial first attempt. For calibration, the entire hotpot campaign was
$40–60; bcplus runs hotter because every rollout is a multi-turn tool agent
plus a frontier judge call — the **judge model is the #1 cost lever** (swap
requires re-validating agreement and holding it fixed within a campaign).

## 7. Open threads for the next owner

1. **Phase C (the claim)**: 2 more seeds × 3 arms + the pre-registered
   holdout100 eval (~$150). holdout100 is still frozen; commands in
   temper-design.md §5. Explicitly *not* green-lit — the user declined once;
   ask, don't assume.
2. **TEMPER search aggressiveness**: raise/remove `length_cap` for bootstrap
   regimes (it killed winning-lineage rewrites), or cap only after first
   promotion. Ablation knobs already exist.
3. **RATCHET + TEMPER hybrid**: RATCHET's search with the variant-champion
   endgame — the 2-for-2 selection result says that combination is the
   obvious v3. One builder config, no new code.
4. **Contrastive variation (T5 candidate)**: champion clause-ablation
   (race single-clause removals on fresh columns, ~30–80 rollouts) and
   decompose-and-test for multi-edit wins — makes ledger lessons causal
   instead of teacher-claimed. Design sketched in conversation 2026-08-23.
5. **RATCHET's paraphrase gap** (+.125 sig on gpt-oss): score finalists on
   variant columns in *every* method (TEMPER's T3 as a default), or add a
   paraphrase term to champion racing.
6. **toast-1 native rig variant**: corpus into a mxbai store, server-side
   `store_search` (example 27 pattern) — tests their model in its trained
   scaffold instead of ours; also fixes the metadata under-reporting.
7. Engine parity items still open from the hotpot handoff §9: `on_rollout_error`,
   traces on cache hits, dominated-candidate removal, dead `num_trials`.

## 8. Artifact map (ALL uncommitted on `v1-final-sprint`)

| Path | What |
|---|---|
| `crates/dspy-rs/src/optimizer/ratchet.rs` | TEMPER knobs + `temper()` preset + tests |
| `crates/dspy-rs/examples/31-bcplus.rs` | rig binary: optimize + serve modes, mxbai student, stop tool, parse fallback |
| `benchmarks/bcplus/` | rig: fetch/build/search/variants/common/run_python_gepa/eval_holdout; data (gitignored) + index (gitignored) |
| winners | `{temper,ratchet}_winner*_s0.txt`, `python_winner_s0.txt` (gpt-oss); `toast_*_winner*_s0.txt`, `python_winner_toast_s0.txt` |
| raw evals | `eval_phaseB_results.jsonl` (gpt-oss), `eval_toastB_results.jsonl` (toast-1), pilot files |
| logs | `*_run_s*.log`, `eval_*.log` |
| `docs/temper-design.md` | design + literature synthesis + P1–P5 + both results tables |
| `docs/docs/components/optimizers.mdx` | user-facing TEMPER section |

Suggested commit split: (1) `optimizer/` + tests + mdx, (2) `benchmarks/bcplus/`
scripts (+ winners; logs optional), (3) `docs/*.md`.

## 9. Validation checklist

0. `cargo test -p dspy-rs` — green, no API.
1. `search_server.py` up → example 31 smoke: `BCPLUS_BUDGET=40 BCPLUS_ITERS=2
   BCPLUS_VAL_FILE=val12.jsonl` (~$0.50) — expect a clean result block and
   seed val ≈ .25–.35.
2. Any new comparison: fresh session, all arms together, paired, via
   `eval_holdout.py`; check the byte-identical-arm delta first (noise floor).
3. Leak audit and paraphrase gap are printed per arm — a GEPA-reference run
   should reproduce the leak signature immediately (19/winner at budget 600
   on gpt-oss).
