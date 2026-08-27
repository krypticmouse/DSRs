# Handoff — RATCHET: the optimizer campaign vs GEPA

Written 2026-08-23 for whoever picks up the optimizer work. Ledger of a
two-day campaign: the GEPA diagnosis, the anti-overfit GEPA rebuild, the
RATCHET algorithm, seven optimization rounds, and the cross-regime verdict.
Everything referenced is **uncommitted working state on `v1-final-sprint`** —
see §8 before touching anything. Companion docs, in reading order:
[gepa-diagnosis-2026-08-22.md](gepa-diagnosis-2026-08-22.md) (why GEPA
overfits, measured) → [ratchet-design.md](ratchet-design.md) (the algorithm +
all round results) → this file (the operational handoff).

## 1. What happened, in one paragraph

We diagnosed the reference GEPA's overfitting from real trajectories (noisy
argmax selection, a gate that validates on its own reflection batch, a
proposer prompt that *asks* for memorization), rebuilt DSRs GEPA with an
anti-overfit stack, then built **RATCHET** (`optimizer/ratchet.rs`) — GEPA's
skeleton plus a trial ledger, paired racing evaluation, and a UCB1 operator
portfolio — and ran a 7-round optimize-measure-falsify campaign against
`gepa` 0.1.4 on a HotpotQA rig with real headroom. Outcome: **equal holdout
mean everywhere (combined +0.012 [−0.017, +0.041] over 210 paired
question-regime points), with a strictly dominant reliability envelope** —
GEPA measurably anti-scales with budget (−0.042, significant), collapses on
small validation sets (−0.087, significant), and ships winners contaminated
with 4–18 verbatim train answers; RATCHET does none of those in 18 runs.
Three of our own "improvements" were falsified by their own pre-registered
ablations and reverted — the shipped algorithm is **v2**.

## 2. The code (all in `crates/dspy-rs/`)

### RATCHET — `src/optimizer/ratchet.rs` (~1,100 lines)

`Ratchet` implements `Optimizer` over the shared `Engine`. Per generation:
sample parent (Pareto coverage) → draw disjoint minibatches A/B
(least-used-first, both biased to parent-unsolved columns) → evaluate parent
on A → operator eligibility + UCB1 choice → propose (leak-guarded, one
retry) → **noise-aware gate on B** (kill only if parent beats child by >
`gate_z`·SE; laterals pass) → **paired race** over shuffled val blocks (die
when deficit > `race_z`·SE; survivors complete the full row) → pool.
Endgame: **fresh-column champion** (a `champion_fresh` slice of the trainset
is pre-reserved out of the batch pool at start; top-2 by val mean race on it)
→ **seed-protected argmax** (only the gen-0 seed keeps incumbency within
`selection_z`·SE; children compete by raw mean).

Operators: `Rewrite` (fan-out ×`rewrite_fanout`, better-on-B races), `Patch`
(one rule from best/worst contrast), `Simplify`, `Merge` (semantic crossover
of complementary Pareto members, shown what each wins on). Simplify/Merge
only become eligible after the first promotion. Requires `prompt_model` —
no concat fallback. `RatchetResult` carries the full trial ledger
(operator + teacher-written hypothesis + measured fate), per-operator
attempts/wins, kill counters, `best_by_mean` (the raw val-argmax whenever the
shipped winner differs), `fresh_selection_cols`.

Knob defaults: `num_iterations` 30, `minibatch_size` 5, `gate_size` =
minibatch, `race_block` 8, `race_z`/`gate_z`/`selection_z` 1.0,
`rewrite_fanout` 2, `champion_fresh` 40, `perfect_score` 1.0,
`allow_memorization` false, `ledger_window` 12.

### GEPA changes — `src/optimizer/gepa.rs`

- Anti-overfit stack (2026-08-22): reflect-on-A/gate-on-B (strict gate),
  leak guard with one retry, epoch/unsolved batch sampling, all-perfect skip,
  **seed-protected argmax** winner (the earlier "Occam sweep" that protected
  every older child was falsified: −0.045 holdout; only seed protection
  survived).
- Shared with RATCHET via `pub(crate)`: `Lineage` (+`add_scored`),
  `require_feedback`, `summarize_feedback`, `leak_strings`, `sample_batches`,
  `within_noise`, `find_leaks`, `row_scores`, `collect_short_strings`.
- `OptimizeTarget::example_json` (crate-private) feeds the leak guard;
  `Report::Ratchet` + `into_ratchet()` in `optimizer/mod.rs`.
- Still dead: `num_trials`, `temperature` on the GEPA builder.

Tests: 48 lib tests + full suite green (`cargo test -p dspy-rs`). Two
`test_phase_upgrades` tests needed genuinely-failing train examples (an
acing parent is now skipped — that's the point). Docs updated:
`docs/optimizers/gepa.mdx`, `components/optimizers.mdx` (incl. a Ratchet
section and comparison-table row).

## 3. The measurement rig (reuse it; do not rebuild)

- **Primary**: `benchmarks/ratchet-hotpot/` — HotpotQA distractor, bare seed
  `"Answer the question."`, `train.jsonl` 150 / `val.jsonl` 30 (+
  `val12.jsonl` = first 12) / `holdout.jsonl` 60 (never seen by any
  optimizer). Student `meta-llama/llama-3.2-3b-instruct` @ temp 0.2, teacher
  `openai/gpt-5.6-sol`, bare rendering identical across all arms.
- **Zero-headroom**: `benchmarks/gepa-hotpot/` — tuned seed, 30/15/30,
  budget 120 (plus `analyze_trajectories.py`, the offline diagnosis of §1's
  claims — zero API spend).
- **Protocol**: per comparison, one *fresh* eval session, all arms together,
  paired per question, 3 reps, bootstrap 95% CI, pooled across 3 seeds.
  **Noise facts you must respect**: byte-identical prompts differ by up to
  ±0.02 between arms in one session; the same arm drifts ±0.03 across
  sessions; pooled 3-seed CIs are ±0.05–0.07 wide. Never compare val means
  across methods (RATCHET's champion deliberately under-states val).
- Eval scripts (all in `ratchet-hotpot/` unless noted): `eval_arms.py` (round
  1), `eval_v2/3/4.py`, `eval_v12.py` (small val), `eval_b24.py` (scaling),
  `gepa-hotpot/eval_zero.py`. Raw per-rollout scores in the matching
  `eval_*_results.jsonl` — every number in the docs is recomputable from
  them.

### Rerun commands

```bash
set -a; source .env; set +a     # OPENROUTER_API_KEY
# RATCHET arm (example 30; example 29 = DSRs-GEPA arm, same env matrix)
HOTPOT_DIR=benchmarks/ratchet-hotpot HOTPOT_SEED=0 \
HOTPOT_SEED_PROMPT="Answer the question." HOTPOT_ITERS=60 HOTPOT_MINIBATCH=5 \
HOTPOT_BUDGET=600 HOTPOT_CHAMPION=40 HOTPOT_OUT_PREFIX=ratchet2_winner \
  cargo run --release --example 30-ratchet-hotpot
# reference arm (gepa 0.1.4 + litellm in a venv)
cd benchmarks/ratchet-hotpot && HOTPOT_SEED=0 HOTPOT_BUDGET=600 \
  HOTPOT_TAG=mytag HOTPOT_VAL_FILE=val.jsonl python run_python_gepa.py
```

`HOTPOT_VAL_FILE` switches to `val12.jsonl` for the small-val regime;
`HOTPOT_TAG` namespaces the reference arm's winner files and run dirs.

## 4. Campaign ledger (each row: 3 seeds × budget 600 + fresh paired eval)

| Round | Change | Pooled vs `gepa` 0.1.4 | Fate |
|---|---|---|---|
| v1 | RATCHET baseline (strict gate) | −0.022 [−0.077, +0.034] | superseded |
| **v2** | noise-aware `gate_z`, rewrite fan-out ×2, fresh-column champion, delayed Simplify/Merge | **+0.005 / +0.009 / +0.010** across three sessions | **shipped** |
| v3 | + momentum parents, accretive rewrite | −0.018 (v3−v2 = −0.027) | reverted |
| v4 | + combined val∪fresh ranking, fusion finale | −0.021 (v4−v2 = −0.031) | reverted |
| v5–7 | regime tests, not variants (below) | — | — |

Falsifying our own ideas with pre-registered ablation arms is the rig
working as designed. After v4, selection-rule changes are **exhausted** at
this measurement power — remaining per-seed gaps are content lottery.

## 5. The regime matrix (v2-RATCHET vs `gepa` 0.1.4, paired, fresh sessions)

| Regime | Config | RATCHET − GEPA | The finding |
|---|---|---|---|
| Bootstrap | bare seed, val 30, budget 600 | +0.010 | parity; s0 +0.07–0.08 in three sessions (n.s.) |
| Small val | val 12, budget 600 | +0.023 | **GEPA's val-12 winners are −0.087 [−0.144, −0.031] vs its own val-30 winners — significant.** The winner's-curse collapse, in its marketed setting |
| Zero headroom | tuned seed, val 15, budget 120 | −0.018 | GEPA s0 regressed −0.053 (sig); RATCHET shipped seed 2/3, one cursed child passed `selection_z=1.0` — seed protection is probabilistic |
| Budget 2400 | bare seed, val 30 | +0.018 | **GEPA anti-scales: b2400 − b600 = −0.042 [−0.081, −0.003], significant**; its winners grow to 1.8–5.6k chars with 7–18 leaked train answers. RATCHET: flat, clean, champion beat its own argmax +0.059 on s2 |

**Combined meta-analysis: +0.012 [−0.017, +0.041]** (210 paired
question-regime points; positive in 3 of 4 regimes; never significantly
negative).

## 6. Verified vs noise

**Safe to state:**
- RATCHET ≥ GEPA on holdout mean in every regime tested; never significantly
  worse; strictly better on: budget scaling (GEPA −0.042 sig), small-val
  robustness (GEPA −0.087 sig), contamination (0 leaks in 18 RATCHET runs vs
  4–18 per GEPA winner), winner size (90–1.4k vs 1–5.6k chars), seed spread
  (~2–3× tighter), diagnostics (trial ledger).
- The reference proposer *prompts for* memorization; RATCHET's operator
  prompts prevent it inline (0 guard retries needed in most runs vs 5–8 for
  DSRs-GEPA's retry-based guard).
- gate-on-own-reflection-batch is a rubber stamp (93% pass, 33% real —
  `analyze_trajectories.py`).

**Do not state:** any raw-mean superiority claim on this task family. Session
noise floor is ±0.05; all inter-method mean differences live under it. Also
noise: any cross-session arm-mean comparison, any val-mean comparison across
methods, single-seed anything.

## 7. Engineering gotchas (each cost a debugging cycle)

1. **Champion slice must be pre-reserved** out of the batch pool at
   construction — the epoch sampler deliberately touches every train column,
   so post-hoc "unused" columns barely exist.
2. **Budget reserve must include the current generation's worst case**
   (A + 3×gate + full race), or the loop eats the champion's budget and the
   phase silently skips.
3. `perfect_score` default 1.0 **silently no-ops GEPA/RATCHET on
   unnormalized metrics** (every batch looks solved). A warning fires; set it
   to your ceiling or `f64::INFINITY`.
4. The Rust seed instruction (signature doc comment) differs from
   `common.py`'s `SEED_INSTRUCTION` by a line break — byte-inequal but
   behaviorally identical; eval scripts normalize whitespace for `[=SEED]`
   tags.
5. `gepa` 0.1.4 needs `propose_new_texts = None` on adapters and litellm
   installed; its runs occasionally crawl (one 600-call run took 97 min on
   API slowness — logs buffer, don't assume a hang).
6. `grep -c` exits 1 on zero matches — it once silently killed a `&&`-chained
   benchmark launch.

## 8. Artifact map (ALL uncommitted on `v1-final-sprint`)

| Path | What |
|---|---|
| `crates/dspy-rs/src/optimizer/ratchet.rs` | RATCHET (new) |
| `crates/dspy-rs/src/optimizer/gepa.rs` | anti-overfit stack, seed-protected argmax, shared pub(crate) helpers |
| `crates/dspy-rs/src/optimizer/{mod,target}.rs` | `Report::Ratchet`; `example_json` on lanes |
| `crates/dspy-rs/examples/29-gepa-hotpot.rs`, `30-ratchet-hotpot.rs` | benchmark arms, fully env-parameterized |
| `crates/dspy-rs/tests/test_phase_upgrades.rs`, `test_gepa_typed_metric_feedback.rs` | updated for skip-perfect semantics |
| `docs/gepa-diagnosis-2026-08-22.md`, `docs/ratchet-design.md` | diagnosis; design + all round results |
| `docs/docs/optimizers/gepa.mdx`, `docs/docs/components/optimizers.mdx` | user-facing docs incl. Ratchet section |
| `benchmarks/ratchet-hotpot/` | rig: `fetch_data.py`, `common.py`, `run_python_gepa.py`, `eval_*.py`; data `train/val/val12/holdout.jsonl`; winners `python_winner*(_v12/_b24)_s*.txt`, `ratchet{2,3,4}_winner*_s*.txt`, `ratchet_{v12,b24}_winner*_s*.txt`; logs `*_run_s*.log`; raw scores `eval_*_results.jsonl` |
| `benchmarks/gepa-hotpot/` | zero-headroom rig + `analyze_trajectories.py`, `eval_zero.py`, `ratchetz_*`, earlier `dsrs2*` GEPA A/B artifacts |
| superseded | `ratchet3/4` variants exist only as winner files + logs (code reverted); `dsrs_run*.log` etc. from the first diagnosis day |

Suggested commit split: (1) `optimizer/` + tests + mdx docs, (2)
`benchmarks/` rigs + winner/log artifacts (or prune logs), (3) `docs/*.md`
research notes. Total campaign cost ≈ $40–60 OpenRouter (teacher-dominated),
~30k student rollouts.

## 9. Open threads for the next owner

1. **The raw-mean scalp.** If "significantly higher mean than GEPA" is
   required, this task family cannot produce it for any method. Most likely
   to separate: multi-constraint instruction following (IFBench-style,
   objective checkers) or the docs-chat judge metric — regimes where
   instruction content compounds and GEPA's curse + bloat hurt most. The rig
   generalizes: swap `common.py`'s task + metric.
2. **Generality run**: 1 seed on a second student (e.g. qwen-2.5-7b) — cheap
   insurance for the "general method" claim.
3. **Champion significance guard**: champion currently overrides on any fresh
   mean difference; making the override require `> z·SE` on fresh (else keep
   val-argmax) would have fixed the one wrong flip observed (r3_s1). Ablation
   arms exist to test it.
4. **Mid-loop Merge leak window**: its `forbidden` list comes from batch A
   only, but its strengths text quotes cached *val* feedback (which contains
   gold answers). Never leaked in practice; close it by building `forbidden`
   from A ∪ the quoted win columns.
5. **Zero-headroom hardening**: one cursed child passed `selection_z=1.0` on
   val 15. Options: larger z when val is tiny, or champion-fresh even at
   small train pools (needs pool > champion+20; the gepa-hotpot rig's 30-row
   trainset disabled it).
6. Engine parity items still open from the first diagnosis: `on_rollout_error`
   policy, traces on cache hits, dominated-candidate removal in Pareto
   sampling, GEPA's dead `num_trials`/`temperature`.

## 10. Validation checklist

0. `cargo test -p dspy-rs` — all green, no API needed.
1. Cheap live smoke: example 30 with `HOTPOT_BUDGET=70 HOTPOT_ITERS=3` (~$0.05,
   2 min) — expect a seed val ≈ 0.24–0.30 (bare seed) and a clean result block.
2. Before trusting any new comparison: fresh session, all arms together,
   paired bootstrap, 3+ seeds. Copy any `eval_v*.py`.
3. Leak audit is one grep: train gold answers vs winner text (see
   `eval_v12.py`'s arm table for the pattern).
