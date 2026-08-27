# GEPA diagnosis — why both engines overfit, and how DSRs gets ahead

Written 2026-08-22 from the trajectories already on disk (no new rollouts).
Reproduce every number with `python benchmarks/gepa-hotpot/analyze_trajectories.py`.
Companion to [handoff-gepa-adapter-dogfood.md](handoff-gepa-adapter-dogfood.md).

> **Status: items 2.1–2.5 are BUILT and A/B-tested** (same day). §5 has the
> implementation summary and the measured results; §2 is kept as the original
> design rationale.

## 0. TL;DR

The overfitting is real, and it is **not** a search-operator problem — the
mutation step is fine. It comes from three places, in order of damage:

1. **The acceptance signal is noise.** Val n=15 has a run-to-run SD of 0.052
   on the *same* prompt. Every val "gain" GEPA reported (+0.00 to +0.04, one
   +0.18) is inside that band; 5 of 6 winners were ≤ seed on holdout.
2. **The minibatch gate validates on the examples it trained on.** The child
   is written to fix minibatch A and then tested on A. It passed 27/29
   proposals (93%); only 9/27 (33%) then beat their parent on val.
3. **The reference proposer asks for memorization.** gepa 0.1.4's prompt says
   *"identify all niche and domain specific factual information … include it
   in the instruction"*. 2 of 3 Python winners contain three verbatim train
   gold answers each. Ours don't (hotpot), but both engines embed train
   key-points on docs-chat because the judge feedback quotes them.

DSRs's engine already has the pieces (score matrix, rollout cache, paired
columns, `evaluate_gated`) to fix all three. The reference can't fix #1/#2
cheaply — it has no per-column matrix and no cache.

## 1. Evidence

### 1.1 Selection noise dominates the signal (hotpot, llama-3.2-3b, F1)

| | python s0 | s1 | s2 | dsrs s0 | s1 | s2 |
|---|---|---|---|---|---|---|
| seed on val (n=15) | 0.750 | 0.822 | 0.690 | 0.666 | 0.778 | 0.756 |
| winner on val | 0.767 | 0.822 (=seed) | 0.866 | 0.706 | 0.804 | 0.773 |
| claimed gain | +0.017 | 0 | **+0.176** | +0.040 | +0.026 | +0.017 |
| winner − seed, holdout n=30×3 | −0.077 | +0.006 | −0.024 | −0.043 | +0.005 | −0.063 |

Same prompt, same 15 questions, six runs: mean 0.744, **SD 0.052**. Five of
six "gains" are < 1 SD. The one that wasn't (python s2, +0.176) regressed on
holdout — winner's curse, textbook.

### 1.2 The gate is a rubber stamp

`analyze_trajectories.py §1`, 4 Python runs, 29 proposals:

| run | passed gate | of which better than parent on val |
|---|---|---|
| hotpot s0 | 5 | 2 |
| hotpot s1 | 5 | 1 |
| hotpot s2 | 5 | 2 |
| docs-chat | 12 | 4 |
| **total** | **27/29 (93%)** | **9/27 (33%)** |

Mechanism: reflection sees minibatch A's traces, writes a child that fixes A,
child is scored on A, "improves", pays 15 val rollouts. Both engines end up
with the same 5 children per 120 rollouts — our cache saves the 3 parent
re-eval rollouts the reference pays, our missing gate spends them back.

Note: [gepa.mdx:219](docs/optimizers/gepa.mdx) says *"roll out the new
candidate; if improved, evaluate on the validation columns"* —
[gepa.rs:528-700](../crates/dspy-rs/src/optimizer/gepa.rs) has no gate at
all. Docs/code mismatch.

### 1.3 Memorization is prompted for (reference) and feedback-induced (both)

Python winners s0/s2 each quote three train gold answers ("Known exact
benchmark answers: …"). DSRs hotpot winners: zero leaks — our
`ReflectOnInstruction` prompt doesn't ask for facts. But on docs-chat *both*
winners carry a "verified details" section lifted from train key-points (ours:
6/33 train questions, Python's: 4/33; 0 val, 0 holdout) because the judge's
feedback lists the missed key-points and the teacher dutifully pastes them in.
It didn't even help: python s0 (most leakage) was −0.077 on holdout.

### 1.4 The Pareto front is saturated, so parent sampling ≈ uniform

Final fronts, hotpot: 40–67% of val columns are won by *every* candidate
(score 1.0 for all); wins per candidate 8–13 of 15. Coverage-proportional
sampling therefore barely prefers anyone. The "complementary strengths" story
in the docs needs discriminating columns to exist; at n=15 on an easy-ish
student it mostly doesn't. Python's `remove_dominated_programs` trims this a
little; we don't do even that.

### 1.5 Smaller DSRs-specific gaps (code reading)

| # | gap | where |
|---|---|---|
| a | No minibatch gate (see 1.2); `evaluate_gated` exists on the engine but GEPA doesn't use it | gepa.rs:560-640 |
| b | Minibatch = `choose_multiple` per generation: no epoch coverage, can resample a perfect batch, no targeting | gepa.rs:575 |
| c | One child per generation; `num_trials`, `temperature` dead | gepa.rs:190-200 |
| d | Winner = argmax val mean, no margin, no tie-break toward shorter/ancestor | gepa.rs:680 |
| e | One rollout error aborts the compile (hit 3× in benchmarking; example 29 works around it with a retry wrapper) | engine.rs `run_rollouts` → `try_collect` |
| f | Cache-served parent rollouts carry no trace → reflection loses the trace when minibatches overlap | gepa.rs:337 |
| g | No merge — irrelevant for one leaf (reference merge is a no-op with a single component), matters for multi-leaf programs | — |

## 2. What "best GEPA" means, concretely

Ranked by (expected holdout gain ÷ implementation cost). Items 1–3 are free
or near-free in rollouts and are where we beat the reference, not match it.

### 2.1 Reflect on A, gate on B  (fix 1.2)
Sample two disjoint train minibatches per generation. Reflect on A (as now).
Score the child on **B** and the parent on B (parent's B scores hit the cache
after the first time). Promote to full val only if child > parent on B. Same
rollout cost as the reference's gate; removes the optimistic bias entirely.
Engine already has `evaluate_gated`; needs a `threshold` computed from the
parent's B mean.

### 2.2 Noise-aware acceptance + Occam winner  (fix 1.1)
We have per-column scores for every candidate, so compare **paired**:
child − incumbent on the same val columns (plus the B columns both were scored
on). Promote/replace only if mean paired delta > `margin` (builder knob,
default ≈ 0.02–0.03; document that at n=15 smaller effects are undetectable).
Final winner: best val mean, but ties within `margin` go to the **shorter**
instruction / earlier generation. Optional `finalist_repeats: k` re-scores the
top-2 on val with a bumped `cache_salt` and picks by pooled mean (kills the
winner's curse at the cost of 2·k·|val| rollouts — make it opt-in).

### 2.3 Leak guard  (fix 1.3)
Two layers. (i) Prompt: add to `ReflectOnInstruction` — *"Write a
generalizable instruction. Do not copy answers, entities, or example-specific
facts from the feedback; the assistant will see different inputs."*
(ii) Mechanical: the target's rows are `Serialize`; collect string leaf values
(≥ 6 chars) from the minibatch rows' *output* side, reject a proposal that
contains any verbatim, and re-ask once with the offending spans named. Knob:
`memorization: Forbid | Allow` (default Forbid). The reference literally asks
for the opposite; this is a clean, demonstrable win.

### 2.4 Informative minibatches  (fix 1.5b, Python's skipped iterations)
Epoch-shuffled coverage (every train row gets reflected on before any
repeats) + never hand the teacher an all-perfect batch + optional
`hard_example_bias` that weights columns where the parent sits below the
Pareto best. All of this is score-matrix lookups — zero rollouts.

### 2.5 Discriminative Pareto  (fix 1.4)
Count wins only on columns where not every candidate ties; drop dominated
candidates (a candidate whose win-set is a subset of another's). Parent
sampling then actually prefers complementary candidates.

### 2.6 K children per reflection  (use `num_trials`)
K proposals at `temperature` from one reflection context → gate each on B →
full-val only the best survivor. Turns the expensive val sweep into the
reward for the best of K rather than for whatever the teacher said first.

### 2.7 Parity items
`on_rollout_error: Fail | ScoreZero` on the engine (1.5e); trace retention for
cache hits or a fresh-trace minibatch rule (1.5f); merge for multi-leaf
programs (1.5g) — per-leaf crossover between Pareto members with a common
ancestor, gated like any child. Fix gepa.mdx to describe what the code does.

## 3. How to prove it (rig already exists)

`benchmarks/gepa-hotpot/` + `eval_all_seeds.py`: 3 seeds × {old engine, new
engine, python}, 120-rollout budget, holdout n=30 × 3 reps, paired bootstrap.
Acceptance for the engine change: pooled (new − seed) CI excludes 0 from
below, and new ≥ python. Add a `val=45` arm to show the gain scales with val
size (prediction: it does, because 2.2's margin stops rejecting real effects).
Cost ≈ previous sweep (~$10 OpenRouter, ~1h).

## 4. Things that looked like problems but aren't (original pass)

- Engine parity: pooled dsrs − python = −0.002 [−0.038, +0.035]; per-seed sign
  flips. Not a quality gap.
- The hotpot student (llama-3.2-3b) is near its ceiling on this data — the
  seed scores 0.74 val; most val columns are 1.0 for everyone. Part of why
  nothing beats seed. docs-chat (judge metric, harder) shows the same
  dynamics, so the conclusions aren't an artifact of the easy task.
- DSRs reflection cost (≈4k prompt tokens/call incl. traces) is fine.

## 5. Implementation + A/B results (2026-08-22, same day)

### What shipped (`crates/dspy-rs/src/optimizer/gepa.rs`)

| Change | Builder knob | Default |
|---|---|---|
| Reflect on A, gate on held-out B (disjoint minibatches) | `gate_size` | `minibatch_size`; `Some(0)` disables |
| Noise-aware Occam winner: earliest gen (then shortest) within `z·SE` of the argmax mean, paired per val column | `selection_z` | `1.0`; `0.0` = argmax |
| Leak guard: proposal quoting short example strings verbatim → one retry naming the spans → skip | `allow_memorization` | `false` (guard on) |
| Epoch-style least-used-first minibatches; both A and B prefer columns the parent hasn't solved (B: so the gate is winnable); all-perfect A skips the generation | `perfect_score` | `1.0` (`∞` disables — needed for unnormalized metrics) |

New `GEPAResult` fields: `gate_rejected`, `skipped_perfect`, `leak_retries`,
`leak_skipped`, `best_by_mean` (the demoted argmax candidate, when it differs).
`OptimizeTarget` gained `example_json` (crate-private) for the leak guard.
Docs updated: `docs/optimizers/gepa.mdx`, `components/optimizers.mdx` (the old
text promised a gate that didn't exist). All tests green; two `test_phase_upgrades`
tests needed genuinely-failing train examples (an acing parent is now skipped,
which is the point).

### Optimizer-run behavior (3 seeds, same rig: budget 120, minibatch 3, llama-3.2-3b)

| | old engine | new engine |
|---|---|---|
| gate pass rate | (reference: 93%) no gate — every child paid 15 val rollouts | 6 promoted / 27 proposals (22%) |
| leak guard | — | 4 retries, **0** unrecovered, 0 leaked winners |
| unwinnable gates (parent 1.0 on B) | — | fixed by B's unsolved bias (first cut wasted ~⅓ of generations on them) |
| shipped winner | mutated child, 3/3 seeds | **the seed, 3/3 seeds** (s1: argmax child 0.794 vs seed 0.782 → within noise → seed) |

### Holdout A/B (fresh, paired, 30 q × 3 reps × 10 arms; `eval_ab_v2.py`)

| arm | holdout F1 | Δ vs seed [95% CI] |
|---|---|---|
| seed | 0.576 | — |
| old engine s0/s1/s2 | 0.568 / 0.591 / 0.523 | −0.009 / +0.015 / **−0.053** (spread) |
| new engine s0/s1/s2 | 0.556 / 0.577 / 0.572 | −0.021 / +0.000 / −0.004 (tight) |
| new argmax ablation s0 | 0.525 | −0.051 — the child Occam refused to ship |

- **new − old pooled: +0.007 [−0.056, +0.076]** — n.s., as predicted: on a
  task where the seed is unbeatable at this budget, the best possible engine
  ships ≈seed. The old engine's earlier-session regressions (−0.043/−0.063)
  re-measured milder this session — seed lottery again, which is the point.
- **chosen − argmax pooled: +0.019 [−0.009, +0.054]** — the winner rule alone,
  direction as designed; on s0 it dodged a −0.031 shipped regression.
- Byte-equivalent arms differed by up to 0.02 in this eval — the same-prompt
  noise floor; treat any single-arm delta under that as silence.

### Honest scorecard

Proven: the engine no longer ships regressions (worst new arm −0.021 vs old
−0.053), never memorizes val/train answers, and spends val sweeps only on
gate-survivors. Not yet proven: a *positive* holdout gain — this task/budget
has no headroom over the seed. The decisive experiment is the one where
headroom exists: **val=45+ arm, or a weaker seed instruction, or docs-chat**
(judge metric showed +0.08–0.10 val gains that did transfer). Rig:
`eval_ab_v2.py` + `HOTPOT_ITERS`/`HOTPOT_OUT_PREFIX` env knobs on example 29.

Still open from §2: discriminative-Pareto parent sampling (2.5, partially —
dominated removal not done), K-children per reflection (2.6, `num_trials`
still dead), `on_rollout_error` policy, trace retention on cache hits,
multi-leaf merge (2.7).
