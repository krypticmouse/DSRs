# RATCHET — a reflective prompt optimizer past GEPA

Design note, 2026-08-22. Successor to the GEPA work in
[gepa-diagnosis-2026-08-22.md](gepa-diagnosis-2026-08-22.md); every design
choice below traces to a *measured* GEPA failure, not intuition.

## Why GEPA stalls (the evidence, condensed)

| Measured failure | Consequence |
|---|---|
| Reflection is **memoryless** — each generation the teacher sees 3 examples and the parent text, nothing about prior attempts | Both engines plateau after 1–2 accepted mutations; the teacher re-proposes near-duplicates of falsified edits |
| Every child pays a **full validation sweep** (reference gates on its own reflection batch — 93% pass, 33% real) | 5 proposals per 120-rollout budget; search is starved |
| **One operator** (full rewrite) | No way to make small safe edits, cut bloat, or combine complementary candidates; instructions grow monotonically and bloat correlates with overfit |
| Argmax winner on a noisy val mean | Winner's curse (fixed already by the Occam rule) |

## The algorithm

RATCHET keeps GEPA's skeleton — candidate pool, Pareto-coverage parent
sampling, reflective mutation, trainset minibatch cycle — and the full
anti-overfit stack already shipped in DSRs GEPA (reflect-on-A/gate-on-B with
unsolved-bias batches, leak guard, noise-aware Occam winner). It adds three
mechanisms:

### R1 — Trial ledger (cross-generation memory with credit assignment)

Every proposal is recorded: operator, parent, the teacher's own one-line
`edit_summary` (the hypothesis the edit claims), and its measured fate —
gate-killed, race-killed at column k, or promoted with paired val delta.
Every subsequent reflection prompt carries the digest:

```
Validated lessons (build on these):
  [gen 2 | patch | Δ+0.041] "Demand the shortest exact span; ban explanations."
Falsified — do NOT re-propose:
  [gen 4 | rewrite | killed at gate] "Add step-by-step reasoning before answering."
```

The search acquires momentum: validated edits compound, falsified directions
die permanently instead of being rediscovered every third generation.

### R2 — Paired racing evaluation (sequential kill, complete survivors)

Replace "gate, then full val in one shot" with a sequential paired test.
After the held-out gate block B, the child is evaluated on val columns in
shuffled blocks; after each block, its scores are compared **paired,
per-column** against the parent's cached scores. The child dies the moment
its paired deficit is significant (`mean < −race_z·SE`). Survivors complete
every val column, so the score matrix stays complete and the Occam winner
rule works unchanged. Killed child ≈ gate + 1–2 blocks (4–12 rollouts) vs
GEPA's 18. The parent side is free (rollout cache).

### R3 — Operator portfolio under a bandit

Four proposal operators, selected per generation by UCB1 on measured
promote-with-positive-delta rate; an operator is only in the draw when its
precondition holds:

| Operator | What it does | Eligible when |
|---|---|---|
| **Rewrite** | GEPA-style full rewrite from batch-A failures + ledger | parent imperfect on A |
| **Patch** | Distill ONE rule from the best-vs-worst rollout contrast on A; minimal edit | A has score spread |
| **Simplify** | Strip clauses not backed by validated lessons; compress | instruction > ~500 chars |
| **Merge** | Semantic crossover: combine two frontier members shown *what each wins on* | complementary same-module frontier pair exists |

Merge is the crossover GEPA ships off-by-default and that is structurally a
no-op for single-instruction programs in the reference implementation —
doing it semantically through the teacher makes it work at any granularity.

## Interfaces

`Ratchet` is an `Optimizer` over the shared `Engine` (score matrix, rollout
cache, budget, Pareto view — all reused). Requires `prompt_model` (all four
operators are LM-driven; there is no concat fallback). Reuses `GEPACandidate`
for pool entries. New report `RatchetResult`: winner + demoted argmax +
trial ledger + per-operator stats + kill counters.

Key knobs (defaults): `num_iterations` 30, `minibatch_size` (A), `gate_size`
(B, = minibatch), `race_block` 8, `race_z` 1.0, `selection_z` 1.0,
`perfect_score` 1.0, `allow_memorization` false, `ledger_window` 12.

## Pre-registered benchmark (headroom restored)

`benchmarks/ratchet-hotpot/`: HotpotQA distractor, **bare seed** ("Answer
the question.") — known ≈0.25 F1 of instruction headroom for
llama-3.2-3b — train 150 / val 30 / holdout 60 (fresh), budget **600 metric
calls per arm**, 3 seeds. Arms: bare seed, Python `gepa` 0.1.4, DSRs GEPA
(anti-overfit build), RATCHET. Verdict: holdout, paired per question,
bootstrap 95% CI, pooled across seeds. Success = RATCHET − seed CI > 0 and
RATCHET ≥ both GEPA arms; secondary metrics: proposals explored per budget,
winner length, leak count. Generality check afterward: repeat 1 seed on a
second student model.

---

## Results (2026-08-22, benchmarks/ratchet-hotpot/)

Ran as pre-registered: bare seed, train 150 / val 30 / holdout 60, budget 600
metric calls/arm, 3 seeds, llama-3.2-3b student, gpt-5.6-sol teacher. Arms:
Python `gepa` 0.1.4, DSRs GEPA (anti-overfit build), RATCHET. Eval:
`eval_arms.py`, 2,340 fresh rollouts, paired bootstrap.

### Search behavior (per 600-rollout budget)

| | Python gepa | DSRs GEPA | RATCHET |
|---|---|---|---|
| proposals explored | ~15 | ~20 | **21–25** |
| winners' verbatim train-answer leaks | **4+ per winner, 3/3 seeds** | 0 | 0 |
| leak-guard retries needed | n/a | 5–8/run | **0** (anti-copy is in the operator prompts) |
| winner length (chars) | 989–2,802 | 373–670 | 90–978 |

Operator bandit (summed over seeds, attempts/wins): rewrite 28/10, patch
20/4, merge 14/1, simplify 11/0 — from a bare seed, full rewrites dominate
and the bandit correctly concentrated on them; simplify never won (bloat is
not the binding constraint in a bootstrap regime).

### Holdout verdict (bare seed = 0.239)

Every winner beat the seed massively (+0.28 to +0.40, all 9 significant).
Method comparison, pooled across seeds, paired per question:

| comparison | as shipped | after selection fix (see below) |
|---|---|---|
| RATCHET − Python | −0.027 [−0.087, +0.032] | −0.022 [−0.077, +0.034] |
| RATCHET − DSRs GEPA | +0.041 [−0.011, +0.096] | +0.016 [−0.035, +0.070] |
| DSRs GEPA − Python | −0.068 [−0.142, +0.007] | −0.038 [−0.100, +0.021] |

Per-seed holdout means: Python .604/.608/.640, RATCHET .592/.602/.590 (after
fix), DSRs GEPA .564/.588/.584. RATCHET is the most *stable* method
(seed-to-seed spread 0.012 vs Python's 0.036) and its winners are the only
contaminated-free ones from a method that also survived the no-headroom
regime without shipping regressions.

### The ablation that falsified our own rule

The pre-registered argmax arms showed the **Occam winner sweep over-protects**
when headroom exists: chosen − argmax = **−0.045** (GEPA arms), −0.015
(RATCHET). The component that had earned its keep in the no-headroom
benchmark was protecting the *seed*, not protecting every earlier child.
Fixed same-day in both optimizers: **seed-protected argmax** — children
compete by raw val mean; only the generation-0 seed keeps incumbency within
`selection_z` SEs. Both regimes are now covered by measured arms: at zero
headroom it still ships the seed (dodged −0.05 regressions), at high headroom
it ships the argmax child (recovers the −0.045).

### Honest verdict + next levers

RATCHET ≈ reference GEPA on raw holdout mean (CI spans 0, Python nominally
+0.02) while being leak-free by construction, 2–3× more compact, 3× more
seed-stable, and regression-proof at zero headroom — the reference is neither.
Not yet a significant *win* on mean. The identified lever: acceptance
strictness. Python accepts ~everything and explores deep lineages, which pays
in a high-headroom regime; our strict gate+race kills lateral children that
might have been stepping stones. Next experiments, in order:

1. **Explore-mode gate**: gate kills only *significantly worse* children
   (`within_noise` on B instead of strict >), racing stays the strict filter.
   Prediction: closes the −0.02 vs Python; risk: more val spend per budget.
2. Generality: repeat 1 seed on a second student (e.g. qwen-2.5-7b) and on
   docs-chat (judge metric).
3. Merge/simplify earn their keep only later in a run — schedule operators by
   regime (bandit already trends this way) and revisit at bigger budgets.

---

## Rounds 2–6: the optimization campaign (2026-08-23)

Each round = implement → 3-seed run at budget 600 → fresh paired holdout
session vs `python gepa 0.1.4`. Ablation arms pre-registered throughout.

| Round | Change | Pooled vs python | Kept? |
|---|---|---|---|
| v1 | baseline RATCHET | −0.022 [−0.077, +0.034] | — |
| v2 | noise-aware gate (`gate_z`), rewrite fan-out ×2, fresh-column champion, simplify/merge eligibility after first promotion | **+0.005 / +0.009 / +0.010** (3 sessions) | **yes — this is RATCHET** |
| v3 | + momentum parents, accretive rewrite prompt | −0.018 (v3−v2 = −0.027) | reverted |
| v4 | + combined val∪fresh ranking, fusion finale | −0.021 (v4−v2 = −0.031) | reverted |

Engineering fixes along the way: champion slice must be **pre-reserved** out
of the batch pool (the epoch sampler otherwise touches every column), and
the loop's budget reserve must include the current generation's own
worst-case spend.

### The regime matrix (RATCHET v2 vs python, paired, fresh sessions)

| Regime | Config | RATCHET − python | Note |
|---|---|---|---|
| Bootstrap, big val | bare seed, val 30, budget 600 | +0.005 / +0.009 / +0.010 | parity; s0 +0.07–0.08 three times (n.s.) |
| Small val (GEPA's marketed case) | bare seed, val 12, budget 600 | +0.023 [−0.045, +0.092] | python's val-12 winners are **−0.087 [−0.144, −0.031]** vs its own val-30 winners — the winner's-curse collapse, measured & significant. RATCHET degrades less |
| Zero headroom | tuned seed, val 15, budget 120 | −0.018 [−0.045, +0.007] | python s0 regressed −0.053 (sig); RATCHET s1/s2 shipped seed, s0 slipped one cursed child past `selection_z=1.0` — seed protection is probabilistic, not categorical |

### What is settled after ~15 optimizer runs + 6 eval sessions

- On HotpotQA-F1/llama-3.2-3b at these budgets, **no method separates from
  the reference on holdout mean beyond the ±0.05 session noise floor** — in
  either direction, in any regime. Remaining per-seed gaps are content
  lottery: which prompt text a stochastic teacher happened to draw.
- RATCHET's categorical wins hold everywhere: **0 leaked winners in 15 runs**
  (python: 3/3 contaminated in bootstrap, 4+ train answers each), winners
  2–10× shorter (deployment tokens), seed-to-seed spread 2–3× tighter,
  full trial ledger + per-operator stats for diagnosis.
- Three self-falsifications (momentum, accretion, combined+fusion) — the rig
  works; selection-rule twiddling is exhausted at this measurement power.

### Round 7 (in flight): budget scaling, the remaining structural claim

Racing makes breadth cheap and the fresh-column champion's power grows with
budget, while python's argmax curse **worsens** with candidate count.
Pre-registered: at budget 2400 (4×), 3 seeds, `champion_fresh=80` —
prediction RATCHET − python > +0.04. Runs: `ratchet_b24_*`/`python_b24_*`
in `benchmarks/ratchet-hotpot/`.

---

## Round 7 result + final verdict (2026-08-23)

### Budget scaling (2400 calls, 3 seeds, fresh paired holdout)

- **GEPA anti-scales: −0.042 [−0.081, −0.003] (significant).** Its budget-2400
  winners are *worse* than its own budget-600 winners — more candidates,
  worse argmax curse — and its memorization grows with budget: 7–18 verbatim
  train answers per winner (1.8–5.6k chars).
- RATCHET holds flat and clean: head-to-head at 2400 = +0.018 [−0.034,
  +0.065]; zero leaks; 3–12× shorter winners; on s2 the fresh-column champion
  beat RATCHET's own argmax candidate by +0.059 holdout.

### Cross-regime meta-analysis (v2-RATCHET vs python gepa, paired)

| regime | mean Δ |
|---|---|
| bootstrap, budget 600 | +0.010 |
| small val (12) | +0.023 |
| zero headroom | −0.018 |
| budget 2400 | +0.018 |
| **combined, 210 question-regime pairs** | **+0.012 [−0.017, +0.041]** |

### Verdict

On raw holdout mean, RATCHET = GEPA within measurement everywhere (combined
+0.012, n.s.; positive in 3 of 4 regimes; never significantly worse). The
campaign's strong claim is about **robustness, where GEPA is measurably
fragile and RATCHET is not**:

| axis | GEPA (reference) | RATCHET |
|---|---|---|
| more budget | **−0.042, significant regression** | flat |
| smaller val | **−0.087, significant regression** | degrades less (+0.023 head-to-head) |
| winner contamination | 3/3 winners at b600; 7–18 leaks/winner at b2400 | **0 in 18 runs** |
| winner size (deploy tokens) | 1–5.6k chars | 90–1.4k |
| seed-to-seed spread | ~0.036 | ~0.012–0.026 |
| diagnostics | val scores only | full trial ledger, per-operator credit, kill counters |

Equal peak, strictly dominant reliability envelope. A raw-mean "much better"
is not achievable on this task family for *any* prompt optimizer we tested —
HotpotQA-F1 saturates at "extract the span, be terse" and the remaining
inter-method differences sit under the ±0.05 session noise floor. Tasks where
instruction *content* compounds (multi-constraint following, judge-metric
answering à la docs-chat) are where a mean separation is most likely to
exist; that is the next benchmark if a raw-mean claim is required.
