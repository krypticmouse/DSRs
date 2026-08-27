# TEMPER + the BrowseComp-Plus rig — design note

Written 2026-08-23. Successor to [ratchet-design.md](ratchet-design.md) and
[gepa-diagnosis-2026-08-22.md](gepa-diagnosis-2026-08-22.md). The RATCHET
campaign ended with: equal holdout mean vs `gepa` 0.1.4 everywhere, strictly
dominant robustness, and the finding that **HotpotQA-F1 cannot separate any
prompt optimizer on raw mean** (±0.05 session noise floor; "extract the span,
be terse" saturates). This note designs the next swing: a task where
instruction content compounds (agentic search), a measurement rig built for
paired significance, and a method — **TEMPER** — that attacks the GEPA
weaknesses the literature now documents beyond the ones RATCHET already fixed.

Every mechanism below is tagged with its evidence: `[measured]` = our own
campaign data, `[lit]` = external published result.

## 1. Why these four mechanisms (the literature pass, condensed)

A four-agent literature sweep (2025 → mid-2026) over GEPA follow-ups,
curriculum/hardness methods, and evolutionary self-improvement produced five
findings that survive contact with our own data:

1. **GEPA's Pareto parent selection collapses at scale** — gepa-ai/gepa
   PR #432: with binary-ish metrics and large pools, dominance pruning gives
   *zero* selection mass to the best-aggregate candidate (5/9 large runs)
   while noise-minted "sole solver" specialists are unprunable. This is the
   mechanism behind the anti-scaling we measured (−0.042 at budget 2400)
   `[lit+measured]`. RATCHET kept GEPA's coverage-proportional draw, so it
   inherits the same latent failure. Fixes with evidence: lexicase-style
   selection (PR #432), archive-wide novelty-weighted sampling (Darwin Gödel
   Machine ablation: build-only-on-latest stagnates; LEVI beat GEPA on
   HotpotQA at 40% of its rollouts mostly via archive diversity).
2. **Frontier-band example selection is replicated ≥5× in RL/self-training
   and absent from every published prompt optimizer** — weight examples by
   p(1−p) solve variance ("learnability", arXiv 2502.12272; R-Zero's 3–7-of-8
   agreement band; AdaRFT). Our unsolved-bias batches were the crude version:
   they over-weight never-solvable columns, which carry zero signal `[lit]`.
3. **Phrasing overfit is measurable and fixable with paraphrase evaluation** —
   TextGrad: 96% val / 69% test; REMO closed it to 90/90 with cross-run memory
   `[lit]`. LAP's worst-case-paraphrase objective and ACCEL's
   perturb-the-high-regret-buffer give the recipe: score finalists on
   answer-preserving rewordings. Doubles as a leak detector — leaked strings
   fail under paraphrase. Complements our leak guard (which catches verbatim
   copies but not strategy-memorized phrasings).
4. **Length is a first-class generalization knob, not a cosmetic one** —
   Decagon production data: hard 1,500-char cap → 4× compression at −0.8% val
   and *better* holdout; 500 train examples → +75% length, −2% quality.
   PAC-Bayes prompt bounds (arXiv 2510.08413) give it theory `[lit]`. Matches
   our measured bloat-correlates-with-leaks finding (GEPA b2400 winners:
   1.8–5.6k chars, 7–18 leaked answers) `[measured]`.
5. **The guardrail**: arXiv 2607.12227 (Terminal-Bench, matched budgets) —
   harness/prompt evolution often fails to beat budget-matched test-time
   scaling. Any 2026 claim must include a budget-matched best-of-N baseline
   arm `[lit]`.

Explicitly deferred (bigger swings, out of v1 scope): optimizing program
*structure* through the IR (optimize_anything's ARC-AGI 32.5→89.5 with
val≈test is the single largest holdout gain in the whole sweep — "you can't
leak an answer into a control-flow graph"; DSRs' IR makes this uniquely cheap
for us later, but it changes the optimization surface and would make the
GEPA comparison apples-to-oranges); shipping a routed frontier ensemble
instead of one winner (MoP, 81% win rate); evolving the reflector itself
(SePO).

## 2. TEMPER, the algorithm

TEMPER = RATCHET v2's backbone (trial ledger, noise-aware gate-on-B, paired
racing, UCB1 operator portfolio, leak guard, fresh-column champion,
seed-protected argmax — all `[measured]`, kept unchanged) plus four new
mechanisms:

### T1 — ε-lexicase parent selection with novelty pressure
Replaces the Pareto-coverage-proportional draw. Per generation: shuffle the
*discriminative* val columns (columns where pool members differ); walk them,
filtering the pool to candidates within `lexicase_epsilon` of the column
best; stop when one survivor remains or columns are exhausted; break ties by
sampling ∝ 1/(1+children). No dominance pruning anywhere, so a noise-minted
sole solver can no longer monopolize selection mass, and broadly-strong
candidates always retain mass. Attacks: anti-scaling `[measured]`, #432
collapse `[lit]`, lineage concentration (the DGM stagnation ablation) `[lit]`.

### T2 — Learnability-band minibatches
Batch sampling weight for train column c: `w_c = p̂_c(1−p̂_c) + floor`,
where p̂_c is the mean score across all candidates evaluated on c (score
matrix lookup, zero rollouts; unevaluated columns get the prior 0.25 =
maximum-uncertainty weight), composed with the existing least-used epoch
ordering. Reflection and gating concentrate on the frontier band — examples
the pool solves *sometimes* — instead of on never-solvable ones. Attacks:
wasted reflection signal (our unsolved-bias was the blunt version)
`[measured]`, curriculum evidence `[lit]`.

### T3 — Variant tempering (the namesake)
The harness supplies answer-preserving paraphrase twins of a sample of seen
train/val questions (`make_variants.py`, teacher-generated, verified by the
same judge/gold answer). The optimizer receives them as trainset rows with a
`variant_pairs: (source_idx, variant_idx)` map, **excludes them from every
batch/val/race draw** (they are never optimized against), and adds them to
the champion race: final top-2 compete on fresh ∪ variant columns. A prompt
that memorized phrasings or content of seen examples loses exactly there;
the per-column variant deltas are reported as a phrasing-overfit audit.
Attacks: example memorization — the user-visible GEPA complaint — beyond
what the verbatim leak guard can see `[measured: docs-chat key-point
embedding; lit: TextGrad 27-pt val/test gap, LAP, ACCEL]`.

### T4 — Length-budget guard
`length_cap: Option<usize>` (TEMPER default 1500 chars): a proposal over the
cap triggers one retry naming the overage (same shape as the leak-guard
retry); still over → generation abandoned. Operators state the cap inline so
retries are rare. Attacks: bloat→overfit correlation `[measured]`, Decagon's
4×-compression-at-no-cost result `[lit]`.

### Implementation shape
All four mechanisms land in `optimizer/ratchet.rs` behind builder knobs that
**default off** (RATCHET v2 stays byte-identical), plus a `Temper` builder
preset that flips them on. Rationale: the campaign method is ablation arms;
knobs make every ablation a builder config instead of a code fork.

| knob | default (RATCHET) | TEMPER preset |
|---|---|---|
| `parent_selection` | `Coverage` | `Lexicase` |
| `lexicase_epsilon` | — | 0.05 |
| `batch_weighting` | `UnsolvedBias` | `Learnability` |
| `variant_pairs` | empty | from `variants.jsonl` |
| `length_cap` | `None` | `Some(1500)` |

## 3. The rig: `benchmarks/bcplus/`

**Task**: BrowseComp-Plus (ACL 2026; 830 BrowseComp-derived queries over a
fixed, MIT-licensed 100,195-doc corpus). Same question distribution as
BrowseComp, but deterministic and ~10× cheaper than live web — built exactly
to make agentic-search evals reproducible. Live BrowseComp would add a paid
search API we don't have keys for, live-web nondeterminism that poisons
paired stats, and $0.30–0.80/rollout; BC-Plus costs student tokens only.
"BrowseComp-100" therefore = a seeded 100-query holdout slice of BC-Plus
(uniform seeded shuffle, the simple-evals convention; BC-Small and the
BC-Plus-230 calibrated subset establish the practice).

**Splits** (single seed-0 shuffle of 830, contiguous disjoint slices):
`holdout100` (final holdout — *untouched until the pre-registered campaign
eval*), `devholdout` 60 (development-time holdout so holdout100 stays
clean), `train` 150, `val` 30 (+ `val12` = first 12, small-val regime),
`reserve` 490 (fresh columns, future variants).

**Search**: local BM25 (`bm25s`) over the corpus behind a tiny HTTP server
(`search_server.py`, port 8123): `/search?q=&k=5` → docid/url/512-token
snippet (the BC-Plus paper's main config), `/doc?docid=` → fuller text.
One server, every arm — Rust and Python optimizers see byte-identical
retrieval. Deterministic, zero API cost.

**Student**: one mid-tier OpenRouter model chosen by a baseline sweep on
`devholdout` (target: 10–35% seed accuracy = headroom without hopelessness;
candidates: qwen3-30b-a3b, llama-4-scout class). Weak models make <2 search
calls unprompted while strong agents make >20 `[lit: BC-Plus paper]` — the
gap between those behaviors is instruction-content headroom, exactly the
regime the RATCHET campaign said a raw-mean separation needs.

**Agent**: DSRs `Predict` with two tools (`search`, `open_doc`),
`max_turns` ≈ 12. The optimized surface is the agent instruction (search
strategy, decomposition, verification, when to stop) — one leaf, same
surface for every optimizer.

**Metric**: official BrowseComp grader template, fixed judge model across
all arms and all phases. Optimizer-facing score is shaped:
`1.0 if correct else 0.3 × gold-doc recall` (qrels ship with BC-Plus;
"found the evidence but flubbed the answer" ≠ "never found it" — the
ASI/rich-feedback finding `[lit]`). Reported eval numbers are always the
official binary accuracy. Judge feedback text (which quotes the gold answer)
flows to reflection — that's the documented leak vector, which is the point:
the guards have to earn their keep.

**Parity for the reference arm**: `gepa` 0.1.4's adapter calls a DSRs
rollout server (the same agent + metric + student the DSRs optimizers use)
over HTTP, so the Python reference optimizes byte-identical rollouts —
tighter parity than the hotpot rig, where Python re-implemented the student.

## 4. Measurement protocol (hardened from the RATCHET campaign)

- Paired-per-question, fresh-session, all-arms-together evals; 10k-resample
  paired bootstrap CIs **plus McNemar exact** (binary metric now); pooled
  across 3 optimization seeds. n=100 paired resolves ~6–8pp per seed;
  pooling 3 seeds targets ~4–5pp.
- **Budget-matched baselines**, not just the seed: (a) seed prompt, (b) seed
  with majority-vote@3 (test-time-scaling arm whose inference cost ≥ any
  optimized single-shot arm) `[lit: 2607.12227]`.
- **Audits per winner**: verbatim-leak grep (train gold answers/entities vs
  winner text), winner length, tool-calls-per-question distribution
  (strategy readout, free from traces), and the **paraphrase-robustness
  delta**: winner accuracy on `devholdout` originals vs teacher-paraphrased
  twins — the direct measure of "overfits the given examples" that the old
  rig lacked.
- Regimes (after the bootstrap phase): small-val (`val12`), budget scaling
  (600 → 2400), and a 1-seed second-student generality check.

### Pre-registered predictions
1. **P1 (headroom)**: every reflective optimizer beats the seed by >10pp on
   devholdout — instruction content compounds on agentic search where it
   couldn't on span extraction. If P1 fails, the campaign stops and the rig
   is re-examined (student too weak/strong), not the methods.
2. **P2 (main)**: TEMPER − `gepa` 0.1.4 > 0 on holdout100, pooled 3 seeds,
   paired CI excluding 0 in the budget-2400 regime; ≥ parity at 600.
3. **P3 (overfit)**: reference-GEPA winners lose ≥2× more accuracy under
   paraphrase than TEMPER winners; TEMPER variant-race demotes at least one
   argmax candidate across the campaign.
4. **P4 (anti-scaling)**: reference GEPA replicates its hotpot anti-scaling
   on BC-Plus (b2400 ≤ b600); TEMPER is flat-or-up; the TEMPER−T1 ablation
   (coverage selection) anti-scales — isolating lexicase as the cause.
5. **P5 (small val)**: TEMPER's val12 winners degrade less than GEPA's
   (hotpot: GEPA −0.087 significant).

### Cost plan (OpenRouter, teacher-dominated like previous campaigns)
Rollout ≈ $0.01–0.05 (mid-tier student, ~5–12 tool turns, local retrieval).
Optimization: 600 rollouts × 3 seeds × ~5 arms ≈ $90–450 upper phase;
evals: ~8 arms × 100 q × 3 reps ≈ $25–120 per fresh session. Phase-gated:
smoke (~$1) → single-seed pilot (~$15–30) → 3-seed bootstrap → regimes.

## 5. Runbook

One-time setup (done 2026-08-23): `benchmarks/bcplus/.venv` (uv, python
3.12: datasets, bm25s, PyStemmer, requests; add `gepa==0.1.4 litellm` for
the reference arm), `fetch_data.py` (splits + corpus), `build_index.py`
(747 MB BM25 index), `make_variants.py` (24 train twins → `variants.jsonl`,
60 devholdout twins → `devholdout_variants.jsonl`).

```bash
set -a; source .env; set +a                      # OPENROUTER_API_KEY
cd benchmarks/bcplus && .venv/bin/python search_server.py &   # port 8123

# DSRs arms (from repo root; BCPLUS_OPT = temper | ratchet | gepa)
BCPLUS_OPT=temper BCPLUS_SEED=0 BCPLUS_BUDGET=600 \
  cargo run --release --example 31-bcplus

# reference arm (gepa 0.1.4 over the DSRs serve process)
cd benchmarks/bcplus && BCPLUS_SEED=0 BCPLUS_BUDGET=600 \
  .venv/bin/python run_python_gepa.py

# development eval (devholdout; holdout100 ONLY for the pre-registered final)
.venv/bin/python eval_holdout.py --split devholdout --reps 3 --ref python \
  --variants devholdout_variants \
  seed=SEED strong=seed_strong.txt \
  python_s0=python_winner_s0.txt temper_s0=temper_winner_s0.txt
```

TEMPER ablation arms: rerun example 31 with the knob under test reverted via
a small env patch in the example, or (simpler) `BCPLUS_OPT=ratchet` plus
individual field overrides — each mechanism is a public builder field.
Campaign phases: (A) pilot — student-model choice by seed accuracy on
devholdout (target 10–35%); (B) 1-seed all-arms at budget 600, devholdout;
(C) 3-seed pre-registered run, holdout100, plus val12 and budget-2400
regimes; (D) ablations for any mechanism the deltas credit.

## 6. Phase B results (2026-08-23, single seed, devholdout 60q × 2 reps)

Arms optimized at budget 600 (student gpt-oss-120b, teacher/judge gpt-5.6-sol),
evaluated paired in one fresh session (`eval_phaseB_results.jsonl`; byte-identical
seed/temper arms measured the session noise floor at ≈0):

| arm | holdout acc | Δ vs GEPA [95% CI] | McNemar p | len | leaks | paraphrase gap |
|---|---|---|---|---|---|---|
| seed | .308 | −.008 | .51 | 72 | 0 | +.042 |
| strong (hand-written) | .333 | +.017 | 1.0 | 558 | 0 | +.042 |
| `gepa` 0.1.4 | .317 | — | — | 12,679 | **19** | +.042 |
| **RATCHET** | **.458** | **+.142 [+.067, +.225] sig** | **.012** | 338 | 0 | +.125 sig |
| TEMPER (shipped seed) | .308 | −.008 | .45 | 72 | 0 | −.017 |
| TEMPER argmax (demoted) | .317 | +.000 | .55 | 598 | 0 | +.017 |

Findings, single-seed strength only:
1. **The rig separates methods on raw mean** — the thing HotpotQA never could.
   GEPA's claimed val gain (.28→.415) transferred zero to holdout: pure
   overfit, 19 memorized train answers riding along.
2. **RATCHET − GEPA = +14.2pp, CI excludes zero, McNemar p = .012** — the
   campaign's first raw-mean scalp, on the content-compounding task the
   handoff predicted. Its 338-char winner drives real behavior change
   (5.9 tool calls/q vs 3.0; gold recall .42 vs .28).
3. **TEMPER's endgame judged correctly but its search found nothing**: the
   demoted argmax indeed ≈ seed on holdout (+0.0pp — the demotion was right),
   yet RATCHET found +15pp with the same operators and budget. Suspects, in
   ablation order: the length cap twice killed rewrites of the best lineage
   (gen 6/11 `LengthSkipped` on parent 4), lexicase/learnability changed the
   search trajectory, and 24 variant columns shrank the batch pool. Or seed
   luck — n=1.
4. **RATCHET's paraphrase gap (+.125, sig)** says part of its edge is
   phrasing-tuned; TEMPER-lineage arms are the most reword-robust (≈0 gap).
   A method combining RATCHET's search with TEMPER's robustness is the
   obvious v2 — mechanisms are already independent knobs.

Campaign stopped here by decision (no Phase C): single-seed direction, not a
pooled claim. holdout100 remains untouched.

### Second student: toast-1 (mxbai), same suite (2026-08-24)

Adapter work needed first: a `final_answer` stop tool (toast-1 acts only
through tools; `submit_answer` is reserved server-side), 200-word snippets +
18 turns (their API 500s on ~46-message conversations — bug filed verbally),
and a raw-text parse fallback (it emits bare-string stop-tool args; fallback
loses tool metadata, so tools/q and gold-recall under-report for this
student). Seed accuracy ≈.375; its failure mode is not committing to an
answer in budget.

| arm | holdout acc | Δ vs GEPA [95% CI] | McNemar p | paraphrase gap |
|---|---|---|---|---|
| seed | .375 | +.017 | 1.0 | +.017 |
| `gepa` 0.1.4 (shipped seed) | .358 | — | — | +.083 |
| RATCHET (948-char child) | .358 | +.000 | .75 | +.075 |
| **TEMPER (655-char child)** | **.408** | +.050 [−.025, +.133] n.s. | .18 (7–2) | +.050 |
| TEMPER argmax (demoted) | .325 | −.033 | 1.0 | −.033 |

Findings (single seed, all n.s. — direction only):
1. Methods rank differently per student: on toast-1 **TEMPER is the only
   arm above the seed** (+5pp vs GEPA, 7–2 McNemar) while RATCHET's child
   bought nothing (its val .433 was the curse again) and reference GEPA
   found nothing at all (shipped its seed — 14 candidates, none beat val).
2. **TEMPER's variant-champion race is now 2-for-2 on correct ship/no-ship
   calls**: on gpt-oss it demoted a dud (confirmed +0.0 holdout), on
   toast-1 it picked the finalist worth +8.3pp over the demoted argmax.
   The selection machinery works; its *search* aggressiveness (length cap
   killed best-lineage rewrites on gpt-oss) is the tuning target.
3. toast-1 is far less instruction-steerable than gpt-oss (policy-dominated;
   best arm .408 vs gpt-oss's .458) and byte-identical arms differed 1.7pp —
   the session noise floor on this student.

## 7. Falsification discipline

Same rules as the RATCHET campaign, which falsified three of its own
improvements: every mechanism ships with its ablation arm pre-registered;
any mechanism whose ablation shows it ≤ 0 on holdout gets reverted, not
argued for. Session noise floor must be re-measured on this rig (same-prompt
arms in one session) before any cross-arm claim.
