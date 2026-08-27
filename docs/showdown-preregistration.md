# The Showdown — pre-registration (FROZEN 2026-08-24, before any run)

Purpose: close the GEPA vs RATCHET vs TEMPER argument with one decisive,
multi-model, multi-benchmark campaign. This file freezes the matrix, the
protocol, and the decision rule **before the first rollout is spent**. Any
deviation gets logged in §7 with a timestamp and a reason.

## 1. Question

Which optimizer should DSRs ship as its default: reference GEPA, RATCHET v2,
TEMPER, or the RATCHET+TEMPER hybrid?

## 2. Matrix (lean tier, user-approved 2026-08-24)

- **Arms (4)**: `gepa` = reference gepa 0.1.4 (Python, litellm) · `ratchet` =
  RATCHET v2 defaults · `temper` = `temper()` preset · `hybrid` = RATCHET
  defaults + `variant_pairs` (TEMPER's variant-champion endgame only).
  Plus the unoptimized **seed** floor in every eval.
- **Students (3)**: `mxbai:toast-1`, `openrouter:qwen/qwen3.6-35b-a3b`,
  `openrouter:openai/gpt-oss-20b`.
- **Teacher (all arms, all rigs)**: `openrouter:openai/gpt-5.6-terra`.
  Prior sol-teacher results are appendix context, never pooled.
- **Benchmarks (3)**: BrowseComp-Plus (agentic search, judged),
  IFBench (instruction following, deterministic checker),
  PUPA (privacy delegation, judged composite).
- **Seeds**: seed 0 only. Decisiveness comes from pooling 9 independent
  student×benchmark cells, paired per question.
- **Budget**: 600 metric rollouts per arm per cell, identical across arms.

## 3. Fixed models per rig (never swapped mid-campaign)

| Rig | Judge / checker | Other fixed calls |
|---|---|---|
| bcplus | `gpt-5.6-sol` judge (unchanged from prior campaign) | BM25 server |
| ifbench | code checker server (no LLM) | — |
| pupa | `gpt-5.6-sol` judge, temp 0 | untrusted LM `gpt-5.6-luna` |

## 4. Protocol (per cell)

1. Optimize all 4 arms at budget 600, seed 0, same seed prompt, same splits.
2. Eval: **one fresh session, all arms + seed together**, devholdout 60 × 2
   reps, paired per question. Raw rows to `eval_*_results.jsonl`.
3. Audits per winner, reported always: verbatim leak count, winner length,
   paraphrase gap (devholdout_variants), val→devholdout gap.
4. `holdout100` stays frozen everywhere. It is spent exactly once: final
   confirmation of the declared winner vs `gepa`, after §5 is settled on
   devholdout. No iteration afterward.

## 5. Decision rule (the part that closes the argument)

**Primary**: pooled paired delta vs `gepa` across all 9 cells (per-question
paired bootstrap, 95% CI; exact McNemar as the secondary test). A method
"beats GEPA" iff pooled CI excludes 0 in its favor. Best such method wins.

**Tie-break** (pooled CIs overlap 0 or each other), in this fixed order:
1. Never-significantly-worse count across the 9 cells.
2. Robustness envelope: leak count, val→test gap, paraphrase gap, winner
   length — each compared pooled; a method must win ≥3 of 4 to take it.
3. If still tied: RATCHET wins over TEMPER/HYBRID only if simpler-machinery
   (fewer active mechanisms) is not beaten on any envelope item — Occam.

**Anti-hedge clauses**, agreed in advance:
- No post-hoc cell exclusion. A crashed cell is rerun, never dropped.
- No metric changes after the first suite launches.
- Single-cell significance is never quoted as a headline; only pooled.
- If NO method beats `gepa` pooled, the published verdict is "parity on
  mean, decided on robustness" — and the envelope decides the ship.

## 6. Spend plan (lean + hybrid ≈ $290–430, hard cap $450)

| Phase | What | Est. |
|---|---|---|
| P0 | rig pilots (~40 rollouts/rig/student) + variants generation | $10–20 |
| P1 | IFBench 3 students × 4 arms | $30–60 |
| P2 | PUPA 3 students × 4 arms | $75–120 |
| P3 | bcplus 3 students × 4 arms | $150–210 |
| P4 | pooled eval + winner-vs-gepa holdout100 confirmation | $25–40 |

Gate: after each phase, spend is reconciled against the cap; on trajectory
to exceed it, drop reps (2→1) on remaining evals before touching arms.

## 7. Deviations log

- 2026-08-26 ~19:50 UTC — campaign stopped early by the user with 8 of 9
  cells fully evaluated (PUPA/qwen36 was killed mid final reference arm;
  bcplus/qwen36 mid arm 2). The pooled verdict therefore rests on 8 cells:
  mean parity (RATCHET +0.00 [−.02,+.03] pooled), robustness envelope to
  RATCHET (0 leaks vs 11; never-sig-worse 8/8 vs 6/8; winners 3–50×
  smaller). holdout100 remains frozen and unspent. Next phase per user:
  merge RATCHET into the DSRs GEPA implementation + small revalidation vs
  gepa 0.1.4, instead of finishing the ninth cell.

- 2026-08-25 11:30 UTC — bcplus toast reference-GEPA arm invalidated and
  rerun under tag `toast2`: gepa 0.1.4 silently resumed the PRIOR campaign's
  `runs/python-toast-s0` state (Aug 23, sol teacher) and "finished" in 3.5
  min. Its 3 Rust arms are unaffected (fresh 2h runs each). Same collision
  also overwrote the prior campaign's toast winner FILES (results remain
  recorded in temper-design.md and raw eval rows). No other cell shares a
  tag with prior runs.

- 2026-08-25 10:20 UTC — bcplus qwen36 cell runs with `BCPLUS_STOP_TOOL=1`
  (same channel toast-1 uses). Seed probes before any arm completed: without
  it qwen executed 0 tools in 199 calls (val 0.000, mechanical floor); with
  it tools fire (val 0.167). The first launch was killed ~20 min into arm 1
  and the cell restarted clean with the flag; no results were produced or
  discarded. All arms within the cell share the config.
- 2026-08-25 05:15 UTC — P1/P2/P3 run in parallel (all 9 cells at once) on
  the user's request, replacing sequential phase gates with monitoring + a
  kill switch. No statistical impact: each cell's eval remains a later fresh
  session; the $450 cap stands. All P0 pilots had already passed.
