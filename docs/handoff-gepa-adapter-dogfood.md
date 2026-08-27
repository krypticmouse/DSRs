# Handoff — GEPA dogfood + adapter findings (docs-chat harness)

Written 2026-08-21 for whoever picks up the optimizer/adapter work. Ledger of
one day's experiments: what was measured, what is *verified*, what is *noise*,
and the ranked task list that fell out. Everything referenced here is
**uncommitted working state** on `v1-program-unification` — see §7 before
touching anything.

## 1. What happened, in one paragraph

We used the DSRs docs-chat widget (Mixedbread toast-1 grounded in the
`dsrs-docs` + `dsrs-code` stores) as a live optimization target. Built a
50-question evalset with per-question key-point checklists, an LLM judge
(GPT-5.6-sol via OpenRouter) with a coverage-dominant density metric, then ran
GEPA three ways: the Python reference implementation (`gepa` 0.1.4), DSRs' own
Rust GEPA (dogfood), and finally a controlled experiment on whether
ChatAdapter's typed-contract rendering itself helps or hurts the student.
Outcome: both GEPA engines produce equivalent gains (ours at 40% of the
rollout budget); the adapter's marker scaffolding is a small measured *cost*
on single-String-output tasks and actively suppresses refusals on
not-in-the-docs questions. §5 turns that into an adapter roadmap.

## 2. The measurement rig (reuse this, don't rebuild it)

- **Evalset**: `docs-chat-worker/gepa/evalset.jsonl` (44) + `holdout.jsonl`
  (6, never seen by any optimizer). Schema per line:
  `{"question", "key_points": [3-6 facts from the docs pages], "citations"}`.
  Includes 3 adversarial "not in the docs" traps (fine-tuning weights, Python
  interop, JS SDK) whose correct answer is a refusal. Key points were derived
  from the docs pages directly, **not** from toast-1's answers (no
  circularity). Split for optimizer runs: `train.jsonl` (33) / `val.jsonl`
  (11), seed-0 shuffle, identical across both engines.
- **Judge/metric**: `docs-chat-worker/gepa/judge.py` (Python side) and the
  `JudgeAnswer` signature in `crates/dspy-rs/examples/27-gepa-docs-chat.rs`
  (Rust side). Score = 0.70·key-point coverage + 0.10·citations +
  0.20·(1 − fluff fraction) − 0.15·per-contradiction (cap 0.30), clamped to
  [0,1]. Judge returns prose feedback (missed points, fluff quoted verbatim)
  — that text is what GEPA's reflection consumes; don't strip it.
- **Judge variance — the sharp edge**: the same prompt scored 0.474, 0.525,
  0.603, 0.712 on the same 6-question holdout across four sweeps.
  **n=6 single-sweep comparisons are unusable.** The standard set by the
  final experiment: n=50 × 3 repeats, paired per-question deltas, bootstrap
  95% CI (`scaffold_experiment.py`). ~300 rollouts ≈ 35 min ≈ $3 OpenRouter
  + ~6M Mixedbread prompt tokens.

## 3. Results — verified vs noise

| Run | Setup | Val (n=11) | Holdout (n=6) | Rollouts |
|---|---|---|---|---|
| Python `gepa` 0.1.4 | 14 iters, minibatch 3 | 0.472 → 0.573 | 0.474 → 0.601 | 220 |
| DSRs GEPA (`27-gepa-docs-chat.rs`) | 5 iters, minibatch 3, cap 120 | 0.468 → 0.552 | 0.525 → 0.639 | 91 |

**Verified (safe to state publicly):**
- Both engines improved val score by +0.08–0.10 with identical dynamics: the
  full gain lands on the *first* accepted mutation, then plateau. Four
  independent positive deltas of similar size — the optimization gain is real.
- DSRs GEPA = reference-implementation parity at 40% of the rollouts.
- Both engines evolved the same *kind* of winner: generalizable
  search-and-style contract + a memorized-facts section overfit to training
  topics. Faithful algorithm implementation, including its failure mode.

**Noise (do not state):** any cross-run holdout comparison, engine-vs-engine
absolute levels, the earlier "full vs trimmed prompt" gap. See §2 variance.

**Adapter experiment (n=50 × 3, paired, same judge, only variable =
rendering; `scaffold_run.log`, raw scores `scaffold_results.jsonl`):**
- bare 0.530 vs ChatAdapter-rendered 0.495; delta **−0.036 [−0.075, −0.002]**.
- Excluding one measurement-artifact question (asks the model to *describe*
  the marker protocol, so its correct answer contains the markers our
  post-processing strips): **−0.029 [−0.064, +0.002]**.
- Normal questions: −0.027. The 3 refusal traps: **−0.169** — the
  "respond with the corresponding output fields, starting with
  `[[ ## answer ## ]]`" instruction compels answers where refusal is correct.
- Honest summary: *no help, ~3-point cost, real refusal distortion, measured
  on one model (toast-1) only.* This is the adapter's degenerate case (single
  free-text output field) — it does NOT measure parsing reliability, which is
  where typed contracts earn their keep (the 6-field `JudgeAnswer` signature
  parsed cleanly across hundreds of Rust-side calls, zero failures).

## 4. DSRs GEPA engine findings (from the dogfood build)

Ranked; file refs are current as of this branch.

1. **Ambient-candidate footgun.** Candidates reach leaves only via the
   task-local overlay inside `Predict::call`/`build_chat`
   (`optimizer/target.rs:230` → `fx/mod.rs:238`); `PredictorInfo::instruction()`
   (`predictors/predict.rs:1219`) reads instance state only. A custom module
   that hand-rolls its prompt from `instruction()` silently evaluates the
   **baseline** on every candidate — GEPA completes and reports noise.
   `fx::ambient_entry` is `pub(crate)` (`fx/mod.rs:218`), so user code can't
   even detect the situation. Fix ideas: public ambient-read API, or a
   runtime warning when a named leaf is never called during a rollout.
   Workaround used in example 27: route rendering through
   `Predict::build_chat` (public, overlay-resolving).
2. **`PredictError` has no public constructor** for custom-module transport
   errors (`core/errors.rs`); example 27 smuggles reqwest failures through
   the `Params` variant. Add `PredictError::custom(...)`.
3. **No merge/crossover stage.** Both engines plateau after one mutation;
   Python gepa ships (off-by-default) Pareto-member merging. Given how
   spread the Pareto front was, this is the most plausible search unlock.
4. **Stale doctest** `optimizer/gepa.rs:179`: `.max_lm_calls(Some(500))`
   doesn't compile (bon setters take `T`).
5. **Dead config**: `num_trials`, `temperature` on the GEPA builder are
   documented as unused.
6. **Reflection loses trace detail on cache hits** (`gepa.rs:337`) when
   trainset-only + small sets make parent re-evals pure cache reads. Passing
   a distinct valset avoids it (what we did); worth a docs callout.
7. Positives, for the record: `require_feedback` fails fast and clearly;
   minibatch clamps instead of panicking; budget exhaustion is a clean break;
   rollout cache made parent re-evals free (91 rollouts vs naive ~130).

## 5. Adapter roadmap (the "get ahead of DSPy" part)

The marker protocol is inherited DSPy design. Today's data makes format a
*measured* dimension worth ±3–17 points — same order as instruction
optimization (+11). Ranked plan:

1. **Refusal-safe response instruction** (one-line template change: "if the
   information is unavailable, say so in the field"). Expected to recover
   most of the −0.169 trap loss. Acceptance test: rerun
   `scaffold_experiment.py`, traps delta ≥ 0.
2. **Single-`String`-output fast path**: when a signature has exactly one
   String output, render plainly (instruction + inputs, whole completion =
   the field). Today's bare arm *is* this path and it won. DSRs can decide
   this at compile time from the signature shape — DSPy can't (runtime
   introspection). Acceptance: paired CI vs marker rendering ≥ 0.
3. **Native structured outputs for typed signatures**: compile the output
   shape (typesys already renders schemas) into provider
   structured-output/tool schemas; markers become the fallback for providers
   without it. Kills parse failures and the prose tax together.
4. **Rendering as an optimizable slot** (moonshot, novel): the IR already
   addresses instructions/demos/tool-descriptions as overlay params; add
   format choice as a slot kind so optimizers search format × instruction
   jointly per leaf. No DSPy-family framework has this.
5. **Standing adapter-bench**: promote `scaffold_experiment.py` into a
   permanent gate for adapter changes. Also: dedupe the format instruction
   (currently verbatim in both system and user message) and fix the indented
   objective block — cosmetic, but shipped on every call.

Before making general claims: replicate the refusal-suppression finding on
one mainstream model. toast-1 is a RAG-tuned specialist; n=1 on models.

## 6. Validation checklist for the next owner

0. Re-run the wiring test (mocked student/judge, real teacher, ~2¢):
   `cd docs-chat-worker/gepa && python test_wiring.py` → `WIRING OK`.
1. Re-run the Rust dogfood cheaply: drop `num_iterations` to 2 in example 27
   and confirm seed val ≈ 0.47 ± judge noise.
2. Before trusting any new comparison: n=50 × 3 repeats, paired bootstrap.
   Copy the analysis block from `scaffold_experiment.py`.
3. The `predictors!` eval question scored ~0 for **every** candidate in both
   engines — toast-1 cannot answer it from the indexed stores. That's a docs
   content/indexing gap, not a prompt problem. Check what `dsrs-code` store
   actually indexed for `core/module.rs`.

## 7. Artifact map (ALL uncommitted — decide what to keep before committing)

| Path | What |
|---|---|
| `docs-chat-worker/gepa/` | Python harness: `judge.py`, `adapter.py`, `run.py`, `compare.py`, `scaffold_experiment.py`, `test_wiring.py`; data: `evalset.jsonl`, `holdout.jsonl`, `train.jsonl`, `val.jsonl`; results: `best_prompt.txt` (Python winner), `dsrs_best_instruction.txt` (Rust winner), `trimmed_prompt.txt`, `scaffold_results.jsonl`, `compare_detail.json`, `rendered_chat.txt`, `*.log`, `runs/` |
| `docs-chat-worker/gepa-optimization.md` | The how-to guide for the Python loop (written first, still accurate) |
| `docs-chat-worker/worker.js` | **Modified**: `SYSTEM` is now the Python-run winner (JSON-string form). Deploy = `npx wrangler deploy` from that dir; whether it's been deployed is user-side state |
| `docs/toast-chat.js` | **Modified**: markdown renderer rewritten (line-based; fixes ol-numbering, inline-``` fence bug, adds tables/headings/nesting), resizable + restyled panel, plus owner's own edits (suggestions, logo) |
| `crates/dspy-rs/examples/27-gepa-docs-chat.rs` | Rust dogfood harness (student via `build_chat` + raw HTTP, judge + metric, GEPA run, holdout compare) |
| `crates/dspy-rs/examples/28-print-docs-chat.rs` | Dumps exact ChatAdapter rendering (feeds `rendered_chat.txt`) |
| `evalset-small.jsonl`, `evalset.example.jsonl` | Superseded (18-entry subset from an aborted variant; 5-entry doc examples). Safe to delete |

Environment: keys in repo-root `.env` (`MXBAI_API_KEY`, `OPENROUTER_API_KEY`
— OpenRouter model string `openai/gpt-5.6-sol`, ~$2.5/M in, ~$15/M out;
each toast-1 rollout ≈ 20k prompt tokens). Python side needs
`pip install gepa litellm` (gepa 0.1.4, litellm 1.97.0 verified); nothing
loads `.env` automatically — `set -a; source .env; set +a` first. Rust side:
`cargo run --example 27-gepa-docs-chat` from repo root.

Known judge caveats: scores are judge-prompt-relative — never compare numbers
across different judge prompts (Python `judge.py` vs Rust `JudgeAnswer` render
differently); re-baseline after any judge change. GEPA gotcha (Python side):
gepa 0.1.4 requires a `propose_new_texts = None` attribute on custom adapters
or every reflection crashes (already handled in `adapter.py`).
