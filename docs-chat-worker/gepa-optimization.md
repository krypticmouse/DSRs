# Optimizing the docs-chat prompt with GEPA

**TL;DR:** the `SYSTEM` prompt in [worker.js](worker.js) is the only text steering toast-1, and GEPA can evolve it automatically: run each eval question through toast-1 (student), grade the answer with GPT-5.6 Sol (teacher/judge), feed the judge's *textual* feedback to the teacher to propose a better prompt, keep a Pareto frontier of winners. Everything in [gepa/](gepa/) is wired and smoke-tested; the one thing that still needs human work is growing the evalset from 5 example entries to 30–50 real ones.

## The setup

| Role | Model | How it's called |
|------|-------|-----------------|
| Student | `toast-1` | Mixedbread chat completions, non-streaming, same `store_search` + `store_grep` tools over `dsrs-docs` + `dsrs-code` as the deployed worker |
| Judge | `openai/gpt-5.6-sol` | litellm via OpenRouter, JSON mode, returns score + written critique |
| Teacher (reflection) | `openai/gpt-5.6-sol` | passed to `gepa.optimize` as the string `"openrouter/openai/gpt-5.6-sol"` |
| Optimizer | `gepa` 0.1.4 (standalone package) | custom adapter, **not** `dspy.GEPA` — the worker isn't a DSPy program |

One GEPA "metric call" = one toast-1 rollout + one judge call. The optimizer proposes a mutated system prompt, tests it on a 3-question minibatch, and only pays for a full validation sweep when the minibatch shows improvement. Candidates that win on *different* questions both survive (Pareto selection), which prevents collapsing into a prompt that's great at API questions and useless at concept questions.

## Quick start

```bash
cd docs-chat-worker/gepa
python -m venv .venv && .venv/bin/pip install gepa litellm

# keys come from the repo-root .env (MXBAI_API_KEY, OPENROUTER_API_KEY)

# 1. sanity-check the loop without spending Mixedbread tokens (~2 cents of teacher)
.venv/bin/python test_wiring.py            # expect: WIRING OK

# 2. build the real evalset (see below), then:
.venv/bin/python run.py --evalset evalset.jsonl --budget 400
```

`run.py` extracts the seed prompt from `worker.js` at runtime, so every run starts from what is actually deployed. Output: `best_prompt.txt` plus full artifacts under `runs/<timestamp>/`. Deploy = paste the winner into `SYSTEM` in worker.js and `npx wrangler deploy`.

## The files

- **[gepa/adapter.py](gepa/adapter.py)** — the integration point. `evaluate()` runs a batch of questions through toast-1 (4 workers, per-example failures score 0.0 instead of crashing the run) and returns `EvaluationBatch(outputs, scores, trajectories)`. Trajectories keep the answer **and the searches toast-1 ran** (`hosted_tool_calls`), so the teacher sees retrieval behavior, not just final text. `make_reflective_dataset()` packages `{Inputs, Generated Outputs, Feedback}` records — the exact schema gepa's built-in proposer expects.
- **[gepa/judge.py](gepa/judge.py)** — the metric. See next section; this file is where the definition of "good answer" lives, and it matters more than anything else here.
- **[gepa/run.py](gepa/run.py)** — CLI entry point: seed extraction, train/val split (seeded shuffle, 25% val), `gepa.optimize(...)` call, result printout.
- **[gepa/evalset.example.jsonl](gepa/evalset.example.jsonl)** — 5 entries grounded in the signatures/predict/optimizers docs pages. Format per line:

```json
{"question": "What does Predict<S> do and what does .call() return?",
 "key_points": [".call() returns Result<Predicted<Output>, PredictError>", "..."],
 "citations": ["components/predict"]}
```

- **[gepa/test_wiring.py](gepa/test_wiring.py)** — full GEPA loop with mocked student/judge and the real teacher. Cheap, fast, and it already earned its keep (see Gotchas).

## The metric: why it's shaped this way

The goal is "maximum answer, minimum noise" — dense answers in a short window. That decomposes into a coverage-dominant score:

```
score = 0.70 · key-point coverage        ← recall: did it state the facts
      + 0.10 · required citations        ← file paths present
      + 0.20 · (1 − noise fraction)      ← precision: chars not carrying a key point
      − 0.15 per contradicted fact (cap 0.30), clamped to [0, 1]
```

Two deliberate choices:

1. **Coverage dominates conciseness 70/20.** If you weight brevity too high, GEPA reward-hacks into terse-but-lossy answers — the exact opposite of what you want. An answer can only win the conciseness points *after* it has the facts.
2. **The judge returns prose, not just a number.** It quotes fluff spans verbatim, lists missed key points, and writes 2–5 sentences of advice. That text is what the reflection LM actually reads — GEPA's entire sample-efficiency advantage comes from feedback that explains *why* the score is low. A bare `0.4` teaches the teacher nothing.

Verified behavior (2026-08-20): a deliberately padded answer ("Great question! … Hope that helps!") covering 2/5 key points scored **0.40**, with the judge quoting all three filler sentences verbatim and naming the three missing facts.

## Building the real evalset — the actual work

30–50 entries is the target; GEPA is explicitly designed for small trainsets. Guidance:

- **Use real questions.** Mine what you and others have actually asked the widget, plus the questions each docs page was written to answer (one page per component makes this mechanical: 2–3 questions per page across signatures, predict, modules, IR, optimizers, .dsrs files, adapters, evaluation…).
- **Key points, not gold answers.** A checklist of 3–6 facts per question is faster to write, and the judge grades recall against it directly. Pull the facts from the docs page, not from toast-1's own answers (circular).
- **Mix difficulty modes:** single-page lookups, cross-page synthesis ("how do signatures relate to Predict state?"), code-grounded questions that should force `store_grep` into `crates/`, and a few questions the stores *don't* cover — key point: "says the docs don't cover this" (kills hallucination-under-pressure).
- **Citations sparingly.** Only where a path genuinely belongs in a good answer; the judge treats an empty list as "no citation required."

## Budget and cost

- `--budget 400` (metric calls) is a solid first run. With ~35 train / ~12 val that's roughly 15–25 candidate proposals.
- Teacher side: ~$5 per 400-call run at GPT-5.6 Sol pricing measured today ($2.5/M in, $15/M out; judge calls run ~1.2k tokens in, ~400 out).
- Student side: each toast-1 rollout carried ~20k prompt tokens in testing (retrieved context dominates), so a 400-call run is ~8M toast-1 prompt tokens. Check that against the Mixedbread plan before scaling budget up.
- Runs are resumable in spirit: every candidate, score, and output lands in `runs/<timestamp>/`, and `track_best_outputs=True` keeps the best answer per validation question for eyeballing.

## Reading the results

`run.py` prints per-candidate validation scores and writes `best_prompt.txt`. Before deploying, do two checks:

1. **Read the best answers, not just the scores** (`runs/<ts>/`, best outputs per val question). You're checking for metric gaming the judge missed — e.g. answers that became bullet-dumps of key-point-shaped fragments.
2. **Spot-check on held-out questions** that were never in the evalset. Twelve val questions can be overfit; five fresh ones tell you if the prompt generalizes.

If a run plateaus immediately, the usual cause is feedback quality, not budget — make the judge's `advice` field more concrete before buying more metric calls.

## Gotchas (each one cost real debugging time or would have)

- **`propose_new_texts = None` must exist on the adapter.** gepa 0.1.4 probes `adapter.propose_new_texts` with attribute access; a plain class without it crashes every reflection with `AttributeError` and the run silently never improves. The wiring test caught this.
- **`stream: false` on the Mixedbread call.** The worker streams to browsers; the optimizer wants the complete message plus `hosted_tool_calls` (present only on the final object).
- **Scores must stay in [0, 1].** gepa sums minibatch scores for acceptance and averages val scores for Pareto selection; `perfect_score=1.0` also drives its skip-perfect logic.
- **Never raise per-example** in `evaluate()` — score failures 0.0 with the error in the trajectory. One 502 from either API shouldn't kill a $5 run.
- **Judge drift:** if you change judge weights or prompt mid-experiment, old and new run scores aren't comparable. Re-score the seed prompt after any judge change to re-baseline.
- **`reflection_minibatch_size=3`** keeps reflection prompts small enough that the teacher actually reads the trajectories. Default epoch-shuffled batching handles the rest.

## Later: dogfooding

DSRs ships its own GEPA ([docs/optimizers/gepa.mdx](../docs/docs/optimizers/gepa.mdx)). Once this Python loop has produced a prompt worth deploying, the same adapter design — student rollout + judge feedback + instruction component — is a ready-made spec for driving the *Rust* GEPA against the same evalset: the trainset rows are plain structs, the metric returns (score, feedback), and the candidate is one instruction string. That would make the docs widget the first end-to-end DSRs-optimizes-its-own-infrastructure story.

---

*Everything above verified 2026-08-20: toast-1 non-streaming with store tools ✓, `hosted_tool_calls` capture ✓, `openrouter/openai/gpt-5.6-sol` as judge (JSON mode) and as `reflection_lm` string ✓, full loop via `test_wiring.py` — seed 0.35 → evolved candidate 0.85 in one mutation ✓ (gepa 0.1.4, litellm 1.97.0).*
