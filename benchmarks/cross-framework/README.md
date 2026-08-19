# Cross-framework benchmarks

Apples-to-apples framework-overhead comparison: DSRs vs DSPy (Python) vs a
minimal LangChain chain. Every benchmark mocks the LM so provider latency is
excluded — what remains is the framework's own CPU cost per call and its
scheduling efficiency around simulated latency.

## Protocol

**Per-call overhead** — identical signature shape in DSRs and DSPy
(`question, context -> answer, confidence: float`), 0-demo and 2-demo variants,
mocked LM returning a fixed correctly-formatted response, 2000 timed iterations
after warmup. Inputs vary per iteration so no cache layer can short-circuit.
The LangChain pipeline (prompt template → `FakeListChatModel` → `StrOutputParser`)
does structurally *less* — no typed field protocol, no coercion, no constraint
checking — and is included as a floor for minimal Python chain overhead, not as
a like-for-like harness.

**Orchestration** — 256 examples evaluated at concurrency 16, where each "LM
call" is a 20 ms sleep. Ideal wall time is (256/16) × 20 ms = 320 ms; anything
above that is scheduling overhead (DSPy: thread pool via `dspy.Evaluate`;
DSRs: tokio + `buffered(16)` via `evaluate_trainset_with_concurrency`).

## Running

```bash
# Rust (from repo root)
cargo run --release --example 97-perf-microbench
cargo run --release --example 98-orchestration-bench
cargo run --release --example 99-deep-harness-bench

# Python
python3 -m venv venv && venv/bin/pip install dspy langchain-core
venv/bin/python bench_dspy.py
venv/bin/python bench_dspy_deep.py
venv/bin/python bench_langchain.py
```

## Results (2026-07-27, Apple Silicon, dspy 3.2.1, langchain-core 1.5.1)

### Per-call framework overhead (mocked LM)

| framework | 0 demos | 2 demos |
|---|---|---|
| **DSRs** | **2.2 µs** | **2.9 µs** |
| DSPy 3.2.1 | 239.3 µs | 291.0 µs |
| LangChain (minimal chain, fewer features) | 135.1 µs | — |

DSRs is ~100x faster than DSPy per call and ~50x faster than a LangChain chain
that does a fraction of the work.

### Orchestration (256 × 20 ms simulated calls @ 16 concurrent)

| framework | wall | overhead vs 320 ms ideal |
|---|---|---|
| **DSRs** (tokio, `buffered(16)`) | **347.9 ms** | **27.9 ms** |
| DSPy (`Evaluate`, 16 threads) | 396.2 ms | 76.2 ms |

### Startup

| framework | import/startup |
|---|---|
| DSRs (compiled binary) | milliseconds |
| DSPy | 1.02 s (`import dspy`) |
| langchain-core | 0.62 s |

### Deep/heavy harness shapes (mocked LM)

`99-deep-harness-bench.rs` vs `bench_dspy_deep.py` — same pipeline topologies,
per full pipeline run:

| shape | DSRs | DSPy 3.2.1 | ratio |
|---|---|---|---|
| chain, depth 10 | 27.0 µs | 2,407 µs | 89× |
| chain, depth 50 | 99.4 µs | 11,009 µs | 111× |
| fan-out 16 + aggregate | 36.6 µs *(concurrent)* | 3,469 µs *(sequential)* | 95× |
| layered DAG 4×8 | 77.2 µs | — | |
| forward, 16 demos | 6.8 µs | 495.9 µs | 73× |
| forward, 64 demos | 21.3 µs | 1,698 µs | 80× |
| forward, 16 output fields | 7.6 µs | 587.6 µs | 77× |
| forward, 50 KB context | 9.5 µs | 294.7 µs | 31× |
| tool loop, 16 tool iterations | 61.8 µs | not benched¹ | |
| eval 64 × depth-10 chain @ 16 concurrency | 1.3 ms wall (640 LM calls) | — | |

¹ Faithfully mocking DSPy's ReAct protocol through DummyLM is fragile enough
that numbers wouldn't be trustworthy.

Notes:
- The fan-out rows differ structurally by design: DSRs runs branches
  concurrently with `try_join_all` inside the module — idiomatic async Rust.
  A plain DSPy module executes sub-calls sequentially; parallelism requires
  manual threading. Both numbers are "what the natural implementation costs."
- DSRs' margin *widens* with depth (89× → 111×) — per-stage overhead amortizes
  in Rust while Python's per-call floor is fixed.
- DSRs demo scaling (6.8 µs @ 16 → 21.3 µs @ 64) is the residual per-call cost
  of cloning the cached prompt prefix into rig form; caching the rig-converted
  prefix (a known follow-up) would flatten it further.

## Caveats

- This measures **framework overhead**, not task quality. On a live API a
  single call is dominated by 300 ms–2 s of provider latency; per-call CPU is
  invisible there. The gap matters in high-concurrency evaluation, optimizer
  loops over cached responses, and local inference at high QPS.
- Cross-language comparisons carry inherent noise (allocator, interpreter
  warmup). The ~100x margin is far above that noise floor.
- Both orchestration measurements include OS timer slop on the 20 ms sleeps;
  the protocol is identical on both sides.
