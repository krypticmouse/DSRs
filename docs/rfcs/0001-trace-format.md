# RFC 0001: The Trace Format

**Status:** proposed · **Date:** 2026-08-14 · **Branch:** `v1-revamp` · **Depends on:** nothing (deliberately — see §7) · **Blocks:** eval engine, GEPA rewrite, record/replay, counterfactual replay, RL export, flywheel

---

## 0. Summary

One trace format replaces three (`trace::Graph`, `mipro::Trace<S>`, `evaluate::ExecutionTrace`).

- **`Span`** = one `Predict` invocation (one logical LM subcall): rendered prompt, provider round-trips + tool executions as ordered *events*, raw output, parsed output, model ref, usage, timing, error. Loops → multiple spans, disambiguated by `seq`.
- **`Trace`** = one rollout: ordered spans + intern tables (components, model configs, prompt prefixes) + rollout-level input/output/eval.
- Spans are addressed by the **same name the params system uses** (fx name / dotted path). In memory that name is a per-trace interned `CompId(u32)`; on the wire it is the string. When phase-1 `ParamId` lands, it maps through the same table — nothing here waits for it.
- **Capture** is a tokio `task_local!` scope (`capture(...)`), identical in shape to today's `trace::trace()`: zero-cost when inactive (one TLS probe per `Predict` call), innermost-scope-wins when nested.
- **Instrumentation seam:** `Predict::call_and_parse_with_input` opens a span before the LM call (failed calls stay visible with prompt recorded, output absent — same two-phase discipline as today's `record_node`/`record_output`); `LMResponse` grows a per-round-trip `exchanges` field so the tool loop's inner structure lands in the span without task-local plumbing inside `LM`.
- **Serialization:** JSONL (header line, span lines, footer line), `"v": 1`, secrets structurally absent, prefix-interned so demos serialize once, 64 KB truncation with hash for oversized text.
- **Streaming:** v1 records final-only; the ordered `events: Vec<SpanEvent>` with a tagged, skip-unknown enum is the seam that lets streamed partial-output events land later without a format break.
- **Dies:** `trace::{Graph, Node, NodeType, Executor}`, `trace::value::{TrackedValue, IntoTracked}`, `trace::context::*`, `mipro::Trace<S>`, `evaluate::ExecutionTrace` (+ builder), and every `node_id: Option<usize>` field on the data carriers.

Prior art that changed a decision (only): **GEPA's `pred_trace`** → `for_component` slicing keyed by param name with per-invocation `seq`. **Temporal's event-log determinism** → per-span `request_hash` + strict replay verification + divergence-switch mode for counterfactual replay. **Agent Lightning / verifiers span convention** → spans carry full `Message` lists (not flattened text) so RL export is a projection, not a reconstruction.

---

## 1. The record: `Span`

One `Span` per `Predict` invocation. A `Predict` whose tool loop makes 3 provider round-trips and runs 4 tools is **one span with 7 events**, because the `Predict` invocation is the unit optimizers attribute to (GEPA mutates the instruction of a component, not of a round-trip). RL export flattens events back into transitions (§4f). A component called 5 times in an agent loop produces 5 spans with `seq` 0..4.

All types live in `crates/dspy-rs/src/trace/` (new files; old files die in §7). Everything below is serde-derivable with the repo's existing types: `Message`/`ContentBlock` (`core/lm/chat.rs`, already serde), `LmUsage` (`core/lm/usage.rs`, serde), `LMConfig` (`core/lm/mod.rs`, serde, `api_key` skipped).

```rust
use serde::{Deserialize, Serialize};
use crate::{LMConfig, LmUsage, Message};

/// Index of a span within its trace. Dense, assigned in insertion order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SpanId(pub u32);

/// Per-trace interned component index. The component *name* is the same string
/// the params system addresses: an fx slot name (`"drafter"`) or a facet dotted
/// path (`"pipeline.rater"`). `Trace::components` maps CompId -> name.
///
/// This is the id-form the vision report asks for, without waiting on phase 1:
/// when the global `ParamId` lands, `Trace::components` grows a parallel
/// `param_ids: Vec<Option<ParamId>>` column (additive change, no version bump).
/// The serde boundary always speaks the string (§5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CompId(pub u32);

/// Per-trace interned prompt prefix (system message + demo turns). One entry
/// per distinct (component, candidate) configuration — demos serialize once
/// per trace, not once per span. Predict already builds and caches exactly
/// this value (`Predict::prompt_prefix`), so recording it is an Arc clone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PrefixId(pub u32);

/// Per-trace interned model configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub u32);

pub type JsonMap = serde_json::Map<String, serde_json::Value>;

/// One Predict invocation: one rendered prompt in, one parsed output out,
/// with the tool loop's inner structure as ordered events.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub id: SpanId,
    /// Who ran. Resolve to the name via `Trace::components`.
    pub component: CompId,
    /// 0-based invocation index of this component within this rollout.
    /// `(component, seq)` is unique per trace — this is what makes loops
    /// addressable and what replay keys on.
    pub seq: u32,
    /// Enclosing span, if this Predict ran inside another span's tool
    /// execution (Predict-in-tool). Best-effort: set from the innermost open
    /// span at begin time; `None` at top level.
    pub parent: Option<SpanId>,
    /// Dataflow predecessors. v1 records the sequential approximation the old
    /// Graph recorded: `[previous span in this scope]`, empty for the first.
    /// Exact for hand-written sequential pipelines; honest-by-name about being
    /// an approximation elsewhere. Sufficient to reconstruct everything
    /// `trace::Graph` captured (insertion order was already topological).
    pub links: Vec<SpanId>,

    // ---- request (recorded eagerly at span open) ----
    /// Interned system+demos prefix. `None` when the call had no prefix
    /// (e.g. `forward_continue` on a caller-owned chat).
    pub prefix: Option<PrefixId>,
    /// The live suffix of the rendered prompt: the user turn (and, for
    /// multi-turn continuations, the full caller-provided history).
    /// `prefix + suffix` reconstructs the exact `Chat` sent.
    pub suffix: Vec<Message>,
    /// Signature input fields as JSON (what `raw_example_from_input` produces
    /// today). Demo harvesting (§4b) reads this; `None` for continuations
    /// where no typed input exists.
    pub input: Option<JsonMap>,
    pub model: ModelId,
    /// FNV/DefaultHasher over (resolved model config bytes ++ full rendered
    /// prompt), computed once at span close via the same streaming-Debug-hash
    /// technique as `LM::cache_key_for`. The replay key (§4d/e) and the
    /// Temporal-style determinism check.
    pub request_hash: u64,

    // ---- execution (filled at span close) ----
    /// Ordered inner events: provider round-trips and tool executions.
    /// Empty on the no-tools fast path *except* the single final Exchange.
    pub events: Vec<SpanEvent>,
    /// Final assistant text, pre-parse (`LMResponse.output.content()`).
    pub raw_output: Option<String>,
    /// Parsed signature output fields as JSON. `None` = call or parse failed.
    pub output: Option<JsonMap>,
    /// Aggregated across all exchanges (what `LMResponse.usage` already is).
    pub usage: LmUsage,
    pub error: Option<SpanError>,

    // ---- timing ----
    /// Microseconds since UNIX epoch.
    pub started_at_us: u64,
    pub duration_us: u64,

    /// False when any text field was truncated at serialization (§5.4).
    /// Replay refuses incomplete spans.
    #[serde(default = "default_true")]
    pub complete: bool,
}

fn default_true() -> bool { true }

/// Ordered events inside a span. Tagged; readers MUST skip unknown tags
/// (§6 — this is the streaming seam).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SpanEvent {
    /// One provider round-trip: the full assistant message (text, tool-call
    /// blocks, reasoning blocks — `Message` already models all of these) and
    /// that round-trip's own usage.
    Exchange { message: Message, usage: LmUsage },
    /// One tool execution between exchanges.
    ToolRun {
        /// Provider tool-call id (`rig::message::ToolCall::id`).
        id: String,
        name: String,
        args: serde_json::Value,
        /// Tool result text as fed back to the model.
        result: String,
        duration_us: u64,
        /// Tool-level failure that was reported back to the model as text
        /// (the loop today stringifies failures; record both).
        error: Option<String>,
    },
    /// RESERVED for streaming (§6). Not emitted in v1. A streamed predict
    /// will interleave these before the closing `Exchange`.
    Chunk { text: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpanError {
    pub kind: SpanErrorKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpanErrorKind {
    /// Provider/network failure (`PredictError::Lm`).
    Lm,
    /// Response received but output parsing failed (`PredictError::Parse`).
    /// `raw_output` is still populated in this case — parse failures are
    /// prime GEPA reflection material.
    Parse,
    /// A tool execution aborted the loop.
    Tool,
    /// Scope ended while the span was open (task cancelled).
    Cancelled,
}

/// Interned model configuration: LMConfig minus live state, minus secrets.
/// `LMConfig` already `#[serde(skip)]`s `api_key`; additionally `base_url`
/// is reduced to scheme+host+port at intern time (userinfo/query stripped).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelEntry {
    pub config: LMConfig,
    /// Hash of the redacted config; part of `request_hash`'s preimage.
    pub config_hash: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixEntry {
    /// System message + demo user/assistant turns, exactly as
    /// `Predict::build_prompt_prefix` produced them.
    pub messages: Vec<Message>,
}
```

### 1.1 Required change to `LMResponse`

The tool loop lives inside `LM::call_with_toolset`; the span is owned by `Predict`. Rather than threading a recorder through `LM` (task-local events would mis-attribute under `futures::join!` interleaving on one task), `LM` returns the breakdown and `Predict` writes the whole span at close:

```rust
// core/lm/mod.rs — additive
pub struct LMResponse {
    pub output: Message,
    pub usage: LmUsage,           // stays: aggregate
    pub chat: Chat,
    pub tool_calls: Vec<ToolCall>,       // deprecated by `events`, kept through PR-2
    pub tool_executions: Vec<String>,    // deprecated by `events`, kept through PR-2
    /// NEW: ordered per-round-trip record. One entry per provider call;
    /// `ToolRun` entries interleaved in execution order. Built by
    /// `execute_tool_loop` (it already collects all of this, just unordered
    /// and lossily — `ToolLoopResult` gains the same field internally).
    pub events: Vec<SpanEvent>,
}
```

`execute_tool_batch` already has per-tool timing in reach (wrap each call in `Instant::now()`); `ToolRun.duration_us` comes from there. Cache-served responses synthesize a single `Exchange` with the cached usage.

---

## 2. The container: `Trace`

```rust
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub meta: TraceMeta,
    /// CompId -> component name (fx name / dotted path / ParamPath).
    pub components: Vec<String>,
    pub models: Vec<ModelEntry>,
    pub prefixes: Vec<PrefixEntry>,
    /// Insertion order == start order == (for the sequential case) topological
    /// order — the same invariant the old Graph had.
    pub spans: Vec<Span>,
    /// Rollout-level result, filled after the scope closes / metric runs.
    pub outcome: Option<TraceOutcome>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TraceMeta {
    /// Format version. Bump on breaking change. This RFC: 1.
    pub v: u32,
    /// ULID/UUID string, minted at capture start.
    pub trace_id: String,
    pub started_at_us: u64,
    /// Hash of the candidate under evaluation (fx `Params` hash / future
    /// Overlay hash). Lets the flywheel and rollout cache key on
    /// (candidate, example) without parsing spans. Optional: plain runs
    /// have no candidate.
    pub candidate_hash: Option<u64>,
    /// Rollout input (the example's input fields), when the harness entry
    /// point recorded it. Replaces the old Graph's Root node and
    /// ExecutionTrace.inputs.
    pub input: Option<JsonMap>,
    /// Free-form run tags ("optimizer": "gepa", "gen": "3"). String-keyed —
    /// serde-boundary data, per the data-structure principle.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub tags: std::collections::BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceOutcome {
    /// Rollout output fields (the final Predicted, serialized).
    pub output: Option<JsonMap>,
    pub error: Option<String>,
    /// Metric result, if a metric ran over this rollout (§4c).
    pub eval: Option<Eval>,
    pub duration_us: u64,
}

/// The metric result type (§5.4 of the vision report). Replaces
/// `MetricOutcome` + `FeedbackMetric` in the trace-facing contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Eval {
    pub score: f64,
    pub feedback: Option<String>,
}
```

### 2.1 Accessors

```rust
impl Trace {
    pub fn component_id(&self, name: &str) -> Option<CompId>;
    pub fn component_name(&self, id: CompId) -> &str;

    /// GEPA's `pred_trace`: every invocation of one component, in order.
    /// Accepts the string form; id-form overload for hot paths.
    pub fn for_component<'a>(&'a self, name: &str)
        -> impl Iterator<Item = &'a Span> + 'a;
    pub fn for_component_id<'a>(&'a self, id: CompId)
        -> impl Iterator<Item = &'a Span> + 'a;

    /// Reconstructs the exact rendered prompt of a span (prefix ++ suffix).
    pub fn prompt(&self, span: &Span) -> Vec<Message>;

    /// Resolves a span's model config.
    pub fn model(&self, span: &Span) -> &LMConfig;

    /// Spans that completed with parsed output.
    pub fn successes<'a>(&'a self) -> impl Iterator<Item = &'a Span> + 'a;
}
```

`for_component` is a filter over `spans` (O(n)); traces are small (tens of spans). No index is built unless profiling says otherwise.

### 2.2 What is captured eagerly vs lazily

| When | What | Cost |
|---|---|---|
| span open (eager) | component intern, `seq`, `parent`, `links`, prefix intern (Arc-clone of Predict's cached prefix, hashed once per config), suffix messages (already built for the LM call — cloned `Vec<Message>`), `input` JsonMap (already serialized today when tracing), `started_at_us` | ~1 alloc + one map probe; suffix clone is the dominant cost, same as today's `input_data` recording |
| span close (lazy) | `events` (moved out of `LMResponse`, not cloned), `raw_output`, `output` JsonMap, `usage`, `request_hash` (streaming hash, no materialization), `error`, `duration_us` | moves + one hash pass |
| never during capture | JSON serialization, truncation, redaction — all serialization-time (§5) | 0 |
| no scope active | one `task_local` `try_with` probe in `Predict` | same as today's `is_tracing()` |

The eager half exists so a call that dies mid-flight still leaves `(component, seq, prompt, input, started_at)` in the trace — blame assignment for pipeline failures, exactly the property the current two-phase `record_node`/`record_output` protects.

---

## 3. Capture mechanism

`crates/dspy-rs/src/trace/capture.rs`. Same shape as today's `trace::trace()`; renamed `capture` to make the old API's death loud at compile time rather than silent behavior change.

```rust
use std::sync::{Arc, Mutex};
use tokio::task_local;

task_local! {
    static ACTIVE: TraceSink;
}

#[derive(Clone)]
pub struct TraceSink(Arc<Mutex<SinkInner>>);

struct SinkInner {
    trace: Trace,
    /// Stack of open span ids — parent attribution for Predict-in-tool.
    open: Vec<SpanId>,
    /// (CompId -> next seq) counters.
    seqs: Vec<u32>,
    /// Prefix/model intern maps: content-hash -> id.
    prefix_index: std::collections::HashMap<u64, PrefixId>,
    model_index: std::collections::HashMap<u64, ModelId>,
}

/// Runs `f` while recording every Predict call on this task into a Trace.
/// Mirrors the old `trace::trace()` contract exactly: task-local scoping,
/// spawned subtasks do NOT inherit the scope, result + trace returned.
pub async fn capture<F, Fut, R>(f: F) -> (R, Trace)
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = R>,
{
    let sink = TraceSink::new();
    let result = ACTIVE.scope(sink.clone(), f()).await;
    (result, sink.finish())   // Arc::try_unwrap, clone-fallback like today
}

/// Variant that also records rollout input/candidate into TraceMeta.
pub async fn capture_with_meta<F, Fut, R>(meta: TraceMeta, f: F) -> (R, Trace);

pub fn is_capturing() -> bool {
    ACTIVE.try_with(|_| ()).is_ok()
}
```

### 3.1 Instrumentation API (used by `Predict`; public for custom leaf modules)

```rust
/// Opens a span. Returns None when no scope is active — the caller does
/// nothing else in that case (zero-overhead path).
pub fn begin_span(req: SpanRequest<'_>) -> Option<SpanGuard>;

pub struct SpanRequest<'a> {
    pub component: &'a str,          // trace_name / fx name / dotted path
    pub prefix: Option<&'a [Message]>,
    pub suffix: &'a [Message],
    pub input: Option<JsonMap>,
    pub model: &'a LMConfig,
}

/// Owns the SpanId — event/close attribution travels with the guard, never
/// through "innermost open span" lookups, so interleaved Predicts on one
/// task (futures::join!) cannot cross-attribute. Dropping the guard without
/// `finish` marks the span `error: Cancelled`.
pub struct SpanGuard { id: SpanId, sink: TraceSink }

impl SpanGuard {
    pub fn finish(self, out: SpanOutcome);
}

pub struct SpanOutcome {
    pub events: Vec<SpanEvent>,
    pub raw_output: Option<String>,
    pub output: Option<JsonMap>,
    pub usage: LmUsage,
    pub error: Option<SpanError>,
}
```

`Predict::call_and_parse_with_input` becomes:

```rust
let guard = crate::trace::begin_span(SpanRequest {
    component: self.trace_name.as_deref().unwrap_or(FALLBACK), // §8 Q4
    prefix: self.prompt_prefix.get().map(Vec::as_slice),
    suffix: &live_messages,
    input: input_data,             // only serialized when a guard was returned
    model: &lm.config,
});
// ... existing LM call ...
if let Some(guard) = guard {
    guard.finish(SpanOutcome { events: response.events, /* ... */ });
}
```

### 3.2 Nesting semantics

- **Nested `capture` scopes: innermost wins, exclusively.** The inner scope's spans go only to the inner trace; the outer trace records nothing for that region. This is today's `task_local` behavior and it is the correct default: it is exactly how MIPRO already keeps LM-as-judge metric calls out of the execution graph ("metric evaluation happens outside the trace scope"). An LM judge that wants isolation wraps itself in its own `capture`.
- Spawned tasks (`tokio::spawn`) do not inherit the scope — unchanged. A harness that fans out and wants unified traces captures in each subtask and splices: `Trace::absorb(&mut self, other: Trace)` remaps ids/interns and appends (provided for the eval engine's fan-out; ordering across absorbed traces is by `started_at_us`).
- `parent` is set from the top of the open-stack at `begin_span` time. Under single-task sequential execution (the overwhelmingly common case, incl. Predict-inside-tool) this is exact; under same-task interleaving it can mis-parent and `links` degrades to the sequential approximation — the same honesty level as the old Graph's `inputs`, now documented on the field.

---

## 4. Consumers

### (a) GEPA per-component reflection

GEPA today never sees a trace — it gets `MetricOutcome` strings. The rewrite (vision §5.4) gets rollout traces from the eval engine and slices:

```rust
// inside GEPA's reflection step, for component `name`:
let mut examples_text = String::new();
for (i, (trace, eval)) in minibatch_results.iter().enumerate() {
    for span in trace.for_component(name) {
        // rendered instruction context + this invocation's I/O
        let prompt = trace.prompt(span);                  // Vec<Message>
        write!(examples_text,
            "### Example {i}, call {} \nInput: {}\nOutput: {}\n{}Score: {:.3}\nFeedback: {}\n",
            span.seq,
            json(&span.input), 
            span.output.as_ref().map(json).unwrap_or_else(|| 
                format!("<{}: {}>", span.error.as_ref().unwrap().kind_str(),
                        truncate(span.raw_output.as_deref().unwrap_or(""), 500))),
            format_tool_runs(&span.events),               // tool behavior is reflectable too
            eval.score,
            eval.feedback.as_deref().unwrap_or("-"),
        )?;
    }
}
// -> ReflectOnInstructionInput { task_description, current_instruction, execution_feedback: examples_text }
```

Parse failures carry `raw_output` — "the model wrote prose instead of `[[ ## answer ## ]]`" is visible to the reflector. This is the pred_trace contract: per-component, per-invocation, loops included.

### (b) MIPRO-style demo harvesting (replaces `mipro::Trace<S>` and the `instance_keys` join)

`generate_traces_with_bootstrap` currently joins `NodeType::Predict.instance_key` (a raw pointer!) against facet paths, and separately accumulates `mipro::Trace<S>` triples. Both collapse into:

```rust
let (result, trace) = capture(|| module.call(input.clone())).await;
let eval = metric.evaluate(&example, &predicted, Some(&trace)).await?;

if eval.score >= self.min_demo_score {
    for span in trace.successes() {
        let name = trace.component_name(span.component);
        if let (Some(inp), Some(out)) = (&span.input, &span.output) {
            demo_candidates.entry(name.to_string())
                .or_default()
                .push((eval.score, demo_from_json(inp, out)));  // -> Example row
        }
    }
}
// whole-program "trace" for candidate generation = (trace.meta.input,
// trace.outcome.output, eval.score) — the mipro::Trace triple is just a
// projection of TraceMeta/TraceOutcome. Delete the struct.
```

No pointer identity, no `predictor_instance_keys`, works identically for fx and struct harnesses.

### (c) Metric feedback (replaces `evaluate::ExecutionTrace` — which is already dead: zero consumers outside its own file)

The metric trait gains trace access; the eval engine owns the capture:

```rust
// evaluate/evaluator.rs
#[allow(async_fn_in_trait)]
pub trait TypedMetric<S, M>: Send + Sync
where S: Signature, M: Module<Input = S::Input>,
{
    async fn evaluate(
        &self,
        example: &Example<S>,
        prediction: &Predicted<M::Output>,
        trace: Option<&Trace>,           // NEW — None when caller didn't capture
    ) -> Result<Eval>;
}

// eval engine inner loop:
let (result, mut trace) = capture_with_meta(meta_for(example), || module.call(input)).await;
let predicted = result?;
let eval = metric.evaluate(example, &predicted, Some(&trace)).await?;
trace.outcome = Some(TraceOutcome { output: Some(json_of(&predicted)),
                                    eval: Some(eval.clone()), .. });
// (candidate_hash, example_id) -> trace lands in the rollout cache / flywheel
```

Metrics that inspect intermediate steps ("did the retriever return the gold doc?") read `trace.for_component("retriever")` instead of the never-populated `ExecutionTrace::intermediate_steps`. `FeedbackMetric`'s formatting helpers (`feedback_helpers.rs`) keep working — they just return `Eval` (`score` widened to f64, `metadata` folded into the feedback text or dropped; see §7).

### (d) Record/replay test fixtures

A trace is a complete set of canned LM responses. Replay is a task-local scope symmetric to `capture`, intercepted in `Predict` **above** the LM (so no client is constructed, no tool executes — tool effects are already baked into the recorded events):

```rust
// trace/replay.rs
pub async fn replay<F, Fut, R>(trace: &Trace, mode: ReplayMode, f: F) -> (R, ReplayReport);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayMode {
    /// Every span request must match its recording: next span for
    /// (component, seq) must exist and request_hash must equal. Mismatch =
    /// error (Temporal's determinism check). For fixtures/CI.
    Strict,
    /// Serve recorded spans while request_hash matches; on first mismatch,
    /// that component call and ALL subsequent calls go live. For
    /// counterfactual replay (§e).
    UntilDivergence,
}

pub struct ReplayReport {
    pub served: usize,
    pub live: usize,
    pub diverged_at: Option<SpanId>,
}
```

`Predict::call_and_parse_with_input` checks the replay scope first: if active, it computes `request_hash` for the chat it *would* send, asks the scope for the next span of `(component)`; on hash match it deserializes `span.output` into `S::Output` (via the same `serde_json::from_value` path `typed_example_from_raw` uses) and returns a synthesized `Predicted` without touching `LM`. Spans with `complete == false` or `output == None` are refused in `Strict` (error) and treated as divergence in `UntilDivergence`. Fixture ergonomics:

```rust
let (out, trace) = capture(|| pipeline(input.clone())).await;      // once, with a key
std::fs::write("fixtures/qa.trace.jsonl", trace.to_jsonl()?)?;

// in tests, forever, zero API calls:
let trace = Trace::from_jsonl(&std::fs::read_to_string("fixtures/qa.trace.jsonl")?)?;
let (out, report) = trace::replay(&trace, ReplayMode::Strict, || pipeline(input)).await;
assert_eq!(report.live, 0);
```

### (e) Counterfactual replay from step k

Counterfactual replay = `UntilDivergence` + a mutated candidate. No "step k" parameter exists or is needed — divergence is *detected*, not declared, because a mutated parameter changes the rendered prompt (and therefore `request_hash`) of exactly the calls it affects:

```rust
let mut candidate = base_params.clone();
candidate.set_instruction("refiner", mutated_instruction);
let ((result, new_trace), report) = trace::replay(&base_trace, ReplayMode::UntilDivergence, || {
    capture(|| fx::with_params(candidate, pipeline(input)))
}).await;
// spans before the first "refiner" call: served free from base_trace
// "refiner" and everything after: live LM calls only
```

What must be recorded for prefix replay to be exact — all already in the format:
1. `request_hash` preimage = redacted model config + full rendered prompt → any change to instruction, demos, model, temperature, or upstream output changes the hash. Prefix intern does not weaken this: the hash is over reconstructed `prefix ++ suffix`.
2. Full `output` JsonMap per span (the value fed downstream) — replay reproduces downstream prompts byte-identically.
3. `(component, seq)` ordering — loop iterations replay in order.
4. Tool effects: recorded `ToolRun` results are *inside* served spans; nothing re-executes. **Limitation to document:** once diverged, live tool calls re-run against the real world; counterfactual replay of side-effectful harnesses requires the phase-4 resettable `Environment`. The format is ready; the world is not.
5. Sampling nondeterminism: a live temperature>0 prefix would not reproduce itself — irrelevant, because the prefix is *served from the recording*, never re-sampled. That is the Temporal insight: record the nondeterministic activity results, replay them, never re-execute.

### (f) RL export (Agent Lightning / verifiers convention)

One rollout → one JSONL record: message lists + reward + per-subcall spans. Pure projection, feature-gated nothing — it's ~80 lines over the format:

```rust
// trace/export.rs
#[derive(Serialize)]
pub struct RlRollout<'a> {
    pub trace_id: &'a str,
    pub reward: f64,                       // trace.outcome.eval.score
    pub transitions: Vec<RlTransition<'a>>,
    pub metadata: &'a std::collections::BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct RlTransition<'a> {
    /// The optimizable unit this subcall belongs to ("drafter") — Agent
    /// Lightning's per-agent credit assignment key.
    pub component: &'a str,
    pub seq: u32,
    /// Full prompt as messages (prefix ++ suffix), provider-agnostic roles.
    pub messages: Vec<Message>,
    /// Everything the policy emitted for this subcall: each Exchange's
    /// assistant message, with ToolRuns as intervening tool-role context.
    pub completion: Vec<Message>,
    pub usage: LmUsage,
    pub model: &'a str,
}

impl Trace {
    pub fn to_rl_rollout(&self) -> Option<RlRollout<'_>>; // None if no eval recorded
}
```

`completion` is rebuilt from `events`: `Exchange.message` verbatim; `ToolRun` → a user-role tool-result message (same shape `push_tool_results` builds). Because spans keep full `Message` structure (tool-call blocks, reasoning blocks), the export needs no lossy text munging — this is why spans store `Message`, not strings.

### (g) OTel export

Feature `otel`. One-way mapping onto the GenAI semantic conventions:

| Trace format | OTel |
|---|---|
| `trace_id` | trace id (hashed to 128-bit) |
| `Span` | span, name = component name, kind = `CLIENT` |
| `span.parent` (else trace root span) | parent span id |
| `started_at_us` / `duration_us` | start/end timestamps |
| `models[span.model].config.model` | `gen_ai.request.model` |
| `config.temperature`, `config.max_tokens` | `gen_ai.request.temperature`, `gen_ai.request.max_tokens` |
| `usage.prompt_tokens` / `completion_tokens` | `gen_ai.usage.input_tokens` / `output_tokens` |
| `prefix ++ suffix` | `gen_ai.prompt` events (opt-in, content capping per OTel content-capture switch) |
| `raw_output` | `gen_ai.completion` event |
| `SpanEvent::ToolRun` | child span, name = `tool:{name}`, `gen_ai.tool.name`, args/result as attributes |
| `error` | span status `ERROR` + `exception.message` |
| `seq`, `component`, `request_hash`, `candidate_hash` | `dsrs.seq`, `dsrs.component`, `dsrs.request_hash`, `dsrs.candidate_hash` attributes |

`utils/telemetry.rs` (tracing-subscriber based) is unaffected; this is batch export of finished traces (`Trace::to_otel(&self, tracer: &opentelemetry::trace::Tracer)`), which is what the serving host (§6.3 of the vision) calls per request.

---

## 5. Serialization

### 5.1 Wire format: JSONL, one trace per file/stream

```
{"h":{"v":1,"trace_id":"01J...","started_at_us":...,"candidate_hash":...,"input":{...},"tags":{...}},"components":["drafter","refiner"],"models":[{...}],"prefixes":[{...}]}
{"id":0,"component":0,"seq":0,"prefix":0,"suffix":[...],"input":{...},"model":0,"request_hash":...,"events":[...],"raw_output":"...","output":{...},"usage":{...},"started_at_us":...,"duration_us":...}
{"id":1,"component":1,"seq":0,...}
{"f":{"output":{...},"eval":{"score":0.8,"feedback":"..."},"duration_us":...}}
```

- Line 1: header (meta + intern tables). Span lines follow in order. Optional footer line (`TraceOutcome`) — appended after the metric runs, which is why it is a separate line: the flywheel can write spans as rollouts finish and attach evals later without rewriting.
- `Trace` also derives plain `Serialize`/`Deserialize` as a single document — used when embedding traces in other artifacts (`GEPAResult`, checkpoint files). JSONL is canonical for disk/wire (`.trace.jsonl`); the single-doc form is a convenience, same types.
- Components are written as table indices on span lines (compact) but the header carries the string table, so files stay greppable via one line. The in-memory form is always the interned id — string-keyed data exists only at this boundary, per the vision's data-structure principle.
- Unknown fields and unknown `SpanEvent` tags are skipped on read (`#[serde(other)]` fallback variant `SpanEvent::Unknown` marked `#[doc(hidden)]`, dropped on re-serialize). Same-major-version forward compatibility = additive fields only; anything else bumps `v`.

### 5.2 Versioning

`h.v: u32`, this RFC = `1`. Readers reject `v` greater than they know. Additive evolution (new optional span fields, new event tags) does not bump. Semantics changes (span granularity, hash preimage) do.

### 5.3 Redaction

- **LM credentials never enter the types**: `LMConfig.api_key` is `#[serde(skip)]` already; `ModelEntry` interning additionally reduces `base_url` to origin (strip userinfo, path, query — vLLM tokens hide in query strings).
- Everything else in a span is user/task content by definition (prompts, outputs, tool args/results). The format does not guess at PII; the flywheel harvester owns content policy. One structural hook: `Trace::redact(&mut self, f: impl FnMut(&mut Span))` for callers that must scrub before persisting. Redaction invalidates `request_hash` replay — `redact` sets `complete = false` on touched spans.

### 5.4 Size budget and truncation

Targets (serialization-time, never during capture):

| Field | Budget | Overflow behavior |
|---|---|---|
| prefix entry | unbounded (interned once/trace) | — (demos are the payload; do not truncate) |
| `suffix` per message text block | 64 KB | truncate, `complete=false`, append `"…[truncated sha256=<hex> len=<n>]"` |
| `raw_output` | 64 KB | same |
| `ToolRun.result` | 16 KB | same (matches what context engineering feeds back anyway) |
| `ToolRun.args` | 16 KB | same, serialized form |
| `output` / `input` JsonMaps | unbounded | — (these are the data; if they're huge, the harness is) |
| whole span, typical | ~2–16 KB | expected envelope, not enforced |
| whole trace | ~10–200 KB for a 5-predict pipeline with demos | prefix interning is what keeps the demo-heavy case at the low end |

Hashes (`sha256` of full pre-truncation content) accompany every truncation so fixtures can detect drift even when content is cut. Truncated spans refuse `Strict` replay (§4d).

---

## 6. Streaming decision

**Decision: v1 records final-only; the format is streaming-ready via `events`, and this is frozen now** (vision §6.1 requires deciding before the format freezes).

- A streamed predict, when it lands, appends `SpanEvent::Chunk { text }` entries as deltas arrive, then closes with the same final `Exchange` + `raw_output` + `output` fields as today. Non-streamed consumers (GEPA, MIPRO, replay, RL export) read only the final fields and are oblivious. Trace-tailing consumers (serving host debuggers) read chunks. No version bump: `SpanEvent` is `#[non_exhaustive]` + skip-unknown on the wire (§5.1).
- BAML-style *semantic* streaming (typed partial outputs) adds `SpanEvent::PartialOutput { fields: JsonMap }` under the same rule — additive tag.
- What the type carries today to make that safe: (1) `events` is ordered and heterogeneous, (2) event tags are strings on the wire, (3) `Span` final fields are `Option`, so a span that is open-and-streaming serializes coherently mid-flight (this is also what makes the execution-state artifact of vision §6.2 a prefix of a trace file — a checkpoint is a header + complete span lines + one incomplete span).
- Capture cost note: chunk events buffer in the `TraceSink` mutex like everything else; if profiling shows contention under high-fanout streaming, the fix is a per-span `Vec` behind the `SpanGuard` (guard-local, merged at `finish`) — an implementation change, not a format change.

---

## 7. Migration plan

### 7.1 What dies

| Deleted | Replaced by |
|---|---|
| `trace::dag::{Graph, Node, NodeType}` | `Trace`, `Span` (`links` covers `inputs`; `TraceMeta.input` covers Root; Operator/Map node types die with the already-dead `trace::Executor`) |
| `trace::context::{trace, is_tracing, last_node_id, record_node, record_output}` | `capture`, `is_capturing`, `begin_span`/`SpanGuard` |
| `trace::value::{TrackedValue, IntoTracked}` | nothing (only consumer is `Prediction::get_tracked`, which dies with `Prediction`) |
| `optimizer::mipro::Trace<S>` | `TraceMeta.input` + `TraceOutcome` projection (§4b) |
| `evaluate::{ExecutionTrace, ExecutionTraceBuilder}` | `Trace` + `Eval` (already zero consumers — pure deletion) |
| `node_id: Option<usize>` on `Prediction`, `data::Example`, `CallMetadata` | `CallMetadata.span_id: Option<SpanId>` (set when a scope was active) |
| `MetricOutcome`/`FeedbackMetric` *in the metric return position* | `Eval { score: f64, feedback: Option<String> }`; `feedback_helpers.rs` constructors return `Eval` (metadata maps fold into feedback text — GEPA only ever read the text) |
| `LMResponse.{tool_calls, tool_executions}` (after PR-2) | `LMResponse.events` |
| `predictor_instance_keys` (pointer-identity join) | span `component` names |

### 7.2 PR sequence

1. **PR-1 `trace-core`** — new `src/trace/{span,capture,serialize}.rs` types + `capture()` + `begin_span`; `LMResponse.events` populated in `execute_tool_loop`/`call_with_toolset`; `Predict` records into the new sink *and* the old Graph (dual-write; both are task-local probes, negligible). Golden-file JSONL round-trip tests. Nothing deleted; CI green throughout.
2. **PR-2 `consumers`** — `TypedMetric::evaluate` gains `trace: Option<&Trace>` param and returns `Eval` (mechanical churn across metrics/tests); MIPRO demo harvesting moves to §4b and `mipro::Trace` is deleted; `ExecutionTrace` deleted; GEPA's `summarize_feedback` optionally enriched with `for_component` (behavioral no-op if not).
3. **PR-3 `delete-old`** — remove `trace::{dag,context,value}` and dual-write; `node_id` fields removed, `CallMetadata.span_id` added; examples 12/14 and tests move to `capture()`; `predictor_instance_keys` deleted.
4. **PR-4 `replay`** — `replay()` scope, `Strict` fixtures (convert one live test to a recorded fixture as proof), `UntilDivergence` (counterfactual foundation).
5. **PR-5 `export`** — `to_rl_rollout` JSONL + `otel` feature mapping.

PR-1/2/3 land before or in parallel with the eval engine (vision phase 2) — the engine consumes `capture` + `Eval` from day one. PR-4/5 are phase-5 flywheel work that can trail.

### 7.3 Carrier-collapse interaction (5→2)

The format's type dependencies, chosen so the two efforts cannot deadlock:

- **Depends on (stable, survives collapse):** `Message`/`ContentBlock`/`Chat` (serde since the LM split), `LmUsage`, `LMConfig`, `serde_json::Map`.
- **Deliberately does NOT depend on:** `RawExample` (`data::Example`), `Prediction`, `Example<S>`, `Predicted<O>`. Spans store `JsonMap` for parsed input/output — the old Graph's `input_data: Option<RawExample>` / `output: Option<Prediction>` were the trace system's only structural grip on the two carriers most likely to die in the collapse; this RFC severs it. Consumers (§4) touch typed carriers only at their own edges (`Example<S>` in metric signatures, `Predicted` in replay synthesis), where the collapse will rename but not reshape.
- **Flags for the collapse effort:** (1) `Prediction::get_tracked`/`TrackedValue` and all three `node_id` fields are trace-owned — PR-3 removes them, shrinking `Prediction`/`data::Example` before the collapse touches them; do not collapse those fields "for" the trace. (2) `PredictState.demos: Vec<RawExample>` (persistence format) is untouched by this RFC — if the collapse replaces `RawExample` there, prefix interning is unaffected (prefixes store rendered `Message`s, not demos-as-data). (3) Replay synthesizes `Predicted<S::Output>` via `serde_json::from_value` — whatever the collapse renames `Predicted` to must keep a `(output, metadata)` constructor.

---

## 8. Open questions

1. **Span granularity: is a tool-looping Predict one span or N?** — *Recommended (and specified above): one span with N events.* The Predict invocation is the attribution unit for every optimizer consumer; RL export flattens events losslessly. Revisit only if a trainer needs per-round-trip rewards, which is a projection change, not a format change.
2. **Do nested `capture` scopes propagate to the outer scope?** — *Recommended: no (innermost-only), as specified.* Matches current semantics and the LM-judge-isolation idiom; `Trace::absorb` covers deliberate merging. Broadcast-to-all-scopes buys nothing today and costs a probe per scope per event.
3. **What component name does an unnamed struct-field `Predict` record?** — *Recommended:* PR-1 falls back to `signature_name` with a `dsrs.unnamed_component` warning; PR-3 makes the derive/`#[predict]` macros emit the field's dotted path as `trace_name` at construction (the static-lane parameter-enumeration work already planned in vision §5.2), making the fallback unreachable. Do not ship silent empty names — that is the facet silent-discovery bug wearing a new hat.
4. **`request_hash` algorithm: `DefaultHasher` (like `cache_key_for`) or a stable hash?** — *Recommended: stable `xxhash64` (or FNV-1a) over a canonical byte encoding, from day one.* `DefaultHasher` is not stable across Rust releases; replay fixtures must survive toolchain upgrades. `cache_key_for` should migrate to the same function in PR-1 (one hasher, two callers).
5. **Should `Eval` carry the vector scores the `FeedbackMetric` TODO wants (multi-objective Pareto)?** — *Recommended: not in v1; reserve `Eval.scores: Option<Vec<(String, f64)>>` as an additive field when the eval engine's Pareto bookkeeping needs it.* The trace format only stores `Eval`; widening it later is additive under §5.1's rules and blocks nothing now.
