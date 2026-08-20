# RFC 0004 — Remaining seams (post-unification ledger)

Status: informational. The v1 unification (phases 1–4) put `Predict` on the IR
interpreter, unified the optimizers over one engine with ambient candidate
injection, and collapsed the build-time feature fiction. What follows is the
deliberate remainder: seams we know about, left open on purpose, each with a
suggested shape for whoever closes it.

## 1. Conversation surface — `TODO(dsrs-phase4-conversation)`

**What:** multi-turn conversations still run on the LM-layer compat path.
`Predict::build_chat` / `call_and_parse` hand a caller-owned `Chat` to the LM
client directly, bypassing the interpreter, because the interpreter's only
entry is map-in/map-out (`Interpreter::run_collecting`).

**Why deferred:** giving the interpreter a conversation-in/conversation-out
surface touches its span model (a turn is not a run) and the replay contract,
and nothing in the optimizer stack needs it yet.

**Suggested shape:** an interpreter-native entry that accepts a prior
conversation and returns the extended one alongside outputs — e.g.
`run_conversation(chat, input, overlay, budget) -> (RunOutput, Chat)` — with
the `AgentLoop` machinery reused for turn bookkeeping. `build_chat` /
`call_and_parse` then become thin wrappers and the static prompt-prefix cache
in `Predict` can be deleted.

## 2. Caller-managed tool loop — `TODO(dsrs-phase4-caller-managed)`

**What:** `ToolLoopMode::CallerManaged` (the "return me the tool calls, I'll
execute them" pattern) lives on the LM-layer path
(`core/lm/mod.rs`, used by `Predict::call_and_parse_with_input`). Typed
`call`s with tools already run through the interpreter's `AgentLoop`; the
caller-managed variant does not.

**Why deferred:** it is built on caller-owned chats, so it is blocked on seam
1 — the interpreter cannot yield mid-loop to an external executor today.

**Suggested shape:** once the conversation surface exists, express
caller-managed as an `AgentLoop` that suspends on tool calls instead of
dispatching them: return pending calls plus a resumption token, and let the
caller feed results back in. That keeps trace spans, budget metering, and
stop-tool semantics identical across both modes.

## 3. Shared-pointer traversal — `TODO(dsrs-shared-ptr-policy)` (retired)

**What:** the old facet reflection walker refused to traverse `Rc<T>`/`Arc<T>`
containers when discovering `Predict` leaves, with an explicit error carrying
this marker.

**Why it's gone:** phase 3 deleted the walker entirely — leaf discovery is now
explicit via the `Predictors` trait (`predictors!` macro), so there is no
container traversal left to have a policy about. The marker survives only in
`docs/specs/modules/*` prose describing the deleted design; treat those
documents as historical.

**Suggested shape:** none. If reflection-based discovery ever returns, the
policy question returns with it; the explicit-declaration contract made it
moot.

## 4. Whole-rollout credit assignment (`optimizer/harvest.rs`)

**What:** demo harvesting scores every span in a rollout with the rollout's
single metric score — a good final answer marks *all* intermediate predictor
calls as good demos, including any that a later step had to recover from.

**Why deferred:** per-leaf credit needs either per-span evals in the trace or
a counterfactual scorer, and the shipped optimizers (Bootstrap, MIPROv2) work
acceptably on whole-rollout signal for the shallow programs people build
today.

**Suggested shape:** the trace format already carries per-span `Eval` records;
let metrics optionally attach span-level scores during evaluation
(`TypedMetric` gains a per-trace hook), and have `harvest.rs` prefer a span's
own eval over the rollout score when present. Deeper counterfactual schemes
(ablate-one-span replays) can layer on the replay machinery later.

## 5. Tool membership as a `ParamSlot` (the ToolSet gene)

**What:** which tools an `AgentLoop` carries is structural today
(`AgentLoopNode::tools: Box<[ToolId]>`); only each tool's *description* is an
optimizable slot (`ParamKind::ToolDesc`). An optimizer can rewrite what a tool
says it does, but not drop a distracting tool or add a relevant one.

**Why deferred:** tool membership changes the capability footprint of a node,
so a membership gene has to interact with the program's cap-ceiling validation
— an overlay must not be able to smuggle in a tool the program's grants don't
cover.

**Suggested shape:** a `ParamKind::ToolSet` slot per agent node whose value is
a subset of the *declared* tool table (declaration stays structural, selection
becomes a gene). Validation stays load-time: the legal alphabet is the
declared tools, so any subset is capability-safe by construction. Mutation
proposals then compose with overlays like any other slot value.

## 6. Structural optimizers over `ir::Edit`

**What:** the edit calculus is fully shipped — `Edit`, `Program::edited`,
`Program::legal_edits` (the menu of applicable edits per node), and
`migrate_overlay` for carrying tuned slot values across a structural change —
but no shipped optimizer proposes edits. All five strategies tune
instructions/demos through overlays only.

**Why deferred:** structural search needs an evaluation budget model (every
candidate is a new program that must be re-scored from scratch, minus what
`migrate_overlay` preserves) and a proposal policy; both are research-shaped
rather than plumbing-shaped.

**Suggested shape:** a GEPA-style loop where the reflection step prompts over
`legal_edits` output (the menu is already serializable data), applies the
chosen `Edit` via `Program::edited`, migrates the incumbent overlay, and
scores the child against the parent on a shared minibatch. The engine's
candidate machinery already treats programs as data, so this slots in as a
sixth strategy rather than a new framework.
