# Handoff — `v1-program-unification` branch

Written 2026-08-20 for whoever takes this branch to `main`. This is the working
ledger of what was done, what is *verified*, what is *claimed but unverified*,
and what the next owner must do before trusting or merging it.

## 1. Branch state

- Branch: `v1-program-unification`, off `main` @ `4435b76`.
- Net diff vs main: **203 files, +12,131 / −8,592** (before the seam merges;
  the three seam merges add ~+3,300 more).
- Structure: nine phase merges + three seam merges. Read it top-down with
  `git log --oneline --merges v1-program-unification ^main`.

| Merge | What | Author session |
|---|---|---|
| Wave 1a–1e | Dead-code kill list, sandbox hardening, `dsrs-syntax` crate, `ir/edit.rs` calculus, workspace hygiene | session A (this ledger's author) |
| Phase 2α/2β | ReAct/ModuleExt deleted; interpreter leaf metadata; **Predict demoted onto the IR Interpreter**; adapter static lane deleted | session A |
| Phase 3 | Object-safe `Optimizer`, unified `Engine`, ambient candidate injection, `DynPredictor`/facet-walker/mutation model deleted, **facet fork unpinned** | session A |
| Phase 4a/4b | Feature collapse (+ `data` feature), fx CLOCK cache, `prelude`, `#[agent]` options honored, full docs sweep, RFC 0004 seams ledger | session A |
| Seam 6 (`ca14e08`) | `Structural` — sixth optimizer, LM-guided edit proposals over `ir::Edit` | **session B — NOT reviewed by session A** |
| Seam 4 (`109c571`) | Per-span credit assignment in demo harvesting | **session B — NOT reviewed by session A** |
| Seam 5 (`16f2b27`) | Tool membership as a `ParamSlot` (`ParamKind::ToolSet`, touches `.dsrs` parse/print) | **session B — NOT reviewed by session A** |

## 2. What is verified, and how

- **Phases 1–4**: full workspace suite green after every merge (79 suites; the
  count moved 80→77→79 as test files were deleted/added — each delta was
  accounted for). `cargo check -p dspy-rs --no-default-features` compiles;
  35 lib tests pass without default features.
- **Prompt stability**: golden prompt tests byte-identical through the adapter
  collapse (they now exercise the single `*_def` lane).
- **`.dsrs` hash stability through the `dsrs-syntax` extraction**: fixture
  programs printed + hashed before/after — byte-identical (verified in wave 1c).
- **Sandbox fixes**: 7 regression tests (injection rejected, `JSON`-named tool
  safe, capability timeout fires, deregister evicts bytecode, current-thread
  runtime works).
- **Soundness**: `grep -rn "unsafe" crates/dspy-rs/src` → zero. The facet
  `[patch.crates-io]` fork pin is gone; upstream facet 0.43 builds and tests.
- **Seams 4/5/6 (session B)**: the suite at HEAD was re-run by session A after
  discovering these merges — result recorded below in §3 item 0. Beyond that,
  session A has only *skimmed* this code. It has not been reviewed.

## 3. Validation checklist for the next owner (priority order)

0. **Confirm HEAD is green.** `cargo test --workspace` at `16f2b27`.
   Session A's result (2026-08-20): exit 0, 80/80 suites ok, zero failures —
   the seam merges pass on top of the phases. Cheap to re-run; do so anyway.
1. **LIVE LM smoke test — the single biggest gap.** Every test in every phase
   ran against mocked completions (`TestCompletionModel`). The demoted
   `Predict` path, the AgentLoop-backed `with_tools` path, and the
   `Structural` optimizer have **never talked to a real provider** on this
   branch. Run with a real key: examples 01–05, 22 (frontdesk module),
   18 (code-mode), 09 (GEPA, small budget), and a small `Structural` run.
2. **Review the three seam merges** (`ca14e08`, `109c571`, `16f2b27`) — they
   are unreviewed by anyone except their authoring session. Specifically:
   - Seam 5 changed `ir/text/parse.rs` + `print.rs` (the ToolSet gene). The
     canonical text is the **program-hash preimage**: confirm whether the
     grammar change invalidates pre-existing `.dsrs` artifacts and whether
     that was deliberate (pre-1.0 it's acceptable, but it must be a decision,
     not an accident). Check `test_ir_text` / `test_ir_bake` diffs in those
     merges for regenerated expectations — regenerated goldens are a red flag
     to inspect, not a proof of correctness.
   - Seam 4 changes demo quality for Bootstrap/MIPRO/SIMBA (per-span credit
     instead of whole-rollout). That is a *behavioral* change to optimizer
     output — eyeball a before/after demo harvest on a real trainset.
   - Seam 6 (`optimizer/structural.rs`): review the accept/reject gate, the
     menu serialization the reflection LM sees, and `migrate_overlay` usage
     across accepted edits. This is the flagship feature; it deserves the
     most careful read.
3. **Performance before/after.** Never measured. Run
   `examples/97-perf-microbench` (and 98/99 orchestration benches) on `main`
   vs this branch. The demoted Predict adds program-cache lookup + serde
   round-trip + error translation per call; the claim that this is noise
   against network latency is *plausible, not proven* — and it is NOT noise
   for cache-hit or replay-served calls, which skip the network entirely.
4. **The task-local footgun.** Ambient candidates (`fx::with_params` /
   `with_ambient_overlay`) do not propagate into `tokio::spawn`ed tasks. A
   module whose `forward` spawns tasks silently evaluates the *baseline*
   during optimization. Write the demonstrating test, then decide: document
   loudly, detect-and-warn (e.g. a capture-scope generation counter), or fix
   (explicit context handle instead of task-locals). Until then any
   user-written module with internal spawns gets silently wrong optimization.
5. **Publishability.** `cargo publish --dry-run` for `dsrs-syntax`,
   `dsrs-tools`, `dsrs_macros`, `dspy-rs` (in that order). The fork pin is
   gone but rig-core + minijinja are still git dependencies — crates.io will
   reject those; decide the strategy (vendor, fork-publish, or wait).
6. **Docs build.** `docs/` was fully swept and API pages regenerated
   (`docs/scripts/gen_api.py` — script-generated, do not hand-edit), but
   nobody ran the Mintlify build/link check. Also verify session B's
   `docs/docs/optimizers/structural.mdx` against the actual implementation,
   and note RFC 0004 now has status annotations added by session A.
7. **Clippy + version bumps.** `cargo clippy --workspace --all-targets`;
   versions are skewed (dspy-rs 0.7.3 / macros 0.7.2 / tools & cli & syntax
   0.1.0) — pick a coherent 0.8.0 story and write a CHANGELOG before the PR.

## 4. Known sharp edges (deliberate trade-offs, documented not fixed)

- **Explicit leaf discovery**: a leaf omitted from `predictors!` is silently
  not optimized/persisted. A `#[derive(Module)]` would close this; not built.
- **`Report::Custom(json)`** on the new Optimizer trait trades type precision
  for object safety.
- **Capability timeouts cancel by drop** (dsrs-tools): a tool mid-side-effect
  can be interrupted half-done. Tool authors own cancellation safety; not
  loudly documented.
- **Reserved JS names mangle silently** (`JSON` → `JSON_tool`) rather than
  erroring.
- **Compat shims still bypass the interpreter**: conversation seam
  (`TODO(dsrs-phase4-conversation)`) and caller-managed tool loop
  (`TODO(dsrs-phase4-caller-managed)`) — RFC 0004 §1–2 has suggested shapes.

## 5. Findings from the original architecture review that were NEVER fixed

These were found in the pre-refactor audit, judged out of scope, and are still
true at HEAD (except where noted, but re-verify before working on them —
session B's merges may have touched some):

- Replay is O(n²) (linear scan + deep `Span` clone per intercepted call) and
  replay/caching identity hashes the `Debug` output of rig's types — any rig
  `Debug` change silently invalidates every fixture. `request_hash` also
  includes operational knobs (`max_retries`), so ops changes invalidate
  replays.
- Trace span `parent` uses the innermost-open-span heuristic — wrong under
  `futures::join!` on one task.
- Dataloader: blocking I/O (`reqwest::blocking`, sync hf-hub) inside an async
  library — calling `load_hf` inside a runtime can stall or panic; `println!`
  + `verbose: bool` threaded through five signatures alongside `tracing`.
- `Message::content()` is lossy and used as canonical `raw_response`/cache
  payload; rig→Message conversion silently drops image/audio/document blocks.
- `LM::default()` calls `Handle::current().block_on()` — panics outside a
  runtime, deadlocks a current-thread runtime.
- COPRO and MIPROv2 still don't call an LM to propose candidates (template
  strings + a hardcoded tip list). Preserved deliberately in phase 3;
  making them real proposers is feature work.
- `CallMetadata` is not extensible (no place for a future BestOfN/Refine to
  record its decision).
- `dsrs-cli` has no `optimize`/`bake`/`run` commands; it can only serve
  hand-written `.dsrs` (host tools/holes refused).
- `#[module]` accepts only straight-line `let` bodies — RFC 0003 M-4
  (match→Route, for→Loop, join!→ForkJoin) was never built.
- The crate-root glob re-exports remain (prelude is additive); the public
  surface is still large.

## 6. Session B addendum — the seam merges, from their author

Written 2026-08-20 by session B (the seam-closure session §1's table calls
"NOT reviewed by session A"). Everything below is author-side knowledge: it
does not upgrade the review status of these merges, but it answers questions
§3 item 2 asks and records sharp edges the diffs won't volunteer.

### 6.1 Provenance and verification

Each seam was implemented by a dedicated agent in an isolated worktree off
`3fcef48`, then merged by session B in landing order `ca14e08` (seam 6) →
`109c571` (seam 4) → `16f2b27` (seam 5). All three were clean auto-merges.
Verified by session B:

- Baseline `cargo test --workspace` at `3fcef48`: green (exit 0) *before* any
  seam landed.
- Each agent ran the full `cargo test -p dspy-rs` suite (63 binaries, zero
  failures) plus `cargo clippy --workspace --all-targets` in its worktree
  before committing; each reported zero introduced warnings (the ~173
  baseline clippy warnings predate this branch's seam work).
- `cargo test -p dspy-rs` re-run green (exit 0) after the seam-6 merge and
  again at `16f2b27` with all three seams combined.
- NOT done, echoing §3: no live-LM run of any seam code, no independent
  review, no workspace-wide suite by session B at `16f2b27` (session A's §3
  item 0 run covers that).

**IN FLIGHT: seams 1+2 are not merged yet.** A fourth agent is implementing
the interpreter conversation surface (`run_conversation`) + caller-managed
tool loop as a suspending `AgentLoop` on branch
`worktree-agent-a5e33e674b89edb65`. If this document is the last word and
those seams look untouched at HEAD, check that branch before redoing the
work. When it lands, §1's table, RFC 0004 §1–2, and this section need rows.

### 6.2 Answers to §3 item 2's specific questions

- **Seam 5 grammar change**: pre-ToolSet `.dsrs` artifacts still parse and
  their canonical text + program hashes are **byte-identical** — the
  `tool_set [a b]` agent option is printed only when the selection differs
  from the full declared table, and the freshly-minted slot defaults to the
  full table. What *does* break is JSON serde of previously-serialized
  `Program` values (new `AgentLoopNode.tool_set` field + new slot kind); that
  was deliberate — canonical text is the wire form per RFC 0002.
- **Seam 5 goldens**: the only changed pre-existing expectation is one line
  in `test_trace_attach.rs` (`Trace::attach_program` joins all node-owned
  slots, so the new slot appears — by design). Everything else in
  `test_ir_text`/`test_ir_bake` is additive new tests, not regenerated goldens.
- **Seam 4 parity**: with no span evals present, the per-span gate reduces to
  the old whole-rollout gate *exactly*; there is a test asserting parity, so
  before/after demo harvests differ only for metrics that opt into
  `evaluate_spans`.

### 6.3 Author-side sharp edges per seam

Seam 6 (`Structural`, `ca14e08`):

- It is **not an `Optimizer` trait impl**. Entry points are
  `Structural::compile_program(&Interpreter, &[DemoRow], &metric, env_factory)`
  (+ `_with_overlay`): every accepted edit yields a *new program* that must be
  re-loaded with host bindings, hence the `RuntimeEnv` factory argument the
  trait can't express. Consequences: no `Report::Structural` variant; the
  winner is returned for `bake`, never installed.
- Acceptance is a SIMBA-style strict minibatch gate (hill climb), not Pareto —
  cross-program score-matrix columns would be incoherent.
- The proposal menu deliberately excludes `SetStop` and
  `SetInstructionDefault` (free-form values belong to the value-level
  optimizers); `AugmentSig` only materializes as the canonical CoT reasoning
  field and is pre-filtered when present.
- `StructuralReport.edits` is a **lineage** (node ids are per-parent), not a
  batch replayable against the root program.

Seam 4 (per-span credit, `109c571`):

- The RFC's premise was wrong: the trace format did *not* already carry
  per-span `Eval` records (eval was rollout-level only). `Span.eval:
  Option<Eval>` was added as an additive field (`skip_serializing_if`), no
  format version bump; eval-free traces serialize byte-identically.
- The hook is `TypedMetric::evaluate_spans(&self, example, prediction, trace)
  -> Result<Vec<(SpanId, Eval)>>`, default = empty (module lane only;
  `ProgramMetric` untouched — extending it later is purely additive).
  Unknown ids are ignored; duplicate ids last-write-win.
- SIMBA's rollout-level pre-gate (append-demo vs append-rule choice) is
  untouched — span evals only exclude bad spans *within* a harvested rollout.

Seam 5 (ToolSet gene, `16f2b27`):

- The selection **is** the surface: stop tools are *not* auto-unioned in. A
  selection that drops a stop tool with `until_parse=false` degrades to
  max_turns exhaustion — bounded and scored poorly, but worth knowing.
- Duplicates in a ToolSet value are a validation error (beyond the RFC, for
  canonical-text/hash unambiguity).
- Overlay membership is enforced inside `Overlay::set` (so the serde load
  path `from_named` is guarded too), via
  `Overlay::set_tool_set(&mut self, program, slot, tools)` — unlike other
  typed setters it takes the program and is fallible.
- `migrate_overlay` re-mints ToolSet entries by tool *name* intersected with
  the child's declared table; a non-empty selection whose intersection is
  empty drops the entry (falls back to full table).
- The Code Mode JS-collision check at interpreter load still runs over the
  full declared table (conservative superset; the overlay isn't known at
  load).

All three agents independently hit the same wall: the repo is not
rustfmt-clean (repo-wide `cargo fmt` reformats ~80 pre-existing files). Each
scoped formatting to its own diff and reverted the churn. Add "deliberate
repo-wide fmt pass" to §3 item 7's version-bump chore.

### 6.4 Deferred by session B on purpose

- `docs/docs/api/` was **not** regenerated by any seam agent (avoids
  cross-worktree conflicts). Someone must run `docs/scripts/gen_api.py` after
  the last seam lands — the API pages currently predate all seam surfaces.
- No optimizer proposes ToolSet mutations yet (the slot exists, validates,
  and is honored; proposal policy is follow-up work).
- Multi-edit batches per Structural generation; program-lane
  `evaluate_spans`; counterfactual/ablation credit — all explicitly out of
  scope per the RFC's own deferrals.

## 7. Housekeeping

- ~14 agent worktrees live under `.claude/worktrees/` with merged
  `worktree-agent-*` branches. After review:
  `git worktree list`, `git worktree remove <path>` for each, then
  `git branch --merged v1-program-unification | grep worktree-agent | xargs git branch -d`.
- `docs/archive/` holds the superseded CURRENT_PLAN/CURRENT_SPEC.
- The docs site search index (Mixedbread `dsrs-docs`/`dsrs-code` stores) needs
  re-indexing after the docs sweep — see `docs/README.md`.
