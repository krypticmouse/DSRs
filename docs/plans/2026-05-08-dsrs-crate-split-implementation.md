# DSRs Crate Split — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Decompose `crates/dspy-rs` into 9 layered crates (`dsrs-core`, `dsrs-lm`, `dsrs-trace`, `dsrs-cache`, `dsrs-predict`, `dsrs-evaluate`, `dsrs-gepa`, `dsrs-data`, `dsrs-leaven`) per the design at [`2026-05-08-dsrs-crate-split-design.md`](2026-05-08-dsrs-crate-split-design.md). No facade. Hard cutover. Delete COPRO and MIPROv2 along the way.

**Architecture:** Layer-aligned. `dsrs-core` is the small foundation that exposes abstract bridge traits (`DynPredictor`, `TraceSink`, `CacheBackend`, `LmClient`) and the Facet walker. Concrete trace/cache/lm crates implement those traits. `dsrs-predict` depends on core+lm. `dsrs-evaluate` depends on core only. `dsrs-gepa` is a leaf optimizer (sunset candidate). `dsrs-leaven` provides the leaven integration. The `dspy-rs` aggregator is dissolved.

**Tech Stack:** Rust 2024, `cargo` workspace, `jj` for VCS, `uv`-managed Python out of scope here. Existing crates use `bamltype`, `bamltype-derive`, `dsrs-macros`, `facet`, `rig-core`, `foyer`, `parquet`, `arrow`, `hf-hub`.

**Verification discipline:** Each task ends with `cargo check --workspace`, `cargo test --workspace` (or a tighter scope when justified), and a `jj` commit. The discipline is *preserve the test suite while moving code*. If a test was load-bearing for COPRO/MIPROv2 specifically, it gets deleted with the optimizer. Otherwise, every test that passes today must pass at the end of every task.

**One-time skill reads (engineer should do these once before starting):**
- `using-jj` (this is a jj repo; no `git add`, no staging area)
- `systematic-debugging` (for when something doesn't compile and you need to find why)

---

## Task 0: Preflight — branch, snapshot, baseline

**Files:** none modified.

**Step 1: Create a working change for this work, off `main`.**

```bash
jj new main -m "wip: dsrs crate split"
```

**Step 2: Confirm a clean baseline.**

```bash
cargo check --workspace
cargo test --workspace --no-run
```

Expected: both succeed. If `cargo test --workspace --no-run` (compile-only) fails, **stop**. The split assumes a green baseline. Investigate before continuing.

**Step 3: Capture the baseline test count for parity-checking later.**

```bash
cargo test --workspace -- --list 2>/dev/null | grep -c ': test$' > /tmp/dsrs-baseline-test-count
cat /tmp/dsrs-baseline-test-count
```

Record the number. After the split (minus deleted COPRO/MIPRO tests), the count must be `baseline − (count of deleted tests)` exactly.

**Step 4: Identify the COPRO/MIPRO tests that will be deleted.**

```bash
ls crates/dspy-rs/tests/ | grep -iE "copro|mipro"
```

Expected: `test_optimize_mipro.rs` (or similar). Write the names into `/tmp/dsrs-deleted-tests` for accounting.

**Step 5: Commit the (no-op) preflight as a marker.**

```bash
jj describe -m "chore: dsrs crate split — preflight baseline (no code changes)"
jj new
```

(The describe attaches a marker to the empty change so the work has a clear starting point in `jj log`.)

---

## Task 1: Create empty crate skeletons in workspace

**Goal:** Register all 9 new crates in the workspace `Cargo.toml` with empty `lib.rs` files. The workspace builds. No code has moved yet.

**Files:**
- Create: `crates/dsrs-core/Cargo.toml`, `crates/dsrs-core/src/lib.rs`
- Create: `crates/dsrs-lm/Cargo.toml`, `crates/dsrs-lm/src/lib.rs`
- Create: `crates/dsrs-trace/Cargo.toml`, `crates/dsrs-trace/src/lib.rs`
- Create: `crates/dsrs-cache/Cargo.toml`, `crates/dsrs-cache/src/lib.rs`
- Create: `crates/dsrs-predict/Cargo.toml`, `crates/dsrs-predict/src/lib.rs`
- Create: `crates/dsrs-evaluate/Cargo.toml`, `crates/dsrs-evaluate/src/lib.rs`
- Create: `crates/dsrs-gepa/Cargo.toml`, `crates/dsrs-gepa/src/lib.rs`
- Create: `crates/dsrs-data/Cargo.toml`, `crates/dsrs-data/src/lib.rs`
- Create: `crates/dsrs-leaven/Cargo.toml`, `crates/dsrs-leaven/src/lib.rs`
- Modify: `Cargo.toml` (workspace root) — register the new members

**Step 1: Create each new crate's `Cargo.toml` with the minimum viable manifest.**

For each crate, the manifest looks like (substitute crate name and dependencies per design § 3):

```toml
[package]
name = "dsrs-core"
version = "0.0.0"
edition = "2024"
rust-version = "1.85"
license = "Apache-2.0"
repository = "https://github.com/krypticmouse/DSRs"
description = "DSRs core: signature, module, schema, abstract bridges."

[dependencies]
# Empty for now; deps land as code is moved into the crate.
```

For `dsrs-leaven`, add path-dep stubs for the leaven crates:

```toml
[dependencies]
# Path-dep into the sibling leaven workspace.
leaven-core      = { path = "../../../leaven/crates/leaven-core" }
leaven-surface   = { path = "../../../leaven/crates/leaven-surface" }
leaven-engine    = { path = "../../../leaven/crates/leaven-engine" }
leaven-evidence  = { path = "../../../leaven/crates/leaven-evidence" }
```

(Once `dsrs-leaven` actually imports types, add `dsrs-core`, `dsrs-evaluate`, `dsrs-predict` too.)

**Step 2: Each `src/lib.rs` is a single line.**

```rust
//! Empty placeholder — code is migrated into this crate by a later task.
```

**Step 3: Register members in workspace `Cargo.toml`.**

Modify `Cargo.toml:3-7`:

```toml
[workspace]
resolver = "3"
members = [
    "crates/*",
    "vendor/baml/crates/*",
]
```

Already uses `crates/*` glob, so the new crates auto-register. Verify with:

```bash
cargo metadata --format-version=1 --no-deps | python3 -c "import sys,json; m=json.load(sys.stdin); print('\n'.join(p['name'] for p in m['packages']))" | sort
```

Expected: lists all current crates plus the 9 new ones.

**Step 4: Verify the workspace still builds with the empty crates registered.**

```bash
cargo check --workspace
```

Expected: success. The new crates have no code, no deps, build instantly.

**Step 5: Run the baseline tests.**

```bash
cargo test --workspace
```

Expected: same pass count as Task 0 step 3.

**Step 6: Commit.**

```bash
jj describe -m "feat(workspace): register 9 empty dsrs-* crate skeletons

dsrs-core, dsrs-lm, dsrs-trace, dsrs-cache, dsrs-predict, dsrs-evaluate,
dsrs-gepa, dsrs-data, dsrs-leaven. Empty lib.rs each. No code moved yet."
jj new
```

---

## Task 2: Extract `dsrs-core` — types and traits foundation

**Goal:** Move the typed-substrate types out of `dspy-rs/src/core/`, `dspy-rs/src/augmentation.rs`, and the legacy boundary types out of `dspy-rs/src/data/{example,prediction}.rs` into `dsrs-core`. Re-export from `dspy-rs/src/lib.rs` so downstream code is unaffected for now.

**Why this re-export step:** This is the hard task. Doing it cleanly first means subsequent extractions just move re-exports around.

**Files:**
- Move: `crates/dspy-rs/src/core/{mod.rs, signature.rs, module.rs, module_ext.rs, schema.rs, predicted.rs, errors.rs, dyn_predictor.rs, specials.rs, settings.rs}` → `crates/dsrs-core/src/`
- Move: `crates/dspy-rs/src/augmentation.rs` → `crates/dsrs-core/src/augmentation.rs`
- Move: `crates/dspy-rs/src/data/example.rs` → `crates/dsrs-core/src/example.rs`
- Move: `crates/dspy-rs/src/data/prediction.rs` → `crates/dsrs-core/src/prediction.rs`
- Modify: `crates/dsrs-core/Cargo.toml` (add `bamltype`, `facet`, `serde`, `serde_json`, `thiserror`, `async-trait`, `indexmap`, `tokio`, `bon`, `tracing`)
- Modify: `crates/dsrs-core/src/lib.rs` (declare modules + pub re-exports)
- Modify: `crates/dspy-rs/Cargo.toml` (add `dsrs-core = { path = "../dsrs-core" }`)
- Modify: `crates/dspy-rs/src/lib.rs` (delete moved `mod core; mod augmentation;` lines, replace with `pub use dsrs_core::*;` for compatibility within the crate)
- Modify: `crates/dspy-rs/src/data/mod.rs` (drop `mod example; mod prediction;`)

**Step 1: Read the existing module hierarchy to confirm what's where.**

```bash
ls crates/dspy-rs/src/core/
cat crates/dspy-rs/src/core/mod.rs
cat crates/dspy-rs/src/lib.rs | head -100
```

This task touches every module currently re-exported by `dspy-rs/src/core/mod.rs` *except* `lm/`, which stays for Task 4. Note what `core/mod.rs` re-exports — the same things must be re-exported from `dsrs-core/src/lib.rs` and from `dspy-rs/src/lib.rs` (via `pub use dsrs_core::*`) for the re-export step to be transparent.

**Step 2: Move files.** Use `jj` for the moves so file history is preserved.

```bash
jj file track crates/dsrs-core/src/lib.rs   # ensure tracked

mkdir -p crates/dsrs-core/src
git mv crates/dspy-rs/src/core/signature.rs        crates/dsrs-core/src/signature.rs
git mv crates/dspy-rs/src/core/module.rs           crates/dsrs-core/src/module.rs
git mv crates/dspy-rs/src/core/module_ext.rs       crates/dsrs-core/src/module_ext.rs
git mv crates/dspy-rs/src/core/schema.rs           crates/dsrs-core/src/schema.rs
git mv crates/dspy-rs/src/core/predicted.rs        crates/dsrs-core/src/predicted.rs
git mv crates/dspy-rs/src/core/errors.rs           crates/dsrs-core/src/errors.rs
git mv crates/dspy-rs/src/core/dyn_predictor.rs    crates/dsrs-core/src/dyn_predictor.rs
git mv crates/dspy-rs/src/core/specials.rs         crates/dsrs-core/src/specials.rs
git mv crates/dspy-rs/src/core/settings.rs         crates/dsrs-core/src/settings.rs
git mv crates/dspy-rs/src/augmentation.rs          crates/dsrs-core/src/augmentation.rs
git mv crates/dspy-rs/src/data/example.rs          crates/dsrs-core/src/example.rs
git mv crates/dspy-rs/src/data/prediction.rs       crates/dsrs-core/src/prediction.rs
```

(`jj` snapshots the workspace after each command; renames inside a colocated repo are picked up correctly.)

`crates/dspy-rs/src/core/mod.rs` is left behind — keep it for now, it'll be deleted in step 6 when nothing references it.

**Step 3: Add the abstract bridge trait stubs to `dsrs-core`.**

These traits don't exist yet — they're being introduced as part of the split, replacing today's tighter coupling.

Create `crates/dsrs-core/src/bridges.rs`:

```rust
//! Abstract bridge traits implemented by dsrs-trace, dsrs-cache, dsrs-lm.
//!
//! These exist so downstream crates (dsrs-predict, dsrs-evaluate, dsrs-gepa,
//! dsrs-leaven) depend only on dsrs-core, not on concrete observability or LM
//! crates. Each capability crate provides a concrete impl.

use async_trait::async_trait;
use std::sync::Arc;

/// Sink for execution-graph events. Implemented by `dsrs-trace::ExecutionGraph`.
pub trait TraceSink: Send + Sync + 'static {
    fn record(&self, event: TraceEvent);
}

/// One trace event. Concrete shape lives in dsrs-trace; this is the public
/// boundary type so producers don't depend on the concrete crate.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub kind: TraceEventKind,
    pub at_ns: u64,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone)]
pub enum TraceEventKind {
    PredictStart,
    PredictEnd,
    LmRequest,
    LmResponse,
    ParseFailure,
    Custom(&'static str),
}

/// LM response cache backend. Implemented by `dsrs-cache::LmCache`.
#[async_trait]
pub trait CacheBackend: Send + Sync + 'static {
    async fn get(&self, key: &str) -> Option<String>;
    async fn put(&self, key: String, value: String);
}

/// LM client trait. Implemented by `dsrs-lm::LM` (which wraps rig-core).
/// Predict and ChainOfThought depend on this trait, not on rig directly.
#[async_trait]
pub trait LmClient: Send + Sync + 'static {
    async fn complete(&self, request: LmRequest) -> Result<LmResponse, crate::errors::LmError>;
}

#[derive(Debug, Clone)]
pub struct LmRequest { /* fill in from existing core/lm/mod.rs request shape */ }

#[derive(Debug, Clone)]
pub struct LmResponse { /* fill in from existing core/lm/mod.rs response shape */ }
```

Leave `LmRequest` / `LmResponse` shapes as TODO — they get filled in when Task 4 (`dsrs-lm`) extracts the concrete LM and you can see the existing shape. Mark with:

```rust
// TODO(dsrs-bridges): fill from crates/dspy-rs/src/core/lm/mod.rs after Task 4.
```

**Step 4: Write `crates/dsrs-core/src/lib.rs`.**

```rust
//! DSRs core: typed signatures, modules, predicted outputs, augmentation,
//! abstract bridge traits, and the Facet walker. No LM, no formats, no
//! observability — those are concrete crates that depend on this one.

pub mod augmentation;
pub mod bridges;
pub mod dyn_predictor;
pub mod errors;
pub mod example;
pub mod module;
pub mod module_ext;
pub mod predicted;
pub mod prediction;
pub mod schema;
pub mod settings;
pub mod signature;
pub mod specials;

// Stable public surface — match what the old dspy-rs/src/lib.rs re-exported.
// (Refer to crates/dspy-rs/src/lib.rs at HEAD~1 to enumerate.)
pub use augmentation::*;
pub use bridges::{CacheBackend, LmClient, LmRequest, LmResponse, TraceEvent, TraceEventKind, TraceSink};
pub use dyn_predictor::*;
pub use errors::{ConversionError, ErrorClass, LmError, ParseError, PredictError};
pub use example::Example;
pub use module::Module;
pub use module_ext::*;
pub use predicted::{CallMetadata, Predicted};
pub use prediction::Prediction;
pub use schema::SignatureSchema;
pub use settings::*;
pub use signature::Signature;
pub use specials::*;
```

**Step 5: Update `crates/dsrs-core/Cargo.toml` deps.**

Read the current `crates/dspy-rs/Cargo.toml` `[dependencies]` and copy across only what core actually needs (the moved files import them):

```toml
[dependencies]
async-trait    = "0.1.83"
bamltype       = { path = "../bamltype" }
bon            = "3.7.0"
facet          = { git = "https://github.com/darinkishore/facet", rev = "cc8613c97cd1ec03e63659db34a947989b45c8a5", default-features = false, features = ["std"] }
indexmap       = "2.10.0"
serde          = { version = "1.0.219", features = ["derive"] }
serde_json     = { version = "1.0.140", features = ["preserve_order"] }
thiserror      = "2.0.17"
tokio          = { version = "1.46.1", features = ["sync"] }
tracing        = "0.1.44"
```

`dsrs_macros` is NOT a dependency of `dsrs-core`; only end-user crates depend on the macros.

**Step 6: Wire `dsrs-core` back into `dspy-rs` as a transparent re-export.**

Modify `crates/dspy-rs/Cargo.toml`:

```toml
[dependencies]
dsrs-core = { path = "../dsrs-core" }
# ... existing deps stay (still used by lm/, predictors/, optimizer/, etc.)
```

Modify `crates/dspy-rs/src/lib.rs` — delete `mod augmentation;` and `mod core;` lines, replace with:

```rust
// Transparent re-export of dsrs-core (extracted in Task 2). Subsequent tasks
// will move more code into dedicated crates and shrink this file further.
pub use dsrs_core::*;
```

The `pub mod core;` declaration in `lib.rs` is gone. But code inside `dspy-rs` that says `use crate::core::Foo` needs updating to `use dsrs_core::Foo`. Find and update:

```bash
grep -rn "crate::core::" crates/dspy-rs/src/ | wc -l
```

Replace `crate::core::` with `dsrs_core::` (and `crate::augmentation::` with `dsrs_core::`).

```bash
grep -rln "crate::core::\|crate::augmentation::" crates/dspy-rs/src/ | xargs sed -i '' \
  -e 's|crate::core::|dsrs_core::|g' \
  -e 's|crate::augmentation::|dsrs_core::|g'
```

(macOS BSD `sed` syntax above. On Linux: `sed -i 's|...|...|g'`.)

Modify `crates/dspy-rs/src/data/mod.rs` — drop `pub mod example;` and `pub mod prediction;` lines (the files are gone). Add `pub use dsrs_core::{Example, Prediction};` if anything inside `data/` imports them.

Delete `crates/dspy-rs/src/core/mod.rs` once you've confirmed nothing in `dspy-rs` still says `use crate::core`:

```bash
grep -rn "crate::core" crates/dspy-rs/src/  # expect empty
rm crates/dspy-rs/src/core/mod.rs
rmdir crates/dspy-rs/src/core               # only if empty (lm/ should still be inside)
```

`crates/dspy-rs/src/core/lm/` stays — it moves in Task 4.

If `core/` still has `lm/` in it, that's fine — `mod core { pub mod lm; }` in `lib.rs` handles it. Check the current state of `lib.rs` and adjust.

**Step 7: Build the new crate first, then the workspace.**

```bash
cargo check -p dsrs-core
```

Expected: success. If errors mention missing imports, the moved files reference siblings (e.g. `signature.rs` uses `super::module`). Fix to use `crate::module` (now flat in `dsrs-core`).

```bash
cargo check --workspace
```

Expected: success. Any errors here mean some `dspy-rs` consumer still references `crate::core::X` or `crate::augmentation::X`. Fix.

**Step 8: Run the full test suite.**

```bash
cargo test --workspace
```

Expected: same pass count as Task 0. The re-export keeps tests working unchanged.

**Step 9: Commit.**

```bash
jj describe -m "refactor(dsrs-core): extract typed-substrate foundation from dspy-rs

Moves core/{signature, module, module_ext, schema, predicted, errors,
dyn_predictor, specials, settings}.rs, augmentation.rs, and data/{example,
prediction}.rs into the new dsrs-core crate. Adds abstract bridge traits
(TraceSink, CacheBackend, LmClient) ready for concrete impls in subsequent
tasks. dspy-rs becomes a transparent re-export shell — every existing import
path still resolves."
jj new
```

---

## Task 3: Update `dsrs-macros` to emit `dsrs-core` paths

**Goal:** Macros (`#[derive(Signature)]`, etc.) currently emit code that resolves `dspy_rs::TypeIR`, `dspy_rs::Constraint`, etc. After Task 2 those re-export from `dsrs-core`, so it works — but the proper path is `dsrs_core::*`. Fix the resolution at the source.

**Files:**
- Modify: `crates/dsrs-macros/src/runtime_path.rs`
- Modify: `crates/dsrs-macros/Cargo.toml` (rename the path-resolution target)

**Step 1: Read current resolver.**

```bash
cat crates/dsrs-macros/src/runtime_path.rs
```

**Step 2: Rewrite to resolve `dsrs-core`.**

```rust
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;

pub(crate) fn resolve_dsrs_core_path() -> syn::Result<syn::Path> {
    match crate_name("dsrs-core") {
        Ok(FoundCrate::Itself) => Ok(syn::parse_quote!(::dsrs_core)),
        Ok(FoundCrate::Name(name)) => {
            let ident = syn::Ident::new(&name.replace('-', "_"), Span::call_site());
            Ok(syn::parse_quote!(::#ident))
        }
        Err(_) => Err(syn::Error::new(
            Span::call_site(),
            "could not resolve `dsrs-core`; add it as a dependency (renamed dependencies are supported)",
        )),
    }
}
```

**Step 3: Update every callsite of the old function name.**

```bash
grep -rn "resolve_dspy_rs_path" crates/dsrs-macros/src/
```

Replace each call with `resolve_dsrs_core_path`. Update any local variable names that mention `dspy_rs` — call them `dsrs_core_path` for clarity.

**Step 4: Verify the macros still expand correctly by building a downstream user.**

```bash
cargo check -p dspy-rs
```

Expected: success. Macro-generated code now targets `dsrs_core::*` paths, which `dspy-rs` re-exports.

**Step 5: Run macro contract tests.**

```bash
cargo test -p dspy-rs --test test_field_macro --test test_bamltype_attr_contract --test test_bamltype_docs_contract
```

Expected: all pass.

**Step 6: Commit.**

```bash
jj describe -m "refactor(dsrs-macros): emit dsrs-core paths instead of dspy-rs

Macro-generated code now references ::dsrs_core::* directly. The dspy-rs
re-export still works for source-level imports, but the canonical path the
proc macro emits is the new core crate."
jj new
```

---

## Task 4: Extract `dsrs-trace`

**Goal:** Move execution-graph recording into `dsrs-trace`. Implements `dsrs_core::TraceSink`.

**Files:**
- Move: `crates/dspy-rs/src/trace/{mod, dag, executor, value, context}.rs` → `crates/dsrs-trace/src/`
- Modify: `crates/dsrs-trace/Cargo.toml`
- Modify: `crates/dsrs-trace/src/lib.rs`
- Modify: `crates/dspy-rs/Cargo.toml`
- Modify: `crates/dspy-rs/src/lib.rs` (drop `pub mod trace;`, add `pub use dsrs_trace as trace;`)

**Step 1: Move files.**

```bash
git mv crates/dspy-rs/src/trace/mod.rs       crates/dsrs-trace/src/lib.rs
git mv crates/dspy-rs/src/trace/dag.rs       crates/dsrs-trace/src/dag.rs
git mv crates/dspy-rs/src/trace/executor.rs  crates/dsrs-trace/src/executor.rs
git mv crates/dspy-rs/src/trace/value.rs     crates/dsrs-trace/src/value.rs
git mv crates/dspy-rs/src/trace/context.rs   crates/dsrs-trace/src/context.rs
rmdir crates/dspy-rs/src/trace
```

**Step 2: Update `dsrs-trace/Cargo.toml`.**

```toml
[dependencies]
dsrs-core   = { path = "../dsrs-core" }
serde       = { version = "1.0.219", features = ["derive"] }
serde_json  = "1.0.140"
tokio       = { version = "1.46.1", features = ["sync"] }
tracing     = "0.1.44"
```

(Add others as compile errors reveal.)

**Step 3: Implement `TraceSink` for the concrete graph.**

In `crates/dsrs-trace/src/lib.rs`, after the `pub mod` declarations:

```rust
use dsrs_core::{TraceEvent, TraceSink};

impl TraceSink for ExecutionGraph {
    fn record(&self, event: TraceEvent) {
        // Existing recording logic moved here.
    }
}
```

Adjust based on the actual `ExecutionGraph` type from `dag.rs`. If `record` doesn't quite fit today's API, add a thin adapter method.

**Step 4: Wire `dsrs-trace` back into `dspy-rs`.**

`crates/dspy-rs/Cargo.toml`:

```toml
dsrs-trace = { path = "../dsrs-trace" }
```

`crates/dspy-rs/src/lib.rs`:

```rust
// Replace existing `pub mod trace;` with:
pub use dsrs_trace as trace;
```

Update internal imports inside `dspy-rs/src/`:

```bash
grep -rn "crate::trace::" crates/dspy-rs/src/ | wc -l
grep -rln "crate::trace::" crates/dspy-rs/src/ | xargs sed -i '' 's|crate::trace::|dsrs_trace::|g'
```

**Step 5: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

Expected: same pass count.

**Step 6: Commit.**

```bash
jj describe -m "refactor(dsrs-trace): extract execution-graph recording

Moves trace/ into the dsrs-trace crate. ExecutionGraph implements
dsrs_core::TraceSink so producers depend on the trait, not the concrete
crate. dspy-rs re-exports dsrs-trace as ::trace for compatibility within
this transitional period."
jj new
```

---

## Task 5: Extract `dsrs-cache`

**Goal:** Move foyer-backed LM response cache into `dsrs-cache`. Implements `dsrs_core::CacheBackend`.

**Files:**
- Move: `crates/dspy-rs/src/utils/cache.rs` → `crates/dsrs-cache/src/lib.rs`
- Possibly move: `crates/dspy-rs/src/utils/telemetry.rs` (if it's only used by cache; otherwise leave)
- Modify: `crates/dsrs-cache/Cargo.toml`
- Modify: `crates/dspy-rs/src/utils/mod.rs` (drop `pub mod cache;`)
- Modify: `crates/dspy-rs/Cargo.toml`

**Step 1: Audit `telemetry.rs` usage.**

```bash
grep -rn "utils::telemetry\|crate::utils::telemetry" crates/dspy-rs/src/
```

If it's only referenced from cache: move both. If broader: leave `telemetry.rs` in `dspy-rs/utils/` for now (a later task can fold it into `dsrs-trace` or `dsrs-core` as appropriate).

**Step 2: Move and wire.**

```bash
git mv crates/dspy-rs/src/utils/cache.rs crates/dsrs-cache/src/lib.rs
```

`crates/dsrs-cache/Cargo.toml`:

```toml
[dependencies]
dsrs-core   = { path = "../dsrs-core" }
async-trait = "0.1.83"
foyer       = { version = "0.20.0", features = ["serde"] }
serde       = { version = "1.0.219", features = ["derive"] }
serde_json  = "1.0.140"
tempfile    = "3.23.0"
tokio       = { version = "1.46.1", features = ["full"] }
```

In `crates/dsrs-cache/src/lib.rs`, add `impl CacheBackend for LmCache { ... }` (use the existing put/get methods).

**Step 3: Update `dspy-rs`.**

`crates/dspy-rs/Cargo.toml`:

```toml
dsrs-cache = { path = "../dsrs-cache" }
```

`crates/dspy-rs/src/utils/mod.rs`: drop `pub mod cache;`. Add `pub use dsrs_cache as cache;` if any external code references `dspy_rs::utils::cache`.

```bash
grep -rn "crate::utils::cache\|dspy_rs::utils::cache" crates/dspy-rs/
```

Adjust imports.

**Step 4: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

**Step 5: Commit.**

```bash
jj describe -m "refactor(dsrs-cache): extract foyer-backed LM response cache

Moves utils/cache.rs into dsrs-cache. Implements dsrs_core::CacheBackend so
dsrs-lm depends on the trait, not the foyer-backed concrete impl."
jj new
```

---

## Task 6: Extract `dsrs-lm`

**Goal:** Move the LM client (rig-core wrapper), `ChatAdapter`, `GLOBAL_SETTINGS`, `configure`, `with_lm` into `dsrs-lm`.

**Files:**
- Move: `crates/dspy-rs/src/core/lm/{mod, chat, client_registry, usage}.rs` → `crates/dsrs-lm/src/`
- Move: `crates/dspy-rs/src/adapter/{mod, chat}.rs` → `crates/dsrs-lm/src/adapter/{mod, chat}.rs`
- Modify: `crates/dsrs-lm/Cargo.toml`
- Modify: `crates/dsrs-lm/src/lib.rs`
- Modify: `crates/dspy-rs/src/lib.rs` (drop `pub mod adapter;`, drop `pub mod core { pub mod lm; }` references)
- Modify: `crates/dspy-rs/Cargo.toml`

**Step 1: Move LM files.**

```bash
git mv crates/dspy-rs/src/core/lm/mod.rs              crates/dsrs-lm/src/lib.rs
git mv crates/dspy-rs/src/core/lm/chat.rs             crates/dsrs-lm/src/chat.rs
git mv crates/dspy-rs/src/core/lm/client_registry.rs  crates/dsrs-lm/src/client_registry.rs
git mv crates/dspy-rs/src/core/lm/usage.rs            crates/dsrs-lm/src/usage.rs
rmdir crates/dspy-rs/src/core/lm
rmdir crates/dspy-rs/src/core 2>/dev/null || true
```

**Step 2: Move adapter files into `dsrs-lm/src/adapter/`.**

```bash
mkdir -p crates/dsrs-lm/src/adapter
git mv crates/dspy-rs/src/adapter/mod.rs   crates/dsrs-lm/src/adapter/mod.rs
git mv crates/dspy-rs/src/adapter/chat.rs  crates/dsrs-lm/src/adapter/chat.rs
rmdir crates/dspy-rs/src/adapter
```

**Step 3: Wire `dsrs-lm/src/lib.rs`.**

The existing `mod.rs` (now `lib.rs`) needs adjustment — it's becoming a crate root. Add module declarations at the top:

```rust
//! DSRs LM crate: rig-core wrapper, ChatAdapter, settings.

pub mod adapter;
pub mod chat;
pub mod client_registry;
pub mod usage;

// Existing `mod.rs` content (LM struct, configure, with_lm, GLOBAL_SETTINGS, ...)
```

**Step 4: Update `dsrs-lm/Cargo.toml`.**

Copy the rig-core, reqwest, regex, minijinja, anyhow, tokio, async-trait, schemars deps from `dspy-rs/Cargo.toml`:

```toml
[dependencies]
dsrs-core   = { path = "../dsrs-core" }
dsrs-cache  = { path = "../dsrs-cache" }
dsrs-trace  = { path = "../dsrs-trace" }
anyhow      = "1.0.99"
async-trait = "0.1.83"
bamltype    = { path = "../bamltype" }
bon         = "3.7.0"
facet       = { git = "https://github.com/darinkishore/facet", rev = "cc8613c97cd1ec03e63659db34a947989b45c8a5", default-features = false, features = ["std"] }
indexmap    = "2.10.0"
minijinja   = { git = "https://github.com/boundaryml/minijinja.git", branch = "main", default-features = false, features = ["builtins", "serde"] }
regex       = "1.11.2"
reqwest     = { version = "0.13", features = ["blocking"] }
rig-core    = { git = "https://github.com/0xPlaygrounds/rig", rev = "aee3b8bf6576ce41c9ac1dd82520752a65fa0127" }
schemars    = "1.0.4"
serde       = { version = "1.0.219", features = ["derive"] }
serde_json  = { version = "1.0.140", features = ["preserve_order"] }
thiserror   = "2.0.17"
tokio       = { version = "1.46.1", features = ["full"] }
tracing     = "0.1.44"
```

**Step 5: Update `LmClient` trait in `dsrs-core/src/bridges.rs`.**

Now that you can see `crates/dsrs-lm/src/lib.rs`, fill in `LmRequest` / `LmResponse` shapes (see TODO from Task 2 step 3). Then have the concrete `LM` struct in `dsrs-lm` implement `dsrs_core::LmClient`:

```rust
use dsrs_core::{LmClient, LmRequest, LmResponse};

#[async_trait::async_trait]
impl LmClient for LM {
    async fn complete(&self, request: LmRequest) -> Result<LmResponse, dsrs_core::LmError> {
        // Adapt to existing call path.
    }
}
```

**Step 6: Update `dspy-rs`.**

`crates/dspy-rs/Cargo.toml`:

```toml
dsrs-lm = { path = "../dsrs-lm" }
```

`crates/dspy-rs/src/lib.rs`:
- Drop `pub mod adapter;` (and `pub mod core { pub mod lm; }` if it still exists).
- Add `pub use dsrs_lm as lm;` and `pub use dsrs_lm::adapter;` to keep external imports working during the transition.

Update imports:

```bash
grep -rn "crate::core::lm::\|crate::adapter::" crates/dspy-rs/src/
sed -i '' \
  -e 's|crate::core::lm::|dsrs_lm::|g' \
  -e 's|crate::adapter::|dsrs_lm::adapter::|g' \
  $(grep -rln "crate::core::lm::\|crate::adapter::" crates/dspy-rs/src/)
```

**Step 7: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

Expected: all tests pass. The LM extraction is the heaviest concrete move (~1700 lines).

**Step 8: Commit.**

```bash
jj describe -m "refactor(dsrs-lm): extract LM client + ChatAdapter into dsrs-lm

Moves core/lm/ and adapter/ into the new dsrs-lm crate. The LM struct
implements dsrs_core::LmClient so Predict (extracted later) depends on the
trait, not on rig-core directly. dsrs-lm pulls in rig, reqwest, minijinja,
schemars — none of those are pulled by dsrs-core consumers anymore."
jj new
```

---

## Task 7: Extract `dsrs-evaluate`

**Goal:** Move the typed-metric surface into `dsrs-evaluate`.

**Files:**
- Move: `crates/dspy-rs/src/evaluate/{mod, evaluator, feedback, feedback_helpers, metrics}.rs` → `crates/dsrs-evaluate/src/`
- Modify: `crates/dsrs-evaluate/Cargo.toml`
- Modify: `crates/dspy-rs/src/lib.rs` (drop `pub mod evaluate;`, add re-export)
- Modify: `crates/dspy-rs/Cargo.toml`

**Step 1: Move.**

```bash
git mv crates/dspy-rs/src/evaluate/mod.rs               crates/dsrs-evaluate/src/lib.rs
git mv crates/dspy-rs/src/evaluate/evaluator.rs         crates/dsrs-evaluate/src/evaluator.rs
git mv crates/dspy-rs/src/evaluate/feedback.rs          crates/dsrs-evaluate/src/feedback.rs
git mv crates/dspy-rs/src/evaluate/feedback_helpers.rs  crates/dsrs-evaluate/src/feedback_helpers.rs
git mv crates/dspy-rs/src/evaluate/metrics.rs           crates/dsrs-evaluate/src/metrics.rs
rmdir crates/dspy-rs/src/evaluate
```

**Step 2: `dsrs-evaluate/Cargo.toml`.**

```toml
[dependencies]
dsrs-core   = { path = "../dsrs-core" }
async-trait = "0.1.83"
serde       = { version = "1.0.219", features = ["derive"] }
serde_json  = "1.0.140"
tokio       = { version = "1.46.1", features = ["full"] }
tracing     = "0.1.44"
futures     = "0.3.31"
```

**Step 3: Update `dsrs-evaluate/src/lib.rs` to declare the modules.**

```rust
//! DSRs typed-metric surface. Permanent (leaven-dsrs adapts this).

pub mod evaluator;
pub mod feedback;
pub mod feedback_helpers;
pub mod metrics;

pub use evaluator::*;
pub use feedback::*;
pub use feedback_helpers::*;
pub use metrics::*;
```

(Mirror what the old `evaluate/mod.rs` re-exported.)

**Step 4: Update `dspy-rs` re-export.**

```rust
// crates/dspy-rs/src/lib.rs
pub use dsrs_evaluate as evaluate;
```

```bash
grep -rn "crate::evaluate::" crates/dspy-rs/src/ | wc -l
sed -i '' 's|crate::evaluate::|dsrs_evaluate::|g' $(grep -rln "crate::evaluate::" crates/dspy-rs/src/)
```

**Step 5: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

`tests/test_evaluate_trainset_typed.rs` is the load-bearing parity check here.

**Step 6: Commit.**

```bash
jj describe -m "refactor(dsrs-evaluate): extract TypedMetric and feedback helpers"
jj new
```

---

## Task 8: Extract `dsrs-predict`

**Goal:** Move `Predict<S>`, `ChainOfThought<S>`, `ReAct<S>` into `dsrs-predict`. This is the L1 leaf — only thing that calls the LM.

**Files:**
- Move: `crates/dspy-rs/src/predictors/{mod, predict}.rs` → `crates/dsrs-predict/src/`
- Move: `crates/dspy-rs/src/modules/{mod, chain_of_thought, react}.rs` → `crates/dsrs-predict/src/modules/`
- Modify: `crates/dsrs-predict/Cargo.toml`
- Modify: `crates/dspy-rs/src/lib.rs`, `Cargo.toml`

**Step 1: Move.**

```bash
mkdir -p crates/dsrs-predict/src/modules
git mv crates/dspy-rs/src/predictors/predict.rs    crates/dsrs-predict/src/predict.rs
git mv crates/dspy-rs/src/predictors/mod.rs        crates/dsrs-predict/src/predictors_mod_DELETE.rs
# The old predictors/mod.rs is just `pub mod predict; pub use predict::*;` — fold into lib.rs.
rm crates/dsrs-predict/src/predictors_mod_DELETE.rs
rmdir crates/dspy-rs/src/predictors

git mv crates/dspy-rs/src/modules/chain_of_thought.rs  crates/dsrs-predict/src/modules/chain_of_thought.rs
git mv crates/dspy-rs/src/modules/react.rs             crates/dsrs-predict/src/modules/react.rs
git mv crates/dspy-rs/src/modules/mod.rs               crates/dsrs-predict/src/modules/mod.rs
rmdir crates/dspy-rs/src/modules
```

**Step 2: `dsrs-predict/src/lib.rs`.**

```rust
//! DSRs predict crate: the L1 leaf. Predict<S>, ChainOfThought<S>, ReAct<S>.
//! Only crate that actually calls the LM.

pub mod modules;
pub mod predict;

pub use modules::*;
pub use predict::*;
```

**Step 3: `dsrs-predict/Cargo.toml`.**

```toml
[dependencies]
dsrs-core    = { path = "../dsrs-core" }
dsrs-lm      = { path = "../dsrs-lm" }
dsrs-trace   = { path = "../dsrs-trace" }
async-trait  = "0.1.83"
bamltype     = { path = "../bamltype" }
bon          = "3.7.0"
facet        = { git = "https://github.com/darinkishore/facet", rev = "cc8613c97cd1ec03e63659db34a947989b45c8a5", default-features = false, features = ["std"] }
futures      = "0.3.31"
indexmap     = "2.10.0"
serde        = { version = "1.0.219", features = ["derive"] }
serde_json   = { version = "1.0.140", features = ["preserve_order"] }
tokio        = { version = "1.46.1", features = ["full"] }
tracing      = "0.1.44"
dsrs_macros  = { path = "../dsrs-macros" }   # ChainOfThought derives Augmentation etc.
```

**Step 4: Wire and update imports.**

```bash
grep -rn "crate::predictors::\|crate::modules::" crates/dspy-rs/src/
sed -i '' \
  -e 's|crate::predictors::|dsrs_predict::|g' \
  -e 's|crate::modules::|dsrs_predict::|g' \
  $(grep -rln "crate::predictors::\|crate::modules::" crates/dspy-rs/src/)
```

`crates/dspy-rs/src/lib.rs`:

```rust
pub use dsrs_predict as predict_crate;
pub use dsrs_predict::*;        // Predict, ChainOfThought, ReAct at top level
pub mod modules { pub use dsrs_predict::modules::*; }
```

`crates/dspy-rs/Cargo.toml`:

```toml
dsrs-predict = { path = "../dsrs-predict" }
```

**Step 5: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

The Predict test surface is the largest. Tests `test_chain_of_thought_swap`, `test_chat_prompt_composition`, `test_chat_prompt_golden` are core.

**Step 6: Commit.**

```bash
jj describe -m "refactor(dsrs-predict): extract Predict, ChainOfThought, ReAct"
jj new
```

---

## Task 9: Extract `dsrs-gepa`, delete COPRO and MIPROv2

**Goal:** Move GEPA + pareto into `dsrs-gepa`. Delete COPRO and MIPROv2 source files. Delete COPRO/MIPRO test files. Delete the `08-optimize-mipro.rs` example.

**Files:**
- Move: `crates/dspy-rs/src/optimizer/{gepa, pareto}.rs` → `crates/dsrs-gepa/src/`
- Move: `crates/dspy-rs/src/optimizer/mod.rs` → `crates/dsrs-gepa/src/lib.rs` (with COPRO/MIPRO declarations removed)
- **Delete:** `crates/dspy-rs/src/optimizer/copro.rs`
- **Delete:** `crates/dspy-rs/src/optimizer/mipro.rs`
- **Delete:** `crates/dspy-rs/tests/test_optimize_mipro.rs` (or whichever exists)
- **Delete:** any COPRO test
- **Delete:** `crates/dspy-rs/examples/04-optimize-hotpotqa.rs` (this uses COPRO — verify, possibly rewrite to GEPA later as separate work)
- **Delete:** `crates/dspy-rs/examples/08-optimize-mipro.rs`

**Step 1: Confirm what to delete.**

```bash
grep -ln "COPRO\|MIPROv2\|MIPRO" crates/dspy-rs/{tests,examples}/*.rs
```

Inventory the hits and confirm with the user before deleting if unsure. Per the design, COPRO and MIPROv2 are out — their tests and examples go with them.

**Step 2: Delete the optimizer source files first.**

```bash
rm crates/dspy-rs/src/optimizer/copro.rs
rm crates/dspy-rs/src/optimizer/mipro.rs
```

**Step 3: Move GEPA + pareto.**

```bash
git mv crates/dspy-rs/src/optimizer/gepa.rs    crates/dsrs-gepa/src/gepa.rs
git mv crates/dspy-rs/src/optimizer/pareto.rs  crates/dsrs-gepa/src/pareto.rs
git mv crates/dspy-rs/src/optimizer/mod.rs     crates/dsrs-gepa/src/lib.rs
rmdir crates/dspy-rs/src/optimizer
```

**Step 4: Edit `dsrs-gepa/src/lib.rs` to drop COPRO/MIPRO references.**

Open the file, delete every `pub mod copro;`, `pub mod mipro;`, `pub use copro::*;`, `pub use mipro::*;`. Keep the `Optimizer` trait, the GEPA exports, the pareto exports.

**Step 5: `dsrs-gepa/Cargo.toml`.**

```toml
[package]
name = "dsrs-gepa"
description = "GEPA optimizer for DSRs (sunset candidate; replaced by leaven once dsrs-leaven is real)."
# ... usual fields ...

[dependencies]
dsrs-core      = { path = "../dsrs-core" }
dsrs-predict   = { path = "../dsrs-predict" }
dsrs-evaluate  = { path = "../dsrs-evaluate" }
async-trait    = "0.1.83"
indexmap       = "2.10.0"
rand           = "0.8.5"
rayon          = "1.10.0"
serde          = { version = "1.0.219", features = ["derive"] }
serde_json     = "1.0.140"
tokio          = { version = "1.46.1", features = ["full"] }
tracing        = "0.1.44"
kdam           = "0.6.3"
```

**Step 6: Delete COPRO/MIPRO tests and examples.**

```bash
rm -f crates/dspy-rs/tests/test_optimize_mipro.rs
# (Add others as inventory in step 1 reveals — confirm each name first.)
rm -f crates/dspy-rs/examples/08-optimize-mipro.rs
# 04-optimize-hotpotqa: read first; if it's COPRO-only, delete; if salvageable for GEPA, leave a TODO comment in the file.
head -30 crates/dspy-rs/examples/04-optimize-hotpotqa.rs
```

If `04-optimize-hotpotqa.rs` is hard-wired to COPRO, delete it. Removing examples is fine — they're not in the test suite gate.

**Step 7: Update `dspy-rs` re-export.**

```rust
// crates/dspy-rs/src/lib.rs
pub use dsrs_gepa as optimizer;
```

```bash
grep -rn "crate::optimizer::" crates/dspy-rs/src/
sed -i '' 's|crate::optimizer::|dsrs_gepa::|g' $(grep -rln "crate::optimizer::" crates/dspy-rs/src/)
```

`crates/dspy-rs/Cargo.toml`:

```toml
dsrs-gepa = { path = "../dsrs-gepa" }
```

**Step 8: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

Expected pass count = baseline − (deleted COPRO/MIPRO test count). If anything else fails, an import wasn't migrated.

**Step 9: Commit.**

```bash
jj describe -m "refactor(dsrs-gepa): extract GEPA; delete COPRO and MIPROv2

Moves optimizer/gepa.rs and optimizer/pareto.rs into dsrs-gepa. Deletes
COPRO (optimizer/copro.rs) and MIPROv2 (optimizer/mipro.rs) along with
their tests (test_optimize_mipro.rs et al) and the 08-optimize-mipro
example. Both were marked outdated.

dsrs-gepa is a sunset candidate — deleted once leaven-gepa ships a
runnable optimizer and dsrs-leaven ships real impls."
jj new
```

---

## Task 10: Extract `dsrs-data` with feature flags

**Goal:** Move `DataLoader` + format readers into `dsrs-data`. Format-specific deps (parquet, hf-hub, csv) go behind feature flags so light users don't pay.

**Files:**
- Move: `crates/dspy-rs/src/data/{mod, dataloader, serialize, utils}.rs` → `crates/dsrs-data/src/`
- Modify: `crates/dsrs-data/Cargo.toml`

(Note: `example.rs` and `prediction.rs` already moved in Task 2.)

**Step 1: Move.**

```bash
git mv crates/dspy-rs/src/data/dataloader.rs  crates/dsrs-data/src/dataloader.rs
git mv crates/dspy-rs/src/data/serialize.rs   crates/dsrs-data/src/serialize.rs
git mv crates/dspy-rs/src/data/utils.rs       crates/dsrs-data/src/utils.rs
git mv crates/dspy-rs/src/data/mod.rs         crates/dsrs-data/src/lib.rs
rmdir crates/dspy-rs/src/data
```

**Step 2: Edit `dsrs-data/src/lib.rs`** — remove the `pub mod example;` and `pub mod prediction;` lines (those types live in `dsrs-core` now). Add `pub use dsrs_core::{Example, Prediction};` if anything in `dsrs-data` references them.

**Step 3: `dsrs-data/Cargo.toml` with feature flags.**

```toml
[package]
name = "dsrs-data"
# ... usual fields ...

[features]
default = ["json"]
json    = []
csv     = ["dep:csv"]
parquet = ["dep:parquet", "dep:arrow"]
hf      = ["dep:hf-hub", "dep:reqwest"]
all     = ["json", "csv", "parquet", "hf"]

[dependencies]
dsrs-core  = { path = "../dsrs-core" }
serde      = { version = "1.0.219", features = ["derive"] }
serde_json = "1.0.140"
tokio      = { version = "1.46.1", features = ["full"] }
tracing    = "0.1.44"
indexmap   = "2.10.0"

# Optional, gated:
csv     = { version = "1.3.1",  optional = true }
parquet = { version = "56.1.0", optional = true }
arrow   = { version = "56.1.0", optional = true }
hf-hub  = { version = "0.4.3",  features = ["tokio"], optional = true }
reqwest = { version = "0.13",   features = ["blocking"], optional = true }
```

**Step 4: Gate format-specific code in `dataloader.rs`.**

The existing `dataloader.rs` unconditionally uses csv, parquet, hf-hub. Wrap each format-specific function/impl with `#[cfg(feature = "csv")]` etc:

```rust
#[cfg(feature = "csv")]
pub fn load_csv(...) -> ... { ... }

#[cfg(feature = "parquet")]
pub fn load_parquet(...) -> ... { ... }

#[cfg(feature = "hf")]
pub async fn load_hf_dataset(...) -> ... { ... }
```

**Step 5: Update `dspy-rs` re-export and consumer.**

```rust
// crates/dspy-rs/src/lib.rs — replace `pub mod data;`
pub use dsrs_data as data;
```

`crates/dspy-rs/Cargo.toml`: depend on `dsrs-data` with `features = ["all"]` so the existing test suite (which exercises all formats) still works.

```toml
dsrs-data = { path = "../dsrs-data", features = ["all"] }
```

```bash
grep -rn "crate::data::" crates/dspy-rs/src/
sed -i '' 's|crate::data::|dsrs_data::|g' $(grep -rln "crate::data::" crates/dspy-rs/src/)
```

**Step 6: Build matrix.**

```bash
cargo check -p dsrs-data --no-default-features
cargo check -p dsrs-data --no-default-features --features json
cargo check -p dsrs-data --no-default-features --features csv
cargo check -p dsrs-data --no-default-features --features parquet
cargo check -p dsrs-data --no-default-features --features hf
cargo check -p dsrs-data --features all
```

Expected: each succeeds.

**Step 7: Workspace tests.**

```bash
cargo test --workspace
```

`tests/test_dataloader.rs` is the gate — it should still pass with `dspy-rs` requesting `features = ["all"]`.

**Step 8: Commit.**

```bash
jj describe -m "refactor(dsrs-data): extract DataLoader with feature-gated format readers

Moves data/{dataloader,serialize,utils}.rs into dsrs-data. Format-specific
deps (csv, parquet, arrow, hf-hub, reqwest) are now feature-gated:
  - default = json
  - csv, parquet, hf, all
Light users skip arrow/parquet/hf-hub. dspy-rs depends with features=[all]
during the transitional period."
jj new
```

---

## Task 11: Skeleton `dsrs-leaven`

**Goal:** Lay down `dsrs-leaven` with type signatures and `unimplemented!()` bodies. Real implementations land in a follow-up plan once at least one leaven-side piece (e.g. `leaven-gepa` real impl) is in place.

**Files:**
- Modify: `crates/dsrs-leaven/Cargo.toml`
- Modify: `crates/dsrs-leaven/src/lib.rs`
- Create: `crates/dsrs-leaven/src/{artifact, change, surface, evaluator, evidence}.rs`

**Step 1: Verify leaven path.**

```bash
ls /Users/darin/src/personal/leaven/crates/{leaven-core,leaven-surface,leaven-engine,leaven-evidence}/
```

If any are missing, **stop** and confirm with the user.

**Step 2: `dsrs-leaven/Cargo.toml`.**

```toml
[package]
name = "dsrs-leaven"
description = "DSRs's implementation of leaven's capability traits — Artifact, EditSurface, Evaluator, Evidence."

[dependencies]
dsrs-core      = { path = "../dsrs-core" }
dsrs-evaluate  = { path = "../dsrs-evaluate" }
dsrs-predict   = { path = "../dsrs-predict" }

leaven-core      = { path = "../../../leaven/crates/leaven-core" }
leaven-surface   = { path = "../../../leaven/crates/leaven-surface" }
leaven-engine    = { path = "../../../leaven/crates/leaven-engine" }
leaven-evidence  = { path = "../../../leaven/crates/leaven-evidence" }

async-trait = "0.1.83"
serde       = { version = "1.0.219", features = ["derive"] }
serde_json  = "1.0.140"
thiserror   = "2.0.17"
```

**Step 3: `src/lib.rs`.**

```rust
//! DSRs ⇄ leaven integration. DSRs implements leaven's capability traits
//! directly so leaven optimizers (leaven-gepa, etc.) can drive DSRs programs.
//!
//! Bodies are `unimplemented!()` until the leaven side is real. See
//! `docs/plans/2026-05-08-dsrs-crate-split-design.md` § 6 for the sunset
//! trigger that lets us delete dsrs-gepa.

pub mod artifact;
pub mod change;
pub mod surface;
pub mod evaluator;
pub mod evidence;

pub use artifact::DsrsProgramArtifact;
pub use change::DsrsProgramChange;
pub use surface::DsrsProgramSurface;
pub use evaluator::DsrsEvaluator;
pub use evidence::DsrsEvidence;
```

**Step 4: Each module is a stub** with type signatures only, no bodies yet:

```rust
// artifact.rs
use dsrs_core::{Module, Signature};

pub struct DsrsProgramArtifact<S, M>
where
    S: Signature,
    M: Module,
{
    _phantom: std::marker::PhantomData<(S, M)>,
}

impl<S, M> leaven_core::Artifact for DsrsProgramArtifact<S, M>
where
    S: Signature,
    M: Module + Send + Sync + 'static,
{
    type Change = crate::change::DsrsProgramChange;

    fn identity(&self) -> leaven_core::ArtifactIdentity {
        unimplemented!("dsrs-leaven: identity — fill once leaven-gepa is real")
    }

    fn apply_change(&self, _change: &Self::Change)
        -> Result<Self, leaven_core::ApplyError>
    {
        unimplemented!("dsrs-leaven: apply_change")
    }
}
```

Repeat the pattern for the other 4 files. The point is: they compile against the current leaven trait shapes. If leaven changes its trait API, this crate breaks immediately and we adjust.

**Step 5: Build.**

```bash
cargo check -p dsrs-leaven
cargo check --workspace
```

If `leaven-core` traits have moved or changed shape, fix the impls to match. The `unimplemented!()` bodies don't run, so failure to build = signature mismatch only.

**Step 6: Commit.**

```bash
jj describe -m "feat(dsrs-leaven): add skeleton crate with leaven trait impls

Type signatures only — bodies are unimplemented!(). Compiles against the
current leaven-core / leaven-surface / leaven-engine / leaven-evidence
trait shapes. Real impls land in a follow-up plan once leaven-gepa is a
runnable optimizer (see design doc § 6 sunset trigger)."
jj new
```

---

## Task 12: Dissolve `dspy-rs` — relocate examples and tests

**Goal:** With every code module moved, the `dspy-rs` crate is now a thin re-export shell. Per design D1 (no facade), it gets deleted. Examples and tests need a home.

**Files:**
- Delete: `crates/dspy-rs/` entirely
- Move: `crates/dspy-rs/examples/*.rs` → `crates/dspy-rs/examples/` lives where? Pick a destination per step below.
- Move: `crates/dspy-rs/tests/*.rs` → distribute across the new crates that own the surfaces being tested.
- Modify: workspace `Cargo.toml` (no change needed — `crates/*` glob)

**Step 1: Inventory the remaining `dspy-rs/src/lib.rs`.**

```bash
cat crates/dspy-rs/src/lib.rs
wc -l crates/dspy-rs/src/lib.rs
```

By this point it should be ~30 lines: just `pub use` re-exports of the new crates. Confirm there's no leftover code that didn't get extracted.

**Step 2: Pick destinations for tests.**

```bash
ls crates/dspy-rs/tests/
```

Distribute by what's tested:

| Test | Destination |
|------|-------------|
| `test_field_macro.rs`, `test_bamltype_*` | `crates/dsrs-macros/tests/` |
| `test_chat*.rs`, `test_lm.rs`, `test_adapters.rs`, `test_input_format.rs`, `test_message_roundtrip.rs` | `crates/dsrs-lm/tests/` |
| `test_evaluate_trainset_typed.rs` | `crates/dsrs-evaluate/tests/` |
| `test_call_outcome.rs`, `test_caller_managed_conversation.rs`, `test_chain_of_thought_swap.rs`, `test_example.rs`, `test_flatten_roundtrip.rs` | `crates/dsrs-predict/tests/` |
| `test_dataloader.rs` | `crates/dsrs-data/tests/` |
| `test_gepa.rs`, `test_gepa_typed_metric_feedback.rs` | `crates/dsrs-gepa/tests/` |
| (rest) | inspect each — most belong to `dsrs-predict` or `dsrs-lm` |

For each destination, `mkdir -p crates/<dst>/tests` then `git mv`. Update the test's `use` statements: `dspy_rs::X` → the right new crate's path.

```bash
# Example for one test:
mkdir -p crates/dsrs-lm/tests
git mv crates/dspy-rs/tests/test_chat.rs crates/dsrs-lm/tests/test_chat.rs
sed -i '' 's|use dspy_rs::|use dsrs_lm::|g; s|dspy_rs::|dsrs_lm::|g' crates/dsrs-lm/tests/test_chat.rs
```

(Each test will probably need 2-3 different replacements as it imports from multiple crates. Read the test, do the imports manually.)

**Step 3: Pick a destination for examples.**

Two options:
- (A) New top-level `examples/` directory with each example mapped to the crate that owns its surface.
- (B) Distribute under `crates/<dst>/examples/`.

(B) is more idiomatic for cargo workspaces — examples auto-discover from each crate. Pick (B).

```bash
# Example:
mkdir -p crates/dsrs-predict/examples
git mv crates/dspy-rs/examples/01-simple.rs crates/dsrs-predict/examples/01-simple.rs
sed -i '' 's|use dspy_rs::|use dsrs_predict::|g' crates/dsrs-predict/examples/01-simple.rs
# (Likely needs additional crates pulled — read the example to see what it imports.)
```

The examples that touch GEPA go to `dsrs-gepa`; tracing examples to `dsrs-trace`; the smoke-* slices to wherever the surface they smoke-test lives.

**Step 4: Delete `dspy-rs`.**

```bash
rm -rf crates/dspy-rs
```

**Step 5: Build and test.**

```bash
cargo check --workspace
cargo test --workspace
```

Expected: `cargo metadata` no longer lists `dspy-rs`. Test pass count = baseline − deleted COPRO/MIPRO tests. Any test that fails because an import didn't get rewritten — fix it.

**Step 6: Commit.**

```bash
jj describe -m "refactor: dissolve dspy-rs aggregator crate

All code has migrated to dsrs-{core, lm, trace, cache, predict, evaluate,
gepa, data, leaven}. Tests and examples redistributed to the crates that
own their surfaces. No facade re-export — users depend on the leaf crates
explicitly."
jj new
```

---

## Task 13: Update docs and READMEs

**Files:**
- Modify: `README.md` (top-level)
- Modify: `CURRENT_PLAN.md`, `CURRENT_SPEC.md` (mark superseded; point to the new design doc)
- Modify: `sub-agents.md`
- Modify: `docs/specs/modules/breadboard.md`, `docs/specs/modules/design_reference.md` (annotate that the topology is now mechanically enforced via crate boundaries; cross-link to the new design doc)
- Modify: `docs/docs/getting-started/*.md` (any quick-start showing `use dspy_rs::*` — replace with the new imports)

**Step 1: Update top-level `README.md`.**

Replace any `dspy-rs = "..."` snippets and `use dspy_rs::*` examples with the new layout. The mental model section should reference the layered crates:

```markdown
## Crates

| Crate | Purpose |
|-------|---------|
| `dsrs-core` | Signatures, modules, schema, errors, abstract bridges. |
| `dsrs-lm` | LM client + ChatAdapter. |
| `dsrs-predict` | Predict, ChainOfThought, ReAct. |
| `dsrs-evaluate` | TypedMetric and feedback helpers. |
| `dsrs-gepa` | GEPA optimizer (sunset; replaced by leaven). |
| `dsrs-data` | DataLoader (csv / parquet / hf-hub feature-gated). |
| `dsrs-trace` | Execution-graph recording. |
| `dsrs-cache` | Foyer LM cache. |
| `dsrs-leaven` | Leaven integration. |
```

**Step 2: Mark `CURRENT_PLAN.md` and `CURRENT_SPEC.md` superseded.**

Add a banner at the top of each:

```markdown
> **Superseded** by [`docs/plans/2026-05-08-dsrs-crate-split-design.md`](docs/plans/2026-05-08-dsrs-crate-split-design.md). Retained for historical context.
```

**Step 3: Verify no doc still says `crates/dspy-rs`.**

```bash
grep -rn "crates/dspy-rs\|dspy-rs/src" docs/ README.md *.md
```

Inventory hits and decide per-line whether to update or annotate as historical.

**Step 4: Update example imports** in `docs/docs/getting-started/`, `docs/docs/tutorials/`, `docs/docs/building-blocks/`, `docs/docs/optimizers/`.

```bash
grep -rln "use dspy_rs::" docs/
# For each hit, decide: rewrite with the new crate paths, or annotate as
# historical (if the doc page is itself outdated).
```

**Step 5: Commit.**

```bash
jj describe -m "docs: update READMEs and references for the dsrs crate split

Top-level README lists the 9 crates and their roles. CURRENT_PLAN.md and
CURRENT_SPEC.md flagged as superseded by the new split design doc. Docs
under docs/docs/* updated where they showed dspy_rs imports."
jj new
```

---

## Task 14: Clean up the leaven side

**Files (in the leaven workspace, separate jj repo):**
- Modify: `/Users/darin/src/personal/leaven/Cargo.toml` (drop `crates/leaven-dsrs` member, drop the workspace-deps entry)
- Delete: `/Users/darin/src/personal/leaven/crates/leaven-dsrs/`

**Step 1: Coordinate with the user before touching leaven.**

This step modifies a sibling repository. Confirm with the user that it's OK to delete `leaven-dsrs` from the leaven workspace, since DSRs's `dsrs-leaven` now owns those impls.

**Step 2: In leaven repo:**

```bash
cd /Users/darin/src/personal/leaven
jj st
jj new -m "chore: drop leaven-dsrs in favor of dsrs-leaven (in DSRs workspace)"

# Remove from workspace members and workspace deps
# (Edit Cargo.toml manually — drop both lines.)

rm -rf crates/leaven-dsrs

cargo check --workspace
```

Expected: leaven workspace builds without `leaven-dsrs`. (Nothing inside leaven imports it; it was a stub.)

**Step 3: Commit (in leaven repo).**

```bash
jj describe -m "chore: drop leaven-dsrs

DSRs now owns the integration via crates/dsrs-leaven in the DSRs
workspace. Bridge crate ownership flipped: a downstream consumer
implements leaven's capability traits, rather than leaven owning a
third-party-shaped bridge."
jj new
```

---

## Task 15: Final verification and squash

**Step 1: From the DSRs repo:**

```bash
cargo check --workspace
cargo test --workspace
cargo build --release --workspace   # full release build smoke-test
```

Expected: all green. The release build catches anything that's wrong with cfg-gates or feature interactions.

**Step 2: Per-crate smoke checks.**

```bash
for c in dsrs-core dsrs-lm dsrs-trace dsrs-cache dsrs-predict dsrs-evaluate dsrs-gepa dsrs-data dsrs-leaven; do
  echo "=== $c ==="
  cargo check -p $c
  cargo test  -p $c
done
```

Expected: each crate builds and tests independently.

**Step 3: Test count parity.**

```bash
cargo test --workspace -- --list 2>/dev/null | grep -c ': test$'
```

Expected: `baseline_count − deleted_copro_mipro_count`. Any other delta means a test was lost in transit. Investigate.

**Step 4: Look at the change graph.**

```bash
jj log -r 'trunk()..@'
```

Expected: a chain of ~14 commits, one per task, each with a clear message.

**Step 5: Decide on commit shape.**

Two options:

(A) **Keep the chain.** 14 commits, easier review and easier revert per-task.

(B) **Squash into one big commit.**

```bash
jj squash --from <first-task-change> --into <preflight-change>
# repeat as needed
```

Recommendation: keep the chain (option A). Each task is a logical unit; merge as a stack of commits.

**Step 6: Final commit if anything was tweaked in step 1-3.**

```bash
jj describe -m "chore: final cleanup after split"
```

If the working copy is empty, you're done.

---

## Open follow-ups (not in this plan — separate work)

1. **Real `dsrs-leaven` impls.** Once `leaven-gepa` is runnable and the leaven optimization run-loop has an ergonomic entry point, replace the `unimplemented!()` bodies in Task 11 with real implementations. Write a parity test: optimize the same DSRs program via `dsrs-gepa` and via leaven-driven `dsrs-leaven`, confirm equal-or-better results from leaven.

2. **Delete `dsrs-gepa`.** When the parity test passes (per design § 6), remove the crate from the workspace. Update the topology doc and README.

3. **Move `dsrs-macros` to emit per-target paths.** Today the macros emit `::dsrs_core::*`. If `dsrs-leaven` ever needs different macro emit-paths (unlikely), the resolver in `dsrs-macros/src/runtime_path.rs` becomes a small lookup table.

4. **Rewrite `04-optimize-hotpotqa.rs` (deleted in Task 9) as a GEPA example** if the COPRO version is missed.

5. **Audit `dsrs-data` features** after a few weeks of use. If `default = ["json"]` is wrong (e.g. csv is more common), adjust.

---

## References

- Design: [`2026-05-08-dsrs-crate-split-design.md`](2026-05-08-dsrs-crate-split-design.md)
- Visual companion: [`2026-05-08-dsrs-crate-split-topology.html`](2026-05-08-dsrs-crate-split-topology.html)
- Predecessor topology spec: `docs/specs/modules/breadboard.md`, `docs/specs/modules/design_reference.md`
- Leaven principles: `/Users/darin/src/personal/leaven/AGENTS.md`, `/Users/darin/src/personal/leaven/docs/specs/guiding_principles.md`
