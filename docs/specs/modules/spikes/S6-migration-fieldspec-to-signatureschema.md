# S6 Spike: Incremental migration from `FieldSpec`/`MetaSignature` to Facet-derived `SignatureSchema`

## Context
The shaping docs call out S6 as the migration question for replacing static macro-emitted `FieldSpec` metadata and `MetaSignature` JSON maps with Facet-derived `SignatureSchema` (`docs/specs/modules/shapes.md:145`).

Current runtime has both:
- Typed `Predict<S: Signature>` path using static `FieldSpec` arrays.
- Legacy `LegacyPredict` path using `MetaSignature` trait objects and JSON field maps.

Design docs target Facet-derived metadata as end state (`docs/specs/modules/design_reference.md:31`, `docs/specs/modules/design_reference.md:240`, `docs/specs/modules/design_reference.md:249`).

## Goal
Identify a safe, incremental migration path that:
- Enables path-aware/flatten-aware metadata consumption for typed modules first.
- Preserves existing optimizer and legacy behaviors until replacement surfaces are ready.
- Avoids a single high-risk cutover across adapter, predictor, macro, optimizer, and examples.

## Questions

| # | Question |
|---|---|
| **S6-Q1** | Where is `FieldSpec` wired into typed prompt formatting/parsing today? |
| **S6-Q2** | Where is `MetaSignature` wired into legacy adapter/optimizer paths today? |
| **S6-Q3** | Which interfaces make big-bang migration high risk? |
| **S6-Q4** | Can Facet provide the metadata primitives needed for `SignatureSchema` (paths, attrs, flatten, traversal)? |
| **S6-Q5** | What compatibility seams allow incremental rollout without breaking optimizers/examples? |
| **S6-Q6** | What phased migration sequence minimizes regressions and duplicate rewrites? |

## Findings

1. `Signature` is currently hard-coupled to static `FieldSpec` arrays and `output_format_content`.
   - Evidence: trait requires `input_fields()`, `output_fields()`, `output_format_content()` (`crates/dspy-rs/src/core/signature.rs:44`, `crates/dspy-rs/src/core/signature.rs:45`, `crates/dspy-rs/src/core/signature.rs:46`).
   - `FieldSpec` is top-level and pathless (`name`, `rust_name`, `description`, `type_ir`, `constraints`, `format`) (`crates/dspy-rs/src/core/signature.rs:6`).

2. `#[derive(Signature)]` still emits static `FieldSpec` arrays and returns them via trait methods.
   - Evidence: `generate_field_specs` creates static arrays (`crates/dsrs-macros/src/lib.rs:444`, `crates/dsrs-macros/src/lib.rs:567`).
   - Signature impl returns those statics (`crates/dsrs-macros/src/lib.rs:660`, `crates/dsrs-macros/src/lib.rs:664`).
   - Derive helper attrs still exclude `flatten` (`crates/dsrs-macros/src/lib.rs:22`).

3. Typed ChatAdapter path is currently top-level-key based, not path-based.
   - Formatting reads input by `field_spec.rust_name` from top-level map (`crates/dspy-rs/src/adapter/chat.rs:530`, `crates/dspy-rs/src/adapter/chat.rs:531`).
   - Assistant formatting uses top-level output keys (`crates/dspy-rs/src/adapter/chat.rs:555`, `crates/dspy-rs/src/adapter/chat.rs:556`).
   - Parsing inserts directly into top-level `output_map` by rust field name (`crates/dspy-rs/src/adapter/chat.rs:607`, `crates/dspy-rs/src/adapter/chat.rs:720`).

4. Typed `Predict<S>` still implements `MetaSignature` via `FieldSpec -> serde_json::Value` conversion.
   - Bridge function converts typed `FieldSpec` arrays into legacy JSON maps (`crates/dspy-rs/src/predictors/predict.rs:251`, `crates/dspy-rs/src/predictors/predict.rs:265`).
   - `Predict<S>` exposes this through `MetaSignature` impl (`crates/dspy-rs/src/predictors/predict.rs:443`, `crates/dspy-rs/src/predictors/predict.rs:473`, `crates/dspy-rs/src/predictors/predict.rs:478`).

5. Legacy adapter/predictor surfaces are explicitly `MetaSignature`-based.
   - Adapter trait is defined on `&dyn MetaSignature` (`crates/dspy-rs/src/adapter/mod.rs:15`, `crates/dspy-rs/src/adapter/mod.rs:24`).
   - Legacy system prompt/user/assistant formatting uses `MetaSignature::input_fields/output_fields` JSON objects (`crates/dspy-rs/src/adapter/chat.rs:295`, `crates/dspy-rs/src/adapter/chat.rs:360`, `crates/dspy-rs/src/adapter/chat.rs:394`).
   - `LegacyPredict` stores `Arc<dyn MetaSignature>` and calls adapter with it (`crates/dspy-rs/src/predictors/predict.rs:513`, `crates/dspy-rs/src/predictors/predict.rs:587`).

6. Optimizer APIs still assume `MetaSignature` and manual `Optimizable::parameters()` traversal.
   - `Optimizable` contract exposes `get_signature() -> &dyn MetaSignature` and `parameters()` (`crates/dspy-rs/src/core/module.rs:85`, `crates/dspy-rs/src/core/module.rs:89`).
   - MIPRO/COPRO read `input_fields`/`output_fields` JSON and mutate instructions through that interface (`crates/dspy-rs/src/optimizer/mipro.rs:491`, `crates/dspy-rs/src/optimizer/copro.rs:73`, `crates/dspy-rs/src/optimizer/copro.rs:223`).
   - Optimizable derive is still `#[parameter]`-driven and flatten-by-recursive-calls, with unsafe pointer casts (`crates/dsrs-macros/src/optim.rs:41`, `crates/dsrs-macros/src/optim.rs:49`, `crates/dsrs-macros/src/optim.rs:72`, `crates/dsrs-macros/src/optim.rs:92`).

7. Legacy usage is still broad in examples/tests, so immediate removal would be disruptive.
   - `LegacyPredict` appears across optimizer code, examples, and tests (`crates/dspy-rs/src/optimizer/mipro.rs:289`, `crates/dspy-rs/src/optimizer/gepa.rs:321`, `crates/dspy-rs/tests/test_optimizable.rs:1`).
   - `MetaSignature` is directly used in adapter and optimizer tests (`crates/dspy-rs/tests/test_adapters.rs:8`, `crates/dspy-rs/tests/test_miprov2.rs:455`).

8. Existing spike docs already identify path-aware metadata as migration-critical and suggest phased compatibility.
   - S1: path-aware runtime metadata is needed; macro-only change is insufficient (`docs/specs/modules/spikes/S1-generic-signature-derive.md:35`, `docs/specs/modules/spikes/S1-generic-signature-derive.md:101`, `docs/specs/modules/spikes/S1-generic-signature-derive.md:110`).
   - S3: composed augmentation rollout is blocked by top-level `FieldSpec` model (`docs/specs/modules/spikes/S3-augmentation-deref-composition.md:41`, `docs/specs/modules/spikes/S3-augmentation-deref-composition.md:84`).
   - S2: optimizer migration needs compatibility shims (`docs/specs/modules/spikes/S2-dynpredictor-handle-discovery.md:26`, `docs/specs/modules/spikes/S2-dynpredictor-handle-discovery.md:94`, `docs/specs/modules/spikes/S2-dynpredictor-handle-discovery.md:111`).

9. Facet-backed primitives needed for `SignatureSchema` are already exercised in this workspace, but flatten semantics still need local migration tests.
   - Facet is the schema substrate for `BamlSchema` (`.workspaces/facet/crates/bamltype/src/lib.rs:87`).
   - Runtime schema construction already walks `Shape`/`Def` variants (`Option`, `List`, `Map`) and field/type attrs (`.workspaces/facet/crates/bamltype/src/schema_builder.rs:8`, `.workspaces/facet/crates/bamltype/src/schema_builder.rs:113`, `.workspaces/facet/crates/bamltype/src/schema_builder.rs:123`, `.workspaces/facet/crates/bamltype/src/schema_builder.rs:127`, `.workspaces/facet/crates/bamltype/src/schema_builder.rs:129`, `.workspaces/facet/crates/bamltype/src/schema_builder.rs:622`).
   - Typed attribute payload extraction is in active use via `Attr::get_as` and local attr grammar (`.workspaces/facet/crates/bamltype/src/facet_ext.rs:37`, `.workspaces/facet/crates/bamltype/src/facet_ext.rs:43`, `.workspaces/facet/crates/bamltype/src/facet_ext.rs:55`).
   - Reflect conversion already recurses through containers using `Partial`/`Peek` (`.workspaces/facet/crates/bamltype/src/convert.rs:4`, `.workspaces/facet/crates/bamltype/src/convert.rs:117`, `.workspaces/facet/crates/bamltype/src/convert.rs:132`, `.workspaces/facet/crates/bamltype/src/convert.rs:191`, `.workspaces/facet/crates/bamltype/src/convert.rs:215`, `.workspaces/facet/crates/bamltype/src/convert.rs:522`, `.workspaces/facet/crates/bamltype/src/convert.rs:593`, `.workspaces/facet/crates/bamltype/src/convert.rs:602`).
   - Flatten-specific behavior is still migration-sensitive because current typed adapter path remains top-level-key based (Finding 3), so flatten support must be proven by parity tests, not assumed.

## Migration options

| Option | Approach | Pros | Risks |
|---|---|---|---|
| **A. Big-bang cutover** | Replace `FieldSpec` + `MetaSignature` + `LegacyPredict` in one release. | Clean architecture quickly. | High break risk across adapter, optimizers, examples, and tests; hard rollback boundary. |
| **B. Typed-path first, compatibility bridge** | Introduce `SignatureSchema` for typed path; keep legacy `MetaSignature` path and provide conversion shims during transition. | Unblocks flatten/path-aware typed work (S1/S3) with controlled blast radius. | Temporary dual surfaces and conversion code. |
| **C. Optimizer-first replacement** | Replace `Optimizable`/`MetaSignature` discovery before typed adapter migration. | Early investment in long-term optimizer model. | Delays typed schema wins; still blocked by pathless typed adapter metadata. |

## Decision

**Subsumed by S1 → Option C.** There is no incremental migration. `SignatureSchema` is built from Facet, `FieldSpec`/`MetaSignature`/`LegacyPredict` are deleted. The 4-phase plan below is preserved as reference for understanding the dependency surface, but it will not be executed as phased work.

## Original Recommendation (not adopted)

The spike originally recommended Option B in four gated phases. Each phase has a compatibility contract and explicit rollback lever.

### Phase plan (tightened)

| Phase | Scope change | Compatibility points that must remain true | Rollback control | Exit gate |
|---|---|---|---|---|
| **Phase 1: Introduce typed `SignatureSchema` (internal only)** | Add `SignatureSchema`/`FieldPath` model and cache, but do not switch typed call sites yet. | `Signature` trait API stays unchanged (`input_fields`, `output_fields`, `output_format_content`); all typed and legacy tests remain green with old adapter path. | Keep schema read-path behind internal feature gate; immediate fallback is old `FieldSpec` readers only. | Schema parity snapshots for non-flatten signatures match current `FieldSpec`/output-format behavior. |
| **Phase 2: Cut typed adapter to path-aware schema reads/writes** | Switch `format_*_typed` + `parse_response_typed` to schema path traversal and path insertion. | Legacy adapter remains `&dyn MetaSignature`; `Predict<S>: MetaSignature` bridge remains unchanged for optimizer callers; typed prompt text remains byte-for-byte stable for non-flatten signatures. | Dual-path adapter switch (`typed_adapter_v1`/`typed_adapter_v2`) with runtime/compile-time kill switch. | Typed parity tests pass for flat signatures and new path tests pass for nested/flatten-shaped fixtures. |
| **Phase 3: Migrate derive to Facet-shape schema generation** | Extend derive for `flatten`, and generate schema from `S::Input`/`S::Output` Facet shape metadata. | Temporary dual-emission: keep old `FieldSpec` outputs for legacy call sites while typed adapter consumes `SignatureSchema`; deterministic field ordering and alias/constraint/format parity preserved. | Keep macro fallback mode that emits legacy-only `FieldSpec` path if schema generation fails in release branch. | Generated-schema coverage includes flatten + generic signatures from S1/S3 fixtures; no typed regressions. |
| **Phase 4: Migrate optimizers/discovery and retire legacy path** | Introduce schema-backed optimizer handles/discovery (S2/S5), migrate COPRO/MIPRO/GEPA incrementally, then remove legacy surfaces. | During migration, optimizer entry points can consume both old (`MetaSignature`/`Optimizable`) and new schema-backed handles; `LegacyPredict` remains available until final removal checkpoint. | Per-optimizer rollback: keep old entrypoint active until each optimizer passes parity suite; no cross-optimizer big-bang switch. | All optimizers/examples/tests run without `MetaSignature`/`LegacyPredict`; removal checklist completed and legacy shims deleted in one final cleanup PR. |

### Compatibility checkpoints (cross-phase)

1. Keep legacy adapter interface (`Adapter::format`/`parse`) stable until Phase 4 is complete (`crates/dspy-rs/src/adapter/mod.rs:15`, `crates/dspy-rs/src/adapter/mod.rs:24`).
2. Preserve typed-to-legacy bridge while optimizer code depends on `MetaSignature` (`crates/dspy-rs/src/predictors/predict.rs:443`, `crates/dspy-rs/src/predictors/predict.rs:473`, `crates/dspy-rs/src/predictors/predict.rs:478`).
3. Treat optimizer migration as N small cutovers (COPRO, MIPRO, GEPA) instead of one global switch (`crates/dspy-rs/src/optimizer/copro.rs:73`, `crates/dspy-rs/src/optimizer/mipro.rs:491`, `crates/dspy-rs/src/optimizer/gepa.rs:321`).
4. Keep examples/tests as canaries during dual-path operation (`crates/dspy-rs/tests/test_adapters.rs:8`, `crates/dspy-rs/tests/test_miprov2.rs:455`, `crates/dspy-rs/tests/test_optimizable.rs:1`).

### Rollback risk controls

1. **Dual-path toggles**: keep old and new typed adapter paths side-by-side through Phase 2.
2. **Dual-emission window**: keep derive outputting legacy `FieldSpec` metadata during Phase 3 while schema path stabilizes.
3. **Per-surface canaries**: require parity test suites for typed adapter, derive, and each optimizer before advancing phase gate.
4. **No irreversible deletions before Phase 4 exit**: do not remove `MetaSignature`/`LegacyPredict` until all consumer references are gone.
5. **Branch-safe fallback**: each phase merge must be revertible without data migration scripts or serialized-state rewrites.

## Concrete implementation steps

1. Add `signature_schema.rs` in runtime core with `SignatureSchema`, `FieldSchema`, and `FieldPath`.
2. Add `SignatureSchema::of::<S>()` cache entrypoint and parity adapter that can project schema fields back to legacy `FieldSpec`-shaped JSON.
3. Introduce typed adapter path helpers: `read_at_path` and `insert_at_path` for nested read/write.
4. Add dual-path typed adapter switch and parity tests asserting non-flatten prompt/parse equivalence against current behavior.
5. Extend `#[derive(Signature)]` parser to accept `flatten` and preserve field ordering, alias, constraints, and format data.
6. Add schema-generation path in derive from `S::Input`/`S::Output` Facet metadata, while retaining temporary `FieldSpec` emission.
7. Add focused migration tests:
   - flat signature parity,
   - flatten roundtrip (input + output),
   - alias/constraint/format parity,
   - deterministic ordering across rebuilds.
8. Add optimizer compatibility shim so current `MetaSignature` consumers and new schema-backed handles can coexist.
9. Migrate COPRO first, then MIPRO, then GEPA, keeping per-optimizer fallback entrypoints until parity passes.
10. Remove legacy surfaces only after a final zero-reference check for `MetaSignature`, `LegacySignature`, and `LegacyPredict`.

## Acceptance
S6 is complete when:

- All S6 questions are answered with code/doc evidence.
- Typed adapter/predict path can operate from `SignatureSchema` (including path-aware parse/format) with parity for non-flatten signatures and explicit coverage for flatten signatures.
- Legacy/optimizer flows continue functioning during transition via compatibility shims, with per-optimizer cutover evidence.
- Every phase exit gate and rollback lever in this document has a passing validation artifact (test/snapshot/checklist item).
- A final cleanup checkpoint removes `FieldSpec`/`MetaSignature`/`LegacyPredict` only after zero-reference and parity checks pass.

## Open Risks

1. **S2/S5 are still hard blockers for full optimizer migration**: handle extraction + container discovery details are not finalized (`docs/specs/modules/spikes/S2-dynpredictor-handle-discovery.md:26`).
2. **Flatten rollout risk remains concentrated in Phase 3**: current typed adapter is still top-level-key based until Phase 2 lands (`crates/dspy-rs/src/adapter/chat.rs:530`, `crates/dspy-rs/src/adapter/chat.rs:607`).
3. **Dual-path drift risk**: temporary coexistence of schema path + legacy path can diverge if parity tests are not required at every phase gate.
