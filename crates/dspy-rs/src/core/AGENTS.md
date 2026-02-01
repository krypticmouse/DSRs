# AGENTS.md - core/

## Boundary

Core abstractions for DSPy-RS: LM client, traits, error types, and global configuration.

**Depends on:** `baml-bridge` (type system), `rig` (LLM providers), `bon` (builders)
**Depended on by:** `predictors/`, `optimizer/`, `adapter/`, `rlm/`, all user code
**NEVER:** Put predictor logic here (belongs in `predictors/`); put optimization logic here (belongs in `optimizer/`)

### What lives here
- `lm/` - `LM` struct, `LMClient` registry, `Chat`, `Message`, usage tracking
- `module.rs` - `Module` and `Optimizable` traits
- `signature.rs` - `Signature` trait, `FieldSpec`, type metadata (`SigTypeMeta`)
- `settings.rs` - `GLOBAL_SETTINGS`, `configure()`, `get_lm()`
- `call_result.rs` - `CallResult<O>`, constraint tracking
- `errors.rs` - `PredictError`, `LmError`, `ParseError`, `ConversionError`
- `specials.rs` - Placeholder types (`NoTool`, `History`)

---

## How to work here

### Key traits

**`Signature`** - Compile-time contract for LLM I/O. Derived via `#[derive(Signature)]`.
```rust
pub trait Signature: Send + Sync + 'static {
    type Input: BamlType;
    type Output: BamlType;
    fn instruction() -> &'static str;
    fn input_fields() -> &'static [FieldSpec];
    fn output_fields() -> &'static [FieldSpec];
}
```

**`Module`** - Async forward pass for composable predictors.
**`Optimizable`** - Mutable signature access for optimizers.

### Global config pattern

Always use `configure()` to set up; predictors read via `GLOBAL_SETTINGS`.

```rust
// Correct: at app startup
configure(LM::builder().model("openai:gpt-4o").build().await?, ChatAdapter);

// Internal predictor code uses:
let guard = GLOBAL_SETTINGS.read().unwrap();
let settings = guard.as_ref().unwrap();
let lm = Arc::clone(&settings.lm);
```

### Golden examples
- `settings.rs` lines 23-30 - How `get_lm()` and `configure()` work
- `lm/mod.rs` lines 67-88 - `LM` struct with builder
- `call_result.rs` lines 87-123 - Unit test for `CallResult` accessors

---

## Verification

```bash
cargo test -p dspy-rs call_result     # CallResult unit tests
cargo test -p dspy-rs test_settings   # Global config tests
cargo test -p dspy-rs test_lm         # LM builder/client tests
cargo test -p dspy-rs test_signature  # Signature trait tests
```

---

## Don't do this

- **Direct `GLOBAL_SETTINGS` access from user code** - Use `configure()` at startup, `.with_lm()` for overrides
- **`unwrap()` in new code** - Use `?` or explicit error handling
- **Adding predictor logic** - That belongs in `predictors/`
- **Provider-specific code in `lm/`** - Provider dispatch is via `rig`; don't add OpenAI/Anthropic specific logic

---

## Gotchas

1. **`LM::builder().build()` is async** - The builder's `build()` method returns `impl Future`. Always `.await` it.
   ```rust
   let lm = LM::builder().model("openai:gpt-4o").build().await?;  // Correct
   // LM::builder().model("...").build() alone is NOT an LM
   ```

2. **`configure()` must be called before any `Predict::call()`** - Otherwise panic on `None` unwrap.

3. **`LM::default()` blocks the runtime** - Uses `block_on` internally. Prefer explicit async builder.

4. **Reasoning models need `temperature: None`** - o1, o3, gpt-5.2 etc. reject temperature parameter.
   ```rust
   LM::builder().model("openai:o3").temperature(None).build().await?
   ```

5. **`get_lm()` returns `Arc<LM>`** - Clone is cheap; don't try to get `&LM` directly.
