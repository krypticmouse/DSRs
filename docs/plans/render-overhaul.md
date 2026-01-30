Below is a concrete implementation plan that maps your spec onto *this* repo’s structure (baml-bridge + dspy-rs + macros), with exact files, key new types/functions, and the critical decisions you’ll have to lock in.

I’m going to assume the target outcome is:

* `Signature`-typed prompt formatting in `ChatAdapter` becomes “boring templates” + typed values.
* Type-level rendering and field-level overrides are first-class.
* Rendering can be strict/permissive and returns diagnostics.
* The `"json"|"yaml"|"toon"` pathways become *styles* and always use `PromptValue.ty` as the schema target.

---

# 0. Architectural placement decision (make once)

You have two viable homes for the new machinery:

## Option 1 (recommended): put PromptValue/World/Renderer in `baml-bridge`

Why:

* `baml-bridge` already owns `BamlValue`, `TypeIR`, `Registry`, `internal_baml_jinja`, `minijinja` env setup.
* It’s the natural layer to “carry type info and render it”.
* `dspy-rs` then only needs “signature compilation + message templates”.

**Consequence:**
`dspy-rs` becomes a consumer of a stable `baml_bridge::prompt::*` API.

## Option 2: put everything in `dspy-rs`

Why:

* Short-term convenience (you’re already editing adapters there).
* But it’s conceptually less clean: prompt rendering becomes tied to DSPy concerns.

**I’ll plan assuming Option 1**, and call out the deltas if you choose Option 2.

---

# 1. Add new “prompt” module surface in `baml-bridge`

## Files to add

Create these files:

* `crates/baml-bridge/src/prompt/mod.rs`
* `crates/baml-bridge/src/prompt/world.rs`
* `crates/baml-bridge/src/prompt/value.rs`
* `crates/baml-bridge/src/prompt/renderer.rs`
* `crates/baml-bridge/src/prompt/jinja.rs`

Then export them from:

* `crates/baml-bridge/src/lib.rs`

### `crates/baml-bridge/src/lib.rs`

Add (near the other `pub use`):

```rust
pub mod prompt;
pub use prompt::*;
```

---

# 2. Extend `Registry` so it can collect renderers (type-level)

Your spec explicitly wants: “Registry can remain the builder concept, but it now also registers renderers.”

That is a *perfect fit* for your existing `crates/baml-bridge/src/registry.rs`.

## Modify: `crates/baml-bridge/src/registry.rs`

### Add new fields

Add a renderer database to `Registry`:

```rust
use crate::prompt::renderer::{RendererSpec, RendererKey};

#[derive(Debug, Default)]
pub struct Registry {
    // existing fields...
    renderers: IndexMap<RendererKey, RendererSpec>,
}
```

### Define keys/specs (new types)

You’ll add these in `prompt/renderer.rs`, but `Registry` will use them.

A good key shape:

```rust
pub struct RendererKey {
    pub type_key: TypeKey, // e.g. internal class/enum name + streaming mode (if relevant)
    pub style: &'static str,
}
```

`TypeKey` should be derived from `TypeIR`:

* Class: `(name, mode)`
* Enum: `name`
* Everything else: you *can* allow, but most type-level renderers will be named types.

### Add registration methods

In `Registry`:

```rust
pub fn register_renderer(&mut self, key: RendererKey, spec: RendererSpec) {
    self.renderers.insert(key, spec);
}
```

### Add a “build world” path

Right now, `Registry::build(self, target: TypeIR) -> OutputFormatContent`.

You want an additional path that also returns renderer defs:

```rust
pub fn build_with_renderers(self, target: TypeIR) -> (OutputFormatContent, RendererDbSeed) { ... }
```

Where `RendererDbSeed` is a simple container of the registered renderer specs. For example:

```rust
pub struct RendererDbSeed {
    pub specs: IndexMap<RendererKey, RendererSpec>,
}
```

**Side effect:** existing callers of `build()` don’t need to change. Keep `build()` as-is and implement it in terms of `build_with_renderers().0`.

---

# 3. Implement PromptWorld (type universe + renderer registry + env + settings)

## New file: `crates/baml-bridge/src/prompt/world.rs`

### Types to add

```rust
pub struct PromptWorld {
    pub types: TypeDb,
    pub renderers: RendererDb,
    pub jinja: minijinja::Environment<'static>,
    pub settings: RenderSettings,
    pub union_resolver: UnionResolver,
}
```

#### TypeDb (wrap OutputFormatContent’s arcs)

You already basically have this in `OutputFormatContent`:

* `enums: Arc<IndexMap<String, Enum>>`
* `classes: Arc<IndexMap<(String, StreamingMode), Class>>`
* `structural_recursive_aliases: Arc<IndexMap<String, TypeIR>>`

So your TypeDb can just be a thin wrapper around those arcs:

```rust
pub struct TypeDb {
    pub enums: Arc<IndexMap<String, Enum>>,
    pub classes: Arc<IndexMap<(String, StreamingMode), Class>>,
    pub structural_recursive_aliases: Arc<IndexMap<String, TypeIR>>,
    pub recursive_classes: Arc<IndexSet<String>>,
}
```

Add helper methods you’ll need for typed traversal:

* `fn find_class(&self, name: &str, mode: StreamingMode) -> Option<&Class>`
* `fn class_field_type(&self, name: &str, mode: StreamingMode, field: &str) -> Option<TypeIR>`
* `fn resolve_recursive_alias(&self, name: &str) -> Option<&TypeIR>`

#### RenderSettings / RenderFailureMode

Put these in `prompt/renderer.rs` or `prompt/world.rs` (your choice, but keep public).

```rust
pub enum RenderFailureMode { Strict, Permissive }

pub struct RenderSettings {
    pub max_total_chars: usize,
    pub max_string_chars: usize,
    pub max_list_items: usize,
    pub max_map_entries: usize,
    pub max_depth: usize,
    pub max_union_branches_shown: usize,
    pub failure_mode: RenderFailureMode,
}
```

Give sensible defaults.

### Constructing a PromptWorld

Add a constructor that takes the registry output:

```rust
impl PromptWorld {
    pub fn from_registry(
        output_format: OutputFormatContent,
        renderers: RendererDbSeed,
        settings: RenderSettings,
    ) -> Self { ... }
}
```

Inside:

* Build `TypeDb` from the arcs inside `output_format`.
* Build `RendererDb` (compile templates into the env, store references).
* Build `jinja` via the existing `jsonish::jinja_helpers::get_env()` and then extend it with your “prompt” filters (see section 6).
* Choose a default `union_resolver` function.

### Critical decision: env ownership and template compilation

You have two routes:

1. **Compile templates into `Environment` by name** (recommended):

   * `env.add_template("renderer::<key>", source)`
   * Store the template name in your renderer handle.
   * Rendering uses `env.get_template(name)`.

2. **Use `env.render_str(source, ctx)` every time**:

   * Much simpler, but slower and less “compiled”.

If you want compile-time validation and stable template identity for diagnostics (`renderer: "type:jinja:<name>"`), route (1) is cleaner.

---

# 4. Implement PromptValue (typed runtime value + traversal)

## New file: `crates/baml-bridge/src/prompt/value.rs`

### Core struct

Match your spec:

```rust
pub struct PromptValue {
    pub value: BamlValue,
    pub ty: TypeIR,
    pub world: Arc<PromptWorld>,
    pub override_renderer: Option<RendererRef>,
    pub path: PromptPath,
}
```

Where:

* `PromptPath` is just a small wrapper that builds strings like `inputs.history.entries[3]`.
* `RendererRef` points at a renderer (either a key lookup into the world DB, or a direct handle).

### Typed child navigation

Add methods (these will be used by the Jinja wrapper):

```rust
impl PromptValue {
    pub fn child_field(&self, field: &str) -> Option<PromptValue>;
    pub fn child_index(&self, idx: usize) -> Option<PromptValue>;
    pub fn child_map_value(&self, key: &str) -> Option<PromptValue>;

    pub fn resolved_ty(&self) -> TypeIR; // union resolution hook point
}
```

Implementation details:

* For `TypeIR::Class { name, mode, .. }`, look up the class in `world.types`, find the declared field type, and fetch the child from the underlying `BamlValue::Class(_, map)`.
* For `TypeIR::List(inner, _)`, child type is `*inner`.
* For `TypeIR::Map(_, value_ty, _)`, child type is `*value_ty` (key typing is rarely needed for rendering, but keep it).
* For `TypeIR::RecursiveTypeAlias { name, .. }`, resolve via `world.types.structural_recursive_aliases`.
* For `TypeIR::Union(...)`, call `world.union_resolver` and either:

  * return a “resolved” view type for traversal, or
  * keep it as union and limit traversal (see union policy below).

### Union resolution caching

Determinism requirement implies: “pick once, reuse always”.

So `PromptValue` should internally memoize union resolution. Since `PromptValue` is usually cheap to clone, the memo should be in an `Arc<Inner>`:

```rust
struct PromptValueInner {
   value: BamlValue,
   ty: TypeIR,
   union_resolution: OnceLock<UnionResolution>,
   // ...
}
```

---

# 5. Implement renderer pipeline + diagnostics + fallback (the heart)

## New file: `crates/baml-bridge/src/prompt/renderer.rs`

### Renderer types

Your spec model:

```rust
pub enum Renderer {
    Jinja { template_name: String },
    Func(fn(&PromptValue, &RenderCtx) -> Result<String, RenderError>),
}
```

### Specs vs compiled renderers

You’ll want a “seed spec” stored in Registry, and then a compiled form in PromptWorld:

```rust
pub enum RendererSpec {
    Jinja { source: &'static str },
    Func { f: fn(&PromptValue, &RenderCtx) -> Result<String, RenderError> },
}
```

Then compile Jinja specs into templates on world creation.

### RenderResult + diagnostics

Exactly as you wrote:

```rust
pub struct RenderResult {
    pub text: String,
    pub diagnostics: Vec<RenderDiagnostic>,
}

pub struct RenderDiagnostic {
    pub path: String,
    pub ty: String,
    pub style: String,
    pub renderer: String,
    pub error: String,
    pub fell_back_to: String,
}
```

### RenderCtx

This is where you connect “value rendering inside Jinja” to a shared diagnostic sink:

```rust
pub struct RenderCtx {
    pub settings: RenderSettings,
    pub diagnostics: Arc<std::sync::Mutex<Vec<RenderDiagnostic>>>,
    pub depth: usize,
}
```

You can optionally add:

* `stack: Vec<(TypeKey, Style)>` for recursion guard (“renderer recursion loop”).

### Renderer resolution pipeline

Implement something like:

```rust
impl PromptWorld {
    pub fn render_value(&self, pv: &PromptValue, style: Option<&str>) -> Result<RenderResult, RenderError>;
}
```

Inside, execute the pipeline as an ordered iterator of steps:

1. **Determine requested style**

   * if `style.is_some()`: use it
   * else if `pv.override_renderer` includes a style override: use it
   * else: `"default"`

2. **Try renderers in precedence**:

   * per-field override renderer (if present and matches style)
   * type-level renderer (lookup by `TypeKey + style`)
   * built-in style renderer if style is `"json"|"yaml"|"toon"`
   * structural fallback
   * last-ditch JSON stringify

3. **Failure handling**:

   * Strict: return `Err(RenderError::RendererFailed { ... })`
   * Permissive: push diagnostic + proceed to next fallback

4. **Budget enforcement**:

   * enforce `max_total_chars` on *each render result* (and optionally on whole message in compiled signature)

### Built-in format styles (“json/yaml/toon”)

This is where you fix the schema-target bug.

Implement:

```rust
fn render_format_style(pv: &PromptValue, fmt: &str) -> Result<String, RenderError> {
    let view = pv.world.types.output_format_view_for(&pv.ty);
    internal_baml_jinja::format_baml_value(&pv.value, &view, fmt)
        .map_err(|e| RenderError::FormatFailed(e))
}
```

`output_format_view_for` is a helper on TypeDb / PromptWorld that builds:

```rust
OutputFormatContent {
  enums: self.enums.clone(),
  classes: self.classes.clone(),
  recursive_classes: self.recursive_classes.clone(),
  structural_recursive_aliases: self.structural_recursive_aliases.clone(),
  target: ty.clone(),
}
```

This is exactly the “wrong schema context” fix you called out.

### Structural fallback renderer

Implement:

```rust
fn render_structural(pv: &PromptValue, ctx: &RenderCtx) -> String
```

Rules:

* Truncate strings to `max_string_chars`
* Limit lists/maps to `max_list_items`/`max_map_entries`
* Stop at `max_depth`
* For classes: use schema order from `Class.fields`
* For recursion: show `Type { ... }`
* For union:

  * if resolved: render as that branch type
  * else print:

    * `one of: A | B | C`
    * then `render_format_style(pv, "json")` with budgets applied

**Critical decision:** does traversal (Jinja iteration) also honor list/map caps?
I recommend “yes by default” to prevent prompt blowups from custom templates; allow overrides later.

---

# 6. Implement Jinja integration: typed traversal object

## New file: `crates/baml-bridge/src/prompt/jinja.rs`

This replaces the current “serialize BamlValue into minijinja Value” approach for prompt rendering.

### Object wrapper

Implement a custom `minijinja::value::Object`:

```rust
pub struct JinjaPromptValue {
    pv: PromptValue,
}
```

And a conversion:

```rust
impl PromptValue {
    pub fn to_jinja(&self) -> minijinja::Value {
        minijinja::Value::from_object(JinjaPromptValue { pv: self.clone() })
    }
}
```

### JinjaPromptValue behavior mapping

Implement these `Object` hooks:

* `repr()`:

  * Class/Map -> `ObjectRepr::Map`
  * List -> `ObjectRepr::Seq`
  * Everything else -> `ObjectRepr::Plain`

* `get_value(key)`:

  * Special keys:

    * `"render"`: return a callable object that implements `.render("style")`
    * `"raw"`: return the raw underlying value converted to a plain minijinja `Value`
  * Otherwise:

    * If class/map: treat key as field name and return `child_field(key)`
    * If seq: treat key as index and return `child_index(idx)`

* `enumerate()` + `enumerator_len()`:

  * For lists: cap to `max_list_items`
  * For maps/classes: expose keys (optionally cap to `max_map_entries`)

* `is_true()`:

  * false for null/empty/zero-ish, true otherwise (you can mirror Rust truthiness or Python-ish; just pick one and document)

* `render(f)`:

  * This is invoked when template does `{{ value }}`.
  * Call `pv.world.render_value(&pv, None)`; write resulting text into formatter.
  * In permissive mode, render never errors; it only logs diagnostics.

### `.render("style")` support

Implement a separate callable object:

```rust
pub struct JinjaRenderMethod { pv: PromptValue }

impl Object for JinjaRenderMethod {
    fn call(&self, ..., args: &[Value]) -> Result<Value, Error> { ... }
}
```

Where:

* arg0 = style string
* call `world.render_value(&pv, Some(style))`
* return `Value::from(rendered_text)`

### Filters to ship in PromptWorld env

Extend `jsonish::jinja_helpers::get_env()` with:

* `truncate(value, n)` (string truncation)
* `format_count(i64)` (already used in REPLHistory)
* `slice_chars(value, n)` (already used)
* maybe `to_json(value)` (calls `.render("json")` internally)

Implementation detail:

* You can add these filters in `PromptWorld::from_registry()` after calling `get_env()`.

---

# 7. Union resolver hook (one policy in one place)

## New file section: likely in `prompt/value.rs` or `prompt/world.rs`

Define:

```rust
pub enum UnionResolution {
    Resolved(TypeIR),
    Ambiguous { candidates: Vec<TypeIR> },
}

pub type UnionResolver = fn(value: &BamlValue, union: &baml_types::UnionType, world: &PromptWorld) -> UnionResolution;
```

Implementation outline (scoring):

* If value is `BamlValue::Class(_, map)`:

  * For each class branch, score by overlap between map keys and class fields.
* If value is string:

  * enum branch: score if it matches a known variant/alias
  * string primitive: score high
* If value is list/map: match list/map branches, etc.

**Architectural decision:**
When ambiguous, keep union type and forbid field traversal by name unless value is class-like and you can treat it dynamically. The spec says “keep the union type for traversal”. That implies:

* `child_field("x")` on unresolved union should either:

  * attempt resolution again (but must stay deterministic), or
  * return `None` / `null` plus a diagnostic.

I’d implement: “attempt resolution once; if ambiguous, refuse typed field traversal (return None)”.

---

# 8. Replace “DefaultJinjaRender” with type-level renderer registration

You currently have:

* `crates/baml-bridge/src/render_trait.rs` (DefaultJinjaRender)
* and a special-case-ish REPLHistory renderer in `crates/dspy-rs/src/rlm/history.rs`

## What to do with `DefaultJinjaRender`

Two choices:

### Choice A (clean): deprecate it, replace with `#[prompt(renderer(...))]`

* Keep the trait around temporarily as compatibility.
* Stop relying on it for new rendering.

### Choice B (bridge): make it integrate into the new world

* Add a helper function/macro that registers `T::DEFAULT_TEMPLATE` into the registry.

Example helper:

```rust
pub fn register_default_jinja_renderer<T: DefaultJinjaRender>(reg: &mut Registry) {
    reg.register_renderer(RendererKey::for_type::<T>("default"), RendererSpec::Jinja { source: T::DEFAULT_TEMPLATE });
}
```

But you’d still need call sites to invoke it, which violates “no manual registry spelunking”.

**Recommendation:** Choice A.

---

# 9. Add type-level renderer attributes in `baml-bridge-derive`

## Modify: `crates/baml-bridge-derive/src/lib.rs`

### Parse new container attribute: `#[prompt(...)]`

Extend the derive macro’s attribute parsing to accept a `prompt` attribute on structs/enums.

Support at least:

* `#[prompt(renderer(default = r#"..."#))]`
* `#[prompt(renderer_fn(default = "path::to::func"))]`

Generate in `BamlTypeInternal::register`:

```rust
reg.register_renderer(
   RendererKey { type_key: <Self as BamlTypeInternal>::baml_internal_name(), style: "default" },
   RendererSpec::Jinja { source: <template literal> },
);
```

For function renderer, store fn pointer (must be in scope).

### Compile-time template validation

In the proc-macro crate, you can:

* parse Jinja source with minijinja’s parser (syntax check)
* validate filter/test names against your shipped env filters

This requires:

* adding `minijinja` as a dependency of `baml-bridge-derive`
* maintaining a known list of filter names (regex_match, sum, truncate, slice_chars, format_count, etc.)

**Note:** you won’t validate field names like `value.entries`; that’s runtime schema-aware.

---

# 10. Add per-field overrides in Signature field metadata

Right now, your `FieldSpec` is:

```rust
pub struct FieldSpec {
    pub name: &'static str,
    pub rust_name: &'static str,
    pub description: &'static str,
    pub type_ir: fn() -> TypeIR,
    pub constraints: &'static [ConstraintSpec],
    pub format: Option<&'static str>,
}
```

## Modify: `crates/dspy-rs/src/core/signature.rs`

### Add rendering fields

Add:

```rust
pub struct FieldSpec {
    // existing
    pub format: Option<&'static str>, // treat as STYLE for now

    // new
    pub style: Option<&'static str>, // optional, can alias format to style or replace format
    pub renderer: Option<FieldRendererSpec>,
    pub render_settings: Option<FieldRenderSettings>,
}
```

Where `FieldRendererSpec` is a copyable static description:

```rust
pub enum FieldRendererSpec {
    Jinja { template: &'static str },
    Func { f: fn(&PromptValue, &RenderCtx) -> Result<String, RenderError> },
}
```

And `FieldRenderSettings` could be:

```rust
pub struct FieldRenderSettings {
    pub max_string_chars: Option<usize>,
    pub max_list_items: Option<usize>,
    pub max_map_entries: Option<usize>,
    pub max_depth: Option<usize>,
}
```

**If you want to keep churn minimal:**

* keep `format` and interpret it as style
* add only `renderer` + `render_settings`
* later rename `format -> style`

## Modify: `crates/dsrs-macros/src/lib.rs`

Extend the `#[derive(Signature)]` macro to parse new attributes on fields, e.g.:

* `#[prompt(style = "compact")]`
* `#[prompt(renderer = r#"..."#)]`
* `#[prompt(renderer_fn = "path::to::func")]`
* `#[prompt(max_list_items = 5)]` etc.

Then emit those into the generated `FieldSpec`.

This is the “per-field override attached during signature compilation” part of your spec.

---

# 11. Build CompiledSignature + boring templates in `dspy-rs`

This is the layer that makes external usage clean.

## Files to add

* `crates/dspy-rs/src/signature/compiled.rs` (or `crates/dspy-rs/src/prompt/compiled_signature.rs`)

## New types

```rust
pub struct CompiledSignature<S: Signature> {
    pub world: Arc<baml_bridge::PromptWorld>,
    pub system_template: String, // or compiled template name
    pub user_template: String,
    pub sig_meta: SigMeta,
    _phantom: std::marker::PhantomData<S>,
}

pub struct RenderedMessages {
    pub system: String,
    pub user: String,
    pub diagnostics: Vec<RenderDiagnostic>,
}
```

`SigMeta` should contain the lists that boring templates iterate over:

```rust
pub struct SigFieldMeta {
    pub llm_name: &'static str,
    pub rust_name: &'static str,
    pub type_name: String,   // simplified for prompt
    pub schema: Option<String>,
}

pub struct SigMeta {
    pub inputs: Vec<SigFieldMeta>,
    pub outputs: Vec<SigFieldMeta>,
}
```

## Compile path

Expose:

```rust
pub trait CompileExt: Signature {
    fn compile() -> CompiledSignature<Self>;
}
impl<T: Signature> CompileExt for T {}
```

Inside `compile()`:

1. Build a `Registry`
2. Register `Self::Input` and `Self::Output` into it
3. `build_with_renderers(TypeIR::string() or TypeIR::Top)` (target doesn’t matter for the *db*, only for schema rendering)
4. Create `PromptWorld::from_registry(...)`
5. Build `SigMeta`:

   * `type_name` from `TypeIR::diagnostic_repr()` + your simplifier
   * `schema` computed per output field:

     * build view `OutputFormatContent { target: field_ty }`
     * call `.render(RenderOptions::default().with_prefix(None or Some))`

## Render messages

Implement:

```rust
impl<S: Signature> CompiledSignature<S> {
    pub fn render_messages(&self, input: &S::Input) -> Result<RenderedMessages, RenderError>;
}
```

Steps:

1. Convert `input` to `BamlValue` via `ToBamlValue`.
2. For each input field spec:

   * extract raw `BamlValue` for that field
   * create a `PromptValue { value, ty: (field.type_ir)(), world, override_renderer: <from FieldSpec>, path: "inputs.<field>" }`
3. Build a Jinja context:

   * `sig` = `sig_meta` (serialized)
   * `inputs` = map `rust_name -> prompt_value.to_jinja()`
4. Render:

   * system: template only uses `sig`, no data
   * user: uses `sig + inputs`
5. Pull diagnostics from the shared sink used during prompt value rendering.

### Default templates

Put these as constants in `compiled.rs`:

```jinja
{# user template #}
{% for f in sig.inputs %}
[[ ## {{ f.llm_name }} ## ]]
{{ inputs[f.rust_name] }}

{% endfor %}
```

```jinja
{# system template #}
Your input fields are:
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}

Your output fields are:
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% if f.schema %}{{ "\n" }}{{ f.schema }}{% endif %}
{% endfor %}
```

(These match your “templates become boring” goal.)

---

# 12. Integrate into `ChatAdapter` typed formatting

## Modify: `crates/dspy-rs/src/adapter/chat.rs`

### Replace the current typed input formatting path

Right now, `format_user_message_typed`:

* converts input to `BamlValue`
* formats each field using `format_baml_value_for_prompt_typed(value, input_output_format, field_spec.format)`

This is exactly what you want to delete.

Instead:

* Create a `CompiledSignature::<S>` (either cached or built on demand)
* Call `compiled.render_messages(input)`
* Use `RenderedMessages.system` and `RenderedMessages.user`

Concretely:

* keep `ChatAdapter::format_system_message_typed::<S>() -> Result<String>`
* but rewrite it to call `S::compile()` and return `rendered.system`

Same for `format_user_message_typed`.

### Caching (optional but clean)

You already have `schema_fingerprint` in baml-bridge. Use it as cache key:

* fingerprint includes schema render + target type string
* extend it to include renderer templates maybe (or treat renderers as part of type identity)

Store compiled signatures in your existing `Cache` (`crate::utils::cache`).

---

# 13. Convert REPLHistory renderer to the new system (remove special casing)

## Modify: `crates/dspy-rs/src/rlm/history.rs`

Right now:

* `impl DefaultJinjaRender for REPLHistory { DEFAULT_TEMPLATE = ... }`
* plus a custom `render()` method that builds env and adds filters.

With the new system:

* Move the template to a type-level renderer attribute:

```rust
#[derive(Debug, Clone, Default, BamlType)]
#[prompt(renderer(default = r#"...jinja..."#))]
pub struct REPLHistory { ... }
```

* Delete the special `render()` method (or keep it as a thin wrapper around `PromptWorld` if you want ergonomics).
* Ensure your PromptWorld env includes the filters used by that template (format_count, slice_chars, truncate, etc).

This makes it the poster child for “no name-based hacks”.

---

# 14. Summary of the exact file touch list

## Add (new files)

* `crates/baml-bridge/src/prompt/mod.rs`
* `crates/baml-bridge/src/prompt/world.rs`
* `crates/baml-bridge/src/prompt/value.rs`
* `crates/baml-bridge/src/prompt/renderer.rs`
* `crates/baml-bridge/src/prompt/jinja.rs`
* `crates/dspy-rs/src/signature/compiled.rs` (location can vary)

## Modify (core plumbing)

* `crates/baml-bridge/src/lib.rs` (export prompt module)
* `crates/baml-bridge/src/registry.rs` (store renderers + build_with_renderers)
* `crates/baml-bridge-derive/src/lib.rs` (parse `#[prompt(...)]`, register renderers, validate templates)
* `crates/dspy-rs/src/core/signature.rs` (extend FieldSpec with render overrides/settings)
* `crates/dsrs-macros/src/lib.rs` (parse field-level prompt attrs, emit into FieldSpec)
* `crates/dspy-rs/src/adapter/chat.rs` (use CompiledSignature rendering instead of format_baml_value_for_prompt_typed)
* `crates/dspy-rs/src/rlm/history.rs` (convert to type-level renderer attribute; remove ad-hoc env hacks)
* (Optional) `crates/baml-bridge/src/render_trait.rs` (deprecate DefaultJinjaRender or adapt)

---

# 15. The few “don’t dodge these” decisions

These are the knobs you should decide early because they shape APIs:

1. **Where PromptWorld lives**: baml-bridge (recommended) vs dspy-rs.
2. **Template compilation strategy**: store templates in env by name vs `render_str` on demand.
3. **Does Jinja traversal respect budgets?**

   * If yes, custom templates are safer but sometimes surprising.
   * If no, default structural renderer is safe but custom templates can explode.
4. **How to expose raw values**:

   * I recommend: `value.raw` in Jinja to bypass renderer/budgets when necessary.
5. **Ambiguous union traversal behavior**:

   * I recommend: “no field traversal unless resolved”, and print union summary + safe json render.

---

If you want the next step to be even more concrete, I can write the “method-by-method behavior contract” for `JinjaPromptValue` (for each `BamlValue`/`TypeIR` combination), including how `repr`, `get_value`, `enumerate`, and `.render(style)` behave, in a way that you can translate almost 1:1 into the Rust implementations.


===

Got it. Let’s lock this all the way in: no “maybe”, no “later”, no “it depends”. Below is the **finalized spec + implementation contract** for `JinjaPromptValue` and the surrounding rendering pipeline, written so you can translate it straight into Rust with minimal second-guessing.

I’m going to do two things:

1. **Resolve ambiguities** by making explicit choices (and explaining why those are the best default choices).
2. Provide the **method-by-method behavior contract** for `JinjaPromptValue` (plus the exact renderer pipeline hooks it relies on).

This assumes the architectural shape from the previous plan: prompt system lives in `baml-bridge`, and `dspy-rs` consumes it via `CompiledSignature`.

---

# Locked decisions and invariants

These are the “I refuse to re-litigate this later” calls.

## Prompt system location

**Decision:** implement `PromptWorld`, `PromptValue`, renderer pipeline, and Jinja wrapper in `crates/baml-bridge/src/prompt/*`.

Why: `baml-bridge` already owns the type universe (`OutputFormatContent`), value model (`BamlValue`), and has Jinja/format tooling. Keeping prompt rendering there makes it reusable and avoids DSPy coupling.

## Jinja environment strategy

**Decision:** `PromptWorld` stores a `minijinja::Environment<'static>` configured for prompts. It may register templates that are `'static` (type and field renderer templates from attributes). It should render non-static message templates via `env.render_str(...)`.

Why: avoids lifetime pain while still allowing “compiled templates by name” for renderer templates.

## How strict mode actually works

**Decision:** Strict vs permissive is enforced by the **environment formatter**, not `Object::render`.

Reason: `minijinja::value::Object::render` cannot return rich errors (it’s `fmt::Result`). The environment formatter can return `minijinja::Error`, so it’s the right choke point to make strict mode truly fail prompt rendering.

Implementation implication: `PromptWorld` must install a formatter that:

* Detects `JinjaPromptValue` values and routes them through the renderer pipeline.
* In strict mode, bubbles errors as `minijinja::Error`.
* In permissive mode, records diagnostics and prints fallback text.

## Budgets apply to Jinja iteration semantics

**Decision:** budgets are enforced not only in rendering output text, but also in:

* `enumerator_len()` / `enumerate()` for lists and maps
* `get_value(index)` for lists (indices beyond cap are treated as missing)

This is to prevent “accidental JSON landfill via `{% for x in huge_list %}`”.

Escape hatch: `.raw` bypasses these caps.

## Reserved keys

**Decision:** reserve these keys on `JinjaPromptValue`:

* `render` (method)
* `raw` (untyped value)
* `__type__` (debug string)
* `__path__` (debug string)
* `__full_len__` (actual container size without caps)

If a class has a field literally named `render` or `raw`, you must access it via `.raw["render"]` or `.raw["raw"]`.

This is explicit and documented. It is rare and acceptable.

## Class field access supports both real and rendered names

**Decision:** `value.foo` will resolve `foo` against:

1. real field name (`Name.real_name()`)
2. rendered field alias (`Name.rendered_name()`)

This mirrors the existing conversion helper pattern (`get_field(name, alias)`) and makes templates more robust.

## Union resolution determinism

**Decision:** union resolution is memoized per `PromptValue` instance. A union may resolve to a single branch or remain ambiguous. Ambiguous unions:

* Render as “one of: …” + safe fallback.
* Do not allow typed `.field` access. Accessing `.field` yields `Undefined` and emits a diagnostic.

No “resolve differently at different call sites”.

## Schema context bug is fixed by construction

**Decision:** all format styles (`json|yaml|toon`) must use an `OutputFormatContent` view whose `target = PromptValue.ty`.

No exceptions. No “parent input format”.

---

# Data structures you should implement

These are the minimal structs you need to make the contract real.

## PromptWorld

```rust
pub struct PromptWorld {
    pub types: TypeDb,
    pub renderers: RendererDb,
    pub env: minijinja::Environment<'static>,
    pub settings: RenderSettings,
    pub union_resolver: UnionResolver,
}
```

`TypeDb` is essentially the arcs from `OutputFormatContent` without a meaningful `target`.

## RenderSettings

```rust
pub struct RenderSettings {
    pub max_total_chars: usize,
    pub max_string_chars: usize,
    pub max_list_items: usize,
    pub max_map_entries: usize,
    pub max_depth: usize,
    pub max_union_branches_shown: usize,
    pub failure_mode: RenderFailureMode,
}
pub enum RenderFailureMode { Strict, Permissive }
```

## Render session and diagnostics

You want one sink per “render messages” call.

```rust
pub struct RenderSession {
    diagnostics: std::sync::Mutex<Vec<RenderDiagnostic>>,
}

pub struct RenderDiagnostic {
    pub path: String,
    pub ty: String,
    pub style: String,
    pub renderer: String,
    pub error: String,
    pub fell_back_to: String,
}
```

You should pass `Arc<RenderSession>` through every PromptValue so all nested render calls accumulate into one list.

## PromptValue

```rust
#[derive(Clone)]
pub struct PromptValue {
    pub value: BamlValue,
    pub ty: TypeIR,
    pub world: std::sync::Arc<PromptWorld>,
    pub session: std::sync::Arc<RenderSession>,
    pub override_renderer: Option<RendererOverride>, // per-field compiled override
    pub path: PromptPath,
}
```

`PromptPath` builds human strings like `inputs.history.entries[3].output`.

---

# Renderer pipeline contract

`PromptWorld` must provide one primary entrypoint:

```rust
impl PromptWorld {
    pub fn render_prompt_value(
        &self,
        pv: &PromptValue,
        style: Option<&str>, // call-site override
    ) -> Result<String, RenderError>;
}
```

### Renderer precedence order

When rendering `pv`:

1. Call-site style override (if `style` passed)
2. Per-field override renderer/style (if exists on `pv.override_renderer`)
3. Type-level renderer for concrete type key + style
4. Built-in style handlers (`json|yaml|toon`)
5. Structural fallback renderer
6. Last-ditch fallback: bounded JSON-ish stringification

### Failure handling

* In `Strict`: any failure in (1)-(4) aborts the render and returns `Err`.
* In `Permissive`: failures push a `RenderDiagnostic` and continue to next fallback step.

### Budget enforcement

* Structural renderer must respect per-structure budgets (`max_*`) and depth.
* After any renderer produces a string, enforce `max_total_chars`:

  * If exceeded:

    * For custom/type renderers: truncate and add a suffix `"... (truncated)"`.
    * For `json|yaml|toon`: do not emit half-structured output. Instead, treat overflow as a failure and fall back to structural. Emit a diagnostic with `fell_back_to = "structural"`.

That last rule makes `render("json")` always produce something meaningful even under budgets.

---

# JinjaPromptValue contract

This is the piece you asked for: method-by-method behavior, with exact semantics.

You will implement a custom minijinja object:

```rust
pub struct JinjaPromptValue {
    pv: PromptValue,
}
```

Create it via:

```rust
impl PromptValue {
    pub fn as_jinja_value(&self) -> minijinja::Value {
        minijinja::Value::from_object(JinjaPromptValue { pv: self.clone() })
    }
}
```

## Environment formatter contract

Your prompt Jinja environment MUST install a formatter like:

* If value is `None` (minijinja “none”): print `"null"` (keep existing behavior).
* Else if value is a `JinjaPromptValue`: call `pv.world.render_prompt_value(&pv, None)`:

  * If success: write string
  * If error:

    * strict mode: return minijinja Error
    * permissive: record diagnostic and write fallback like `"[render error]"` or better: render structural fallback immediately
* Else: default formatting (likely `minijinja::escape_formatter`)

This is the strict/permissive enforcement point.

### Downcasting

You need to detect your object in the formatter. Your formatter can do:

* if `value.as_object()` and `downcast_ref::<JinjaPromptValue>()` succeeds, handle it.

This works because `Object` is dyn and supports downcasting.

## Object repr

### `repr() -> ObjectRepr`

Return based on **effective shape**:

1. If `pv.ty` is `TypeIR::Class` or `pv.value` is `BamlValue::Class`: `ObjectRepr::Map`
2. Else if `pv.ty` is `TypeIR::Map` or `pv.value` is `BamlValue::Map`: `ObjectRepr::Map`
3. Else if `pv.ty` is `TypeIR::List` or `pv.value` is `BamlValue::List`: `ObjectRepr::Seq`
4. Else: `ObjectRepr::Plain`

This “type-first, value-fallback” rule avoids weirdness when types mismatch.

## get_value

### `get_value(key: &Value) -> Option<Value>`

There are three key cases.

### Case A: string keys

Let `k = key.as_str()?`.

Reserved keys:

* `k == "render"`: return a callable method object (see below)
* `k == "raw"`: return raw untyped view (see below)
* `k == "__type__"`: return `Value::from(pv.ty.diagnostic_repr().to_string())`
* `k == "__path__"`: return `Value::from(pv.path.to_string())`
* `k == "__full_len__"`: return actual container length if container else `0`

If not reserved, resolve based on effective shape:

#### If effective shape is class-like

This means:

* `pv.ty` is `TypeIR::Class { name, mode, .. }` (preferred) OR
* `pv.value` is `BamlValue::Class(class_name, fields)` (fallback)

Resolution steps:

1. Determine `class_key`:

   * If `pv.ty` is Class: use its (name, mode)
   * Else if value is Class: use (class_name, NonStreaming) first, and fallback to Streaming if not found
2. Look up `Class` in `pv.world.types`.
3. Resolve `k` to a declared field:

   * First match `Name.real_name() == k`
   * Else match `Name.rendered_name() == k`
   * If no match: return `None` (Undefined)
4. Let `real_name = matched_field_name.real_name()`.
5. Fetch the child value:

   * If `pv.value` is Class: `fields.get(real_name)` (and optionally also try alias key just in case parsed values come in with aliases, but prefer real)
   * If missing: return `None`
6. Determine child type:

   * `field_type` from the class definition (the `TypeIR` stored in `Class.fields`)
7. Return `Some(child_pv.as_jinja_value())` where:

   * child `PromptValue.value = child_value.clone()`
   * child `PromptValue.ty = field_type.clone()`
   * child inherits `world`, `session`
   * child inherits `override_renderer = None` (unless you later want nested per-field overrides inside classes, which is optional)
   * child path appends `.real_name` (not alias)

Diagnostics:

* If class not found or field type not found but value exists: in permissive mode, emit a diagnostic about “type mismatch / missing schema” and return a child with `ty = TypeIR::Top` or treat as untyped. Since `TypeIR::Top` panics in schema rendering elsewhere, prefer:

  * `ty = inferred_ty_from_value(child_value)` (see fallback typing rules below)

#### If effective shape is map-like

This means:

* `pv.ty` is `TypeIR::Map(_, value_ty, ..)` OR
* `pv.value` is `BamlValue::Map(map)`

Resolution:

1. Fetch map value by string key `k` (exact match)
2. Determine child type:

   * If `pv.ty` is Map: `value_ty.clone()`
   * Else: fallback type inference from value
3. Return child PromptValue with path `["k"]` or `.k`?
   Use bracket form: `["key"]` to avoid confusion with field access:

   * path appends `["{k}"]`

Budgets:

* If `k` exists but map entry is beyond `max_map_entries` when iterating, direct key access still works.
* If you want strict “no bypass”, enforce cap by refusing keys not in the first N keys. I recommend **do not** do that. It’s surprising and it’s not how caps are usually expected to work.

### Case B: integer keys for sequences

Let `idx = key.as_usize()?`.

Applicable if effective shape is list-like.

Rules:

1. If `pv.value` is `BamlValue::List(items)`:

   * If `idx >= items.len()`: return `None`
   * If `idx >= world.settings.max_list_items`: return `None` (cap enforced)
   * child_value = items[idx].clone()
2. Determine child type:

   * If `pv.ty` is `TypeIR::List(inner, ..)`: `*inner.clone()`
   * Else: fallback inference from child_value
3. Return child PromptValue with path appending `[idx]`

This makes loops safe and also prevents accidental indexing beyond cap.

Escape hatch: `pv.raw[idx]` should allow it, because `.raw` bypasses caps.

### Case C: anything else

Return `None`.

## enumerate and enumerator_len

These are what make `for` loops and `|length` behave.

### `enumerate() -> Enumerator`

#### If list-like

Let `n = min(actual_len, settings.max_list_items)`.

Return `Enumerator::Seq(n)`.

#### If map-like

Return `Enumerator::Values(keys)` where `keys` is a `Vec<Value>`:

* Determine key list:

  * If class-like: keys are declared fields in schema order (real names), capped to `max_map_entries`.
  * If map-like: keys are map keys sorted lexicographically, capped to `max_map_entries`.
* Convert each key to `Value::from(key_str)`.

Sorting map keys gives determinism even if insertion order differs between construction paths.

### `enumerator_len() -> Option<usize>`

Return `Some(n)` corresponding exactly to the enumerate cap.

This implies `value|length` is capped length, not true length.

To allow users to get the true length, implement `__full_len__` and document:

* `value.__full_len__` gives actual count
* `value|length` gives capped iteration length

## is_true

Truthiness is used by `{% if value %}`.

Rules:

* Null -> false
* Bool -> itself
* String -> `!is_empty()`
* Int -> `!= 0`
* Float -> `!= 0.0` (treat NaN as true or false? Pick one. I recommend true to avoid surprises)
* List/Map/Class -> `len > 0`
* Enum -> true if variant string non-empty
* Media -> true

If type and value mismatch, value wins.

## raw

`value.raw` returns an untyped Jinja value representation of the underlying `BamlValue`, bypassing typed traversal and budgets.

Contract for `.raw`:

* `.raw` is a standard minijinja `Value` that behaves like plain serialized data.
* `.raw` does not carry types or renderer overrides.
* `.raw` for `BamlValue::Null` must be `Value::from(())` so `{% if raw is none %}` works.
* `.raw` should be built via a dedicated conversion function, not serde serialize, so you can preserve class/map/list shapes.

Minimal conversion is fine:

* String -> Value::from(String)
* Int/Float/Bool -> Value::from(...)
* Null -> Value::from(())
* List -> Value::from_iter(...)
* Map/Class -> Value::from_iter((k, v)) using string keys
* Enum -> Value::from(variant_string) (real name)
* Media -> Value::from(serde_json_string or map)

## render method object

`value.render("json")` is implemented by returning a callable object from `get_value("render")`.

### Call signature

Support:

* `render(style: string) -> string`

Optionally later:

* `render(style: string, *, max_total_chars: int, ...)` (keyword args)
  Not necessary now. Lock the simple one.

### Behavior

When called:

1. Parse style string. If missing or not string, return minijinja error.
2. Call `pv.world.render_prompt_value(&pv, Some(style))`.
3. If strict mode and render fails: return minijinja error.
4. If permissive mode and render fails: record diagnostic and return fallback text.

`render(style)` is **call-site override** and must take precedence over field/type renderers.

---

# Structural fallback renderer contract

This is not JinjaPromptValue per se, but it’s what makes default `{{ value }}` safe. This must be deterministic and typed.

Entry:

```rust
fn render_structural(pv: &PromptValue, depth: usize) -> String
```

Rules by type:

## Primitive types

* string: truncate to `max_string_chars`, add suffix `"... (truncated)"` if truncated
* int/float/bool: format normally
* null: `"null"`

## Enum

Render the rendered alias if possible:

* If `pv.ty` is Enum and the world has enum def with alias: show alias name
* Else show stored variant string

## List

* If depth >= max_depth: return `"List { ... }"` (or `"<depth limit>"`)
* Show first `max_list_items` items:

  * Each item is rendered via `render_structural(child, depth+1)`
* If more: append line like `"... (+N more)"`

## Map

* If depth >= max_depth: `"Map { ... }"`
* Show first `max_map_entries` entries, keys sorted:

  * `key: value`
* If more: `"... (+N more keys)"`

## Class

* If depth >= max_depth: `"{TypeName} { ... }"`
* Use schema field order:

  * For each field in class schema:

    * If present, render `field: <child structural>`
    * If absent, skip (or show `field: null`? Decide now)

      * **Decision:** skip absent fields. Prompts prefer signal over noise.
* If more than `max_map_entries` fields, cap similarly with “...”.

## Union

If union is resolved: treat as that resolved type and render accordingly.

If ambiguous:

* Emit header: `"one of: A | B | C"` showing up to `max_union_branches_shown`
* Then render the raw value in a safe way:

  * **Decision:** use `render_structural` on the underlying value inferred type, not JSON, to keep bounded and meaningful.

This avoids “ambiguous union prints 10KB JSON”.

---

# Fallback typing rules when schema is missing

This removes a common “oops, type db missing” footgun.

When you have a `BamlValue` but cannot reliably compute a child `TypeIR`:

* Infer a minimal TypeIR from the value itself:

  * String -> TypeIR::string()
  * Int -> TypeIR::int()
  * Float -> TypeIR::float()
  * Bool -> TypeIR::bool()
  * Null -> TypeIR::primitive(null)
  * List -> TypeIR::list(TypeIR::Top?) but Top is dangerous; instead:

    * If non-empty: infer from first element, union-including-null across sample? Keep simple: first element type
    * If empty: list[string] or list[ANY]? You don’t have ANY safely. Use list[string] and treat as untyped.
  * Map/Class -> map<string, string> or class name if present. For Class with unknown schema: treat as map<string, inferred>.
* Emit a diagnostic: renderer `"structural"`, error `"missing schema for typed traversal"`, fell_back_to `"value_inference"`.

This makes the system resilient even if registry misses a type.

---

# Concrete glue between PromptValue and Jinja templates

## Renderer template context

Type-level and field-level renderer templates should see:

* `value`: the typed `PromptValue` (as JinjaPromptValue)
* `ctx`: a serializable struct with settings you expect templates to use

Example ctx fields to expose:

```rust
#[derive(Serialize)]
pub struct JinjaRenderCtx {
    pub max_total_chars: usize,
    pub max_string_chars: usize,
    pub max_list_items: usize,
    pub max_map_entries: usize,
    pub max_depth: usize,
}
```

If you want RLM-specific stuff like `max_output_chars`, pass it via a separate “user ctx” map that `CompiledSignature` can merge in, but do not couple PromptWorld to RLM.

## Filters you must ship

In PromptWorld env, add:

* `truncate(s, n)` -> string truncation
* `slice_chars(s, n)` -> first n chars
* `format_count(n)` -> 1,234 formatting
* keep existing: `regex_match`, `sum`

These match patterns already used in `REPLHistory` templates.

---

# The best way to implement this in code with minimal pain

Here’s the “do it in this order so you don’t cry” approach.

## Step 1: implement PromptValue and PromptPath

* PromptPath string formatting is straightforward
* Child constructors should be pure and cheap

## Step 2: implement JinjaPromptValue get_value and enumerate

Do not implement rendering yet. First prove traversal works:

* class field lookup
* list indexing and cap
* map key lookup
* reserved keys

## Step 3: implement PromptWorld env and formatter

* Start with `jsonish::jinja_helpers::get_env()`
* Override formatter with your logic for `JinjaPromptValue`
* Confirm strict mode errors out cleanly

## Step 4: implement structural renderer and wire it as fallback

At this point `{{ value }}` prints something reasonable, with caps.

## Step 5: implement renderer pipeline resolution

* Field override support
* Type-level renderer support
* Built-in format styles with correct schema target
* Diagnostics capture in permissive mode

## Step 6: migrate one real type renderer

Move `REPLHistory` to type-level renderer attribute and delete the ad-hoc rendering code.

That is your end-to-end proof.

---

# What you will end up with in practice

After this is implemented, these are the invariant behaviors you’ll be able to rely on:

* `{{ inputs.history }}` always prints a safe, bounded representation with type-aware traversal.
* `{{ inputs.history.render("json") }}` uses `inputs.history.ty` as schema target, never the input struct format.
* Broken type renderers won’t silently fail in permissive mode: you’ll get diagnostics.
* Broken type renderers will hard fail in strict mode: prompt rendering errors.
* Template authors can loop over lists and maps without accidentally iterating millions of items.
* If someone *really* wants raw behavior, `.raw` exists and is explicit.

---

If you want, I can also write the exact pseudocode for each of these methods (`get_value`, `enumerate`, union resolver scoring) in Rust-like form that you can copy into the actual files, including the downcast-based formatter closure.


===

TODO/FIXME:
- no lenient mode, wtf thats stupid

