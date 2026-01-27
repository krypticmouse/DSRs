# Typed RLM with DSRs Integration - Implementation Plan

> **Note:** Ignore the sub-skill requirement; proceed without it.

**Goal:** Build a Rust-native Recursive Language Model (RLM) system that surpasses DSPy's Python implementation by leveraging DSRs' typed signature system, with first-class PyO3 integration for rich Python REPL ergonomics.

**Architecture:**
- **All types must be Rust + RlmType** - no native Python types as signature inputs
- `rlm-derive` + `rlm-core` live in DSRs and are re-exported by `dspy-rs`
- PyO3 conversion utilities live in DSRs (feature-gated in `baml-bridge` + `dspy-rs`), not in rig-rlm
- `#[rlm_type]` attribute macro **adds** `#[pyclass]` automatically and injects `#[derive(BamlType, RlmType)]` (optional convenience)
- Users may instead write `#[pyclass] #[derive(BamlType, RlmType)]` directly (preferred for minimal annotation)
- **Note:** `#[derive(RlmType)]` cannot inject `#[pyclass]`; it requires `#[pyclass]` to already be present on the struct.
- Standard DSRs `#[derive(Signature)]` defines both input AND output fields
- `dsrs_macros` implements `RlmInputFields` on the generated `<Sig>Input` struct **when `dspy-rs` feature `rlm` is enabled** (module‑level opt‑in via `TypedRlm`)
- `TypedRlm::call(input: S::Input)` takes the signature's input struct directly - no builder pattern for context; returns `CallResult<S>` with optional `RlmMeta`
- Input fields are converted to Python variables via `RlmInputFields` + `IntoPyObject`
- `SUBMIT(**kwargs)` converts Python values to `BamlValue` via schema-guided conversion (jsonish only for string coercion), then validates constraints; orjson/json fallback for exotic objects
- Variable descriptions auto-generated from `RlmDescribe` implementations on input types

**Tech Stack:**
- Rust (edition 2024), PyO3 0.27+, rig-core, DSRs (dspy-rs, baml-bridge, baml-bridge-derive)
- proc-macro2, syn, quote, darling for proc macros
- tokio for async runtime

**Explicit UX constraints/goals from discussion:**
- **DSPy parity:** Match DSPy RLM system prompt wording/structure, code‑fence conventions, SUBMIT guidance, and final display format.
- **Module‑level opt‑in (DSPy‑like):** Users opt into RLM by **choosing the module** (`TypedRlm::<Sig>`) not by annotating the Signature. Signatures remain unchanged.
- **Derive‑first ergonomics:** Users can simply write `#[pyclass] #[derive(BamlType, RlmType)]` on their types (same line). `#[rlm_type]` remains a convenience macro, not a requirement.
- **Single entrypoint:** `TypedRlm::call(...)` returns the full `RlmResult` (meta included); no separate `call_with_meta`.
- **No extra boilerplate:** Users should not add `#[pyclass]` or other PyO3 boilerplate; `#[rlm_type]` handles it.
- **Default SUBMIT usage:** Prefer `SUBMIT(field=variable)`; prompt should emphasize “minimize retyping” like DSPy.
- **Arbitrary Python outputs:** Model may return arbitrary Python objects in SUBMIT; we best‑effort normalize with clear errors.
- **Our types always convertible:** Any `#[rlm_type]` value is convertible via `__baml__` and should “just work.”
- **DSRs‑first integration:** PyO3 conversion lives in DSRs (feature‑gated in baml‑bridge/dspy‑rs), not in rig‑rlm.

---

## User UX: Predict vs TypedRlm (DSPy‑like)

**Current DSRs UX (example 17 style):**
- Define `BamlType` structs + `#[derive(Signature)]`
- Use `Predict::<Sig>` for a single LM call and typed parsing

```rust
#[derive(Signature)]
struct InsuranceClaimInfo {
  #[input] claim_text: String,
  #[output] claim: InsuranceClaim,
}

let predict = Predict::<InsuranceClaimInfo>::new();
let output = predict.call(InsuranceClaimInfoInput { claim_text }).await?;
```

**RLM UX (opt‑in by module choice, same signature):**
- Same signature + types
- Use `TypedRlm::<Sig>` with a **`call(...)`** method (matches `Predict`)
- `call(...)` runs the DSPy‑style REPL loop + `SUBMIT(...)`

```rust
let rlm = TypedRlm::<InsuranceClaimInfo>::with_agent(agent);
let result = rlm.call(InsuranceClaimInfoInput { claim_text }).await?;
```

**User‑visible difference:**
- `Predict` = one‑shot LM call + parse.
- `TypedRlm` = iterative REPL + tool calls + `SUBMIT` + trajectory metadata.

---

## Table of Contents

1. [Phase 1: Rlm Core + Derive (DSRs)](#phase-1-rlm-core--derive-dsrs) - `#[rlm_type]` + `#[derive(RlmType)]`
2. [Phase 2: Type Description + Signature Integration (DSRs)](#phase-2-type-description--signature-integration-dsrs) - `RlmDescribe` + `RlmInputFields`
3. [Phase 3: SUBMIT Function](#phase-3-submit-function) - PyO3 validation bridge
4. [Phase 4: TypedRlm Core](#phase-4-typedrlm-core) - The main RLM orchestrator
5. [Phase 5: Prompt Generation](#phase-5-prompt-generation) - Rich variable descriptions for LLM
6. [Phase 6: Integration & Examples](#phase-6-integration--examples) - Wire everything together
7. [Phase 7: Testing & Verification](#phase-7-testing--verification) - End-to-end tests

---

## Phase 1: Rlm Core + Derive (DSRs)

### Overview

Create new crates in the DSRs workspace:
- `rlm-core` (shared traits + variable description helpers)
- `rlm-derive` (proc-macros)

`rlm-derive` provides:
- `#[rlm_type]` attribute macro that **adds `#[pyclass]` automatically** and injects `#[derive(BamlType, RlmType)]`
- `#[derive(RlmType)]` derive macro that generates `#[pymethods]` impls with getters and Python ergonomics
- `__repr__` from template or auto-inferred
- `__len__`, `__iter__`, `__getitem__` for collections
- Filter properties (e.g., `user_steps` from `steps` where `source == "user"`)
- Flatten properties (e.g., `all_tool_calls` flattened from nested `tool_calls`)
- Computed properties: optional container-level `#[rlm(property(...))]` metadata (methods still implemented by user)
- `BamlType` derived via `baml-bridge-derive` (injected by `#[rlm_type]`)
- `__baml__()` method on every `#[rlm_type]` to expose a stable JSON-like representation (used by SUBMIT conversion)
- **Python‑side docs/metadata:** generate class‑level docs + per‑field docs (where supported) and a `__rlm_schema__` map for REPL discovery
- `RlmDescribe` trait for prompt description generation

**`__baml__` details (important for SUBMIT):**
- `__baml__(self)` is generated on every `#[rlm_type]` under the `rlm` feature.
- It **does not** serialize to JSON; it returns a **JSON-like Python object** (dict/list/primitive).
- Implementation uses `BamlType`/`ToBamlValue` → `BamlValue` → `baml_bridge::py::baml_value_to_py`.
- Shape guarantees:
  - Struct/class → Python `dict` of fields
  - Enum → Python `str` (variant name)
  - Option/Null → `None`
  - Vec → `list`
  - Primitives → `bool/int/float/str`
- This makes our types **always convertible** for SUBMIT, even if the model returns the object directly.
- If a type is not `BamlType` or if conversion fails, SUBMIT emits a clear error (no silent fallback).

**Python docstrings / field metadata (for better REPL UX):**
- Class `__doc__` should summarize the type, fields, and usage hints (len/indexable/properties).
- Each field should carry its `desc`/constraints where possible (e.g., property docstrings).
- Always expose a machine‑readable `__rlm_schema__` dict so `dir()`/`help()`/`print(obj.__rlm_schema__)` is informative even if per‑field docstrings are limited by PyO3.

### Task 1.1: Create rlm-derive crate structure (DSRs)

**Files (DSRs repo):**
- Create: `crates/rlm-derive/Cargo.toml`
- Create: `crates/rlm-derive/src/lib.rs`
- Create: `crates/rlm-derive/src/rlm_attr.rs`
- Modify: `/Users/darin/src/personal/DSRs/Cargo.toml` (workspace)

**Step 1: Create the crate directory**

```bash
mkdir -p /Users/darin/src/personal/DSRs/crates/rlm-derive/src
```

**Step 2: Write Cargo.toml for rlm-derive**

```toml
# /Users/darin/src/personal/DSRs/crates/rlm-derive/Cargo.toml
[package]
name = "rlm-derive"
version = "0.1.0"
edition = "2024"

[lib]
proc-macro = true

[dependencies]
proc-macro2 = "1"
quote = "1"
syn = { version = "2", features = ["full", "parsing", "extra-traits"] }
darling = "0.20"  # For attribute parsing
```

**Step 3: Write initial lib.rs skeleton**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-derive/src/lib.rs
use proc_macro::TokenStream;

mod attrs;
mod generators;
mod rlm_attr;
mod rlm_type;

/// Attribute macro for ergonomic usage.
///
/// Expands to:
/// - `#[pyo3::pyclass]` on the struct (with optional name override)
/// - `#[derive(baml_bridge::BamlType, RlmType)]`
/// - preserves existing derives/attrs
#[proc_macro_attribute]
pub fn rlm_type(attr: TokenStream, item: TokenStream) -> TokenStream {
    rlm_attr::expand(attr, item)
}

/// Derive macro for RLM-compatible types (used by `#[rlm_type]`).
#[proc_macro_derive(RlmType, attributes(rlm))]
pub fn derive_rlm_type(input: TokenStream) -> TokenStream {
    rlm_type::derive(input)
}
```

**Step 4: Add rlm_attr.rs for struct rewriting**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-derive/src/rlm_attr.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Attribute, ItemStruct};

/// Parse #[rlm(...)] on the struct to pull out pyclass_name (if any).
fn extract_pyclass_name(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("rlm") {
            continue;
        }
        let mut found: Option<String> = None;
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("pyclass_name") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                found = Some(lit.value());
            }
            Ok(())
        });
        if found.is_some() {
            return found;
        }
    }
    None
}

pub fn expand(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let mut attrs = input.attrs.clone();

    let pyclass_name = extract_pyclass_name(&attrs);
    if let Some(name) = pyclass_name {
        let lit = syn::LitStr::new(&name, proc_macro2::Span::call_site());
        attrs.push(syn::parse_quote!(#[pyo3::pyclass(name = #lit)]));
    } else {
        attrs.push(syn::parse_quote!(#[pyo3::pyclass]));
    }

    // Inject BamlType + RlmType derives (merge with existing derive list to avoid duplicates)
    let mut merged = false;
    for attr in attrs.iter_mut() {
        if attr.path().is_ident("derive") {
            let mut paths: syn::punctuated::Punctuated<syn::Path, syn::token::Comma> =
                attr.parse_args_with(syn::punctuated::Punctuated::parse_terminated)
                    .unwrap_or_default();
            let to_add: [syn::Path; 2] = [
                syn::parse_quote!(baml_bridge::BamlType),
                syn::parse_quote!(RlmType),
            ];
            for p in to_add {
                if !paths.iter().any(|existing| existing == &p) {
                    paths.push(p);
                }
            }
            *attr = syn::parse_quote!(#[derive(#paths)]);
            merged = true;
            break;
        }
    }
    if !merged {
        attrs.push(syn::parse_quote!(#[derive(baml_bridge::BamlType, RlmType)]));
    }

    let ident = &input.ident;
    let generics = &input.generics;
    let vis = &input.vis;
    let fields = &input.fields;

    let expanded = quote! {
        #(#attrs)*
        #vis struct #ident #generics #fields
    };
    expanded.into()
}
```

**Step 4: Add to DSRs workspace Cargo.toml**

Add these entries to the existing `[workspace].members` in `/Users/darin/src/personal/DSRs/Cargo.toml`:

```toml
[workspace]
members = [
  "crates/rlm-derive",
  "crates/rlm-core"
]
```

**Step 5: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/ /Users/darin/src/personal/DSRs/Cargo.toml
git commit -m "feat: scaffold rlm-derive crate"
```

---

### Task 1.2: Implement attribute parsing with darling

**Files:**
- Create: `crates/rlm-derive/src/attrs.rs`

**Step 1: Write the attribute structs**

```rust
// crates/rlm-derive/src/attrs.rs
use darling::{FromDeriveInput, FromField, FromMeta};
use syn::{Ident, Type, Visibility};

/// Container-level attributes: #[rlm(...)] on the struct
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(rlm), supports(struct_named))]
pub struct RlmTypeAttrs {
    pub ident: Ident,
    pub vis: Visibility,
    pub data: darling::ast::Data<(), RlmFieldAttrs>,

    /// Custom __repr__ template. Uses {self.field} and {len(self.field)} interpolation.
    /// Example: "Trajectory({len(self.steps)} steps, session={self.session_id:12}...)"
    #[darling(default)]
    pub repr: Option<String>,

    /// Field name to iterate over for __iter__ and __len__
    #[darling(default)]
    pub iter: Option<String>,

    /// Field name to index into for __getitem__
    #[darling(default)]
    pub index: Option<String>,

    /// Python class name override (defaults to Rust struct name)
    #[darling(default)]
    pub pyclass_name: Option<String>,

    /// Computed properties metadata (methods implemented manually in #[pymethods])
    /// Usage: #[rlm(property(name = "foo", desc = "Human description"))]
    #[darling(default, multiple, rename = "property")]
    pub properties: Vec<RlmPropertyAttrs>,
}

/// Field-level attributes: #[rlm(...)] on struct fields
#[derive(Debug, FromField)]
#[darling(attributes(rlm))]
pub struct RlmFieldAttrs {
    pub ident: Option<Ident>,
    pub ty: Type,
    pub vis: Visibility,

    /// Human-readable description for prompt generation
    #[darling(default)]
    pub desc: Option<String>,

    /// Generate a filter property on the parent struct.
    /// Example: filter_property = "user_steps" with filter_value = "user"
    /// generates: fn user_steps(&self) -> Vec<T> { self.field.iter().filter(...) }
    #[darling(default)]
    pub filter_property: Option<String>,

    /// The value to filter by (used with filter_property)
    #[darling(default)]
    pub filter_value: Option<String>,

    /// The field on the child to filter by (defaults to "source")
    #[darling(default)]
    pub filter_field: Option<String>,

    /// Flatten this nested collection into a parent property.
    /// Example: flatten_property = "all_tool_calls" on tool_calls: Option<Vec<ToolCall>>
    /// generates: fn all_tool_calls(&self) -> Vec<ToolCall> { self.items.iter().flat_map(...) }
    #[darling(default)]
    pub flatten_property: Option<String>,

    /// Explicit parent field for flatten_property (required when ambiguous)
    /// Example: flatten_parent = "steps"
    #[darling(default)]
    pub flatten_parent: Option<String>,

    /// Skip this field in Python (don't generate getter)
    #[darling(default)]
    pub skip_python: bool,

    /// Skip this field in BamlType schema
    #[darling(default)]
    pub skip_schema: bool,
}

/// Container-level computed property metadata
#[derive(Debug, FromMeta)]
pub struct RlmPropertyAttrs {
    pub name: String,

    #[darling(default)]
    pub desc: Option<String>,
}

impl RlmTypeAttrs {
    pub fn fields(&self) -> Vec<&RlmFieldAttrs> {
        match &self.data {
            darling::ast::Data::Struct(fields) => fields.iter().collect(),
            _ => vec![],
        }
    }
}
```

**Step 2: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/attrs.rs
git commit -m "feat(rlm-derive): add attribute parsing with darling"
```

---

### Task 1.3: Implement PyO3 code generation

**Files:**
- Create: `crates/rlm-derive/src/generators/mod.rs`
- Create: `crates/rlm-derive/src/generators/pyclass.rs`

**Step 1: Create generators module**

```rust
// crates/rlm-derive/src/generators/mod.rs
pub mod pyclass;
pub mod repr;
pub mod iter;
pub mod properties;
pub mod describe;
pub mod schema;
pub mod schema;
```

**Step 2: Implement pyclass generator**

```rust
// crates/rlm-derive/src/generators/pyclass.rs
use proc_macro2::TokenStream;
use quote::{quote, format_ident};
use crate::attrs::{RlmTypeAttrs, RlmFieldAttrs};

/// Generate the #[pyclass] implementation with getters
pub fn generate_pyclass(attrs: &RlmTypeAttrs) -> TokenStream {
    let struct_name = &attrs.ident;

    let getters = generate_getters(attrs);
    let repr_impl = super::repr::generate_repr(attrs);
    let iter_impl = super::iter::generate_iter(attrs);
    let properties = super::properties::generate_properties(attrs);
    let schema_impl = super::schema::generate_schema_metadata(attrs);

    quote! {
        // NOTE: #[pyclass] is injected by #[rlm_type]
        #[pyo3::pymethods]
        impl #struct_name {
            #getters
            #repr_impl
            #iter_impl
            #properties
            #schema_impl

            /// Convert this value to a JSON-like Python object for SUBMIT normalization
            fn __baml__(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<::pyo3::PyObject> {
                let value = ::baml_bridge::ToBamlValue::to_baml_value(self);
                Ok(::baml_bridge::py::baml_value_to_py(py, &value))
            }
        }
    }
}

fn generate_getters(attrs: &RlmTypeAttrs) -> TokenStream {
    let getters: Vec<TokenStream> = attrs.fields()
        .iter()
        .filter(|f| !f.skip_python)
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            let field_ty = &field.ty;

            // Determine if we need to clone or can return reference
            let (return_ty, body) = if is_copy_type(field_ty) {
                (quote! { #field_ty }, quote! { self.#field_name })
            } else if is_string_type(field_ty) {
                (quote! { &str }, quote! { &self.#field_name })
            } else {
                // Clone for complex types
                (quote! { #field_ty }, quote! { self.#field_name.clone() })
            };

            quote! {
                #[getter]
                fn #field_name(&self) -> #return_ty {
                    #body
                }
            }
        })
        .collect();

    quote! { #(#getters)* }
}

fn is_copy_type(ty: &syn::Type) -> bool {
    let ty_str = quote!(#ty).to_string();
    matches!(ty_str.as_str(),
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" |
        "u8" | "u16" | "u32" | "u64" | "u128" | "usize" |
        "f32" | "f64" | "bool" | "char"
    )
}

fn is_string_type(ty: &syn::Type) -> bool {
    let ty_str = quote!(#ty).to_string();
    ty_str == "String"
}
```

**Step 3: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/
git commit -m "feat(rlm-derive): implement pyclass code generation"
```

---

### Task 1.4: Implement __repr__ generation

**Files:**
- Create: `crates/rlm-derive/src/generators/repr.rs`

**Step 1: Write repr generator with template parsing**

```rust
// crates/rlm-derive/src/generators/repr.rs
use proc_macro2::TokenStream;
use quote::quote;
use crate::attrs::RlmTypeAttrs;

/// Generate __repr__ implementation from template or auto-infer
pub fn generate_repr(attrs: &RlmTypeAttrs) -> TokenStream {
    match &attrs.repr {
        Some(template) => generate_from_template(attrs, template),
        None => generate_auto_repr(attrs),
    }
}

/// Parse template like "Trajectory({len(self.steps)} steps, session={self.session_id:12}...)"
/// Supported interpolations:
/// - {self.field} - field value
/// - {self.field:N} - field value truncated to N chars
/// - {len(self.field)} - length of collection/string
fn generate_from_template(attrs: &RlmTypeAttrs, template: &str) -> TokenStream {
    let struct_name = &attrs.ident;

    // Parse the template into format string + args
    let (format_str, args) = parse_repr_template(template);

    quote! {
        fn __repr__(&self) -> String {
            format!(#format_str, #(#args),*)
        }
    }
}

fn parse_repr_template(template: &str) -> (String, Vec<TokenStream>) {
    let mut format_str = String::new();
    let mut args: Vec<TokenStream> = vec![];
    let mut chars = template.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' {
            // Parse interpolation
            let mut expr = String::new();
            let mut truncate: Option<usize> = None;

            while let Some(&next) = chars.peek() {
                if next == '}' {
                    chars.next();
                    break;
                }
                if next == ':' {
                    chars.next();
                    // Parse truncation length
                    let mut num = String::new();
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            num.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    truncate = num.parse().ok();
                } else {
                    expr.push(next);
                    chars.next();
                }
            }

            // Generate the argument
            let arg = if expr.starts_with("len(") && expr.ends_with(")") {
                // len(self.field)
                let inner = &expr[4..expr.len()-1];
                let field = inner.trim_start_matches("self.");
                let field_ident = syn::Ident::new(field, proc_macro2::Span::call_site());
                quote! { self.#field_ident.len() }
            } else if expr.starts_with("self.") {
                // self.field
                let field = expr.trim_start_matches("self.");
                let field_ident = syn::Ident::new(field, proc_macro2::Span::call_site());

                if let Some(n) = truncate {
                    quote! {
                        self.#field_ident.chars().take(#n).collect::<String>()
                    }
                } else {
                    quote! { self.#field_ident }
                }
            } else {
                // Unknown - just use as-is
                let expr_tokens: TokenStream = expr.parse().unwrap_or_else(|_| quote! { "" });
                expr_tokens
            };

            format_str.push_str("{}");
            args.push(arg);
        } else {
            format_str.push(c);
        }
    }

    (format_str, args)
}

fn generate_auto_repr(attrs: &RlmTypeAttrs) -> TokenStream {
    let struct_name = &attrs.ident;
    let name_str = struct_name.to_string();

    // Find a good preview field (first string field or first field)
    let preview_field = attrs.fields()
        .iter()
        .find(|f| {
            let ty_str = quote!(#f.ty).to_string();
            ty_str == "String"
        })
        .or_else(|| attrs.fields().first())
        .and_then(|f| f.ident.as_ref());

    if let Some(field) = preview_field {
        quote! {
            fn __repr__(&self) -> String {
                let preview: String = format!("{:?}", self.#field)
                    .chars()
                    .take(40)
                    .collect();
                let suffix = if format!("{:?}", self.#field).len() > 40 { "..." } else { "" };
                format!("{}({}{})", #name_str, preview, suffix)
            }
        }
    } else {
        quote! {
            fn __repr__(&self) -> String {
                format!("{}(...)", #name_str)
            }
        }
    }
}
```

**Step 2: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/repr.rs
git commit -m "feat(rlm-derive): implement __repr__ generation with templates"
```

---

### Task 1.5: Implement __iter__, __len__, __getitem__ generation

**Files:**
- Create: `crates/rlm-derive/src/generators/iter.rs`

**Step 1: Write iterator generation**

```rust
// crates/rlm-derive/src/generators/iter.rs
use proc_macro2::TokenStream;
use quote::{quote, format_ident};
use crate::attrs::RlmTypeAttrs;

/// Generate __iter__, __len__, __getitem__ based on #[rlm(iter = "field", index = "field")]
pub fn generate_iter(attrs: &RlmTypeAttrs) -> TokenStream {
    let mut impls = TokenStream::new();

    // __len__ from iter field
    if let Some(ref iter_field) = attrs.iter {
        let field_ident = format_ident!("{}", iter_field);
        impls.extend(quote! {
            fn __len__(&self) -> usize {
                self.#field_ident.len()
            }
        });
    }

    // __getitem__ from index field
    if let Some(ref index_field) = attrs.index {
        let field_ident = format_ident!("{}", index_field);
        let struct_name = &attrs.ident;

        // Determine the item type from the field
        let item_type = attrs.fields()
            .iter()
            .find(|f| f.ident.as_ref().map(|i| i.to_string()) == Some(index_field.clone()))
            .map(|f| extract_vec_inner_type(&f.ty))
            .flatten();

        if let Some(item_ty) = item_type {
            impls.extend(quote! {
                fn __getitem__(&self, idx: isize) -> pyo3::PyResult<#item_ty> {
                    let len = self.#field_ident.len() as isize;
                    let actual_idx = if idx < 0 { len + idx } else { idx };

                    if actual_idx < 0 || actual_idx >= len {
                        return Err(pyo3::exceptions::PyIndexError::new_err(
                            format!("index {} out of range for length {}", idx, len)
                        ));
                    }

                    Ok(self.#field_ident[actual_idx as usize].clone())
                }
            });
        }
    }

    // __iter__ - return a Python iterator (generate a private iterator type per struct)
    if let Some(ref iter_field) = attrs.iter {
        let field_ident = format_ident!("{}", iter_field);

        let item_type = attrs.fields()
            .iter()
            .find(|f| f.ident.as_ref().map(|i| i.to_string()) == Some(iter_field.clone()))
            .map(|f| extract_vec_inner_type(&f.ty))
            .flatten();

        if let Some(item_ty) = item_type {
            let iter_struct = format_ident!("__{}Iter", attrs.ident);
            impls.extend(quote! {
                #[pyo3::pyclass]
                struct #iter_struct {
                    items: Vec<#item_ty>,
                    index: usize,
                }

                #[pyo3::pymethods]
                impl #iter_struct {
                    fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyRef<'_, Self> {
                        slf
                    }

                    fn __next__(mut slf: pyo3::PyRefMut<'_, Self>) -> Option<#item_ty> {
                        if slf.index >= slf.items.len() {
                            return None;
                        }
                        let item = slf.items[slf.index].clone();
                        slf.index += 1;
                        Some(item)
                    }
                }

                fn __iter__(slf: pyo3::PyRef<'_, Self>) -> pyo3::PyResult<pyo3::Py<#iter_struct>> {
                    let items: Vec<#item_ty> = slf.#field_ident.clone();
                    let iter = #iter_struct { items, index: 0 };
                    pyo3::Py::new(slf.py(), iter)
                }
            });
        }
    }

    impls
}

/// Extract T from Vec<T>
fn extract_vec_inner_type(ty: &syn::Type) -> Option<TokenStream> {
    if let syn::Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident == "Vec" {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return Some(quote! { #inner });
                }
            }
        }
    }
    None
}
```

**Step 2: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/iter.rs
git commit -m "feat(rlm-derive): implement __iter__, __len__, __getitem__ generation"
```

---

### Task 1.6: Implement filter and flatten properties

**Files:**
- Create: `crates/rlm-derive/src/generators/properties.rs`

**Step 1: Write property generation**

```rust
// crates/rlm-derive/src/generators/properties.rs
use proc_macro2::TokenStream;
use quote::{quote, format_ident};
use crate::attrs::{RlmTypeAttrs, RlmFieldAttrs};

/// Generate filter properties (e.g., user_steps) and flatten properties (e.g., all_tool_calls)
pub fn generate_properties(attrs: &RlmTypeAttrs) -> TokenStream {
    let mut impls = TokenStream::new();

    // Collect filter properties from child type annotations
    // These are specified on the field that will be filtered
    for field in attrs.fields() {
        if let Some(ref filter_prop) = field.filter_property {
            impls.extend(generate_filter_property(attrs, field, filter_prop));
        }

        if let Some(ref flatten_prop) = field.flatten_property {
            impls.extend(generate_flatten_property(attrs, field, flatten_prop));
        }
    }

    impls
}

/// Generate: fn user_steps(&self) -> Vec<Step> { self.steps.iter().filter(|s| s.source == "user").cloned().collect() }
fn generate_filter_property(
    attrs: &RlmTypeAttrs,
    field: &RlmFieldAttrs,
    property_name: &str,
) -> TokenStream {
    let field_ident = field.ident.as_ref().unwrap();
    let property_ident = format_ident!("{}", property_name);

    let filter_value = field.filter_value.as_ref()
        .expect("filter_property requires filter_value");
    let filter_field = field.filter_field.as_ref()
        .map(|s| s.as_str())
        .unwrap_or("source");
    let filter_field_ident = format_ident!("{}", filter_field);

    // Extract Vec<T> inner type
    let item_type = extract_vec_inner_type(&field.ty)
        .expect("filter_property only works on Vec<T> fields");

    quote! {
        #[getter]
        fn #property_ident(&self) -> Vec<#item_type> {
            self.#field_ident
                .iter()
                .filter(|item| item.#filter_field_ident == #filter_value)
                .cloned()
                .collect()
        }
    }
}

/// Generate: fn all_tool_calls(&self) -> Vec<ToolCall> { self.steps.iter().filter_map(|s| s.tool_calls.as_ref()).flatten().cloned().collect() }
fn generate_flatten_property(
    attrs: &RlmTypeAttrs,
    field: &RlmFieldAttrs,
    property_name: &str,
) -> TokenStream {
    let property_ident = format_ident!("{}", property_name);

    // The parent field containing the collection (e.g., steps)
    let parent_ident = resolve_parent_collection(attrs, field)
        .expect("flatten_property requires flatten_parent when more than one Vec<T> exists");

    // The nested field name (e.g., tool_calls)
    let nested_field_ident = field.ident.as_ref().unwrap();

    // Extract the innermost type
    // e.g., Option<Vec<ToolCall>> -> ToolCall
    let inner_type = extract_innermost_vec_type(&field.ty)
        .expect("flatten_property requires Option<Vec<T>> or Vec<T>");

    // Check if it's Option<Vec<T>> or Vec<T>
    let is_option = is_option_type(&field.ty);

    if is_option {
        quote! {
            #[getter]
            fn #property_ident(&self) -> Vec<#inner_type> {
                self.#parent_ident
                    .iter()
                    .filter_map(|item| item.#nested_field_ident.as_ref())
                    .flatten()
                    .cloned()
                    .collect()
            }
        }
    } else {
        quote! {
            #[getter]
            fn #property_ident(&self) -> Vec<#inner_type> {
                self.#parent_ident
                    .iter()
                    .flat_map(|item| &item.#nested_field_ident)
                    .cloned()
                    .collect()
            }
        }
    }
}

fn resolve_parent_collection<'a>(
    attrs: &'a RlmTypeAttrs,
    child: &RlmFieldAttrs,
) -> Option<&'a syn::Ident> {
    if let Some(parent) = &child.flatten_parent {
        return attrs
            .fields()
            .iter()
            .find(|f| f.ident.as_ref().map(|i| i == parent).unwrap_or(false))
            .and_then(|f| f.ident.as_ref());
    }

    // Infer only if there's exactly one Vec<T> field
    let vec_fields: Vec<_> = attrs
        .fields()
        .iter()
        .filter(|f| is_vec_type(&f.ty))
        .collect();

    if vec_fields.len() == 1 {
        return vec_fields[0].ident.as_ref();
    }
    None
}

fn is_vec_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Vec";
        }
    }
    false
}

fn extract_vec_inner_type(ty: &syn::Type) -> Option<TokenStream> {
    // Same as in iter.rs - extract T from Vec<T>
    if let syn::Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;
        if segment.ident == "Vec" {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return Some(quote! { #inner });
                }
            }
        }
    }
    None
}

fn extract_innermost_vec_type(ty: &syn::Type) -> Option<TokenStream> {
    // Handle Option<Vec<T>> or Vec<T>
    if let syn::Type::Path(type_path) = ty {
        let segment = type_path.path.segments.last()?;

        if segment.ident == "Option" {
            // Recurse into Option's inner type
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return extract_innermost_vec_type(inner);
                }
            }
        } else if segment.ident == "Vec" {
            if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                    return Some(quote! { #inner });
                }
            }
        }
    }
    None
}

fn is_option_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}
```

**Step 2: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/properties.rs
git commit -m "feat(rlm-derive): implement filter and flatten property generation"
```

---

### Task 1.6b: Expose `__rlm_schema__` metadata for REPL discovery

**Goal:** Make field docs/constraints discoverable at runtime in Python, even if property docstrings are limited.

**Files:**
- Create: `crates/rlm-derive/src/generators/schema.rs`
- Modify: `crates/rlm-derive/src/generators/pyclass.rs`

**Step 1: Generate `__rlm_schema__`**

```rust
// crates/rlm-derive/src/generators/schema.rs
use proc_macro2::TokenStream;
use quote::quote;
use crate::attrs::RlmTypeAttrs;

/// Generate a __rlm_schema__ method returning a dict of field metadata.
pub fn generate_schema_metadata(attrs: &RlmTypeAttrs) -> TokenStream {
    let field_entries: Vec<TokenStream> = attrs.fields()
        .iter()
        .map(|f| {
            let name = f.ident.as_ref().unwrap().to_string();
            let ty = quote::quote!(#f.ty).to_string();
            let desc = f.desc.as_deref().unwrap_or("");
            quote! {
                schema.set_item(#name, (
                    #ty,
                    #desc,
                ))?;
            }
        })
        .collect();

    quote! {
        /// Machine‑readable field schema for REPL discovery.
        fn __rlm_schema__(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<::pyo3::PyObject> {
            let schema = ::pyo3::types::PyDict::new(py);
            #(#field_entries)*
            Ok(schema.into_py(py))
        }
    }
}
```

**Step 2: (Optional) Attempt per‑field docstrings**
- If PyO3 supports property docstrings (`#[pyo3(get, doc = "...")]`), include field descs on getters.
- Otherwise rely on `__rlm_schema__` + variable preview.

**Step 3: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/schema.rs \
        /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/pyclass.rs
git commit -m "feat(rlm-derive): add __rlm_schema__ metadata"
```

---

### Task 1.7: Use baml-bridge-derive instead of custom BamlType codegen

**Why:** baml-bridge already ships a robust `#[derive(BamlType)]`. Re-implementing it here is redundant and risks drift.

**Changes:**
- Remove the `baml` generator module entirely.
- Update `rlm_type.rs` to **not** generate any BamlType impls.
- In `#[rlm_type]`, inject `#[derive(baml_bridge::BamlType)]` so users don't add it manually.

**Step: Update generators/mod.rs + rlm_type.rs**

```rust
// crates/rlm-derive/src/generators/mod.rs
pub mod pyclass;
pub mod repr;
pub mod iter;
pub mod properties;
pub mod describe;
```

```rust
// crates/rlm-derive/src/rlm_type.rs
let pyclass_impl = pyclass::generate_pyclass(&attrs);
let describe_impl = describe::generate_describe(&attrs);

let expanded = quote! {
    #pyclass_impl
    #describe_impl
};
```

---

### Task 1.8: Implement RlmDescribe trait generation

**Files:**
- Create: `crates/rlm-derive/src/generators/describe.rs`

**Step 1: Write description generator for prompts**

```rust
// crates/rlm-derive/src/generators/describe.rs
use proc_macro2::TokenStream;
use quote::{quote, format_ident};
use crate::attrs::RlmTypeAttrs;

/// Generate RlmDescribe implementation for rich prompt descriptions
pub fn generate_describe(attrs: &RlmTypeAttrs) -> TokenStream {
    let struct_name = &attrs.ident;
    let struct_name_str = struct_name.to_string();

    // Collect field descriptions
    let field_descs: Vec<TokenStream> = attrs.fields()
        .iter()
        .filter(|f| !f.skip_schema)
        .map(|field| {
            let field_name = field.ident.as_ref().unwrap();
            let field_name_str = field_name.to_string();
            let field_ty = &field.ty;
            let desc = field.desc.as_ref().map(|s| s.as_str()).unwrap_or("");

            quote! {
                RlmFieldDesc {
                    name: #field_name_str,
                    type_name: std::any::type_name::<#field_ty>(),
                    description: #desc,
                    is_optional: <#field_ty as RlmTypeInfo>::is_optional(),
                }
            }
        })
        .collect();

    // Collect computed property descriptions (container-level metadata)
    let computed_descs: Vec<TokenStream> = attrs.properties
        .iter()
        .map(|prop| {
            let name = prop.name.as_str();
            let desc = prop.desc.as_deref().unwrap_or("");
            quote! {
                RlmPropertyDesc {
                    name: #name,
                    description: #desc,
                }
            }
        })
        .collect();

    // Collect filter properties
    let filter_props: Vec<TokenStream> = attrs.fields()
        .iter()
        .filter_map(|f| {
            f.filter_property.as_ref().map(|prop| {
                let filter_value = f.filter_value.as_ref().unwrap();
                let filter_field = f.filter_field.as_ref().map(|s| s.as_str()).unwrap_or("source");
                let return_type = extract_vec_inner_type_str(&f.ty);

                quote! {
                    RlmPropertyDesc {
                        name: #prop,
                        description: &format!("list[{}] where {} == {:?}", #return_type, #filter_field, #filter_value),
                    }
                }
            })
        })
        .collect();

    // Check for iter/index
    let is_iterable = attrs.iter.is_some();
    let is_indexable = attrs.index.is_some();

    quote! {
        impl RlmDescribe for #struct_name {
            fn type_name() -> &'static str {
                #struct_name_str
            }

            fn fields() -> &'static [RlmFieldDesc] {
                static FIELDS: std::sync::OnceLock<Vec<RlmFieldDesc>> = std::sync::OnceLock::new();
                FIELDS.get_or_init(|| vec![
                    #(#field_descs),*
                ])
            }

            fn properties() -> &'static [RlmPropertyDesc] {
                static PROPS: std::sync::OnceLock<Vec<RlmPropertyDesc>> = std::sync::OnceLock::new();
                PROPS.get_or_init(|| {
                    let mut props = vec![
                        #(#computed_descs),*
                    ];
                    props.extend(vec![
                        #(#filter_props),*
                    ]);
                    props
                })
            }

            fn is_iterable() -> bool {
                #is_iterable
            }

            fn is_indexable() -> bool {
                #is_indexable
            }

            fn describe_value(&self) -> String {
                // Generate preview
                let repr = format!("{:?}", self);
                let preview = if repr.len() > 200 {
                    format!("{}...", &repr[..200])
                } else {
                    repr
                };

                format!(
                    "Type: {}\nFields: {:?}\nPreview: {}",
                    Self::type_name(),
                    Self::fields().iter().map(|f| f.name).collect::<Vec<_>>(),
                    preview
                )
            }
        }
    }
}

fn extract_vec_inner_type_str(ty: &syn::Type) -> String {
    // Extract "Step" from Vec<Step>
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            if segment.ident == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return quote::quote!(#inner).to_string();
                    }
                }
            }
        }
    }
    quote::quote!(#ty).to_string()
}
```

**Step 2: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/generators/describe.rs
git commit -m "feat(rlm-derive): implement RlmDescribe generation for prompts"
```

---

### Task 1.9: Wire up the main derive entry point

**Files:**
- Create: `crates/rlm-derive/src/rlm_type.rs`
- Modify: `crates/rlm-derive/src/lib.rs`

**Step 1: Write the main derive function**

```rust
// crates/rlm-derive/src/rlm_type.rs
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

use crate::attrs::RlmTypeAttrs;
use crate::generators::{pyclass, describe};

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Ensure #[pyclass] is present; derive macros cannot add it.
    if !has_pyclass_attr(&input.attrs) {
        return syn::Error::new_spanned(
            &input.ident,
            "RlmType requires #[pyclass]. Use #[rlm_type] or add #[pyclass] above the struct.",
        )
        .to_compile_error()
        .into();
    }

    let attrs = match RlmTypeAttrs::from_derive_input(&input) {
        Ok(attrs) => attrs,
        Err(e) => return e.write_errors().into(),
    };

    // Generate all implementations
    let pyclass_impl = pyclass::generate_pyclass(&attrs);
    let describe_impl = describe::generate_describe(&attrs);

    let expanded = quote! {
        #pyclass_impl
        #describe_impl
    };

    expanded.into()
}

fn has_pyclass_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("pyclass") || attr.path().is_ident("pyo3::pyclass")
    })
}
```

**Step 2: Update lib.rs to use modules**

```rust
// crates/rlm-derive/src/lib.rs
use proc_macro::TokenStream;

mod attrs;
mod generators;
mod rlm_type;

#[proc_macro_derive(RlmType, attributes(rlm))]
pub fn derive_rlm_type(input: TokenStream) -> TokenStream {
    rlm_type::derive(input)
}
```

**Step 3: Create generators/mod.rs**

```rust
// crates/rlm-derive/src/generators/mod.rs
pub mod pyclass;
pub mod repr;
pub mod iter;
pub mod properties;
pub mod describe;
```

**Step 4: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-derive/src/
git commit -m "feat(rlm-derive): wire up main derive entry point"
```

---

### Acceptance criteria (Phase 1)
- A struct annotated with `#[rlm_type]` compiles and is usable from Python: getters work, `repr()` matches the template, and optional `__len__/__iter__/__getitem__` behave as configured.
- `#[derive(RlmType)]` without `#[pyclass]` fails at compile time with a clear error; with `#[pyclass]` it compiles.
- `__baml__()` returns JSON‑like data consistent with `BamlType::to_baml_value`; a nested sample round‑trips via `try_from_baml_value`.
- `__rlm_schema__()` returns a dict containing field type + description (and constraints where available).
- Filter/flatten properties produce correct values on a representative sample type.
- `RlmDescribe` output includes fields + computed properties and is stable across runs.

## Phase 2: Type Description + Signature Integration (DSRs)

### Overview

Add shared traits in DSRs (`rlm-core`) and integrate them into `dsrs_macros` so generated signature inputs can expose fields to the REPL.

**Also needed:** `RlmInputFields` trait for signature input structs:
- `inject_into_python(py, globals)` - converts input fields to Python variables
- `rlm_variable_descriptions()` - creates prompt descriptions from field metadata
- This trait should be auto-generated by DSRs `#[derive(Signature)]` or by our RLM extension

### Task 2.1: Create rlm-core crate with traits

**Files (DSRs repo):**
- Create: `/Users/darin/src/personal/DSRs/crates/rlm-core/Cargo.toml`
- Create: `/Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs`
- Create: `/Users/darin/src/personal/DSRs/crates/rlm-core/src/describe.rs`

**Step 1: Create Cargo.toml**

```toml
# /Users/darin/src/personal/DSRs/crates/rlm-core/Cargo.toml
[package]
name = "rlm-core"
version = "0.1.0"
edition = "2024"

[dependencies]
pyo3 = { version = "0.27.2", features = ["auto-initialize"] }
baml-bridge = { path = "/Users/darin/src/personal/DSRs/crates/baml-bridge" }
```

**Step 2: Write describe traits**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-core/src/describe.rs

/// Description of a field for prompt generation
#[derive(Debug, Clone)]
pub struct RlmFieldDesc {
    pub name: &'static str,
    pub type_name: &'static str,
    pub description: &'static str,
    pub is_optional: bool,
}

/// Description of a computed property
#[derive(Debug, Clone)]
pub struct RlmPropertyDesc {
    pub name: &'static str,
    pub description: &'static str,
}

/// Trait for types that can describe themselves for RLM prompts
pub trait RlmDescribe {
    /// The type name as shown in prompts
    fn type_name() -> &'static str;

    /// Field descriptions
    fn fields() -> &'static [RlmFieldDesc];

    /// Computed property descriptions
    fn properties() -> &'static [RlmPropertyDesc];

    /// Whether this type is iterable (has __iter__)
    fn is_iterable() -> bool { false }

    /// Whether this type is indexable (has __getitem__)
    fn is_indexable() -> bool { false }

    /// Generate a human-readable description of a value
    fn describe_value(&self) -> String;

    /// Generate full type documentation for prompts
    fn describe_type() -> String {
        let mut desc = format!("{}:\n", Self::type_name());

        // Fields
        for field in Self::fields() {
            let optional = if field.is_optional { " | None" } else { "" };
            desc.push_str(&format!("  .{}: {}{}", field.name, field.type_name, optional));
            if !field.description.is_empty() {
                desc.push_str(&format!("  # {}", field.description));
            }
            desc.push('\n');
        }

        // Properties
        if !Self::properties().is_empty() {
            desc.push_str("  Properties:\n");
            for prop in Self::properties() {
                desc.push_str(&format!("    .{}", prop.name));
                if !prop.description.is_empty() {
                    desc.push_str(&format!(" -> {}", prop.description));
                }
                desc.push('\n');
            }
        }

        // Iteration info
        if Self::is_iterable() {
            desc.push_str("  Iterable: for item in value\n");
        }
        if Self::is_indexable() {
            desc.push_str("  Indexable: value[0], value[-1]\n");
        }

        desc
    }
}

/// Helper trait for type introspection
pub trait RlmTypeInfo {
    fn is_optional() -> bool { false }
}

impl<T> RlmTypeInfo for Option<T> {
    fn is_optional() -> bool { true }
}

impl<T> RlmTypeInfo for Vec<T> {
    fn is_optional() -> bool { false }
}

impl RlmTypeInfo for String {
    fn is_optional() -> bool { false }
}

impl RlmTypeInfo for i32 {
    fn is_optional() -> bool { false }
}

// ... more primitive impls

// Container impls for nicer variable descriptions
impl<T: RlmDescribe> RlmDescribe for Vec<T> {
    fn type_name() -> &'static str { "list" }
    fn fields() -> &'static [RlmFieldDesc] { &[] }
    fn properties() -> &'static [RlmPropertyDesc] { &[] }
    fn is_iterable() -> bool { true }
    fn is_indexable() -> bool { true }
    fn describe_value(&self) -> String {
        format!("list(len={})", self.len())
    }
    fn describe_type() -> String {
        format!("list[{}]", T::type_name())
    }
}

impl<T: RlmDescribe> RlmDescribe for Option<T> {
    fn type_name() -> &'static str { "optional" }
    fn fields() -> &'static [RlmFieldDesc] { &[] }
    fn properties() -> &'static [RlmPropertyDesc] { &[] }
    fn describe_value(&self) -> String {
        match self {
            Some(v) => format!("Some({})", v.describe_value()),
            None => "None".to_string(),
        }
    }
    fn describe_type() -> String {
        format!("{} | None", T::type_name())
    }
}
```

**Step 3: Write lib.rs**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs
pub mod describe;

pub use describe::{RlmDescribe, RlmFieldDesc, RlmPropertyDesc, RlmTypeInfo};
```

**Step 4: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-core/
git commit -m "feat: add rlm-core crate with RlmDescribe trait"
```

---

### Task 2.2: Create variable description for prompts

**Files (DSRs repo):**
- Create: `/Users/darin/src/personal/DSRs/crates/rlm-core/src/variable.rs`

**Step 1: Write variable description struct**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-core/src/variable.rs
use crate::describe::RlmDescribe;

/// A variable description for inclusion in RLM prompts
#[derive(Debug, Clone)]
pub struct RlmVariable {
    pub name: String,
    pub type_desc: String,
    pub description: String,
    pub constraints: Vec<String>,
    pub total_length: usize,
    pub preview: String,
    pub properties: Vec<(String, String)>,  // (name, return_type_desc)
}

impl RlmVariable {
    /// Create from a Rust type implementing RlmDescribe
    pub fn from_rust<T: RlmDescribe>(name: &str, value: &T) -> Self {
        let type_desc = T::describe_type();
        let value_desc = value.describe_value();
        let preview_len = value_desc.len();
        let preview = if preview_len > 500 {
            format!("{}...", &value_desc[..500])
        } else {
            value_desc.clone()
        };

        let properties = T::properties()
            .iter()
            .map(|p| (p.name.to_string(), p.description.to_string()))
            .collect();

        Self {
            name: name.to_string(),
            type_desc,
            description: String::new(),  // Can be overridden
            constraints: Vec::new(),
            total_length: preview_len,
            preview,
            properties,
        }
    }

    /// Add a human description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add constraints (stringified)
    pub fn with_constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Format for prompt inclusion
    pub fn format(&self) -> String {
        let mut output = format!("Variable: `{}` (access it in your code)\n", self.name);
        output.push_str(&format!("Type: {}\n", self.type_desc.lines().next().unwrap_or("unknown")));

        if !self.description.is_empty() {
            output.push_str(&format!("Description: {}\n", self.description));
        }
        if !self.constraints.is_empty() {
            output.push_str(&format!("Constraints: {}\n", self.constraints.join("; ")));
        }

        output.push_str(&format!("Total length: {} characters\n", self.total_length));

        if !self.properties.is_empty() {
            output.push_str("Properties:\n");
            for (name, ret_type) in &self.properties {
                output.push_str(&format!("  .{} -> {}\n", name, ret_type));
            }
        }

        output.push_str(&format!("Preview:\n```\n{}\n```\n", self.preview));

        output
    }
}
```

**Step 2: Update lib.rs**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs
pub mod describe;
pub mod variable;

pub use describe::{RlmDescribe, RlmFieldDesc, RlmPropertyDesc, RlmTypeInfo};
pub use variable::RlmVariable;
```

**Step 3: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-core/src/variable.rs /Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs
git commit -m "feat(rlm-core): add RlmVariable for prompt descriptions"
```

---

### Task 2.4: Add Python conversion helpers in `baml-bridge` (feature-gated)

**Goal:** Keep Python ↔ BAML conversion in DSRs (not rig-rlm), with a `pyo3` feature gate.

**Files (DSRs repo):**
- Modify: `/Users/darin/src/personal/DSRs/crates/baml-bridge/Cargo.toml`
- Modify: `/Users/darin/src/personal/DSRs/crates/baml-bridge/src/lib.rs`
- Create: `/Users/darin/src/personal/DSRs/crates/baml-bridge/src/py.rs`
- Modify: `/Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml`
- Create: `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/py.rs`
- Modify: `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs`

**Step 1: Feature‑gate pyo3 in baml-bridge**

```toml
# /Users/darin/src/personal/DSRs/crates/baml-bridge/Cargo.toml
[features]
pyo3 = ["dep:pyo3"]

[dependencies]
pyo3 = { version = "0.27.2", features = ["auto-initialize"], optional = true }
```

**Step 2: Add `baml_bridge::py` helpers**

```rust
// /Users/darin/src/personal/DSRs/crates/baml-bridge/src/py.rs
use pyo3::{PyAny, PyObject, Python};
use pyo3::types::{PyDict, PyList};

use crate::baml_types::{BamlMap, BamlValue, TypeIR};
use crate::internal_baml_jinja::types::OutputFormatContent;

/// Convert a BamlValue into a JSON-like Python object (dict/list/primitives).
pub fn baml_value_to_py(py: Python<'_>, value: &BamlValue) -> PyObject { /* as before */ }

/// Convert a Python object into BamlValue using a target TypeIR.
/// - Calls __baml__() if present
/// - Handles primitives, list/tuple, dict
/// - Uses jsonish only for string → non-string coercion
pub fn py_to_baml_value(
    py: Python<'_>,
    obj: &pyo3::Bound<'_, PyAny>,
    type_ir: &TypeIR,
    output_format: &OutputFormatContent,
) -> Result<BamlValue, crate::BamlParseError> { /* schema‑guided conversion */ }

/// Best-effort normalization for dataclass / pydantic / attrs, if present.
pub fn normalize_python_object(
    py: Python<'_>,
    obj: &pyo3::Bound<'_, PyAny>,
) -> pyo3::PyResult<pyo3::Bound<'_, PyAny>> { /* best‑effort dict */ }

/// Optional orjson/json fallback: serialize → serde_json → BamlValue
pub fn orjson_fallback_to_baml(
    py: Python<'_>,
    obj: &pyo3::Bound<'_, PyAny>,
) -> pyo3::PyResult<BamlValue> { /* best‑effort */ }
```

**Step 3: Wire module export**

```rust
// /Users/darin/src/personal/DSRs/crates/baml-bridge/src/lib.rs
#[cfg(feature = "pyo3")]
pub mod py;
```

**Step 4: Signature‑aware helpers in dspy-rs**

```rust
// /Users/darin/src/personal/DSRs/crates/dspy-rs/src/py.rs
use pyo3::types::PyDict;
use pyo3::{PyAny, Python};

use baml_bridge::{BamlParseError, BamlValue};
use baml_bridge::jsonish::deserializer::coercer::run_user_checks;

use crate::Signature;

pub fn missing_output_fields<S: Signature>(kwargs: &pyo3::Bound<'_, PyDict>) -> Vec<String> { /* ... */ }

pub fn kwargs_to_baml_value<S: Signature>(
    py: Python<'_>,
    kwargs: &pyo3::Bound<'_, PyDict>,
) -> Result<BamlValue, BamlParseError> { /* uses baml_bridge::py::py_to_baml_value */ }

pub fn collect_checks_for_output<S: Signature>(
    value: &BamlValue,
) -> Result<Vec<baml_bridge::ResponseCheck>, BamlParseError> { /* schema‑guided checks + asserts */ }
```

**Step 5: Enable baml-bridge pyo3 in dspy-rs feature**

```toml
# /Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml
[features]
rlm = ["dep:rlm-core", "dep:rlm-derive", "dep:pyo3", "baml-bridge/pyo3"]
```

**Step 6: Re-export (optional)**

```rust
// /Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs
#[cfg(feature = "rlm")]
pub mod py;
```

**Step 7: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/baml-bridge/Cargo.toml \
        /Users/darin/src/personal/DSRs/crates/baml-bridge/src/lib.rs \
        /Users/darin/src/personal/DSRs/crates/baml-bridge/src/py.rs \
        /Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml \
        /Users/darin/src/personal/DSRs/crates/dspy-rs/src/py.rs \
        /Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs
git commit -m "feat(dsrs): add feature-gated PyAny↔BamlValue conversion helpers"
```

---

### Task 2.3: Add RlmInputFields + wire Signature derive (DSRs)

**Goal:** Make `S::Input` expose its fields to the REPL and prompt builder without manual boilerplate.

**Files (DSRs repo):**
- Create: `/Users/darin/src/personal/DSRs/crates/rlm-core/src/input.rs`
- Modify: `/Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs`
- Modify: `/Users/darin/src/personal/DSRs/crates/dsrs-macros/src/lib.rs`
- Modify: `/Users/darin/src/personal/DSRs/crates/dsrs-macros/Cargo.toml`
- Modify: `/Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml`
- Modify: `/Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs`

**Step 1: Add RlmInputFields trait**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-core/src/input.rs
use pyo3::{Py, PyAny, Python};
use pyo3::types::PyDict;

use crate::variable::RlmVariable;

/// Exposes signature inputs as Python variables + prompt descriptions.
pub trait RlmInputFields {
    fn rlm_py_fields(&self, py: Python<'_>) -> Vec<(String, Py<PyAny>)>;
    fn rlm_variables(&self) -> Vec<RlmVariable>;

    fn rlm_variable_descriptions(&self) -> String {
        self.rlm_variables()
            .iter()
            .map(|v| v.format())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn inject_into_python(&self, py: Python<'_>, globals: &PyDict) -> pyo3::PyResult<()> {
        for (name, obj) in self.rlm_py_fields(py) {
            globals.set_item(name, obj)?;
        }
        Ok(())
    }
}
```

**Step 2: Re-export from rlm-core**

```rust
// /Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs
pub mod describe;
pub mod variable;
pub mod input;

pub use describe::{RlmDescribe, RlmFieldDesc, RlmPropertyDesc, RlmTypeInfo};
pub use variable::RlmVariable;
pub use input::RlmInputFields;
```

**Step 3: Implement RlmInputFields in dsrs-macros**

Add a `#[cfg(feature = "rlm")]` impl for the generated `<Sig>Input` struct:

_(Build `input_field_names`, `input_field_idents`, and `input_field_descs/constraints` from `parsed.input_fields` in `generate_helper_structs`.)_

```rust
// /Users/darin/src/personal/DSRs/crates/dsrs-macros/src/lib.rs
// inside generate_helper_structs(...) after input struct definition

#[cfg(feature = "rlm")]
impl ::rlm_core::RlmInputFields for #input_name {
    fn rlm_py_fields(&self, py: ::pyo3::Python<'_>) -> Vec<(String, ::pyo3::Py<::pyo3::PyAny>)> {
        vec![
            #(
                (
                    #input_field_names.to_string(),
                    ::pyo3::IntoPyObject::into_pyobject(self.#input_field_idents.clone(), py)
                        .expect("IntoPyObject failed for input field")
                        .unbind()
                )
            ),*
        ]
    }

    fn rlm_variables(&self) -> Vec<::rlm_core::RlmVariable> {
        vec![
            #(
                ::rlm_core::RlmVariable::from_rust(#input_field_names, &self.#input_field_idents)
                    .with_description(#input_field_descs)
                    .with_constraints(#input_field_constraints)
            ),*
        ]
    }
}
```

**Step 4: Add deps + feature flags**

```toml
# /Users/darin/src/personal/DSRs/crates/dsrs-macros/Cargo.toml
[features]
rlm = []

[dependencies]
rlm-core = { path = "../rlm-core", optional = true }
pyo3 = { version = "0.27.2", features = ["auto-initialize"], optional = true }
```

```toml
# /Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml
[features]
rlm = ["dep:rlm-core", "dep:rlm-derive", "dep:pyo3", "baml-bridge/pyo3"]

[dependencies]
rlm-core = { path = "../rlm-core", optional = true }
rlm-derive = { path = "../rlm-derive", optional = true }
pyo3 = { version = "0.27.2", features = ["auto-initialize"], optional = true }
```

**Step 5: Re-export from dspy-rs**

```rust
// /Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs
#[cfg(feature = "rlm")]
pub use rlm_core;
#[cfg(feature = "rlm")]
pub use rlm_derive::*;
```

**Step 6: Commit**

```bash
git add /Users/darin/src/personal/DSRs/crates/rlm-core/src/input.rs \
        /Users/darin/src/personal/DSRs/crates/rlm-core/src/lib.rs \
        /Users/darin/src/personal/DSRs/crates/dsrs-macros/src/lib.rs \
        /Users/darin/src/personal/DSRs/crates/dsrs-macros/Cargo.toml \
        /Users/darin/src/personal/DSRs/crates/dspy-rs/Cargo.toml \
        /Users/darin/src/personal/DSRs/crates/dspy-rs/src/lib.rs
git commit -m "feat(dsrs): add RlmInputFields + rlm feature reexports"
```

---

### Acceptance criteria (Phase 2)
- With `dspy-rs` feature `rlm` enabled, `#[derive(Signature)]` generates `RlmInputFields` for `<Sig>Input`; without the feature, signatures compile without PyO3 types.
- `inject_into_python()` exposes input fields by name in the REPL globals.
- `rlm_variable_descriptions()` includes field descriptions + constraints from Signature metadata.
- `RlmVariable::format()` shows description/constraints only when present and includes correct length/preview behavior.
- Container impls (`Vec`, `Option`) produce correct `RlmDescribe` type names and value descriptions.

## Phase 3: SUBMIT Function

### Overview

Create the `SUBMIT` PyO3 function that:
1. Accepts `**kwargs` from Python
2. Converts Python values to `BamlValue` via schema-guided conversion:
   - `__baml__()` hook for `#[rlm_type]` pyclasses
   - dataclass / pydantic / attrs → dict → recurse
   - dict/list/tuple/primitives → recurse
   - jsonish used **only** to coerce string values into non-string expected types
3. Runs constraints (check/assert) over the resulting `BamlValue` + `TypeIR`
4. Optional orjson/json fallback for exotic objects (serialize → serde_json → BamlValue)
5. Returns validation feedback or signals success

### Task 3.1: Create submit module

**Files:**
- Create: `src/submit.rs`

**Step 1: Write SUBMIT handler**

```rust
// src/submit.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use baml_bridge::{BamlParseError, BamlValue};
use dspy_rs::{FieldMeta, Signature};
use dspy_rs::py;

/// Result of a SUBMIT call
#[derive(Debug, Clone)]
pub enum SubmitResult<O> {
    /// Successful submission with validated output
    Success {
        output: O,
        metas: indexmap::IndexMap<String, FieldMeta>,
    },
    /// Validation errors - can be retried
    ValidationError {
        message: String,
        errors: Vec<String>,
    },
    /// Hard assertion failed - configurable whether to allow retry
    AssertionFailed {
        field: String,
        label: String,
        expression: String,
    },
}

/// PyO3-compatible SUBMIT handler
///
/// Generic over the output signature, but erased for Python compatibility
#[pyclass]
pub struct SubmitHandler {
    /// Parse function specialized to S::Output (type-erased for PyO3)
    parse_fn: Arc<dyn for<'py> Fn(Python<'py>, &Bound<'py, PyDict>) -> Result<ParsedDyn, BamlParseError> + Send + Sync>,

    /// Channel to send result back to Rust
    result_tx: Arc<Mutex<Option<SubmitResultDyn>>>,

    /// Human-readable schema for error messages
    schema_description: String,
}

/// Type-erased parsed payload
pub struct ParsedDyn {
    pub baml_value: BamlValue,
    pub checks: Vec<baml_bridge::ResponseCheck>,
}

/// Type-erased result for cross-thread communication
pub type SubmitResultDyn = Result<(BamlValue, indexmap::IndexMap<String, FieldMeta>), SubmitError>;

#[derive(Debug, Clone)]
pub enum SubmitError {
    ValidationError { message: String, errors: Vec<String> },
    AssertionFailed { label: String, expression: String },
}

impl SubmitHandler {
    pub fn new<S: Signature>() -> (Self, Arc<Mutex<Option<SubmitResultDyn>>>) {
        let result_tx = Arc::new(Mutex::new(None));
        let schema_description = generate_schema_description::<S>();
        let parse_fn: Arc<dyn for<'py> Fn(Python<'py>, &Bound<'py, PyDict>) -> Result<ParsedDyn, BamlParseError> + Send + Sync> =
            Arc::new(|py, kwargs| {
                let baml_value = dspy_rs::py::kwargs_to_baml_value::<S>(py, kwargs)?;
                let checks = dspy_rs::py::collect_checks_for_output::<S>(&baml_value)?;
                Ok(ParsedDyn { baml_value, checks })
            });

        let handler = Self {
            parse_fn,
            result_tx: result_tx.clone(),
            schema_description,
        };

        (handler, result_tx)
    }
}

#[pymethods]
impl SubmitHandler {
    /// SUBMIT(field1=value1, field2=value2, ...)
    ///
    /// Validates the provided fields against the output schema.
    /// Returns a string indicating success or describing validation errors.
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        let kwargs = kwargs.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "SUBMIT requires keyword arguments. Usage: SUBMIT(field1=value1, field2=value2)"
            )
        })?;

        // Ensure all required output fields are present
        let missing = dspy_rs::py::missing_output_fields::<S>(kwargs);
        if !missing.is_empty() {
            let error = SubmitError::ValidationError {
                message: "Missing output fields".to_string(),
                errors: vec![format!("Missing fields: {:?}", missing)],
            };
            *self.result_tx.lock().unwrap() = Some(Err(error.clone()));
            return Ok(format!(
                "[SUBMIT FAILED] Missing output fields: {:?}. Use SUBMIT({})",
                missing,
                S::output_fields().iter().map(|f| f.name).collect::<Vec<_>>().join(", ")
            ));
        }

        // Convert kwargs to BamlValue via schema-guided conversion
        let parsed_result = (self.parse_fn)(py, kwargs);

        match parsed_result {
            Ok(parsed) => {
                // Build field metas from flags and checks
                let raw_text = serde_json::to_string(&parsed.baml_value)
                    .unwrap_or_else(|_| "<unserializable>".to_string());
                let metas = build_field_metas(&parsed, &raw_text);

                // Success!
                *self.result_tx.lock().unwrap() = Some(Ok((parsed.baml_value.clone(), metas)));

                // Report any soft check failures
                let check_failures: Vec<_> = parsed.checks
                    .iter()
                    .filter(|c| c.status != "passed")
                    .collect();

                if check_failures.is_empty() {
                    Ok("✓ SUBMIT successful! All validations passed.".to_string())
                } else {
                    let warnings: Vec<String> = check_failures
                        .iter()
                        .map(|c| format!("  - {} ({})",
                            c.name,
                            c.expression))
                        .collect();

                    Ok(format!(
                        "✓ SUBMIT successful with warnings:\n{}\n\
                         (These are soft constraints - output accepted but noted)",
                        warnings.join("\n")
                    ))
                }
            }
            Err(BamlParseError::ConstraintAssertsFailed { failed }) => {
                let failure = &failed[0];
                let error = SubmitError::AssertionFailed {
                    label: failure.name.clone(),
                    expression: failure.expression.clone(),
                };
                *self.result_tx.lock().unwrap() = Some(Err(error.clone()));

                Ok(format!(
                    "[SUBMIT FAILED] Assertion '{}' failed: {}\n\
                     Please fix and try again.",
                    failure.name,
                    failure.expression,
                ))
            }
            Err(e) => {
                let errors = format_parse_errors(&e);
                let error = SubmitError::ValidationError {
                    message: e.to_string(),
                    errors: errors.clone(),
                };
                *self.result_tx.lock().unwrap() = Some(Err(error));

                Ok(format!(
                    "[SUBMIT FAILED] Validation errors:\n{}\n\n\
                     Expected schema:\n{}\n\n\
                     Please fix and try again.",
                    errors.join("\n"),
                    self.schema_description,
                ))
            }
        }
    }

    /// Get the expected output schema as a string
    fn schema(&self) -> String {
        self.schema_description.clone()
    }
}


fn build_field_metas(parsed: &ParsedDyn, raw_json: &str) -> indexmap::IndexMap<String, FieldMeta> {
    // baml_bridge checks/flags are not field-scoped, so store them under _root for now
    let mut metas = indexmap::IndexMap::new();
    let mut meta = FieldMeta {
        raw_text: raw_json.to_string(),
        flags: Vec::new(),
        checks: Vec::new(),
    };

    meta.flags.extend(parsed.flags.iter().cloned());
    meta.checks.extend(parsed.checks.iter().map(|c| dspy_rs::ConstraintResult {
        label: c.name.clone(),
        expression: c.expression.clone(),
        passed: c.status == "passed",
    }));

    metas.insert("_root".to_string(), meta);
    metas
}

fn format_parse_errors(e: &BamlParseError) -> Vec<String> {
    match e {
        BamlParseError::Jsonish(err) => vec![err.to_string()],
        BamlParseError::ConstraintAssertsFailed { failed } => {
            failed.iter().map(|c| format!(
                "Assertion '{}' failed: {}",
                c.name, c.expression
            )).collect()
        }
        BamlParseError::Convert(err) => vec![err.to_string()],
    }
}

fn generate_schema_description<S: Signature>() -> String {
    let mut desc = String::new();
    desc.push_str("SUBMIT(");

    let fields: Vec<String> = S::output_fields()
        .iter()
        .map(|f| f.name.to_string())
        .collect();

    desc.push_str(&fields.join(", "));
    desc.push_str(") where:\n");

    for field in S::output_fields() {
        let type_ir = (field.type_ir)();
        let type_name = format_type_ir(&type_ir);
        desc.push_str(&format!("  {}: {}", field.name, type_name));

        if !field.description.is_empty() {
            desc.push_str(&format!("  # {}", field.description));
        }
        desc.push('\n');

        // Show constraints
        for constraint in field.constraints {
            let kind = match constraint.kind {
                dspy_rs::ConstraintKind::Check => "check",
                dspy_rs::ConstraintKind::Assert => "ASSERT",
            };
            desc.push_str(&format!("    [{kind}] {}: {}\n",
                constraint.label, constraint.expression));
        }
    }

    desc
}

fn format_type_ir(type_ir: &baml_bridge::baml_types::TypeIR) -> String {
    // Simplified type formatting
    match type_ir {
        baml_bridge::baml_types::TypeIR::String => "string".to_string(),
        baml_bridge::baml_types::TypeIR::Int => "int".to_string(),
        baml_bridge::baml_types::TypeIR::Float => "float".to_string(),
        baml_bridge::baml_types::TypeIR::Bool => "bool".to_string(),
        baml_bridge::baml_types::TypeIR::List(inner) => format!("list[{}]", format_type_ir(inner)),
        baml_bridge::baml_types::TypeIR::Optional(inner) => format!("{} | null", format_type_ir(inner)),
        baml_bridge::baml_types::TypeIR::Class { name, .. } => name.clone(),
        baml_bridge::baml_types::TypeIR::Enum { name, .. } => name.clone(),
        _ => "any".to_string(),
    }
}
```

**Step 2: Commit**

```bash
git add src/submit.rs
git commit -m "feat: implement SUBMIT PyO3 handler with validation"
```

---

### Acceptance criteria (Phase 3)
- `SUBMIT(field=value, ...)` accepts Python variables and returns typed output for matching fields.
- Missing output fields returns DSPy‑exact error: `[Error] Missing output fields: ... Use SUBMIT(...)`.
- Type coercion failures return DSPy‑exact `[Type Error] ...` strings.
- Soft checks are recorded; failed asserts block completion with an error.
- `__baml__()` is honored for `#[rlm_type]` objects; dict/list/primitives work; dataclass/pydantic/attrs normalize to dict.
- jsonish is used **only** for string→non‑string coercion; orjson/json fallback handles exotic objects.

## Phase 4: TypedRlm Core

### Overview

Create the main `TypedRlm<S: Signature>` struct that orchestrates:
1. REPL initialization with signature input fields as Python variables
2. Prompt generation with variable descriptions from RlmDescribe
3. Query loop with LLM
4. SUBMIT handling and result extraction
5. Extraction fallback on max iterations

**Key design:** No builder pattern for context. The signature's `#[input]` fields ARE the context. `TypedRlm::call(input: S::Input)` takes the signature's generated input struct directly.

### Task 4.1: Create TypedRlm struct

**Files:**
- Create: `src/typed_rlm.rs`

**Step 1: Write TypedRlm struct and config**

```rust
// src/typed_rlm.rs
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::{Py, PyAny, Python, types::{PyDict, PyDictMethods}, IntoPyObject};
use rig::agent::Agent;
use rig::completion::Prompt;
use rig::providers::openai::CompletionModel;
use tokio::runtime::Handle;

use dspy_rs::{Signature, FieldMeta, FieldSpec, BamlType};
use baml_bridge::ToBamlValue;
use rlm_core::{RlmDescribe, RlmVariable, RlmInputFields};

use crate::repl::{REPL, Command};
use crate::tools::LlmTools;
use crate::submit::{SubmitHandler, SubmitResultDyn, SubmitError};
use crate::preamble::generate_typed_preamble;
use crate::error::RlmError;

/// Configuration for TypedRlm
#[derive(Debug, Clone)]
pub struct RlmConfig {
    /// Maximum REPL iterations before extraction fallback
    pub max_iterations: usize,

    /// Maximum sub-LLM calls allowed
    pub max_llm_calls: usize,

    /// Whether to attempt extraction on max iterations (vs error)
    pub enable_extraction_fallback: bool,

    /// Whether assertion failures are fatal (vs allowing retry)
    pub strict_assertions: bool,

    /// Maximum characters to include from REPL output
    pub max_output_chars: usize,

    /// Maximum characters to include when rendering REPL history in prompts (DSPy default: 5000)
    pub max_history_output_chars: usize,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_llm_calls: 50,
            enable_extraction_fallback: true,
            strict_assertions: true,
            max_output_chars: 100_000,
            max_history_output_chars: 5_000,
        }
    }
}

/// Result of a TypedRlm execution
#[derive(Debug)]
pub struct RlmResult<S: Signature> {
    /// The typed input (preserved from run() call)
    pub input: S::Input,

    /// The typed output
    pub output: S::Output,

    /// Per-field metadata (flags, constraint checks)
    pub field_metas: IndexMap<String, FieldMeta>,

    /// Number of REPL iterations used
    pub iterations: usize,

    /// Number of sub-LLM calls made
    pub llm_calls: usize,

    /// Whether output was obtained via extraction fallback
    pub extraction_fallback: bool,

    /// Summary of constraint results
    pub constraint_summary: ConstraintSummary,
}

impl<S: Signature> RlmResult<S> {
    /// Reconstruct the full signature struct if needed
    pub fn to_signature(&self) -> S
    where
        S: Clone,
        S::Input: Clone,
        S::Output: Clone,
    {
        S::from_parts(self.input.clone(), self.output.clone())
    }
}

#[derive(Debug, Default)]
pub struct ConstraintSummary {
    pub checks_passed: usize,
    pub checks_failed: usize,
    pub assertions_passed: usize,
}

impl<S: Signature> RlmResult<S> {
    /// Get all failed soft checks
    pub fn failed_checks(&self) -> Vec<&dspy_rs::ConstraintResult> {
        self.field_metas
            .values()
            .flat_map(|meta| &meta.checks)
            .filter(|check| !check.passed)
            .collect()
    }

    /// Whether any soft constraints failed
    pub fn has_constraint_warnings(&self) -> bool {
        !self.failed_checks().is_empty()
    }

    /// Whether this was a fallback extraction
    pub fn is_fallback(&self) -> bool {
        self.extraction_fallback
    }
}

/// Typed Recursive Language Model
///
/// Takes a DSRs Signature and runs it as an RLM:
/// - Input fields become Python variables in the REPL
/// - Output fields define the SUBMIT schema
/// - Types with RlmType get rich Python ergonomics
pub struct TypedRlm<S: Signature> {
    agent: Agent<CompletionModel>,
    config: RlmConfig,
    _marker: PhantomData<S>,
}

impl<S: Signature> TypedRlm<S> {
    /// Create a new TypedRlm with the given agent and config
    pub fn new(agent: Agent<CompletionModel>, config: RlmConfig) -> Self {
        Self {
            agent,
            config,
            _marker: PhantomData,
        }
    }

    /// Create with default config
    pub fn with_agent(agent: Agent<CompletionModel>) -> Self {
        Self::new(agent, RlmConfig::default())
    }
}

impl<S> TypedRlm<S>
where
    S: Signature + Clone + 'static,
    S::Input: ToBamlValue + RlmInputFields + Clone + Send + Sync,
    S::Output: BamlType + Clone,
{
    /// Run the RLM with the given input and return typed output
    ///
    /// Input fields are automatically:
    /// 1. Converted to Python objects via IntoPyObject (from RlmType)
    /// 2. Made available as variables in the REPL
    /// 3. Described in the prompt via RlmDescribe
    pub async fn call(&self, input: S::Input) -> Result<RlmResult<S>, RlmError> {
        let runtime_handle = Handle::current();

        // Create LLM tools with call limit
        let tools = LlmTools::new(
            self.agent.clone(),
            self.config.max_llm_calls,
            runtime_handle.clone(),
        );
        // DSPy parity: LlmTools should raise Python errors (not return "[ERROR] ..." strings).
        // - On call limit: raise PyRuntimeError with the exact DSPy message.
        // - On LM failures: propagate as PyRuntimeError so REPL output shows "[Error] RuntimeError: <message>".
        // DSPy‑parity: exceeding max_llm_calls should raise a Python RuntimeError with:
        // "LLM call limit exceeded: {count} + {n} > {max}. Use Python code for aggregation instead of making more LLM calls."

        // Create SUBMIT handler
        let (submit_handler, submit_rx) = SubmitHandler::new::<S>();

        // Generate the preamble with variable descriptions
        let variable_descriptions = input.rlm_variable_descriptions();
        let preamble = generate_typed_preamble::<S>(&variable_descriptions);

        let mut prompt = format!(
            "{}\n\nYour next action:",
            preamble
        );

        let mut iterations = 0;
        let mut message_history = Vec::new();

        loop {
            iterations += 1;

            if iterations > self.config.max_iterations {
                if self.config.enable_extraction_fallback {
                    return self.extraction_fallback(input.clone(), &message_history, iterations, tools.call_count()).await;
                } else {
                    return Err(RlmError::MaxIterationsReached {
                        max: self.config.max_iterations,
                        partial: None,
                    });
                }
            }

            // Query the LLM
            let response = self.agent.prompt(&prompt).await
                .map_err(|e| RlmError::LlmError { source: Box::new(e) })?;

            // Store in history
            message_history.push((prompt.clone(), response.clone()));

            // Parse command
            let cmd = Command::parse(&response);

            // Handle REPL code execution
            if let Some(code) = cmd.get_code_to_run() {
                let exec_result = Python::with_gil(|py| {
                    // Set up globals with tools and SUBMIT
                    let globals = PyDict::new(py);

                    // Add LLM tools
                    let tools_instance = tools.clone();
                    globals.set_item("_llm_tools", tools_instance.into_pyobject(py)?)?;
                    py.run(
                        c"llm_query = _llm_tools.llm_query\nllm_query_batched = _llm_tools.llm_query_batched",
                        Some(&globals),
                        None,
                    )?;

                    // Add SUBMIT handler
                    globals.set_item("SUBMIT", submit_handler.clone().into_pyobject(py)?)?;

                    // Add input fields as Python variables
                    // RlmInputFields trait (generated by Signature derive) provides field iteration
                    input.inject_into_python(py, &globals)?;

                    // Run the code with Jupyter semantics:
                    // 1) Execute all statements
                    // 2) If the last statement is an expression (and not suppressed by trailing ';'),
                    //    evaluate it and display its repr
                    // 3) Always include stdout
                    //
                    // Combined output order: stdout first, then repr(last_expr) if present.
                    // This matches Jupyter/IPython behavior and is more informative than DSPy’s stdout-vs-result switch.
                    //
                    // After combining, apply capture-time truncation:
                    // - empty => "(no output - did you forget to print?)"
                    // - over max_output_chars => append "\n... (truncated)"
                    //
                    // History formatting will apply a second truncation at max_history_output_chars
                    // with DSPy’s exact marker string.
                    let locals = PyDict::new(py);
                    py.run(pyo3::ffi::c_str!(&code), Some(&globals), Some(&locals))?;
                    Ok(pyo3::types::PyString::new(py, "").into_any().unbind())
                })?;

                // Check if SUBMIT was called
                if let Some(result) = submit_rx.lock().unwrap().take() {
                    match result {
                        Ok((baml_value, metas)) => {
                            // Convert BamlValue to typed output
                            let output = S::Output::try_from_baml_value(baml_value)
                                .map_err(|e| RlmError::ConversionError {
                                    message: e.to_string()
                                })?;

                            // We need input to construct full S - for now use default
                            // In real impl, we'd track the input
                            let constraint_summary = compute_constraint_summary(&metas);

                            return Ok(RlmResult {
                                input: input.clone(),
                                output,
                                field_metas: metas,
                                iterations,
                                llm_calls: tools.call_count(),
                                extraction_fallback: false,
                                constraint_summary,
                            });
                        }
                        Err(SubmitError::ValidationError { message, errors }) => {
                            prompt = format!(
                                "[SUBMIT failed]\n{}\n\nErrors:\n{}\n\nPlease fix and try again.",
                                message,
                                errors.join("\n")
                            );
                        }
                        Err(SubmitError::AssertionFailed { label, expression }) => {
                            if self.config.strict_assertions {
                                return Err(RlmError::AssertionFailed {
                                    field: "_root".to_string(),
                                    label,
                                    expression,
                                    value: serde_json::Value::Null,
                                });
                            }
                            prompt = format!(
                                "[SUBMIT failed] Assertion '{}' failed: {}\n\nPlease fix and try again.",
                                label, expression
                            );
                        }
                    }
                } else {
                    // No SUBMIT - continue with output
                    prompt = Python::with_gil(|py| {
                        exec_result.bind(py)
                            .str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|_| "[Error converting result]".to_string())
                    });

                    // Truncate if needed
                    if prompt.len() > self.config.max_output_chars {
                        prompt = format!(
                            "{}... (truncated to {}/{})",
                            &prompt[..self.config.max_output_chars],
                            self.config.max_output_chars,
                            prompt.len()
                        );
                    }
                }
            } else if cmd.get_run_command().is_some() {
                // Bash command - similar handling
                prompt = "[RUN commands not yet supported in TypedRlm]".to_string();
            } else {
                prompt = "[Invalid command. Use ```repl...``` blocks with SUBMIT() to return output.]".to_string();
            }
        }
    }

    // NOTE: rlm_variable_descriptions() is provided by RlmInputFields trait on S::Input
    // The Signature derive macro generates this trait impl for the Input struct

    async fn extraction_fallback(
        &self,
        input: S::Input,
        message_history: &[(String, String)],
        iterations: usize,
        llm_calls: usize,
    ) -> Result<RlmResult<S>, RlmError> {
        // Build extraction prompt from history
        let history_summary: String = message_history
            .iter()
            .enumerate()
            .map(|(i, (_, response))| format!("=== Iteration {} ===\n{}\n", i + 1, response))
            .collect();

        let extraction_prompt = format!(
            "Based on the work so far, extract the final answer.\n\n\
             {}\n\n\
             Provide output as JSON matching the schema:\n{}\n\n\
             Output JSON:",
            history_summary,
            generate_output_schema::<S>()
        );

        let response = self.agent.prompt(&extraction_prompt).await
            .map_err(|e| RlmError::LlmError { source: Box::new(e) })?;

        // Parse with ChatAdapter
        let chat_adapter = ChatAdapter;
        let (output, metas) = chat_adapter.parse_response_typed::<S>(&response)
            .map_err(|e| RlmError::ExtractionFailed { source: e })?;

        let constraint_summary = compute_constraint_summary(&metas);

        Ok(RlmResult {
            input,
            output,
            field_metas: metas,
            iterations,
            llm_calls,
            extraction_fallback: true,
            constraint_summary,
        })
    }
}

fn compute_constraint_summary(metas: &IndexMap<String, FieldMeta>) -> ConstraintSummary {
    let mut summary = ConstraintSummary::default();

    for meta in metas.values() {
        for check in &meta.checks {
            if check.passed {
                if check.level == dspy_rs::ConstraintLevel::Check {
                    summary.checks_passed += 1;
                } else {
                    summary.assertions_passed += 1;
                }
            } else {
                summary.checks_failed += 1;
            }
        }
    }

    summary
}

fn generate_output_schema<S: Signature>() -> String {
    // Similar to SUBMIT schema description
    let mut schema = String::new();
    for field in S::output_fields() {
        schema.push_str(&format!("  \"{}\": ...,\n", field.name));
    }
    format!("{{\n{}}}", schema)
}
```

**Step 2: Commit**

```bash
git add src/typed_rlm.rs
git commit -m "feat: implement TypedRlm core orchestrator"
```

---

### Acceptance criteria (Phase 4)
- `TypedRlm::call(...)` returns `CallResult<S>` with `RlmMeta` (trajectory, iterations, llm_calls, fallback flag).
- Jupyter semantics: stdout is captured, last‑expression repr is shown unless suppressed by `;`, and both appear in output order.
- Truncation matches DSPy: capture‑time `... (truncated)` at `max_output_chars`, and history `... (truncated to X/Y chars)` at `max_history_output_chars`.
- Code fences are stripped with DSPy regex before execution and re‑added in history formatting.
- `max_iterations` triggers extract fallback (or error if disabled).
- `max_llm_calls` raises PyRuntimeError with DSPy message; empty prompt raises ValueError; missing LM raises RuntimeError.

## Phase 5: Prompt Generation

### Task 5.0: DSPy prompt parity + RLMAdapter

**Goal:** Match DSPy’s `ACTION_INSTRUCTIONS_TEMPLATE` (system prompt) and REPL variable preview format using a dedicated `RlmAdapter`.

**Notes:**
- Use DSPy’s template from `tmp/dspy/dspy/predict/rlm.py` as the canonical baseline.
- Keep the **“MINIMIZE RETYPING (INPUTS & OUTPUTS)”** line verbatim.
- Placeholders are *semantic*, not necessarily literal: we can keep `{inputs}`, `{output_fields}`, `{max_llm_calls}` or substitute directly, as long as the resulting prompt matches DSPy wording/structure.
  - `inputs` = comma‑separated backticked variable names (e.g., `` `context`, `query` ``)
  - `output_fields` = formatted output list (name + type + description + constraints)
  - `max_llm_calls` = `RlmConfig.max_llm_calls`
- Input preview should mirror DSPy `REPLVariable.format()` ordering/labels:
  - variable name, type, description/constraints, total length, preview block
  - use `RlmVariable::format()` and adjust formatting to match DSPy where needed.
- **Adapter choice:** add a new `RlmAdapter` (feature‑gated in `dspy-rs`) rather than modifying `ChatAdapter`.
  - Rust has no inheritance, so `RlmAdapter` should *compose* (wrap) any shared helpers or re‑use parsing utilities from existing adapters.
  - `RlmAdapter` is responsible for: system prompt, variable previews, extraction prompt, and RLM‑specific formatting.

---

### Task 5.1: Create preamble generator

**Files:**
- Create: `src/preamble.rs`

**Step 1: Write preamble generator**

```rust
// src/preamble.rs
use dspy_rs::Signature;

/// Generate the typed RLM preamble with variable descriptions and output schema
pub fn generate_typed_preamble<S: Signature>(variable_descriptions: &str) -> String {
    let instruction = S::instruction();
    let output_schema = generate_output_schema_description::<S>();

    format!(r#"You are tasked with a computation that requires structured output.

## Task
{instruction}

## Input Variables
{variable_descriptions}

## Output Schema
Call SUBMIT() with the following fields when you have your answer:

{output_schema}

## Available Commands

1. Python code in ```repl ``` blocks:
   - Access input variables directly by name
   - `llm_query(prompt)` - Query a sub-LLM for semantic analysis (~500K char capacity)
   - `llm_query_batched(prompts)` - Batch query multiple prompts concurrently
   - `SUBMIT(field1=value1, ...)` - Submit your final answer (validates against schema)
   - `print()` - Print intermediate results

2. SUBMIT validates your output and returns:
   - "✓ SUBMIT successful!" on valid output
   - Detailed error messages if validation fails (fix and retry)

## Guidelines

1. EXPLORE FIRST - Examine input variables before processing
2. ITERATE - Write small code snippets, observe outputs, adjust
3. USE llm_query FOR SEMANTICS - String matching finds WHERE; llm_query understands WHAT
4. VERIFY BEFORE SUBMITTING - Check results look sensible
5. SUBMIT when ready - It validates types and constraints automatically

## Constraints

- Soft checks (⚠): Violations are reported but output is accepted
- Hard asserts (❌): Violations require you to fix and resubmit

Example:
```repl
# Explore the data
print(type(trajectories), len(trajectories))
print(trajectories[0])

# Process
results = []
for t in trajectories[:5]:
    analysis = llm_query(f"Summarize: {{t}}")
    results.append(analysis)

# Submit
SUBMIT(summary=results[0], count=len(results))
```
"#)
}

fn generate_output_schema_description<S: Signature>() -> String {
    let mut desc = String::new();
    desc.push_str("SUBMIT(\n");

    for field in S::output_fields() {
        let type_ir = (field.type_ir)();
        let type_name = format_type_for_prompt(&type_ir);

        desc.push_str(&format!("    {}={},", field.name, type_name));

        if !field.description.is_empty() {
            desc.push_str(&format!("  # {}", field.description));
        }
        desc.push('\n');

        // Show constraints
        for constraint in field.constraints {
            let icon = match constraint.kind {
                dspy_rs::ConstraintKind::Check => "⚠",
                dspy_rs::ConstraintKind::Assert => "❌",
            };
            desc.push_str(&format!("        {} {}: {}\n",
                icon, constraint.label, constraint.expression));
        }
    }

    desc.push_str(")\n");
    desc
}

fn format_type_for_prompt(type_ir: &baml_bridge::baml_types::TypeIR) -> String {
    match type_ir {
        baml_bridge::baml_types::TypeIR::String => "\"...\"".to_string(),
        baml_bridge::baml_types::TypeIR::Int => "123".to_string(),
        baml_bridge::baml_types::TypeIR::Float => "1.23".to_string(),
        baml_bridge::baml_types::TypeIR::Bool => "True/False".to_string(),
        baml_bridge::baml_types::TypeIR::List(inner) => {
            format!("[{}, ...]", format_type_for_prompt(inner))
        }
        baml_bridge::baml_types::TypeIR::Optional(inner) => {
            format!("{} or None", format_type_for_prompt(inner))
        }
        baml_bridge::baml_types::TypeIR::Class { name, fields } => {
            if fields.len() <= 3 {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|f| format!("\"{}\": {}", f.name, format_type_for_prompt(&f.r#type)))
                    .collect();
                format!("{{{}}}", field_strs.join(", "))
            } else {
                format!("{{...}}  # {}", name)
            }
        }
        baml_bridge::baml_types::TypeIR::Enum { name, values } => {
            let vals: Vec<&str> = values.iter().take(4).map(|v| v.as_str()).collect();
            if values.len() > 4 {
                format!("\"{}\" | ... ({} options)", vals.join("\" | \""), values.len())
            } else {
                format!("\"{}\"", vals.join("\" | \""))
            }
        }
        _ => "...".to_string(),
    }
}
```

**Step 2: Commit**

```bash
git add src/preamble.rs
git commit -m "feat: implement typed preamble generation"
```

---

### Acceptance criteria (Phase 5)
- Action prompt text matches DSPy `ACTION_INSTRUCTIONS_TEMPLATE` ordering and wording (incl. “MINIMIZE RETYPING” line).
- Action inputs explicitly include `variables_info`, `repl_history`, and `iteration`, and are rendered in the prompt.
- Variable preview format matches the new spec (Count/Usage/Shape/Preview); repr samples are truncated to 100 chars.
- Shape block uses BAML‑style formatting with depth=3, max_fields=20, and `... (+N more)` for truncation.
- REPL history section matches DSPy formatting and truncation behavior exactly.
- Extract‑fallback prompt uses DSPy‑style wording and includes history + variables.

## Phase 6: Integration & Examples

### Task 6.0: Enable rlm feature on dspy-rs (rig-rlm)

**Files:**
- Modify: `Cargo.toml`

**Step 1: Update dependency**

```toml
# Cargo.toml (rig-rlm)
dspy-rs = { path = "/Users/darin/src/personal/DSRs/crates/dspy-rs", features = ["rlm"] }
```

**Step 2: Commit**

```bash
git add Cargo.toml
git commit -m "chore: enable dspy-rs rlm feature"
```

---

### Task 6.1: Create error types

**Files:**
- Create: `src/error.rs`

**Step 1: Write error types**

```rust
// src/error.rs
use dspy_rs::ParseError;

#[derive(Debug, thiserror::Error)]
pub enum RlmError {
    #[error("Max iterations ({max}) reached without valid submission")]
    MaxIterationsReached {
        max: usize,
        partial: Option<serde_json::Value>,
    },

    #[error("Max LLM calls ({max}) exceeded")]
    MaxLlmCallsExceeded { max: usize },

    #[error("Assertion '{label}' failed on field '{field}': {expression}")]
    AssertionFailed {
        field: String,
        label: String,
        expression: String,
        value: serde_json::Value,
    },

    #[error("Extraction fallback failed: {source}")]
    ExtractionFailed {
        #[source]
        source: ParseError,
    },

    #[error("Failed to convert output: {message}")]
    ConversionError { message: String },

    #[error("Python execution error: {message}")]
    PythonError { message: String },

    #[error("LLM error")]
    LlmError {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

impl From<pyo3::PyErr> for RlmError {
    fn from(err: pyo3::PyErr) -> Self {
        RlmError::PythonError {
            message: err.to_string(),
        }
    }
}
```

**Step 2: Commit**

```bash
git add src/error.rs
git commit -m "feat: add RlmError types"
```

---

### Task 6.2: Update main.rs with example

**Files:**
- Modify: `src/main.rs`

**Step 1: Write example usage**

```rust
// src/main.rs
use dspy_rs::rlm_type;
use dspy_rs::Signature;

pub mod error;
pub mod exec;
pub mod llm;
pub mod preamble;
pub mod repl;
pub mod submit;
pub mod tools;
pub mod typed_rlm;

use typed_rlm::{TypedRlm, RlmConfig};

// ========== Define ALL types in Rust with RlmType ==========

/// A trajectory step (simplified - real version has more fields)
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(repr = "Step({self.source}: {self.content:50}...)")]
pub struct Step {
    pub source: String,  // "user" or "agent"
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[rlm_type]
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: String,
}

/// A trajectory - the main input type
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(
    repr = "Trajectory({len(self.steps)} steps, session={self.session_id:12}...)",
    iter = "steps",
    index = "steps",
)]
pub struct Trajectory {
    pub session_id: String,

    #[rlm(desc = "All conversation steps", filter_property = "user_steps", filter_value = "user")]
    pub steps: Vec<Step>,
}

/// A documentation pattern that reduces tool calls
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(repr = "Pattern({self.name}, {len(self.examples)} examples)")]
pub struct Pattern {
    pub id: String,
    pub name: String,
    pub description: String,
    pub documentation_section: String,
    pub example_trajectory_ids: Vec<String>,
    pub examples: Vec<PatternExample>,
    pub estimated_calls_saved: i32,
}

#[rlm_type]
#[derive(Clone, Debug)]
pub struct PatternExample {
    pub trajectory_id: String,
    pub step_range: String,
    pub description: String,
    pub suggested_doc: String,
}

// ========== Standard DSRs Signature ==========

/// Analyze trajectories to identify documentation patterns that would reduce tool calls
#[derive(Signature, Clone, Debug)]
struct AnalyzeTrajectories {
    #[input(desc = "Trajectories to analyze - use .user_steps, .steps[i], len()")]
    trajectories: Vec<Trajectory>,

    #[input(desc = "Existing patterns to update/extend")]
    existing_patterns: Vec<Pattern>,

    #[output]
    #[check("len(this) >= 1", label = "has_patterns")]
    updated_patterns: Vec<Pattern>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Load trajectories (in real code, deserialize from files)
    let trajectories: Vec<Trajectory> = load_trajectories_from_disk();

    // Initial patterns (could be empty or seeded)
    let existing_patterns: Vec<Pattern> = vec![];

    // Create the RLM
    let agent = llm::create_cerebras_agent();
    let config = RlmConfig {
        max_iterations: 30,
        max_llm_calls: 100,
        ..Default::default()
    };
    let rlm = TypedRlm::<AnalyzeTrajectories>::new(agent, config);

    // Run with standard signature input - no builder pattern!
    let input = AnalyzeTrajectoriesInput {
        trajectories,
        existing_patterns,
    };
    let result = rlm.call(input).await?;

    // Use the typed result
    println!("Found {} patterns:", result.output.updated_patterns.len());
    for pattern in &result.output.updated_patterns {
        println!("  - {}: {} examples, ~{} calls saved",
            pattern.name,
            pattern.examples.len(),
            pattern.estimated_calls_saved
        );
    }

    if result.has_constraint_warnings() {
        println!("\nWarnings:");
        for check in result.failed_checks() {
            println!("  - {}: {}", check.label, check.expression);
        }
    }

    println!("\nStats: {} iterations, {} LLM calls, fallback={}",
        result.iterations,
        result.llm_calls,
        result.is_fallback()
    );

    Ok(())
}

fn load_trajectories_from_disk() -> Vec<Trajectory> {
    // Real implementation would deserialize from JSON/msgpack files
    vec![]
}
```

**Step 2: Commit**

```bash
git add src/main.rs
git commit -m "feat: add typed RLM example with trajectory analysis"
```

---

### Acceptance criteria (Phase 6)
- `rig-rlm` builds with `dspy-rs` feature `rlm` enabled; no extra PyO3 deps required in rig‑rlm.
- Example code compiles and uses `TypedRlm::call(...)` with a standard Signature input struct.
- Re‑exports from `dspy-rs` allow `use dspy_rs::rlm_type;` without direct `rlm-derive` dependency.

## Phase 7: Testing & Verification

### Task 7.1: Create integration test

**Files:**
- Create: `tests/typed_rlm_test.rs`

**Step 1: Write integration test**

```rust
// tests/typed_rlm_test.rs
use dspy_rs::rlm_type;
use dspy_rs::Signature;
use rig_rlm::typed_rlm::{TypedRlm, RlmConfig};

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
#[rlm(repr = "Item({self.name})")]
struct Item {
    pub name: String,
    pub value: i32,
}

#[derive(Signature, Clone, Debug)]
/// Sum the values of items
struct SumItems {
    #[input]
    items: Vec<Item>,

    #[output]
    #[assert("this >= 0", label = "non_negative")]
    total: i32,

    #[output]
    summary: String,
}

#[tokio::test]
async fn test_typed_rlm_basic() {
    // This would need a mock LLM for proper testing
    // For now, just verify the types compile and basic structure works

    let items = vec![
        Item { name: "a".to_string(), value: 10 },
        Item { name: "b".to_string(), value: 20 },
    ];

    let config = RlmConfig {
        max_iterations: 5,
        max_llm_calls: 10,
        ..Default::default()
    };

    // Create input using generated Input struct
    let input = SumItemsInput { items };

    // Verify TypedRlm can be created (full test needs mock agent)
    // let agent = create_mock_agent();
    // let rlm = TypedRlm::<SumItems>::new(agent, config);
    // let result = rlm.call(input).await?;
}

#[test]
fn test_rlm_type_derive_repr() {
    let item = Item {
        name: "test".to_string(),
        value: 42,
    };

    // Verify __repr__ works (would need Python runtime)
    // For compile-time, just check the type exists
    let _ = format!("{:?}", item);
}
```

**Step 2: Commit**

```bash
git add tests/typed_rlm_test.rs
git commit -m "test: add typed RLM integration tests"
```

---

### Task 7.2: Verification checklist

**Manual Testing Steps:**

1. **Compile check**
   ```bash
   cargo build
   ```
   Expected: No errors

2. **Run tests**
   ```bash
   cargo test
   ```
   Expected: All tests pass

3. **Test with real LLM** (requires API key)
   ```bash
   CEREBRAS_API_KEY=xxx cargo run
   ```
   Expected: RLM executes, SUBMIT validates output

4. **Verify Python integration**
   ```bash
   cargo run --example python_types
   ```
   Expected: Rust types work correctly in Python REPL

5. **Test constraint validation**
   - Verify soft checks produce warnings but succeed
   - Verify hard asserts fail and provide retry opportunity

---

### Acceptance criteria (Phase 7)
- Tests cover: `#[rlm_type]` pyclass behavior, `__baml__` round‑trip, `__rlm_schema__`, `RlmInputFields`, SUBMIT conversion, and DSPy‑exact error strings.
- REPL history formatting + truncation tests match DSPy’s `REPLEntry.format` and `_format_output` behavior.
- Jupyter semantics tests verify stdout + repr output and semicolon suppression.
- All automated tests pass; manual checklist produces expected outputs and constraint behavior.

## Summary

This plan implements a typed RLM system with:

1. **`#[rlm_type]` / `#[derive(RlmType)]`** macro generating:
   - PyO3 `#[pyclass]` with getters
   - Custom `__repr__` from templates
   - `__iter__`, `__len__`, `__getitem__` for collections
   - Filter properties (`user_steps`) and flatten properties (`all_tool_calls`)
   - `BamlType` for schema generation and parsing (via injection)
   - `RlmDescribe` for rich prompt descriptions

2. **`TypedRlm<S: Signature>`** orchestrator with:
   - **Standard DSRs signature** - input fields ARE the context, no builder pattern
  - `call(input: S::Input)` takes the generated input struct directly
   - Input fields auto-converted to Python variables via RlmType's IntoPyObject
   - REPL loop with LLM querying
   - SUBMIT handling with validation feedback
   - Extraction fallback on max iterations

3. **`SUBMIT(**kwargs)`** PyO3 function with:
   - Schema-guided Python → BamlValue conversion (`__baml__`, dataclass/pydantic/attrs, dict/list)
   - jsonish only for string coercion into non-string expected types
   - optional orjson/json fallback for exotic objects
   - Constraint validation (check/assert)
   - Detailed error messages for retry
   - Field metadata collection

4. **Rich prompt generation** with:
   - Type structure descriptions from RlmDescribe
   - Computed properties listed
   - Iteration/indexing hints
   - Value previews

This gives you DSPy-like RLM functionality but with:
- Compile-time type safety
- Rust-native performance
- Rich Python ergonomics via PyO3
- **All types in Rust** - no native Python type escape hatches
- Constraint validation with clear feedback
- Full metadata on parsing (flags, checks)

---

## RLMAdapter Requirements (Detailed)

**Purpose:** Provide DSPy‑compatible prompt construction and RLM‑specific formatting without changing `ChatAdapter`.

### Responsibilities
- **System prompt**: Render DSPy `ACTION_INSTRUCTIONS_TEMPLATE` (structure + wording) with:
  - `inputs` = comma‑separated backticked input variable names (e.g., `` `issues`, `query` ``)
  - `output_fields` = bullet list of output fields (name + type + description/constraints)
  - `max_llm_calls` = `RlmConfig.max_llm_calls`
- **Iteration tracking (DSPy‑style):**
  - Provide `iteration` as `"{current}/{max_iterations}"` in the action prompt context.
- **Action‑prompt inputs (DSPy‑style):**
  - Explicitly include `variables_info`, `repl_history`, and `iteration` as inputs to the action signature.
  - Adapter/template must render these fields in the prompt (not just side‑channels).
- **Variables preview**: Produce variable descriptions matching DSPy `REPLVariable.format()` ordering/labels:
  - `Variable: \`name\` (access it in your code)`
  - `Type: ...`
  - `Count: ...` (for list/map inputs)
  - `Description: ...` (if present)
  - `Constraints: ...` (if present)
  - `Total length: ... characters`
  - `Preview:` multi‑line samples (repr‑based, truncated to 100 chars)
- **Token‑dense preview (default for large/nested inputs):**
  - include `len(...)` and 1–2 representative item reprs for lists
  - include “usage hints” (e.g., `len(x)`, `x[i]`, `x.user_steps`, `x.all_tool_calls`) derived from `RlmDescribe` + properties
  - include a **Shape** block using BAML‑style formatting (see below)
  - keep ordering obvious and stable so the LLM can reliably parse it
- **Shape block rules (BAML‑style, capped):**
  - format as `TypeName { field: type, ... }` with `//` doc comments (same style as `render_schema`)
  - depth = **3**, max fields = **20** per type
  - if a type has more than 20 fields, append `... (+N more)`
  - if depth is exhausted, render `TypeName { ... }` without expanding fields
  - expansion order = **field order**, deterministic
  - use a visited set to avoid cycles
  - implementation uses `OutputFormatContent` classes/enums + `TypeIR` to render (no jsonish)
- **Leverage BAML/Signature metadata:** use field docstrings, constraints, and formats from `Signature`/`FieldSpec` to enrich:
  - variable `Description:` and `Constraints:` lines
  - `__rlm_schema__` entries (field name → {type, desc, constraints})
  - optional per‑field getter docstrings where PyO3 supports them
- **REPL history**: Render previous steps in DSPy `REPLEntry.format()` style:
  - `=== Step N ===`
  - Optional `Reasoning: ...`
  - `Code:` then fenced block:
    ```python
    <code>
    ```
  - `Output ({len(output):,} chars):` then output text
  - **Truncation rule (DSPy‑exact, uses `RlmConfig.max_history_output_chars`):**
    - If output length > `max_output_chars`, truncate to that length and append:
      `\n... (truncated to {max_output_chars}/{len(original):,} chars)`
  - **Capture‑time formatting (DSPy‑exact):**
    - If result is empty: `"(no output - did you forget to print?)"`
    - If result length > `max_output_chars`: append `\n... (truncated)`
  - **Code fence handling (DSPy‑exact):**
    - Strip markdown fences using DSPy regex: `^```(?:python|py)?\\s*\\n(.*)\\n```\\s*$` (DOTALL)
    - Execute the stripped code; history formatting will re‑add fences.
- **Extraction prompt**: DSPy‑style fallback prompt with:
  - original task instructions
  - REPL history
  - variable previews

### Inputs / Outputs
- **Inputs**: `S: Signature`, `RlmConfig`, `variables: Vec<RlmVariable>`, `history: Vec<(prompt, response)>` (or a structured REPLHistory)
- **Outputs**: `String` (fully rendered prompt) + optional structured metadata (e.g., `RlmPromptMeta`)

### Complex Python Types (behavior)
- **Inputs**: always Rust types with `RlmType` → exposed as PyO3 pyclasses with getters, `__repr__`, `__len__`, `__iter__`, `__getitem__`, computed properties.  
  Variable preview uses `RlmDescribe` + `__repr__` samples + the capped BAML‑style shape formatter for a stable, readable snapshot.
- **Outputs**: SUBMIT accepts arbitrary Python objects; conversion is **schema‑guided**:
  - `__baml__()` (our pyclasses) → JSON‑like object → BamlValue
  - dataclass / pydantic / attrs → dict → recurse
  - dict/list/tuple/primitives → recurse
  - jsonish only for **string** → non‑string coercion
  - optional orjson/json fallback for exotic objects
  - clear error if unsupported

### Fully Rendered Prompt Example (DSPy‑style)
_Example signature:_
```rust
#[derive(Signature)]
struct UpdateIssues {
  #[input] issues: Vec<IssueGraph>,
  #[input] query: String,
  #[output] updated_issues: Vec<IssueGraph>,
  #[output(desc = "Short summary of changes")] summary: String,
}
```

_Rendered system prompt (excerpt, as passed to the LLM):_
```
You are tasked with producing the following outputs given the inputs `issues`, `query`:
- updated_issues: list[IssueGraph]
- summary: string  # Short summary of changes

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: `issues`, `query` (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT(updated_issues, summary)` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max 50 sub-LLM calls. When done, call SUBMIT() with your output.
```

_Variable preview block (example for `issues`):_
```
Variable: `issues` (access it in your code)
Type: list[IssueGraph]
Count: 204
Total length: 12,483 characters
Usage:
  - len(issues), issues[i]
  - issues[i].events / .assignees / .labels
Shape:
  IssueGraph {
    id: string,
    status: "open" | "triaged" | "closed",
    events: list[IssueEvent],
    assignees: list[Assignee],
    labels: list[string],
    ...
  }
  IssueEvent { ... (+N more) }
Preview (2 of 204, repr<=100 chars):
  - IssueGraph(id="ISSUE-1", status="open", events=12, assignees=2, ...)
  - IssueGraph(id="ISSUE-2", status="triaged", events=4, assignees=1, ...)
  - …
```

### Implementation choices
- **Adapter vs template:** keep a dedicated `RlmAdapter` that composes shared utilities; avoids touching `ChatAdapter`.
- **No `call_with_meta`:** `TypedRlm::call(...)` returns `CallResult<S>` with `RlmMeta` (trajectory, iterations, warnings).
- **Error UX:** missing fields / type coercion / asserts return DSPy‑style error strings, augmented with richer info when available.
- **SUBMIT signature (DSPy‑aligned, keyword‑friendly):**
  - Define SUBMIT with named parameters matching output field names (e.g., `def SUBMIT(a, b): ...`).
  - Encourage keyword usage (`SUBMIT(a=a, b=b)`), but allow positional to match DSPy examples.
- **Custom tools (v1):**
  - Allow user‑provided tools with signature/docs appended to the prompt (DSPy style).
  - Not required for v0; defer to v1.
- **Error UX (DSPy‑exact prefixes):**
  - Missing fields: `[Error] Missing output fields: [..]. Use SUBMIT(a, b, ...)`
  - Wrong FINAL type: `[Error] FINAL returned <type>, expected dict with fields: [...]`
  - Type errors: `[Type Error] field: expected <T>, got <U>: <reason>`
  - Interpreter/runtime errors: `[Error] <ErrorType>: <message>`
  - LLM call limit: `[Error] RuntimeError: LLM call limit exceeded: {count} + {n} > {max}. Use Python code for aggregation instead of making more LLM calls.`
  - Empty prompt in llm_query: `[Error] ValueError: prompt cannot be empty`
  - Missing LM: `[Error] RuntimeError: No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM.`
  - Tool bridge failures: `[Error] RuntimeError: Tool bridge error for '<name>': <message>`
  - We can append richer details after these prefixes (new line), but **prefix + first sentence must match DSPy**.
