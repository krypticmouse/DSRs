//! `#[module]` (RFC 0003 M-3): compiles an ordinary async fn body into an IR
//! program — one parse, two projections (executable fn + `Program` constant),
//! so drift between code and graph is impossible by construction.
//!
//! M-3 supports straight-line bodies:
//!
//! - `let x = step(args).await?;` where `step` is a `#[predict]`/`#[cot]`/
//!   `#[agent]` fn — a leaf node named `x`. Args are ports: fn params, prior
//!   `binding.field` accesses, literals (`.clone()`/`&` stripped).
//! - `let y: SimpleType = <any Rust expr>;` — an extracted **host hole**:
//!   port-shaped subexpressions become typed inputs, the residual runs as
//!   native code bound by name at load, and the artifact records it as
//!   `extern "<hash>"`. Reported in `OPACITY`; refused under `deny_holes`.
//! - `Ok(Struct { field: port, … })` tail — the program's output bindings.
//!
//! Types are never guessed here: the emitted `ModuleSpec` carries structure
//! only, and `dspy_rs::ir::build_module_program` resolves every field type
//! from the step signatures at first use.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::collections::{HashMap, HashSet};
use syn::punctuated::Punctuated;
use syn::visit_mut::VisitMut;
use syn::{Expr, Meta, Token};

use crate::runtime_path::resolve_dspy_rs_path;
use crate::step_support::{fnv1a64, map_simple_type};

pub(crate) fn expand_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };
    match expand_module_inner(attr.into(), &func, &runtime) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

// ---------------------------------------------------------------------------
// Attrs
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ModuleAttrs {
    caps: Vec<String>,
    deny_holes: bool,
}

fn parse_module_attr(attr: TokenStream2) -> syn::Result<ModuleAttrs> {
    let mut out = ModuleAttrs::default();
    if attr.is_empty() {
        return Ok(out);
    }
    let metas: Punctuated<Meta, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr)?;
    for meta in metas {
        match &meta {
            Meta::List(list) if list.path.is_ident("caps") => {
                let strings: Punctuated<syn::LitStr, Token![,]> =
                    list.parse_args_with(Punctuated::parse_terminated)?;
                out.caps.extend(strings.iter().map(syn::LitStr::value));
            }
            Meta::Path(path) if path.is_ident("deny_holes") => out.deny_holes = true,
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "#[module] accepts `caps(\"…\", …)` and `deny_holes`",
                ));
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Ports
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum PortTok {
    Input(String),
    Out { node: String, field: String },
    Lit(syn::Lit),
}

impl PortTok {
    fn key(&self) -> String {
        match self {
            PortTok::Input(name) => format!("$.{name}"),
            PortTok::Out { node, field } => format!("{node}.{field}"),
            PortTok::Lit(lit) => format!("lit:{}", quote!(#lit)),
        }
    }

    fn tokens(&self, runtime: &syn::Path) -> TokenStream2 {
        match self {
            PortTok::Input(name) => quote! { #runtime::ir::PortSpec::Input(#name) },
            PortTok::Out { node, field } => {
                quote! { #runtime::ir::PortSpec::Out { node: #node, field: #field } }
            }
            PortTok::Lit(lit) => quote! {
                #runtime::ir::PortSpec::Lit(#runtime::__macro_support::serde_json::json!(#lit))
            },
        }
    }
}

struct Ctx {
    params: Vec<String>,
    step_bindings: HashSet<String>,
    hole_bindings: HashSet<String>,
}

impl Ctx {
    fn is_binding(&self, name: &str) -> bool {
        self.step_bindings.contains(name) || self.hole_bindings.contains(name)
    }
}

/// Strips `&` and `.clone()` wrappers.
fn peel(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(r) => peel(&r.expr),
        Expr::MethodCall(m) if m.method == "clone" && m.args.is_empty() => peel(&m.receiver),
        Expr::Paren(p) => peel(&p.expr),
        _ => expr,
    }
}

fn single_ident(expr: &Expr) -> Option<String> {
    let Expr::Path(p) = expr else { return None };
    if p.qself.is_some() || p.path.segments.len() != 1 {
        return None;
    }
    Some(p.path.segments[0].ident.to_string())
}

/// A port in argument/tail position; hard error when the expression is not
/// port-shaped (the fix is extracting it to an ascribed `let` — a hole).
fn as_port(expr: &Expr, ctx: &Ctx) -> syn::Result<PortTok> {
    let peeled = peel(expr);
    if let Some(name) = single_ident(peeled) {
        if ctx.params.contains(&name) {
            return Ok(PortTok::Input(name));
        }
        if ctx.hole_bindings.contains(&name) {
            return Ok(PortTok::Out {
                node: name.clone(),
                field: name,
            });
        }
        if ctx.step_bindings.contains(&name) {
            return Err(syn::Error::new_spanned(
                expr,
                format!("reference an output field of `{name}` (e.g. `{name}.answer`)"),
            ));
        }
        return Err(syn::Error::new_spanned(
            expr,
            format!("`{name}` is not a fn param or a prior step binding"),
        ));
    }
    if let Expr::Field(f) = peeled
        && let Some(base) = single_ident(peel(&f.base))
        && let syn::Member::Named(field) = &f.member
    {
        if ctx.is_binding(&base) {
            return Ok(PortTok::Out {
                node: base,
                field: field.to_string(),
            });
        }
        if ctx.params.contains(&base) {
            return Err(syn::Error::new_spanned(
                expr,
                format!("params are scalar inputs — pass `{base}` directly"),
            ));
        }
    }
    if let Expr::Lit(l) = peeled {
        match &l.lit {
            syn::Lit::Str(_) | syn::Lit::Int(_) | syn::Lit::Float(_) | syn::Lit::Bool(_) => {
                return Ok(PortTok::Lit(l.lit.clone()));
            }
            _ => {}
        }
    }
    Err(syn::Error::new_spanned(
        expr,
        "not a port expression (fn param, `binding.field`, or literal); \
         extract computed values into a typed `let` (which becomes a hole)",
    ))
}

// ---------------------------------------------------------------------------
// Hole extraction
// ---------------------------------------------------------------------------

struct Extractor<'a> {
    ctx: &'a Ctx,
    /// (input field name, port, replacement ident) in extraction order.
    inputs: Vec<(String, PortTok, syn::Ident)>,
    seen: HashMap<String, usize>,
    used_names: HashSet<String>,
    errors: Vec<syn::Error>,
}

impl Extractor<'_> {
    fn field_name(&mut self, base: &str) -> String {
        let mut name = base.to_string();
        let mut n = 1;
        while !self.used_names.insert(name.clone()) {
            name = format!("{base}_{n}");
            n += 1;
        }
        name
    }

    fn replace(&mut self, port: PortTok, expr: &mut Expr) {
        let key = port.key();
        if let Some(&index) = self.seen.get(&key) {
            let ident = self.inputs[index].2.clone();
            *expr = syn::parse_quote!(#ident);
            return;
        }
        let base = match &port {
            PortTok::Input(name) => name.clone(),
            PortTok::Out { field, .. } => field.clone(),
            PortTok::Lit(_) => unreachable!("literals are never extracted"),
        };
        let field = self.field_name(&base);
        let ident = format_ident!("__dsrs_p{}", self.inputs.len());
        self.seen.insert(key, self.inputs.len());
        self.inputs.push((field, port, ident.clone()));
        *expr = syn::parse_quote!(#ident);
    }
}

impl VisitMut for Extractor<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // `binding.field` first — must win over the bare-ident case below.
        if let Expr::Field(f) = expr
            && let Some(base) = single_ident(peel(&f.base))
            && let syn::Member::Named(field) = &f.member
            && self.ctx.is_binding(&base)
        {
            let port = PortTok::Out {
                node: base,
                field: field.to_string(),
            };
            self.replace(port, expr);
            return;
        }
        if let Some(name) = single_ident(expr) {
            if self.ctx.params.contains(&name) {
                self.replace(PortTok::Input(name), expr);
                return;
            }
            if self.ctx.hole_bindings.contains(&name) {
                self.replace(
                    PortTok::Out {
                        node: name.clone(),
                        field: name,
                    },
                    expr,
                );
                return;
            }
            if self.ctx.step_bindings.contains(&name) {
                self.errors.push(syn::Error::new_spanned(
                    &*expr,
                    format!(
                        "holes consume output fields, not whole step results — \
                         use `{name}.<field>`"
                    ),
                ));
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }
}

// ---------------------------------------------------------------------------
// Body model
// ---------------------------------------------------------------------------

struct CallStep {
    binding: String,
    callee: syn::Path,
    args: Vec<PortTok>,
}

struct HoleStep {
    binding: String,
    output_ty: syn::Type,
    output_ft: TokenStream2,
    inputs: Vec<(String, PortTok, syn::Ident)>,
    residual: Expr,
    excerpt: String,
    hash: u64,
}

enum StepModel {
    Call(CallStep),
    Hole(Box<HoleStep>),
}

/// `<path>(<args>).await?` — the step-call shape.
fn as_step_call(expr: &Expr) -> Option<(&syn::Path, &Punctuated<Expr, Token![,]>)> {
    let Expr::Try(t) = expr else { return None };
    let Expr::Await(a) = &*t.expr else { return None };
    let Expr::Call(c) = &*a.base else { return None };
    let Expr::Path(p) = &*c.func else { return None };
    if p.qself.is_some() {
        return None;
    }
    Some((&p.path, &c.args))
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

fn expand_module_inner(
    attr: TokenStream2,
    func: &syn::ItemFn,
    runtime: &syn::Path,
) -> syn::Result<TokenStream2> {
    let attrs = parse_module_attr(attr)?;
    let sig = &func.sig;
    if sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(sig, "#[module] functions are async"));
    }
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "#[module] functions cannot be generic",
        ));
    }

    let fn_name = &sig.ident;
    let fn_name_str = fn_name.to_string();
    let vis = &func.vis;
    let doc_attrs: Vec<&syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .collect();

    // Params.
    let mut param_idents = Vec::new();
    let mut param_types = Vec::new();
    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(ident) => {
                    param_idents.push(ident.ident.clone());
                    param_types.push(pat_type.ty.as_ref().clone());
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "#[module] parameters must be plain identifiers",
                    ));
                }
            },
            syn::FnArg::Receiver(receiver) => {
                return Err(syn::Error::new_spanned(
                    receiver,
                    "#[module] functions cannot take self",
                ));
            }
        }
    }

    // Return type: literally `Result<Out, Err>` with `Err: From<RunError>`.
    let (tout, terr) = match &sig.output {
        syn::ReturnType::Type(_, ty) => result_types(ty).ok_or_else(|| {
            syn::Error::new_spanned(
                ty,
                "#[module] functions return `Result<Out, Err>` \
                 (Err: From<dspy_rs::ir::RunError>)",
            )
        })?,
        syn::ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                sig,
                "#[module] functions return `Result<Out, Err>`",
            ));
        }
    };

    // ---- classify the body ------------------------------------------------
    let mut ctx = Ctx {
        params: param_idents.iter().map(ToString::to_string).collect(),
        step_bindings: HashSet::new(),
        hole_bindings: HashSet::new(),
    };
    let mut steps: Vec<StepModel> = Vec::new();

    let stmts = &func.block.stmts;
    let Some((tail, body)) = stmts.split_last() else {
        return Err(syn::Error::new_spanned(
            &func.block,
            "#[module] bodies need at least a tail expression",
        ));
    };

    for stmt in body {
        let syn::Stmt::Local(local) = stmt else {
            return Err(syn::Error::new_spanned(
                stmt,
                "#[module] bodies are `let` statements followed by a tail \
                 (M-3 supports straight-line flow only)",
            ));
        };
        let Some(init) = &local.init else {
            return Err(syn::Error::new_spanned(local, "`let` needs an initializer"));
        };
        if init.diverge.is_some() {
            return Err(syn::Error::new_spanned(local, "`let … else` is not supported"));
        }

        // Binding: `ident` (step) or `ident: Type` (hole ascription).
        let (binding_ident, ascription) = match &local.pat {
            syn::Pat::Ident(p) => (p.ident.clone(), None),
            syn::Pat::Type(t) => match t.pat.as_ref() {
                syn::Pat::Ident(p) => (p.ident.clone(), Some(t.ty.as_ref().clone())),
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "#[module] bindings must be plain identifiers",
                    ));
                }
            },
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "#[module] bindings must be plain identifiers",
                ));
            }
        };
        let binding = binding_ident.to_string();
        if ctx.params.contains(&binding) || ctx.is_binding(&binding) {
            return Err(syn::Error::new_spanned(
                &binding_ident,
                format!("`{binding}` shadows a param or earlier binding"),
            ));
        }

        if let Some((path, args)) = as_step_call(&init.expr) {
            let mut ports = Vec::with_capacity(args.len());
            for arg in args {
                ports.push(as_port(arg, &ctx)?);
            }
            steps.push(StepModel::Call(CallStep {
                binding: binding.clone(),
                callee: path.clone(),
                args: ports,
            }));
            ctx.step_bindings.insert(binding);
            continue;
        }

        // Not a step call → a hole.
        let init_expr = &init.expr;
        let excerpt = quote!(#init_expr).to_string();
        if attrs.deny_holes {
            return Err(syn::Error::new_spanned(
                &init.expr,
                format!(
                    "deny_holes: `{binding}` is not a step call (`step(args).await?`); \
                     expression: `{excerpt}`"
                ),
            ));
        }
        let Some(output_ty) = ascription else {
            return Err(syn::Error::new_spanned(
                local,
                format!(
                    "`{binding}` is not a step call, so it becomes a typed hole — \
                     ascribe its output type: `let {binding}: <SimpleType> = …;` \
                     (String, i64, f64, bool, Vec<…>, Option<…>)"
                ),
            ));
        };
        let Some(output_ft) = map_simple_type(&output_ty, runtime) else {
            return Err(syn::Error::new_spanned(
                &output_ty,
                "hole output types must be simple: String, i64/i32, f64/f32, \
                 bool, Vec<…>, Option<…>",
            ));
        };

        let mut residual = init.expr.as_ref().clone();
        let mut extractor = Extractor {
            ctx: &ctx,
            inputs: Vec::new(),
            seen: HashMap::new(),
            used_names: HashSet::new(),
            errors: Vec::new(),
        };
        extractor.visit_expr_mut(&mut residual);
        if let Some(err) = extractor.errors.into_iter().next() {
            return Err(err);
        }
        let hash = fnv1a64(format!("{excerpt} -> {}", quote!(#output_ty)).as_bytes());
        steps.push(StepModel::Hole(Box::new(HoleStep {
            binding: binding.clone(),
            output_ty,
            output_ft,
            inputs: extractor.inputs,
            residual,
            excerpt,
            hash,
        })));
        ctx.hole_bindings.insert(binding);
    }

    // Tail: `Ok(Struct { field: port, … })`.
    let syn::Stmt::Expr(tail_expr, None) = tail else {
        return Err(syn::Error::new_spanned(
            tail,
            "#[module] bodies end with a tail expression `Ok(Struct { … })`",
        ));
    };
    let outs = parse_tail(tail_expr, &ctx)?;

    // ---- emission ---------------------------------------------------------
    let sj = quote! { #runtime::__macro_support::serde_json };
    let caps = &attrs.caps;

    let param_entries: Vec<TokenStream2> = param_idents
        .iter()
        .zip(&param_types)
        .map(|(ident, ty)| {
            let name = ident.to_string();
            let ft = match map_simple_type(ty, runtime) {
                Some(ft) => quote! { ::core::option::Option::Some(#ft) },
                None => quote! { ::core::option::Option::None },
            };
            quote! { (#name, #ft) }
        })
        .collect();

    let step_entries: Vec<TokenStream2> = steps
        .iter()
        .map(|step| match step {
            StepModel::Call(call) => {
                let binding = &call.binding;
                let callee = &call.callee;
                let args: Vec<TokenStream2> =
                    call.args.iter().map(|p| p.tokens(runtime)).collect();
                quote! {
                    #runtime::ir::ModuleStep {
                        binding: #binding,
                        kind: #runtime::ir::ModuleStepKind::Call {
                            def: super::#callee::__dsrs_step(),
                            args: ::std::vec![#(#args),*],
                        },
                    }
                }
            }
            StepModel::Hole(hole) => {
                let binding = &hole.binding;
                let hash = proc_macro2::Literal::u64_suffixed(hole.hash);
                let output = &hole.output_ft;
                let inputs: Vec<TokenStream2> = hole
                    .inputs
                    .iter()
                    .map(|(field, port, _)| {
                        let port = port.tokens(runtime);
                        quote! { (#field.to_string(), #port) }
                    })
                    .collect();
                quote! {
                    #runtime::ir::ModuleStep {
                        binding: #binding,
                        kind: #runtime::ir::ModuleStepKind::Hole {
                            hash: #hash,
                            output: #output,
                            inputs: ::std::vec![#(#inputs),*],
                        },
                    }
                }
            }
        })
        .collect();

    let out_entries: Vec<TokenStream2> = outs
        .iter()
        .map(|(name, port)| {
            let port = port.tokens(runtime);
            quote! { (#name, #port) }
        })
        .collect();

    let opacity_entries: Vec<TokenStream2> = steps
        .iter()
        .filter_map(|step| match step {
            StepModel::Hole(hole) => {
                let name = &hole.binding;
                let excerpt = &hole.excerpt;
                Some(quote! {
                    #runtime::ir::HoleReport {
                        name: #name,
                        kind: "host",
                        excerpt: #excerpt,
                        reason: "not a step call (`step(args).await?`)",
                    }
                })
            }
            StepModel::Call(_) => None,
        })
        .collect();

    let hole_bindings: Vec<TokenStream2> = steps
        .iter()
        .filter_map(|step| {
            let StepModel::Hole(hole) = step else {
                return None;
            };
            let binding = &hole.binding;
            let residual = &hole.residual;
            let output_ty = &hole.output_ty;
            let extractions: Vec<TokenStream2> = hole
                .inputs
                .iter()
                .map(|(field, _, ident)| {
                    quote! {
                        let #ident = #sj::from_value(
                            __input.remove(#field).unwrap_or(#sj::Value::Null),
                        )
                        .map_err(|e| ::std::format!(
                            "hole `{}` input `{}`: {}", #binding, #field, e
                        ))?;
                    }
                })
                .collect();
            Some(quote! {
                env = env.bind_host_hole(#binding, |mut __input: #runtime::trace::JsonMap| async move {
                    #(#extractions)*
                    let __out: #output_ty = #residual;
                    let __value = #sj::to_value(&__out).map_err(|e| e.to_string())?;
                    ::core::result::Result::Ok(#sj::json!({ #binding: __value }))
                });
            })
        })
        .collect();

    let validation_mod = format_ident!("__dsrs_module_validation_{}", fn_name);
    let param_strs: Vec<String> = param_idents.iter().map(ToString::to_string).collect();
    let run_err = quote! { #runtime::ir::RunError };
    let to_terr = quote! { <#terr as ::core::convert::From<#run_err>>::from };

    Ok(quote! {
        #vis mod #fn_name {
            #![allow(unused_imports)]
            use super::*;

            /// Hole-ized expressions in this module's body (RFC 0003 §6).
            pub const OPACITY: &[#runtime::ir::HoleReport] = &[#(#opacity_entries),*];

            #[doc(hidden)]
            pub fn __spec() -> #runtime::ir::ModuleSpec {
                #runtime::ir::ModuleSpec {
                    name: #fn_name_str,
                    caps: &[#(#caps),*],
                    params: ::std::vec![#(#param_entries),*],
                    steps: ::std::vec![#(#step_entries),*],
                    outs: ::std::vec![#(#out_entries),*],
                }
            }

            /// The lowered program, linked at first use.
            pub fn try_program() -> ::core::result::Result<
                &'static #runtime::ir::Program,
                &'static #runtime::ir::ModuleBuildError,
            > {
                static PROGRAM: ::std::sync::LazyLock<
                    ::core::result::Result<#runtime::ir::Program, #runtime::ir::ModuleBuildError>,
                > = ::std::sync::LazyLock::new(|| {
                    #runtime::ir::build_module_program(__spec())
                });
                PROGRAM.as_ref()
            }

            /// The lowered program. Panics on link errors; see [`try_program`].
            pub fn program() -> &'static #runtime::ir::Program {
                match try_program() {
                    ::core::result::Result::Ok(p) => p,
                    ::core::result::Result::Err(e) => {
                        ::std::panic!("#[module] `{}` failed to link: {e}", #fn_name_str)
                    }
                }
            }

            /// The runtime environment this module needs: extracted host
            /// holes and `#[tool]` implementations bound, and the `default`
            /// model bound from the global settings when configured. Extend
            /// it (named models, sandbox) and load manually for custom
            /// serving.
            pub fn env() -> #runtime::ir::RuntimeEnv {
                let mut env = #runtime::ir::RuntimeEnv::new();
                // Self-authorization (RFC 0003 §5): the author wrote the
                // native code — the module's declared ceiling is granted.
                // Serving the printed artifact elsewhere re-checks grants.
                for cap in __spec().caps {
                    env = env.grant(cap);
                }
                if let ::core::option::Option::Some(lm) = #runtime::ir::default_lm() {
                    env = env.bind_model("default", lm);
                }
                for step in __spec().steps {
                    if let #runtime::ir::ModuleStepKind::Call { def, .. } = step.kind {
                        if let ::core::option::Option::Some(agent) = def.agent {
                            for tool in agent.tools {
                                env = env.bind_host_tool(tool.name, tool.dyn_tool);
                            }
                        }
                    }
                }
                #(#hole_bindings)*
                env
            }

            #[doc(hidden)]
            pub async fn __try_interp()
            -> ::core::result::Result<&'static #runtime::ir::Interpreter, #runtime::ir::RunError>
            {
                static CELL: #runtime::__macro_support::tokio::sync::OnceCell<
                    #runtime::ir::Interpreter,
                > = #runtime::__macro_support::tokio::sync::OnceCell::const_new();
                CELL.get_or_try_init(|| async {
                    let program = try_program()
                        .map_err(|e| #runtime::ir::RunError::Internal {
                            at: #fn_name_str.into(),
                            message: e.to_string(),
                        })?
                        .clone();
                    #runtime::ir::Interpreter::load(program, env()).await.map_err(|e| {
                        #runtime::ir::RunError::Internal {
                            at: #fn_name_str.into(),
                            message: e.to_string(),
                        }
                    })
                })
                .await
            }
        }

        #(#doc_attrs)*
        #vis async fn #fn_name(
            #(#param_idents: #param_types),*
        ) -> ::core::result::Result<#tout, #terr> {
            let __interp = #fn_name::__try_interp().await.map_err(#to_terr)?;
            let mut __input = #runtime::trace::JsonMap::new();
            #(
                __input.insert(
                    #param_strs.to_string(),
                    #sj::to_value(&#param_idents).map_err(|e| #to_terr(
                        #runtime::ir::RunError::Internal {
                            at: #fn_name_str.into(),
                            message: ::std::format!("input encode: {e}"),
                        }
                    ))?,
                );
            )*
            let __overlay = #runtime::ir::current_overlay();
            let __out = __interp
                .run(__input, __overlay, #runtime::ir::Budget::default())
                .await
                .map_err(#to_terr)?;
            #sj::from_value::<#tout>(#sj::Value::Object(__out)).map_err(|e| #to_terr(
                #runtime::ir::RunError::Internal {
                    at: #fn_name_str.into(),
                    message: ::std::format!("output decode: {e}"),
                }
            ))
        }

        #[cfg(test)]
        mod #validation_mod {
            #[test]
            fn module_program_links_and_validates() {
                for hole in super::#fn_name::OPACITY {
                    ::std::eprintln!(
                        "[opacity] {}.{} ({}): {} — `{}`",
                        #fn_name_str, hole.name, hole.kind, hole.reason, hole.excerpt
                    );
                }
                if let ::core::result::Result::Err(e) = super::#fn_name::try_program() {
                    ::std::panic!("#[module] `{}` failed to link: {e}", #fn_name_str);
                }
            }
        }
    })
}

/// `Result<Out, Err>` (literal, by final segment) → `(Out, Err)`.
fn result_types(ty: &syn::Type) -> Option<(syn::Type, syn::Type)> {
    let syn::Type::Path(tp) = ty else { return None };
    let segment = tp.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    });
    Some((types.next()?, types.next()?))
}

/// `Ok(Struct { field: port, … })` → out bindings.
fn parse_tail(expr: &Expr, ctx: &Ctx) -> syn::Result<Vec<(String, PortTok)>> {
    let err = || {
        syn::Error::new_spanned(
            expr,
            "#[module] tails are `Ok(Struct { field: <port>, … })` — every \
             field a fn param, `binding.field`, or literal",
        )
    };
    let Expr::Call(call) = expr else { return Err(err()) };
    if single_ident(&call.func).as_deref() != Some("Ok") || call.args.len() != 1 {
        return Err(err());
    }
    let Expr::Struct(structure) = &call.args[0] else {
        return Err(err());
    };
    if structure.rest.is_some() {
        return Err(syn::Error::new_spanned(
            structure,
            "struct-update syntax (`..`) is not supported in #[module] tails",
        ));
    }
    let mut outs = Vec::with_capacity(structure.fields.len());
    for field in &structure.fields {
        let syn::Member::Named(name) = &field.member else {
            return Err(err());
        };
        outs.push((name.to_string(), as_port(&field.expr, ctx)?));
    }
    Ok(outs)
}
