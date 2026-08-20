//! `#[tool]` and `#[agent]` (RFC 0003 M-2).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{Expr, ExprLit, Lit, Meta, Token};

use crate::runtime_path::resolve_dspy_rs_path;
use crate::step_support::{doc_string, model_ref_value, option_str_tokens};

// ---------------------------------------------------------------------------
// #[tool]
// ---------------------------------------------------------------------------

pub(crate) fn expand_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ItemFn);
    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };
    match expand_tool_inner(attr.into(), &func, &runtime) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn parse_tool_attr(attr: TokenStream2) -> syn::Result<Vec<String>> {
    let mut caps = Vec::new();
    if attr.is_empty() {
        return Ok(caps);
    }
    let metas: Punctuated<Meta, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr)?;
    for meta in metas {
        match &meta {
            Meta::List(list) if list.path.is_ident("caps") => {
                let strings: Punctuated<syn::LitStr, Token![,]> =
                    list.parse_args_with(Punctuated::parse_terminated)?;
                caps.extend(strings.iter().map(syn::LitStr::value));
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "#[tool] accepts only `caps(\"…\", …)`",
                ));
            }
        }
    }
    Ok(caps)
}

fn expand_tool_inner(
    attr: TokenStream2,
    func: &syn::ItemFn,
    runtime: &syn::Path,
) -> syn::Result<TokenStream2> {
    let caps = parse_tool_attr(attr)?;
    let sig = &func.sig;
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "#[tool] functions cannot be generic",
        ));
    }

    let fn_name = &sig.ident;
    let fn_name_str = fn_name.to_string();
    let vis = &func.vis;
    let desc = doc_string(&func.attrs);
    let doc_attrs: Vec<&syn::Attribute> = func
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .collect();

    let mut arg_names = Vec::new();
    let mut arg_types = Vec::new();
    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(ident) => {
                    arg_names.push(ident.ident.clone());
                    arg_types.push(pat_type.ty.as_ref().clone());
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "#[tool] parameters must be plain identifiers",
                    ));
                }
            },
            syn::FnArg::Receiver(receiver) => {
                return Err(syn::Error::new_spanned(
                    receiver,
                    "#[tool] functions cannot take self",
                ));
            }
        }
    }
    if arg_names.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.inputs,
            "#[tool] functions need at least one parameter (the input schema)",
        ));
    }

    let return_ty = match &sig.output {
        syn::ReturnType::Type(_, ty) => ty.as_ref().clone(),
        syn::ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                sig,
                "#[tool] functions need an explicit return type (the output field)",
            ));
        }
    };

    // `Result<T, E>` (fallible) or plain `T`.
    let (output_type, fallible) = match result_ok_type(&return_ty) {
        Some(ok) => (ok, true),
        None => (return_ty.clone(), false),
    };

    let invoke = if sig.asyncness.is_some() {
        quote! { super::#fn_name(#(#arg_names),*).await }
    } else {
        quote! { super::#fn_name(#(#arg_names),*) }
    };
    let call_body = if fallible {
        quote! { #invoke.map_err(|e| __ToolError(::std::format!("{e}"))) }
    } else {
        quote! { ::core::result::Result::Ok(#invoke) }
    };

    Ok(quote! {
        #func

        #vis mod #fn_name {
            #![allow(non_camel_case_types, unused_imports)]
            use super::*;

            #(#doc_attrs)*
            #[derive(#runtime::Signature, Clone, Debug)]
            pub struct Sig {
                #(
                    #[input]
                    pub #arg_names: #arg_types,
                )*
                #[output]
                pub #fn_name: #output_type,
            }

            #[derive(Debug)]
            pub struct __ToolError(pub ::std::string::String);
            impl ::std::fmt::Display for __ToolError {
                fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                    ::std::write!(f, "{}", self.0)
                }
            }
            impl ::std::error::Error for __ToolError {}

            /// The host implementation as a rig tool — what `RuntimeEnv`
            /// binds for `ToolKind::Host`.
            #[derive(Clone, Copy)]
            pub struct __HostTool;

            impl #runtime::__macro_support::rig::tool::Tool for __HostTool {
                const NAME: &'static str = #fn_name_str;
                type Error = __ToolError;
                type Args = SigInput;
                type Output = #output_type;

                async fn definition(
                    &self,
                    _prompt: ::std::string::String,
                ) -> #runtime::__macro_support::rig::completion::ToolDefinition {
                    #runtime::__macro_support::rig::completion::ToolDefinition {
                        name: #fn_name_str.to_string(),
                        description: #desc.to_string(),
                        parameters: #runtime::ir::input_schema_of(
                            #runtime::ir::SignatureDef::of::<Sig>(),
                            #runtime::ir::SignatureDef::types_of::<Sig>(),
                        ),
                    }
                }

                async fn call(
                    &self,
                    args: Self::Args,
                ) -> ::core::result::Result<Self::Output, Self::Error> {
                    let SigInput { #(#arg_names),* } = args;
                    #call_body
                }
            }

            /// RFC 0003 M-2: this tool as data.
            pub fn __dsrs_tool() -> #runtime::ir::ToolStepDef {
                #runtime::ir::ToolStepDef {
                    name: #fn_name_str,
                    desc: #desc,
                    caps: &[#(#caps),*],
                    sig: #runtime::ir::SignatureDef::of::<Sig>(),
                    types: #runtime::ir::SignatureDef::types_of::<Sig>(),
                    dyn_tool: ::std::sync::Arc::new(__HostTool),
                }
            }
        }
    })
}

/// `Result<T, …>` (by final path segment) → `Some(T)`.
fn result_ok_type(ty: &syn::Type) -> Option<syn::Type> {
    let syn::Type::Path(tp) = ty else { return None };
    let segment = tp.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    args.args.iter().find_map(|arg| match arg {
        syn::GenericArgument::Type(t) => Some(t.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// #[agent]
// ---------------------------------------------------------------------------

#[derive(Default)]
struct AgentAttrs {
    model: Option<String>,
    tools: Vec<syn::Path>,
    stop_tools: Vec<String>,
    max_turns: Option<u32>,
    until_parse: Option<bool>,
    budget_calls: Option<u32>,
    budget_tokens: Option<u64>,
    budget_deadline_ms: Option<u64>,
    budget_finalize: bool,
    ctx_history: Option<u32>,
    ctx_result_bytes: Option<u32>,
    ctx_playbook: Option<String>,
}

fn int_value<T: std::str::FromStr>(value: &Expr) -> syn::Result<T> {
    if let Expr::Lit(ExprLit {
        lit: Lit::Int(int), ..
    }) = value
        && let Ok(parsed) = int.base10_parse::<u64>()
        && let Ok(parsed) = parsed.to_string().parse::<T>()
    {
        return Ok(parsed);
    }
    Err(syn::Error::new_spanned(value, "expected an integer literal"))
}

fn parse_agent_attr(attr: TokenStream2) -> syn::Result<AgentAttrs> {
    let mut out = AgentAttrs::default();
    if attr.is_empty() {
        return Ok(out);
    }
    let metas: Punctuated<Meta, Token![,]> =
        syn::parse::Parser::parse2(Punctuated::parse_terminated, attr)?;
    for meta in metas {
        match &meta {
            Meta::NameValue(nv) if nv.path.is_ident("model") => {
                out.model = Some(model_ref_value(&nv.value)?);
            }
            Meta::NameValue(nv) if nv.path.is_ident("max_turns") => {
                let turns: u32 = int_value(&nv.value)?;
                if turns == 0 {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`max_turns` must be > 0 (the loop is mandatory and bounded)",
                    ));
                }
                out.max_turns = Some(turns);
            }
            Meta::NameValue(nv) if nv.path.is_ident("until_parse") => {
                let Expr::Lit(ExprLit {
                    lit: Lit::Bool(b), ..
                }) = &nv.value
                else {
                    return Err(syn::Error::new_spanned(&nv.value, "expected true/false"));
                };
                out.until_parse = Some(b.value);
            }
            Meta::List(list) if list.path.is_ident("tools") => {
                let paths: Punctuated<syn::Path, Token![,]> =
                    list.parse_args_with(Punctuated::parse_terminated)?;
                out.tools.extend(paths);
            }
            Meta::List(list) if list.path.is_ident("stop_tools") => {
                let idents: Punctuated<syn::Ident, Token![,]> =
                    list.parse_args_with(Punctuated::parse_terminated)?;
                out.stop_tools.extend(idents.iter().map(Ident::to_string));
            }
            Meta::List(list) if list.path.is_ident("budget") => {
                let inner: Punctuated<Meta, Token![,]> =
                    list.parse_args_with(Punctuated::parse_terminated)?;
                for meta in inner {
                    match &meta {
                        Meta::NameValue(nv) if nv.path.is_ident("calls") => {
                            out.budget_calls = Some(int_value(&nv.value)?);
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("tokens") => {
                            out.budget_tokens = Some(int_value(&nv.value)?);
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("deadline_ms") => {
                            out.budget_deadline_ms = Some(int_value(&nv.value)?);
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("on_exhausted") => {
                            let Expr::Path(p) = &nv.value else {
                                return Err(syn::Error::new_spanned(
                                    &nv.value,
                                    "expected `fail` or `finalize`",
                                ));
                            };
                            match p.path.get_ident().map(ToString::to_string).as_deref() {
                                Some("finalize") => out.budget_finalize = true,
                                Some("fail") => out.budget_finalize = false,
                                _ => {
                                    return Err(syn::Error::new_spanned(
                                        p,
                                        "expected `fail` or `finalize`",
                                    ));
                                }
                            }
                        }
                        other => {
                            return Err(syn::Error::new_spanned(
                                other,
                                "budget(...) accepts calls/tokens/deadline_ms/on_exhausted",
                            ));
                        }
                    }
                }
            }
            Meta::List(list) if list.path.is_ident("context") => {
                let inner: Punctuated<Meta, Token![,]> =
                    list.parse_args_with(Punctuated::parse_terminated)?;
                for meta in inner {
                    match &meta {
                        Meta::NameValue(nv) if nv.path.is_ident("max_history_turns") => {
                            out.ctx_history = Some(int_value(&nv.value)?);
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("tool_result_max_bytes") => {
                            out.ctx_result_bytes = Some(int_value(&nv.value)?);
                        }
                        Meta::NameValue(nv) if nv.path.is_ident("playbook") => {
                            let Expr::Lit(ExprLit {
                                lit: Lit::Str(s), ..
                            }) = &nv.value
                            else {
                                return Err(syn::Error::new_spanned(
                                    &nv.value,
                                    "expected a string literal",
                                ));
                            };
                            out.ctx_playbook = Some(s.value());
                        }
                        other => {
                            return Err(syn::Error::new_spanned(
                                other,
                                "context(...) accepts max_history_turns/tool_result_max_bytes/playbook",
                            ));
                        }
                    }
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unknown #[agent] option (model/tools/stop_tools/max_turns/until_parse/budget/context)",
                ));
            }
        }
    }
    for stop in &out.stop_tools {
        if !out
            .tools
            .iter()
            .any(|p| p.segments.last().is_some_and(|s| s.ident == *stop))
        {
            return Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                format!("stop tool `{stop}` is not in tools(...)"),
            ));
        }
    }
    Ok(out)
}

use syn::Ident;

pub(crate) fn expand_agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = syn::parse_macro_input!(item as syn::ForeignItemFn);
    let runtime = match resolve_dspy_rs_path() {
        Ok(path) => path,
        Err(err) => return err.to_compile_error().into(),
    };
    match expand_agent_inner(attr.into(), &func, &runtime) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_agent_inner(
    attr: TokenStream2,
    func: &syn::ForeignItemFn,
    runtime: &syn::Path,
) -> syn::Result<TokenStream2> {
    let attrs = parse_agent_attr(attr)?;
    let sig = &func.sig;
    if let Some(asyncness) = &sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            "remove `async` — the generated function is async automatically",
        ));
    }
    if !sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.generics,
            "#[agent] functions cannot be generic",
        ));
    }

    let fn_name = &sig.ident;
    let fn_name_str = fn_name.to_string();
    let vis = &func.vis;
    let doc_attrs: Vec<&syn::Attribute> = func
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .collect();

    let mut arg_names = Vec::new();
    let mut arg_types = Vec::new();
    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(ident) => {
                    arg_names.push(ident.ident.clone());
                    arg_types.push(pat_type.ty.as_ref().clone());
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "#[agent] parameters must be plain identifiers",
                    ));
                }
            },
            syn::FnArg::Receiver(receiver) => {
                return Err(syn::Error::new_spanned(
                    receiver,
                    "#[agent] functions cannot take self",
                ));
            }
        }
    }
    if arg_names.is_empty() {
        return Err(syn::Error::new_spanned(
            &sig.inputs,
            "#[agent] functions need at least one input parameter",
        ));
    }
    let output_type = match &sig.output {
        syn::ReturnType::Type(_, ty) => ty.as_ref().clone(),
        syn::ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                sig,
                "#[agent] functions need an explicit return type (the output field)",
            ));
        }
    };

    let model_tokens = option_str_tokens(attrs.model.as_deref());
    let tool_paths = &attrs.tools;
    let stop_names = &attrs.stop_tools;
    let max_turns = option_num(attrs.max_turns.map(u64::from), quote! { u32 });
    let until_parse = match attrs.until_parse {
        Some(v) => quote! { ::core::option::Option::Some(#v) },
        None => quote! { ::core::option::Option::None },
    };
    let budget_calls = option_num(attrs.budget_calls.map(u64::from), quote! { u32 });
    let budget_tokens = option_num(attrs.budget_tokens, quote! { u64 });
    let budget_deadline = option_num(attrs.budget_deadline_ms, quote! { u64 });
    let budget_policy = if attrs.budget_finalize {
        quote! { #runtime::ir::BudgetPolicy::Finalize }
    } else {
        quote! { #runtime::ir::BudgetPolicy::Fail }
    };
    let ctx_history = option_num(attrs.ctx_history.map(u64::from), quote! { u32 });
    let ctx_bytes = option_num(attrs.ctx_result_bytes.map(u64::from), quote! { u32 });
    let ctx_playbook = match &attrs.ctx_playbook {
        Some(p) => quote! { ::core::option::Option::Some(#p.to_string()) },
        None => quote! { ::core::option::Option::None },
    };

    // The standalone fn executes the same 1-node `AgentLoop` program the
    // `#[module]` lowering produces, so the loop options are honored on both
    // paths. The one exception is `model`: model refs bind only inside a
    // `#[module]` program's model table, and silently falling back to the
    // globally configured LM would misreport which model ran — so with
    // `model = "…"` set, no standalone fn is generated at all and calling it
    // is a compile error ("expected function, found module").
    let standalone = if attrs.model.is_some() {
        quote! {}
    } else {
        quote! {
            #(#doc_attrs)*
            ///
            /// Standalone calls execute the same 1-node `AgentLoop` program
            /// the `#[module]` lowering produces, with the attribute options
            /// honored: `max_turns`/`stop_tools`/`until_parse` land in the
            /// node's `StopSpec`, `budget` in its `NodeBudget`, and `context`
            /// in its `ContextPolicy`. The predictor (and thus the tool set)
            /// is built once, on first call.
            #vis async fn #fn_name(
                #(#arg_names: #arg_types),*
            ) -> ::core::result::Result<
                #runtime::Predicted<#fn_name::SigOutput>,
                #runtime::PredictError,
            > {
                static __PREDICT: ::std::sync::OnceLock<
                    ::std::sync::Arc<#runtime::Predict<#fn_name::Sig>>,
                > = ::std::sync::OnceLock::new();
                let predictor = __PREDICT.get_or_init(|| {
                    let step = #fn_name::__dsrs_step();
                    let agent = step
                        .agent
                        .expect("#[agent] steps always carry agent opts");
                    let spec = #runtime::predictors::AgentLoopSpec {
                        stop_tools: agent
                            .stop_tools
                            .iter()
                            .map(|name| ::std::string::String::from(*name))
                            .collect(),
                        max_turns: agent.max_turns,
                        until_parse: agent.until_parse,
                        budget: agent.budget.clone(),
                        context: agent.context.clone(),
                    };
                    let tools: ::std::vec::Vec<
                        ::std::sync::Arc<dyn #runtime::__macro_support::rig::tool::ToolDyn>,
                    > = agent.tools.into_iter().map(|t| t.dyn_tool).collect();
                    ::std::sync::Arc::new(
                        #runtime::Predict::<#fn_name::Sig>::builder()
                            .named(#fn_name_str)
                            .with_tools(tools)
                            .with_agent_spec(spec)
                            .build(),
                    )
                });
                predictor
                    .call(#fn_name::SigInput { #(#arg_names),* })
                    .await
            }
        }
    };

    Ok(quote! {
        #vis mod #fn_name {
            #![allow(non_camel_case_types, unused_imports)]
            use super::*;

            #(#doc_attrs)*
            #[derive(#runtime::Signature, Clone, Debug)]
            pub struct Sig {
                #(
                    #[input]
                    pub #arg_names: #arg_types,
                )*
                #[output]
                pub #fn_name: #output_type,
            }

            /// RFC 0003 M-2: this agent loop as data.
            pub fn __dsrs_step() -> #runtime::ir::StepDef {
                #runtime::ir::StepDef {
                    name: #fn_name_str,
                    kind: #runtime::ir::StepKind::Agent,
                    sig: #runtime::ir::SignatureDef::of::<Sig>(),
                    types: #runtime::ir::SignatureDef::types_of::<Sig>(),
                    model: #model_tokens,
                    agent: ::core::option::Option::Some(#runtime::ir::AgentStepOpts {
                        tools: ::std::vec![#( super::#tool_paths::__dsrs_tool() ),*],
                        stop_tools: ::std::vec![#(#stop_names),*],
                        max_turns: #max_turns,
                        until_parse: #until_parse,
                        budget: #runtime::ir::NodeBudget {
                            max_lm_calls: #budget_calls,
                            max_tokens: #budget_tokens,
                            deadline_ms: #budget_deadline,
                            on_exhausted: #budget_policy,
                        },
                        context: #runtime::ir::ContextPolicy {
                            max_history_turns: #ctx_history,
                            tool_result_max_bytes: #ctx_bytes,
                            playbook: #ctx_playbook,
                        },
                    }),
                }
            }
        }

        #standalone
    })
}

fn option_num(value: Option<u64>, ty: TokenStream2) -> TokenStream2 {
    match value {
        Some(v) => {
            let lit = proc_macro2::Literal::u64_unsuffixed(v);
            quote! { ::core::option::Option::Some(#lit as #ty) }
        }
        None => quote! { ::core::option::Option::None },
    }
}
