//! RFC 0003 stage M-3 library support: the module "linker".
//!
//! `#[module]` parses an ordinary Rust fn body into a [`ModuleSpec`] at macro
//! expansion — pure structure, no cross-item type information — and the
//! generated code calls [`build_module_program`] at first use. Everything
//! type-shaped is resolved *here*, from the step signatures the callees'
//! `__dsrs_step()` metadata carries: fn-param field types come from the first
//! step input they feed (or the macro-mapped declared type), hole input types
//! come from the ports they extract, and the Main signature's outputs come
//! from the tail's source ports. The macro never guesses a type; the linker
//! looks every one up.

use indexmap::IndexMap;
use serde_json::Value;

use crate::LMConfig;
use crate::ir::builder::{self, BuildError, NodeSpec, Port, ProgramBuilder};
use crate::ir::graph::{ModelId, Program, ToolId};
use crate::ir::sig::{SigError, SignatureDef};
use crate::ir::step::{StepDef, StepKind, ToolStepDef};
use crate::typesys::FieldType;

/// The parsed shape of a `#[module]` fn body.
pub struct ModuleSpec {
    pub name: &'static str,
    pub caps: &'static [&'static str],
    /// Fn params in order: `(name, macro-mapped type when the declared Rust
    /// type is in the simple mappable subset)`.
    pub params: Vec<(&'static str, Option<FieldType>)>,
    /// Body statements, in order.
    pub steps: Vec<ModuleStep>,
    /// Tail bindings: Main output field → port.
    pub outs: Vec<(&'static str, PortSpec)>,
}

pub struct ModuleStep {
    /// The `let` binding — the program-unique leaf name.
    pub binding: &'static str,
    pub kind: ModuleStepKind,
}

pub enum ModuleStepKind {
    /// `let <binding> = step(args).await?;`
    Call {
        def: StepDef,
        /// Positional: `args[i]` feeds the step's `inputs[i]`.
        args: Vec<PortSpec>,
    },
    /// An unmappable expression, extracted as an extern (host) hole with a
    /// single output field named after the binding.
    Hole {
        /// Content hash of the extracted expression (macro-computed; the
        /// `HoleImpl::Host` fingerprint).
        hash: u64,
        /// The ascribed output type.
        output: FieldType,
        /// `(input field name, source port)`, in extraction order.
        inputs: Vec<(String, PortSpec)>,
    },
}

/// A macro-emitted port: name-based, resolved by the linker.
#[derive(Clone, Debug)]
pub enum PortSpec {
    /// A fn param (`$.field`).
    Input(&'static str),
    /// A prior binding's output field.
    Out {
        node: &'static str,
        field: &'static str,
    },
    /// A literal argument.
    Lit(Value),
}

/// Why a `#[module]` body could not be linked into a program.
#[derive(Debug, thiserror::Error)]
pub enum ModuleBuildError {
    #[error(
        "step `{step}` calls `{callee}` with {got} argument(s), but its signature \
         declares {expected} input(s)"
    )]
    Arity {
        step: String,
        callee: String,
        expected: usize,
        got: usize,
    },
    #[error(
        "param `{name}` has no mappable declared type and feeds no step — \
         give it a simple type (String, i64, f64, bool, Vec<…>, Option<…>) \
         or pass it to a step"
    )]
    ParamType { name: String },
    #[error("step `{step}` references `{node}` before it is bound")]
    UnknownNode { step: String, node: String },
    #[error("step `{step}`: `{node}.{field}` is not an output of `{node}`")]
    UnknownField {
        step: String,
        node: String,
        field: String,
    },
    #[error("step `{step}`: stop tool `{name}` is not among its declared tools")]
    UnknownStopTool { step: String, name: String },
    #[error(
        "step `{step}`: literal `{value}` has no signature type \
         (strings, integers, floats, and bools only)"
    )]
    LitType { step: String, value: Value },
    #[error("step `{step}`: `max_turns` must be > 0")]
    ZeroMaxTurns { step: String },
    #[error("signature `{name}` is declared twice with different shapes")]
    SigConflict { name: String },
    #[error("tool `{name}` is declared twice with different signatures")]
    ToolConflict { name: String },
    #[error(transparent)]
    Sig(#[from] SigError),
    #[error(transparent)]
    Build(#[from] BuildError),
}

/// Links a [`ModuleSpec`] into a validated [`Program`]. Deterministic: the
/// same spec produces the same canonical text and therefore the same
/// program hash.
pub fn build_module_program(spec: ModuleSpec) -> Result<Program, ModuleBuildError> {
    // ---- pass 1: resolve fn-param types --------------------------------
    // Declared simple types win; otherwise the first step input the param
    // feeds. A param that neither declares nor feeds is an error.
    let mut param_types: IndexMap<&str, FieldType> = IndexMap::new();
    for (name, declared) in &spec.params {
        if let Some(ty) = declared {
            param_types.insert(name, ty.clone());
        }
    }
    for step in &spec.steps {
        if let ModuleStepKind::Call { def, args } = &step.kind {
            if args.len() != def.sig.inputs.len() {
                return Err(ModuleBuildError::Arity {
                    step: step.binding.to_string(),
                    callee: def.name.to_string(),
                    expected: def.sig.inputs.len(),
                    got: args.len(),
                });
            }
            for (i, arg) in args.iter().enumerate() {
                if let PortSpec::Input(param) = arg
                    && !param_types.contains_key(param)
                {
                    param_types.insert(param, def.sig.inputs[i].ty.clone());
                }
            }
        }
    }
    for (name, _) in &spec.params {
        if !param_types.contains_key(name) {
            return Err(ModuleBuildError::ParamType {
                name: name.to_string(),
            });
        }
    }

    // ---- pass 2: walk steps, resolving port types as bindings appear ---
    enum Outputs<'a> {
        /// Base signature of a call step (cot's `reasoning` augmentation is
        /// lowering-side; its base fields are what downstream code reads).
        Call(&'a SignatureDef),
        /// A hole's single output: (field name == binding, type).
        Hole(FieldType),
    }
    let mut outputs: IndexMap<&str, Outputs<'_>> = IndexMap::new();

    let port_type = |at: &str,
                     port: &PortSpec,
                     outputs: &IndexMap<&str, Outputs<'_>>,
                     param_types: &IndexMap<&str, FieldType>|
     -> Result<FieldType, ModuleBuildError> {
        match port {
            PortSpec::Input(param) => {
                param_types
                    .get(param)
                    .cloned()
                    .ok_or_else(|| ModuleBuildError::ParamType {
                        name: param.to_string(),
                    })
            }
            PortSpec::Out { node, field } => match outputs.get(node) {
                None => Err(ModuleBuildError::UnknownNode {
                    step: at.to_string(),
                    node: node.to_string(),
                }),
                Some(Outputs::Call(sig)) => sig
                    .outputs
                    .iter()
                    .find(|f| &*f.name == *field)
                    .map(|f| f.ty.clone())
                    .ok_or_else(|| ModuleBuildError::UnknownField {
                        step: at.to_string(),
                        node: node.to_string(),
                        field: field.to_string(),
                    }),
                Some(Outputs::Hole(ty)) => {
                    if field == node {
                        Ok(ty.clone())
                    } else {
                        Err(ModuleBuildError::UnknownField {
                            step: at.to_string(),
                            node: node.to_string(),
                            field: field.to_string(),
                        })
                    }
                }
            },
            PortSpec::Lit(value) => match value {
                Value::String(_) => Ok(FieldType::String),
                Value::Number(n) if n.is_i64() => Ok(FieldType::Int),
                Value::Number(_) => Ok(FieldType::Float),
                Value::Bool(_) => Ok(FieldType::Bool),
                other => Err(ModuleBuildError::LitType {
                    step: at.to_string(),
                    value: other.clone(),
                }),
            },
        }
    };

    // Hole input/signature types must be resolved in binding order, before
    // the builder pass consumes the spec.
    let mut hole_sigs: IndexMap<&str, SignatureDef> = IndexMap::new();
    for step in &spec.steps {
        match &step.kind {
            ModuleStepKind::Call { def, .. } => {
                outputs.insert(step.binding, Outputs::Call(def.sig));
            }
            ModuleStepKind::Hole { output, inputs, .. } => {
                let mut sb = SignatureDef::build(&format!("{}_hole", step.binding));
                for (field, port) in inputs {
                    let ty = port_type(step.binding, port, &outputs, &param_types)?;
                    sb = sb.input(field, ty);
                }
                sb = sb.output(step.binding, output.clone());
                hole_sigs.insert(step.binding, sb.finish()?);
                outputs.insert(step.binding, Outputs::Hole(output.clone()));
            }
        }
    }

    // Main signature: params in order, tail fields typed by their ports.
    let mut mb = SignatureDef::build("Main");
    for (name, _) in &spec.params {
        mb = mb.input(name, param_types[name].clone());
    }
    for (name, port) in &spec.outs {
        mb = mb.output(name, port_type("out", port, &outputs, &param_types)?);
    }
    let main_def = mb.finish()?;

    // ---- pass 3: drive the ProgramBuilder ------------------------------
    let mut b = ProgramBuilder::new(spec.name);
    for cap in spec.caps {
        b.cap(cap);
    }

    // Models: unique refs in first-use order; configs are `unbound:<name>`
    // placeholders — real bindings arrive by name through `RuntimeEnv`.
    let mut model_ids: IndexMap<&str, ModelId> = IndexMap::new();
    for step in &spec.steps {
        if let ModuleStepKind::Call { def, .. } = &step.kind {
            let name = def.model.unwrap_or("default");
            if !model_ids.contains_key(name) {
                model_ids.insert(name, b.model(name, unbound_model_config(name)));
            }
        }
    }

    // Signatures dedupe by name (two calls to the same step fn reuse one
    // decl); conflicting shapes under one name are refused. Tools likewise.
    type SigIds = IndexMap<String, (crate::ir::graph::SigId, SignatureDef)>;
    let mut sig_ids: SigIds = IndexMap::new();
    let mut tool_ids: IndexMap<&str, ToolId> = IndexMap::new();

    fn register_sig(
        b: &mut ProgramBuilder,
        sig_ids: &mut SigIds,
        def: &SignatureDef,
    ) -> Result<crate::ir::graph::SigId, ModuleBuildError> {
        if let Some((id, existing)) = sig_ids.get(&*def.name) {
            if existing == def {
                return Ok(*id);
            }
            return Err(ModuleBuildError::SigConflict {
                name: def.name.to_string(),
            });
        }
        let id = b.sig(def.clone());
        sig_ids.insert(def.name.to_string(), (id, def.clone()));
        Ok(id)
    }

    fn register_tool<'a>(
        b: &mut ProgramBuilder,
        sig_ids: &mut SigIds,
        tool_ids: &mut IndexMap<&'a str, ToolId>,
        tool: &'a ToolStepDef,
    ) -> Result<ToolId, ModuleBuildError> {
        if let Some(id) = tool_ids.get(tool.name) {
            return Ok(*id);
        }
        b.add_types(tool.types);
        // Same rename rule as steps: the tool's sig takes the tool name,
        // suffixed to stay clear of a step fn with the same name.
        let mut named = tool.sig.clone();
        named.name = format!("{}_tool", tool.name).into();
        let sid = if let Some((id, existing)) = sig_ids.get(&*named.name) {
            if *existing != named {
                return Err(ModuleBuildError::ToolConflict {
                    name: tool.name.to_string(),
                });
            }
            *id
        } else {
            let id = b.sig(named.clone());
            sig_ids.insert(named.name.to_string(), (id, named.clone()));
            id
        };
        let id = b.host_tool(tool.name, tool.desc, sid, tool.caps);
        tool_ids.insert(tool.name, id);
        Ok(id)
    }

    let as_port = |port: &PortSpec| -> Port {
        match port {
            PortSpec::Input(field) => builder::input(field),
            PortSpec::Out { node, field } => builder::out(*node, field),
            PortSpec::Lit(value) => builder::lit(value.clone()),
        }
    };

    let mut nodes: Vec<NodeSpec> = Vec::new();
    for step in &spec.steps {
        match &step.kind {
            ModuleStepKind::Call { def, args } => {
                b.add_types(def.types);
                // Macro-generated sigs are all named `Sig` (one per module);
                // in the program they take the step fn's name — unique,
                // greppable, and naturally deduped across repeat calls.
                let mut named = def.sig.clone();
                named.name = def.name.into();
                let sid = register_sig(&mut b, &mut sig_ids, &named)?;
                let model = model_ids[def.model.unwrap_or("default")];
                let mut ns = match def.kind {
                    StepKind::Predict => builder::predict(step.binding, sid),
                    StepKind::Cot => builder::cot(step.binding, sid),
                    StepKind::Agent => builder::agent(step.binding, sid),
                };
                ns = ns.model(model);
                if let Some(agent) = &def.agent {
                    let mut tids = Vec::with_capacity(agent.tools.len());
                    for tool in &agent.tools {
                        tids.push(register_tool(&mut b, &mut sig_ids, &mut tool_ids, tool)?);
                    }
                    let mut stop = Vec::with_capacity(agent.stop_tools.len());
                    for name in &agent.stop_tools {
                        let position = agent.tools.iter().position(|t| t.name == *name).ok_or(
                            ModuleBuildError::UnknownStopTool {
                                step: step.binding.to_string(),
                                name: name.to_string(),
                            },
                        )?;
                        stop.push(tids[position]);
                    }
                    ns = ns.tools(tids).stop_tools(stop);
                    if let Some(turns) = agent.max_turns {
                        if turns == 0 {
                            return Err(ModuleBuildError::ZeroMaxTurns {
                                step: step.binding.to_string(),
                            });
                        }
                        ns = ns.max_turns(turns);
                    }
                    if let Some(until_parse) = agent.until_parse {
                        ns = ns.until_parse(until_parse);
                    }
                    ns = ns.budget(agent.budget.clone()).context(agent.context.clone());
                }
                for (i, arg) in args.iter().enumerate() {
                    ns = ns.bind(&def.sig.inputs[i].name, as_port(arg));
                }
                nodes.push(ns);
            }
            ModuleStepKind::Hole { hash, inputs, .. } => {
                let def = hole_sigs
                    .get(step.binding)
                    .expect("pass 2 built a sig for every hole");
                let sid = register_sig(&mut b, &mut sig_ids, def)?;
                let mut ns = builder::extern_hole(step.binding, sid, *hash, &[]);
                for (field, port) in inputs {
                    ns = ns.bind(field, as_port(port));
                }
                nodes.push(ns);
            }
        }
    }

    let main_sid = register_sig(&mut b, &mut sig_ids, &main_def)?;
    let mut root = builder::seq(nodes);
    for (name, port) in &spec.outs {
        root = root.out(name, as_port(port));
    }
    Ok(b.main(main_sid, root)?)
}

/// The placeholder config a module program declares for model ref `name`.
/// Loading it unbound fails loudly; the real client arrives by name through
/// [`RuntimeEnv::bind_model`](crate::ir::RuntimeEnv::bind_model).
pub fn unbound_model_config(name: &str) -> LMConfig {
    LMConfig {
        model: format!("unbound:{name}"),
        ..LMConfig::default()
    }
}

/// The globally-configured LM ([`configure`](crate::configure)), used by
/// generated module code to bind the `default` model ref at load.
pub fn default_lm() -> Option<std::sync::Arc<crate::LM>> {
    let guard = crate::core::settings::GLOBAL_SETTINGS.read().ok()?;
    guard.as_ref().map(|settings| std::sync::Arc::clone(&settings.lm))
}
