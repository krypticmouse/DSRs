//! Load-time validation (RFC 0002 §2.3): builder `finish()`, the loader, and
//! `Program::validate()` run the same code path. Nothing is checked lazily at
//! call time.
//!
//! Rules enforced:
//! 1. A node may reference: its scope's inputs, outputs of nodes *earlier* in
//!    the enclosing `Seq` chain (including ancestors' earlier siblings), and
//!    `^carried` values only inside a `Loop` body.
//! 2. Every leaf input field is bound exactly once; every `Out` field exists
//!    on the referenced node's interface; bound port types must equal (or
//!    widen: Int→Float, T→Optional<T>, T→Union containing T) the destination.
//! 3. Leaf names are unique program-wide. Route arms are type-identical;
//!    `Route.on` is enum-typed with covered variants or a default. All loops
//!    are bounded by construction.
//! 4. Node/tool/hole caps ⊆ `program.caps`.
//! 5. Acyclicity is structural: trees + earlier-sibling references cannot
//!    cycle. Every node is reachable from the root exactly once.

use std::collections::{HashMap, HashSet};

use cranelift_entity::EntityRef;
use indexmap::IndexMap;

use crate::ir::graph::{
    Binding, Node, NodeId, PortRef, Program, SigId, Sym, ToolKind,
};
use crate::ir::params::{ParamId, ParamKind, ParamOwner};
use crate::ir::sig::SignatureDef;
use crate::typesys::{FieldType, TypeTable};

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ValidateError {
    #[error("id out of range at {at}: {what}")]
    IdOutOfRange { at: String, what: String },
    #[error("interner contains duplicate string `{string}`")]
    DuplicateInternedString { string: String },
    #[error("duplicate param path `{path}`")]
    DuplicateParamPath { path: String },
    #[error("duplicate leaf name `{name}` (leaf names are program-unique)")]
    DuplicateLeafName { name: String },
    #[error("duplicate tool name `{name}`")]
    DuplicateToolName { name: String },
    #[error("node {at} is used more than once (nodes form a tree)")]
    NodeReused { at: String },
    #[error("{count} node(s) are not reachable from the root")]
    UnreachableNodes { count: usize },
    #[error("root node must be a Seq")]
    RootNotSeq,
    #[error("signature `{sig}` references unknown type `{token}`")]
    UnknownTypeToken { sig: String, token: String },
    #[error("input `{field}` of `{at}` is not bound")]
    UnboundInput { at: String, field: String },
    #[error("input `{field}` of `{at}` is bound more than once")]
    DuplicateBinding { at: String, field: String },
    #[error("`{at}` binds `{field}`, which is not an input field of its signature")]
    UnknownBindingDst { at: String, field: String },
    #[error("`{at}` references scope input `$.{field}`, which does not exist here")]
    UnknownScopeInput { at: String, field: String },
    #[error("`{at}` references `^{field}` outside a loop body")]
    CarriedOutsideLoop { at: String, field: String },
    #[error("`{at}` references output `{field}` of {node}, which has no such field")]
    UnknownOutField {
        at: String,
        node: String,
        field: String,
    },
    #[error("`{at}` references {node}, which is not visible from here (only earlier siblings are)")]
    NodeNotVisible { at: String, node: String },
    #[error("type mismatch binding `{field}` of `{at}`: expected {expected}, got {got}")]
    BindingTypeMismatch {
        at: String,
        field: String,
        expected: String,
        got: String,
    },
    #[error("literal bound to `{field}` of `{at}` does not match type {expected}")]
    LiteralTypeMismatch {
        at: String,
        field: String,
        expected: String,
    },
    #[error("route at {at}: `on` port must be enum-typed or a literal union, got {got}")]
    RouteNotEnum { at: String, got: String },
    #[error("route at {at}: `{variant}` is not a variant of the routed enum")]
    RouteUnknownVariant { at: String, variant: String },
    #[error("route at {at}: arm `{arm}` exports a different interface than the first arm")]
    RouteArmMismatch { at: String, arm: String },
    #[error("route at {at} does not cover variants {missing:?} and has no default")]
    RouteUncovered { at: String, missing: Vec<String> },
    #[error("caps of `{at}` exceed the program ceiling: missing {missing:?}")]
    CapsExceedProgram { at: String, missing: Vec<String> },
    #[error("param `{path}` referenced by `{at}` has kind {got:?}, expected {expected:?}")]
    ParamKindMismatch {
        at: String,
        path: String,
        expected: ParamKind,
        got: ParamKind,
    },
    #[error("param `{path}` referenced by `{at}` is owned by a different entity")]
    ParamOwnerMismatch { at: String, path: String },
    #[error("param `{path}` default value has kind {got:?}, expected {expected:?}")]
    ParamDefaultMismatch {
        path: String,
        expected: ParamKind,
        got: ParamKind,
    },
    #[error("agent loop `{at}`: stop tool is not in the node's tool list")]
    StopToolNotDeclared { at: String },
    #[error("refine at {at}: judge must be a Predict or Hole leaf")]
    RefineJudgeNotLeaf { at: String },
    #[error("refine at {at}: judge outputs must include score: float and feedback: string")]
    RefineJudgeInterface { at: String },
    #[error("refine at {at}: feedback field `{field}` is not a string input of the child leaf")]
    RefineFeedbackField { at: String, field: String },
    #[error("loop at {at}: `while` port must be bool-typed, got {got}")]
    WhileNotBool { at: String, got: String },
    #[error(
        "loop at {at}: carried field `{field}` must shadow an enclosing scope input of a \
         compatible type (v1 rule: carry gives iteration-0 values from the scope input)"
    )]
    CarryNotScopeInput { at: String, field: String },
    #[error("program output `{field}` is not exported by the root seq (or has an incompatible type)")]
    ProgramOutputMissing { field: String },
}

/// Type interface of a node: exported field name → type, in export order.
pub(crate) type Interface = IndexMap<String, FieldType>;

impl Program {
    /// Full structural validation. Run by builder `finish()` and by the
    /// deserialization load path; hosts may re-run it at will.
    pub fn validate(&self) -> Result<(), ValidateError> {
        Validator::new(self).run()
    }

    /// A human-readable handle for error messages: the leaf name when the
    /// node is a leaf, the entity display (`n3`) otherwise.
    pub fn node_display(&self, id: NodeId) -> String {
        match self.leaf_name(id) {
            Some(name) => name.to_string(),
            None => format!("{id}"),
        }
    }
}

struct Scope<'a> {
    /// `$`-frame: field name → type.
    inputs: &'a IndexMap<String, FieldType>,
    /// Nodes usable through `Out` ports: earlier siblings at every ancestor
    /// level, in order.
    visible: Vec<NodeId>,
    in_loop: bool,
}

impl<'a> Scope<'a> {
    fn child(&self, extra_visible: &[NodeId]) -> Scope<'a> {
        let mut visible = self.visible.clone();
        visible.extend_from_slice(extra_visible);
        Scope {
            inputs: self.inputs,
            visible,
            in_loop: self.in_loop,
        }
    }
}

struct Validator<'p> {
    p: &'p Program,
    visited: HashSet<NodeId>,
    leaf_names: HashSet<&'p str>,
    ifaces: HashMap<NodeId, Interface>,
}

impl<'p> Validator<'p> {
    fn new(p: &'p Program) -> Self {
        Self {
            p,
            visited: HashSet::new(),
            leaf_names: HashSet::new(),
            ifaces: HashMap::new(),
        }
    }

    fn run(mut self) -> Result<(), ValidateError> {
        self.check_ids()?;
        self.check_sigs()?;
        self.check_params()?;
        self.check_tools()?;

        let Node::Seq(_) = &self.p.nodes[self.p.root] else {
            return Err(ValidateError::RootNotSeq);
        };

        let root_inputs = sig_inputs(&self.p.sigs[self.p.sig]);
        let scope = Scope {
            inputs: &root_inputs,
            visible: Vec::new(),
            in_loop: false,
        };
        let root_iface = self.check_node(self.p.root, &scope)?;

        // The root seq's exports are the program's external interface.
        for field in self.p.sigs[self.p.sig].outputs.iter() {
            match root_iface.get(&*field.name) {
                Some(ty) if compat(ty, &field.ty) => {}
                _ => {
                    return Err(ValidateError::ProgramOutputMissing {
                        field: field.name.to_string(),
                    });
                }
            }
        }

        if self.visited.len() != self.p.nodes.len() {
            return Err(ValidateError::UnreachableNodes {
                count: self.p.nodes.len() - self.visited.len(),
            });
        }
        Ok(())
    }

    // -- structural pre-passes ------------------------------------------------

    /// Every entity reference in every arena is range-checked up front so the
    /// tree walk can index without panicking on hostile input.
    fn check_ids(&self) -> Result<(), ValidateError> {
        let p = self.p;
        let n_nodes = p.nodes.len();
        let n_sigs = p.sigs.len();
        let n_params = p.params.len();
        let n_models = p.models.len();
        let n_tools = p.tools.len();
        let n_syms = p.syms.len();

        let err = |at: &str, what: String| ValidateError::IdOutOfRange {
            at: at.to_string(),
            what,
        };
        let node_ok = |at: &str, id: NodeId| {
            (id.index() < n_nodes)
                .then_some(())
                .ok_or_else(|| err(at, format!("{id}")))
        };
        let sym_ok = |at: &str, s: Sym| {
            (s.index() < n_syms)
                .then_some(())
                .ok_or_else(|| err(at, format!("{s}")))
        };
        let sig_ok = |at: &str, s: SigId| {
            (s.index() < n_sigs)
                .then_some(())
                .ok_or_else(|| err(at, format!("{s}")))
        };
        let param_ok = |at: &str, s: ParamId| {
            (s.index() < n_params)
                .then_some(())
                .ok_or_else(|| err(at, format!("{s}")))
        };
        let port_ok = |at: &str, port: &PortRef| match port {
            PortRef::Input(s) | PortRef::Carried(s) => sym_ok(at, *s),
            PortRef::Out { node, field } => {
                node_ok(at, *node)?;
                sym_ok(at, *field)
            }
            PortRef::Lit(_) => Ok(()),
        };
        let binds_ok = |at: &str, binds: &[Binding]| {
            binds.iter().try_for_each(|b| {
                sym_ok(at, b.dst)?;
                port_ok(at, &b.src)
            })
        };

        node_ok("program.root", p.root)?;
        sig_ok("program.sig", p.sig)?;

        for (id, node) in p.nodes.iter() {
            let at = format!("{id}");
            match node {
                Node::Predict(n) => {
                    sym_ok(&at, n.name)?;
                    sig_ok(&at, n.sig)?;
                    param_ok(&at, n.instruction)?;
                    param_ok(&at, n.demos)?;
                    param_ok(&at, n.model)?;
                    binds_ok(&at, &n.binding)?;
                }
                Node::AgentLoop(n) => {
                    sym_ok(&at, n.name)?;
                    sig_ok(&at, n.sig)?;
                    param_ok(&at, n.instruction)?;
                    param_ok(&at, n.demos)?;
                    param_ok(&at, n.model)?;
                    param_ok(&at, n.context_policy)?;
                    for t in n.tools.iter().chain(n.stop.stop_tools.iter()) {
                        if t.index() >= n_tools {
                            return Err(err(&at, format!("{t}")));
                        }
                    }
                    binds_ok(&at, &n.binding)?;
                }
                Node::Hole(n) => {
                    sym_ok(&at, n.name)?;
                    sig_ok(&at, n.sig)?;
                    param_ok(&at, n.code)?;
                    binds_ok(&at, &n.binding)?;
                }
                Node::Seq(n) => {
                    for c in n.body.iter() {
                        node_ok(&at, *c)?;
                    }
                    binds_ok(&at, &n.out)?;
                }
                Node::ForkJoin(n) => {
                    for c in n.branches.iter() {
                        node_ok(&at, *c)?;
                    }
                    binds_ok(&at, &n.join)?;
                }
                Node::Route(n) => {
                    port_ok(&at, &n.on)?;
                    for (v, c) in n.arms.iter() {
                        sym_ok(&at, *v)?;
                        node_ok(&at, *c)?;
                    }
                    if let Some(d) = n.default {
                        node_ok(&at, d)?;
                    }
                }
                Node::Retry(n) => node_ok(&at, n.child)?,
                Node::Refine(n) => {
                    node_ok(&at, n.child)?;
                    node_ok(&at, n.judge)?;
                    sym_ok(&at, n.feedback_field)?;
                }
                Node::Loop(n) => {
                    node_ok(&at, n.body)?;
                    if let Some(w) = &n.while_ {
                        port_ok(&at, w)?;
                    }
                    binds_ok(&at, &n.carry)?;
                    binds_ok(&at, &n.out)?;
                }
            }
        }

        for (id, slot) in p.params.iter() {
            let at = format!("{id}");
            match slot.owner {
                ParamOwner::Node(n) => node_ok(&at, n)?,
                ParamOwner::Tool(t) => {
                    if t.index() >= n_tools {
                        return Err(err(&at, format!("{t}")));
                    }
                }
            }
            if let crate::ir::params::ParamValue::ModelRef { model } = &slot.default
                && model.index() >= n_models
            {
                return Err(err(&at, format!("{model}")));
            }
        }

        for (id, tool) in p.tools.iter() {
            let at = format!("{id}");
            sym_ok(&at, tool.name)?;
            param_ok(&at, tool.desc)?;
            sig_ok(&at, tool.sig)?;
            if let ToolKind::Sandboxed { code } = tool.kind {
                param_ok(&at, code)?;
            }
        }
        Ok(())
    }

    /// What the derive rejects, the loader rejects: every Class/Enum token in
    /// every signature must resolve against the program's type table.
    fn check_sigs(&self) -> Result<(), ValidateError> {
        for (_, sig) in self.p.sigs.iter() {
            for field in sig.inputs.iter().chain(sig.outputs.iter()) {
                check_tokens(&sig.name, &field.ty, &self.p.types)?;
            }
        }
        Ok(())
    }

    fn check_params(&self) -> Result<(), ValidateError> {
        for (_, slot) in self.p.params.iter() {
            if slot.default.kind() != slot.kind {
                return Err(ValidateError::ParamDefaultMismatch {
                    path: slot.path.to_string(),
                    expected: slot.kind,
                    got: slot.default.kind(),
                });
            }
        }
        Ok(())
    }

    fn check_tools(&self) -> Result<(), ValidateError> {
        let mut names: HashSet<&str> = HashSet::new();
        for (id, tool) in self.p.tools.iter() {
            let name = self.p.syms.get(tool.name);
            if !names.insert(name) {
                return Err(ValidateError::DuplicateToolName {
                    name: name.to_string(),
                });
            }
            if !tool.caps.is_subset(&self.p.caps) {
                return Err(ValidateError::CapsExceedProgram {
                    at: format!("tool.{name}"),
                    missing: tool.caps.missing_from(&self.p.caps),
                });
            }
            self.check_param_ref(
                &format!("tool.{name}"),
                tool.desc,
                ParamKind::ToolDesc,
                ParamOwner::Tool(id),
            )?;
            if let ToolKind::Sandboxed { code } = tool.kind {
                self.check_param_ref(
                    &format!("tool.{name}"),
                    code,
                    ParamKind::Code,
                    ParamOwner::Tool(id),
                )?;
            }
        }
        Ok(())
    }

    fn check_param_ref(
        &self,
        at: &str,
        id: ParamId,
        kind: ParamKind,
        owner: ParamOwner,
    ) -> Result<(), ValidateError> {
        let slot = &self.p.params[id];
        if slot.kind != kind {
            return Err(ValidateError::ParamKindMismatch {
                at: at.to_string(),
                path: slot.path.to_string(),
                expected: kind,
                got: slot.kind,
            });
        }
        if slot.owner != owner {
            return Err(ValidateError::ParamOwnerMismatch {
                at: at.to_string(),
                path: slot.path.to_string(),
            });
        }
        Ok(())
    }

    // -- the tree walk --------------------------------------------------------

    fn check_node(&mut self, id: NodeId, scope: &Scope<'_>) -> Result<Interface, ValidateError> {
        if !self.visited.insert(id) {
            return Err(ValidateError::NodeReused {
                at: self.p.node_display(id),
            });
        }
        // Clone the node handle so `self` stays borrowable; nodes are cheap
        // relative to a validation pass and this runs once at load.
        let node = self.p.nodes[id].clone();
        let iface = match &node {
            Node::Predict(n) => {
                let at = self.leaf(n.name)?;
                self.check_param_ref(&at, n.instruction, ParamKind::Instruction, ParamOwner::Node(id))?;
                self.check_param_ref(&at, n.demos, ParamKind::Demos, ParamOwner::Node(id))?;
                self.check_param_ref(&at, n.model, ParamKind::ModelRef, ParamOwner::Node(id))?;
                self.check_leaf_bindings(&at, n.sig, &n.binding, scope)?;
                sig_outputs(&self.p.sigs[n.sig])
            }
            Node::AgentLoop(n) => {
                let at = self.leaf(n.name)?;
                self.check_param_ref(&at, n.instruction, ParamKind::Instruction, ParamOwner::Node(id))?;
                self.check_param_ref(&at, n.demos, ParamKind::Demos, ParamOwner::Node(id))?;
                self.check_param_ref(&at, n.model, ParamKind::ModelRef, ParamOwner::Node(id))?;
                self.check_param_ref(
                    &at,
                    n.context_policy,
                    ParamKind::ContextPolicy,
                    ParamOwner::Node(id),
                )?;
                for stop in n.stop.stop_tools.iter() {
                    if !n.tools.contains(stop) {
                        return Err(ValidateError::StopToolNotDeclared { at: at.clone() });
                    }
                }
                self.check_leaf_bindings(&at, n.sig, &n.binding, scope)?;
                sig_outputs(&self.p.sigs[n.sig])
            }
            Node::Hole(n) => {
                let at = self.leaf(n.name)?;
                if !n.caps.is_subset(&self.p.caps) {
                    return Err(ValidateError::CapsExceedProgram {
                        at: at.clone(),
                        missing: n.caps.missing_from(&self.p.caps),
                    });
                }
                self.check_param_ref(&at, n.code, ParamKind::Code, ParamOwner::Node(id))?;
                self.check_leaf_bindings(&at, n.sig, &n.binding, scope)?;
                sig_outputs(&self.p.sigs[n.sig])
            }
            Node::Seq(n) => {
                let mut inner = scope.child(&[]);
                for &child in n.body.iter() {
                    self.check_node(child, &inner)?;
                    inner.visible.push(child);
                }
                self.check_export_bindings(&format!("{id}"), &n.out, &inner)?
            }
            Node::ForkJoin(n) => {
                for &branch in n.branches.iter() {
                    // Branches cannot see each other: entry scope for each.
                    self.check_node(branch, scope)?;
                }
                let after = scope.child(&n.branches);
                self.check_export_bindings(&format!("{id}"), &n.join, &after)?
            }
            Node::Route(n) => {
                let at = format!("{id}");
                let on_ty = self.port_type(&at, &n.on, scope)?;
                let variants = self.route_variants(&at, &on_ty)?;
                let mut first: Option<(String, Interface)> = None;
                let mut covered: HashSet<String> = HashSet::new();
                for (variant, arm) in n.arms.iter() {
                    let vname = self.p.syms.get(*variant).to_string();
                    if !variants.contains(&vname) {
                        return Err(ValidateError::RouteUnknownVariant {
                            at,
                            variant: vname,
                        });
                    }
                    covered.insert(vname.clone());
                    let iface = self.check_node(*arm, scope)?;
                    match &first {
                        None => first = Some((vname, iface)),
                        Some((_, expect)) => {
                            if !iface_eq(expect, &iface) {
                                return Err(ValidateError::RouteArmMismatch { at, arm: vname });
                            }
                        }
                    }
                }
                let (_, iface) = first.ok_or(ValidateError::RouteUncovered {
                    at: at.clone(),
                    missing: variants.clone(),
                })?;
                match n.default {
                    Some(default) => {
                        let d_iface = self.check_node(default, scope)?;
                        if !iface_eq(&iface, &d_iface) {
                            return Err(ValidateError::RouteArmMismatch {
                                at,
                                arm: "else".to_string(),
                            });
                        }
                    }
                    None => {
                        let missing: Vec<String> = variants
                            .iter()
                            .filter(|v| !covered.contains(*v))
                            .cloned()
                            .collect();
                        if !missing.is_empty() {
                            return Err(ValidateError::RouteUncovered { at, missing });
                        }
                    }
                }
                iface
            }
            Node::Retry(n) => self.check_node(n.child, scope)?,
            Node::Refine(n) => {
                let at = format!("{id}");
                let child_iface = self.check_node(n.child, scope)?;
                let judge_scope = scope.child(&[n.child]);
                let judge_iface = self.check_node(n.judge, &judge_scope)?;
                if !matches!(self.p.nodes[n.judge], Node::Predict(_) | Node::Hole(_)) {
                    return Err(ValidateError::RefineJudgeNotLeaf { at });
                }
                let score_ok = judge_iface
                    .get("score")
                    .is_some_and(|ty| compat(ty, &FieldType::Float));
                let feedback_ok = judge_iface
                    .get("feedback")
                    .is_some_and(|ty| matches!(ty, FieldType::String));
                if !score_ok || !feedback_ok {
                    return Err(ValidateError::RefineJudgeInterface { at });
                }
                // Feedback injection targets an input field of the child leaf.
                let field = self.p.syms.get(n.feedback_field).to_string();
                let child_sig = match &self.p.nodes[n.child] {
                    Node::Predict(c) => Some(c.sig),
                    Node::AgentLoop(c) => Some(c.sig),
                    Node::Hole(c) => Some(c.sig),
                    _ => None,
                };
                let ok = child_sig.is_some_and(|sig| {
                    self.p.sigs[sig]
                        .inputs
                        .iter()
                        .any(|f| *f.name == *field && matches!(f.ty, FieldType::String))
                });
                if !ok {
                    return Err(ValidateError::RefineFeedbackField { at, field });
                }
                child_iface
            }
            Node::Loop(n) => {
                let at = format!("{id}");
                // v1 rule: every carried name shadows an enclosing scope
                // input — that input's value is the iteration-0 carry.
                for b in n.carry.iter() {
                    let name = self.p.syms.get(b.dst);
                    if !scope.inputs.contains_key(name) {
                        return Err(ValidateError::CarryNotScopeInput {
                            at: at.clone(),
                            field: name.to_string(),
                        });
                    }
                }
                let body_scope = Scope {
                    inputs: scope.inputs,
                    visible: scope.visible.clone(),
                    in_loop: true,
                };
                self.check_node(n.body, &body_scope)?;
                let after = Scope {
                    inputs: scope.inputs,
                    visible: {
                        let mut v = scope.visible.clone();
                        v.push(n.body);
                        v
                    },
                    in_loop: true,
                };
                // Carry rebinding must produce values compatible with the
                // shadowed inputs.
                for b in n.carry.iter() {
                    let name = self.p.syms.get(b.dst).to_string();
                    let expected = scope.inputs.get(&name).cloned().expect("checked above");
                    self.check_binding(&at, &name, &expected, &b.src, &after)?;
                }
                if let Some(port) = &n.while_ {
                    let ty = self.port_type(&at, port, &after)?;
                    if !matches!(ty, FieldType::Bool) {
                        return Err(ValidateError::WhileNotBool {
                            at,
                            got: type_label(&ty),
                        });
                    }
                }
                self.check_export_bindings(&at, &n.out, &after)?
            }
        };
        self.ifaces.insert(id, iface.clone());
        Ok(iface)
    }

    fn leaf(&mut self, name: Sym) -> Result<String, ValidateError> {
        let name = self.p.syms.get(name);
        if !self.leaf_names.insert(name) {
            return Err(ValidateError::DuplicateLeafName {
                name: name.to_string(),
            });
        }
        Ok(name.to_string())
    }

    /// Leaf rule: every signature input bound exactly once, types compatible.
    fn check_leaf_bindings(
        &self,
        at: &str,
        sig: SigId,
        binds: &[Binding],
        scope: &Scope<'_>,
    ) -> Result<(), ValidateError> {
        let def = &self.p.sigs[sig];
        let mut bound: HashSet<&str> = HashSet::new();
        for b in binds {
            let dst = self.p.syms.get(b.dst);
            let Some(field) = def.inputs.iter().find(|f| &*f.name == dst) else {
                return Err(ValidateError::UnknownBindingDst {
                    at: at.to_string(),
                    field: dst.to_string(),
                });
            };
            if !bound.insert(dst) {
                return Err(ValidateError::DuplicateBinding {
                    at: at.to_string(),
                    field: dst.to_string(),
                });
            }
            self.check_binding(at, dst, &field.ty, &b.src, scope)?;
        }
        for field in def.inputs.iter() {
            if !bound.contains(&*field.name) {
                return Err(ValidateError::UnboundInput {
                    at: at.to_string(),
                    field: field.name.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Container export rule (`out`/`join`): dst names unique; sources
    /// resolve. The export set becomes the container's interface.
    fn check_export_bindings(
        &self,
        at: &str,
        binds: &[Binding],
        scope: &Scope<'_>,
    ) -> Result<Interface, ValidateError> {
        let mut iface = Interface::new();
        for b in binds {
            let dst = self.p.syms.get(b.dst).to_string();
            let ty = match &b.src {
                PortRef::Lit(value) => infer_lit_type(value).ok_or_else(|| {
                    ValidateError::LiteralTypeMismatch {
                        at: at.to_string(),
                        field: dst.clone(),
                        expected: "an inferable literal type".to_string(),
                    }
                })?,
                src => self.port_type(at, src, scope)?,
            };
            if iface.insert(dst.clone(), ty).is_some() {
                return Err(ValidateError::DuplicateBinding {
                    at: at.to_string(),
                    field: dst,
                });
            }
        }
        Ok(iface)
    }

    fn check_binding(
        &self,
        at: &str,
        dst_name: &str,
        dst_ty: &FieldType,
        src: &PortRef,
        scope: &Scope<'_>,
    ) -> Result<(), ValidateError> {
        if let PortRef::Lit(value) = src {
            if !json_matches_type(value, dst_ty, &self.p.types) {
                return Err(ValidateError::LiteralTypeMismatch {
                    at: at.to_string(),
                    field: dst_name.to_string(),
                    expected: type_label(dst_ty),
                });
            }
            return Ok(());
        }
        let src_ty = self.port_type(at, src, scope)?;
        if !compat(&src_ty, dst_ty) {
            return Err(ValidateError::BindingTypeMismatch {
                at: at.to_string(),
                field: dst_name.to_string(),
                expected: type_label(dst_ty),
                got: type_label(&src_ty),
            });
        }
        Ok(())
    }

    fn port_type(
        &self,
        at: &str,
        port: &PortRef,
        scope: &Scope<'_>,
    ) -> Result<FieldType, ValidateError> {
        match port {
            PortRef::Input(sym) => {
                let name = self.p.syms.get(*sym);
                scope
                    .inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ValidateError::UnknownScopeInput {
                        at: at.to_string(),
                        field: name.to_string(),
                    })
            }
            PortRef::Carried(sym) => {
                let name = self.p.syms.get(*sym);
                if !scope.in_loop {
                    return Err(ValidateError::CarriedOutsideLoop {
                        at: at.to_string(),
                        field: name.to_string(),
                    });
                }
                // v1 shadow rule: carried names are scope inputs.
                scope
                    .inputs
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ValidateError::UnknownScopeInput {
                        at: at.to_string(),
                        field: name.to_string(),
                    })
            }
            PortRef::Out { node, field } => {
                if !scope.visible.contains(node) {
                    return Err(ValidateError::NodeNotVisible {
                        at: at.to_string(),
                        node: self.p.node_display(*node),
                    });
                }
                let name = self.p.syms.get(*field);
                let iface = self
                    .ifaces
                    .get(node)
                    .expect("visible nodes are validated before use");
                iface
                    .get(name)
                    .cloned()
                    .ok_or_else(|| ValidateError::UnknownOutField {
                        at: at.to_string(),
                        node: self.p.node_display(*node),
                        field: name.to_string(),
                    })
            }
            PortRef::Lit(_) => unreachable!("literals are checked against the destination type"),
        }
    }

    /// Variant names of the routed discriminant: an enum's values, or the
    /// members of a literal-string union.
    fn route_variants(&self, at: &str, ty: &FieldType) -> Result<Vec<String>, ValidateError> {
        match ty {
            FieldType::Enum(token) => match self.p.types.enums.get(token) {
                Some(def) => Ok(def.values.iter().map(|v| v.name.clone()).collect()),
                None => Err(ValidateError::UnknownTypeToken {
                    sig: at.to_string(),
                    token: token.clone(),
                }),
            },
            FieldType::Union(items)
                if !items.is_empty()
                    && items.iter().all(|i| matches!(i, FieldType::Literal(_))) =>
            {
                Ok(items
                    .iter()
                    .map(|i| match i {
                        FieldType::Literal(s) => s.clone(),
                        _ => unreachable!(),
                    })
                    .collect())
            }
            other => Err(ValidateError::RouteNotEnum {
                at: at.to_string(),
                got: type_label(other),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

fn sig_inputs(def: &SignatureDef) -> IndexMap<String, FieldType> {
    def.inputs
        .iter()
        .map(|f| (f.name.to_string(), f.ty.clone()))
        .collect()
}

fn sig_outputs(def: &SignatureDef) -> Interface {
    def.outputs
        .iter()
        .map(|f| (f.name.to_string(), f.ty.clone()))
        .collect()
}

fn iface_eq(a: &Interface, b: &Interface) -> bool {
    a.len() == b.len() && a.iter().all(|(name, ty)| b.get(name) == Some(ty))
}

/// Source-to-destination compatibility: exact equality plus the closed
/// widening set — Int→Float, T→Optional<T>, T→Union containing T.
pub(crate) fn compat(src: &FieldType, dst: &FieldType) -> bool {
    if src == dst {
        return true;
    }
    match (src, dst) {
        (FieldType::Int, FieldType::Float) => true,
        (FieldType::Optional(s), FieldType::Optional(d)) => compat(s, d),
        (s, FieldType::Optional(d)) => compat(s, d),
        (s, FieldType::Union(items)) => items.iter().any(|item| compat(s, item)),
        _ => false,
    }
}

fn check_tokens(sig: &str, ty: &FieldType, types: &TypeTable) -> Result<(), ValidateError> {
    match ty {
        FieldType::Class(token) => {
            let def = types.classes.get(token).ok_or_else(|| {
                ValidateError::UnknownTypeToken {
                    sig: sig.to_string(),
                    token: token.clone(),
                }
            })?;
            for field in &def.fields {
                check_tokens(sig, &field.field_type, types)?;
            }
            Ok(())
        }
        FieldType::Enum(token) => types
            .enums
            .get(token)
            .map(|_| ())
            .ok_or_else(|| ValidateError::UnknownTypeToken {
                sig: sig.to_string(),
                token: token.clone(),
            }),
        FieldType::List(inner) | FieldType::Optional(inner) => check_tokens(sig, inner, types),
        FieldType::Map(key, value) => {
            check_tokens(sig, key, types)?;
            check_tokens(sig, value, types)
        }
        FieldType::Union(items) => items.iter().try_for_each(|i| check_tokens(sig, i, types)),
        _ => Ok(()),
    }
}

/// Structural check of a JSON literal against a field type (Int values are
/// accepted where Float is expected).
pub(crate) fn json_matches_type(
    value: &serde_json::Value,
    ty: &FieldType,
    types: &TypeTable,
) -> bool {
    use serde_json::Value;
    match ty {
        FieldType::String => value.is_string(),
        FieldType::Int => value.as_i64().is_some() || value.as_u64().is_some(),
        FieldType::Float => value.is_number(),
        FieldType::Bool => value.is_boolean(),
        FieldType::Literal(expected) => value.as_str() == Some(expected),
        FieldType::List(inner) => value
            .as_array()
            .is_some_and(|items| items.iter().all(|i| json_matches_type(i, inner, types))),
        FieldType::Optional(inner) => value.is_null() || json_matches_type(value, inner, types),
        FieldType::Map(_, inner) => value
            .as_object()
            .is_some_and(|map| map.values().all(|v| json_matches_type(v, inner, types))),
        FieldType::Class(token) => match (value, types.classes.get(token)) {
            (Value::Object(map), Some(def)) => def.fields.iter().all(|field| {
                match map.get(&field.name) {
                    Some(v) => json_matches_type(v, &field.field_type, types),
                    None => field.field_type.is_optional(),
                }
            }),
            _ => false,
        },
        FieldType::Enum(token) => match (value.as_str(), types.enums.get(token)) {
            (Some(s), Some(def)) => def.values.iter().any(|v| v.name == s),
            _ => false,
        },
        FieldType::Union(items) => items.iter().any(|i| json_matches_type(value, i, types)),
    }
}

/// Best-effort literal type inference for export bindings.
fn infer_lit_type(value: &serde_json::Value) -> Option<FieldType> {
    use serde_json::Value;
    match value {
        Value::String(_) => Some(FieldType::String),
        Value::Bool(_) => Some(FieldType::Bool),
        Value::Number(n) if n.is_i64() || n.is_u64() => Some(FieldType::Int),
        Value::Number(_) => Some(FieldType::Float),
        _ => None,
    }
}

/// Human-readable type label for error messages.
pub(crate) fn type_label(ty: &FieldType) -> String {
    crate::typesys::type_name(ty, None)
}
