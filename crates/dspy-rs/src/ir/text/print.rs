//! Canonical printer for the `.dsrs` text format (RFC 0002 §4).
//!
//! `print` is deterministic and total over valid [`Program`]s;
//! `parse(print(p))` reconstructs a program with the identical canonical print
//! and program hash, and `print(parse(t))` is the canonical form of any valid
//! text `t`. The canonical printed text (with the `lineage` block omitted) is
//! also the preimage of [`Program::compute_hash`].
//!
//! # Canonical form rules
//!
//! Stable ordering, so diffs and hashes are meaningful:
//!
//! 1. Header: `dsrs 1`, `program <name>`.
//! 2. Sections in fixed order, separated by exactly one blank line:
//!    `caps` (one line, entries sorted — `CapSet` is a `BTreeSet`), `model`
//!    lines (arena order, one block), `class` blocks (sorted by token),
//!    `enum` blocks (sorted by token), `sig` blocks (arena order), `tool`
//!    blocks (arena order), `lineage`, `main`. Empty sections are omitted.
//! 3. `sig` declarations cover every signature arena entry **except** (a)
//!    cot-augmented signatures re-sugared into `cot` expressions and (b)
//!    signatures referenced only by tools (tool interfaces print inline).
//! 4. Model option blocks print only fields that differ from
//!    `LMConfig::default()`, in fixed key order. `api_key` is never printed
//!    (it is `#[serde(skip)]` — secrets are structurally absent).
//! 5. Node option blocks print only non-default entries in fixed order;
//!    `max_turns` is always printed on `agent` nodes (the bound is
//!    load-bearing), and `@model` references are always explicit.
//! 6. Container nodes have no names in the IR; the printer assigns `_0`,
//!    `_1`, … in order of appearance (skipping any identifier already taken
//!    by a leaf or tool). A container in a non-step position is named only
//!    when some port references it.
//! 7. A `Predict` node prints as `cot <Sig>` when its signature is exactly
//!    `<Sig>.augmented_with([reasoning])` for some other arena signature
//!    with the same name/instruction/inputs; the augmented copy is skipped.
//! 8. Indentation is two spaces per nesting level; binding lists are
//!    comma-separated on one line; option entries are one per line. Code
//!    fences open immediately after `js` and close with a fence at column 0.
//! 9. Demos print as compact one-line JSON arrays (insertion order
//!    preserved); literals print as compact JSON.
//! 10. Ordering within a construct always follows arena/stored order
//!     (bindings, tools, route arms, steps) — parse preserves text order, so
//!     canonical text round-trips byte-for-byte.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::LMConfig;
use crate::ir::builder::cot_reasoning_field;
use crate::ir::graph::{
    AgentLoopNode, Binding, HoleImpl, HoleNode, Node, NodeId, PortRef, PredictNode, Program, SigId,
    ToolKind,
};
use crate::ir::params::{ContextPolicy, ParamId, ParamValue};
use crate::ir::sig::{ConstraintDef, FieldDef, RenderSpec};
use crate::typesys::FieldType;

impl Program {
    /// Renders the canonical `.dsrs` text form of this program.
    pub fn to_dsrs(&self) -> String {
        canonical_text(self, true)
    }
}

/// Canonical text; `include_lineage = false` yields the hash preimage
/// (RFC 0002 §2.1: the program hash covers the canonical text minus lineage).
pub(crate) fn canonical_text(p: &Program, include_lineage: bool) -> String {
    Printer::new(p).render(include_lineage)
}

struct Printer<'p> {
    p: &'p Program,
    out: String,
    /// Names already claimed (leaf + tool names); synthesized container names
    /// must not collide.
    taken: HashSet<String>,
    /// Assigned container names, in order of appearance.
    container_names: HashMap<NodeId, String>,
    next_anon: usize,
    /// Every node targeted by a `PortRef::Out` anywhere in the program.
    referenced: HashSet<NodeId>,
    /// Predict nodes detected as `cot` sugar → the base signature.
    cot_bases: HashMap<NodeId, SigId>,
}

impl<'p> Printer<'p> {
    fn new(p: &'p Program) -> Self {
        let mut taken = HashSet::new();
        for node in p.nodes.values() {
            if let Some(sym) = node.leaf_name() {
                taken.insert(p.syms.get(sym).to_string());
            }
        }
        for tool in p.tools.values() {
            taken.insert(p.syms.get(tool.name).to_string());
        }

        let mut referenced = HashSet::new();
        let mut visit_port = |port: &PortRef| {
            if let PortRef::Out { node, .. } = port {
                referenced.insert(*node);
            }
        };
        for node in p.nodes.values() {
            match node {
                Node::Predict(n) => n.binding.iter().for_each(|b| visit_port(&b.src)),
                Node::AgentLoop(n) => n.binding.iter().for_each(|b| visit_port(&b.src)),
                Node::Hole(n) => n.binding.iter().for_each(|b| visit_port(&b.src)),
                Node::Seq(n) => n.out.iter().for_each(|b| visit_port(&b.src)),
                Node::ForkJoin(n) => n.join.iter().for_each(|b| visit_port(&b.src)),
                Node::Route(n) => visit_port(&n.on),
                Node::Retry(_) => {}
                Node::Refine(_) => {}
                Node::Loop(n) => {
                    if let Some(w) = &n.while_ {
                        visit_port(w);
                    }
                    n.carry.iter().for_each(|b| visit_port(&b.src));
                    n.out.iter().for_each(|b| visit_port(&b.src));
                }
            }
        }

        let mut cot_bases = HashMap::new();
        for (id, node) in p.nodes.iter() {
            if let Node::Predict(n) = node
                && let Some(base) = cot_base(p, n)
            {
                cot_bases.insert(id, base);
            }
        }

        Self {
            p,
            out: String::new(),
            taken,
            container_names: HashMap::new(),
            next_anon: 0,
            referenced,
            cot_bases,
        }
    }

    fn anon_name(&mut self) -> String {
        loop {
            let candidate = format!("_{}", self.next_anon);
            self.next_anon += 1;
            if self.taken.insert(candidate.clone()) {
                return candidate;
            }
        }
    }

    fn name_container(&mut self, id: NodeId) -> String {
        if let Some(name) = self.container_names.get(&id) {
            return name.clone();
        }
        let name = self.anon_name();
        self.container_names.insert(id, name.clone());
        name
    }

    // -- top level ----------------------------------------------------------

    fn render(mut self, include_lineage: bool) -> String {
        let p = self.p;
        let _ = writeln!(self.out, "dsrs {}", p.meta.format);
        let _ = writeln!(self.out, "program {}", p.meta.name);

        if !p.caps.is_empty() {
            let caps: Vec<&str> = p.caps.iter().collect();
            let _ = write!(self.out, "\ncaps {{ {} }}\n", caps.join(" "));
        }

        if !p.models.is_empty() {
            self.out.push('\n');
            for model in p.models.values() {
                let opts = model_opts(&model.config);
                let _ = write!(
                    self.out,
                    "model {} = {}",
                    model.name,
                    json_str(&model.config.model)
                );
                if !opts.is_empty() {
                    let _ = write!(self.out, " {{ {} }}", opts.join(" "));
                }
                self.out.push('\n');
            }
        }

        let mut class_tokens: Vec<&String> = p.types.classes.keys().collect();
        class_tokens.sort();
        for token in class_tokens {
            let class = &p.types.classes[token];
            self.out.push('\n');
            let _ = write!(self.out, "class {token}");
            if class.rendered_name != *token {
                let _ = write!(self.out, " alias {}", json_str(&class.rendered_name));
            }
            self.out.push_str(" {\n");
            if let Some(docs) = &class.docs {
                let _ = writeln!(self.out, "  {}", json_str(docs));
            }
            for field in &class.fields {
                let _ = write!(
                    self.out,
                    "  {}: {}",
                    field.name,
                    type_text(&field.field_type)
                );
                if field.rendered_name != field.name {
                    let _ = write!(self.out, " alias {}", json_str(&field.rendered_name));
                }
                if let Some(docs) = &field.docs {
                    let _ = write!(self.out, " {}", json_str(docs));
                }
                for c in &field.constraints {
                    let label = c.label.as_deref().unwrap_or("");
                    self.out.push(' ');
                    self.out.push_str(&constraint_text(
                        c.level == crate::typesys::ConstraintKind::Check,
                        &c.expression,
                        label,
                    ));
                }
                self.out.push('\n');
            }
            self.out.push_str("}\n");
        }

        let mut enum_tokens: Vec<&String> = p.types.enums.keys().collect();
        enum_tokens.sort();
        for token in enum_tokens {
            let def = &p.types.enums[token];
            self.out.push('\n');
            let _ = write!(self.out, "enum {token}");
            if def.rendered_name != *token {
                let _ = write!(self.out, " alias {}", json_str(&def.rendered_name));
            }
            self.out.push_str(" {\n");
            if let Some(docs) = &def.docs {
                let _ = writeln!(self.out, "  {}", json_str(docs));
            }
            for value in &def.values {
                let _ = write!(self.out, "  {}", value.name);
                if value.rendered_name != value.name {
                    let _ = write!(self.out, " alias {}", json_str(&value.rendered_name));
                }
                if let Some(docs) = &value.docs {
                    let _ = write!(self.out, " {}", json_str(docs));
                }
                self.out.push('\n');
            }
            self.out.push_str("}\n");
        }

        for sig_id in self.printed_sigs() {
            let sig = &p.sigs[sig_id];
            self.out.push('\n');
            let _ = writeln!(self.out, "sig {} {{", sig.name);
            if !sig.instruction.is_empty() {
                let _ = writeln!(self.out, "  {}", json_str(&sig.instruction));
            }
            for field in sig.inputs.iter() {
                self.sig_field("in ", field);
            }
            for field in sig.outputs.iter() {
                self.sig_field("out", field);
            }
            self.out.push_str("}\n");
        }

        for tool in p.tools.values() {
            self.out.push('\n');
            let name = p.syms.get(tool.name);
            let desc = match &p.params[tool.desc].default {
                ParamValue::ToolDesc { text } => text.as_str(),
                _ => "",
            };
            let _ = write!(self.out, "tool {name} {}", json_str(desc));
            if !tool.caps.is_empty() {
                let caps: Vec<&str> = tool.caps.iter().collect();
                let _ = write!(self.out, " caps [{}]", caps.join(" "));
            }
            self.out.push_str(" {\n");
            let sig = &p.sigs[tool.sig];
            for field in sig.inputs.iter() {
                self.sig_field("in ", field);
            }
            for field in sig.outputs.iter() {
                self.sig_field("out", field);
            }
            self.out.push('}');
            if let ToolKind::Sandboxed { code } = tool.kind
                && let ParamValue::Code { source, .. } = &p.params[code].default
            {
                self.out.push_str(" js");
                self.code_fence(source);
            }
            self.out.push('\n');
        }

        if include_lineage && let Some(lineage) = &p.meta.lineage {
            self.out.push('\n');
            self.out.push_str("lineage {\n");
            let _ = writeln!(self.out, "  optimizer {}", json_str(&lineage.optimizer));
            let _ = writeln!(self.out, "  trainset {}", json_str(&lineage.trainset));
            let _ = writeln!(self.out, "  budget {}", json_str(&lineage.budget));
            if let Some(parent) = &lineage.parent {
                let _ = writeln!(self.out, "  parent {}", json_str(parent));
            }
            if let Some(overlay) = &lineage.overlay {
                let _ = writeln!(self.out, "  overlay {}", json_str(overlay));
            }
            let _ = writeln!(self.out, "  date {}", json_str(&lineage.date));
            self.out.push_str("}\n");
        }

        self.out.push('\n');
        let main_sig_name = p.sigs[p.sig].name.to_string();
        let _ = write!(self.out, "main: {main_sig_name} = ");
        self.expr(p.root, 0);
        self.out.push('\n');
        self.out
    }

    /// Signature arena entries printed as `sig` declarations (rule 3).
    fn printed_sigs(&self) -> Vec<SigId> {
        let p = self.p;
        let sugar: HashSet<SigId> = self
            .cot_bases
            .keys()
            .map(|id| match &p.nodes[*id] {
                Node::Predict(n) => n.sig,
                _ => unreachable!("cot bases are Predict nodes"),
            })
            .collect();
        let tool_sigs: HashSet<SigId> = p.tools.values().map(|t| t.sig).collect();
        let mut node_sigs: HashSet<SigId> = HashSet::new();
        node_sigs.insert(p.sig);
        for (id, node) in p.nodes.iter() {
            let sig = match node {
                Node::Predict(n) => {
                    if let Some(base) = self.cot_bases.get(&id) {
                        *base
                    } else {
                        n.sig
                    }
                }
                Node::AgentLoop(n) => n.sig,
                Node::Hole(n) => n.sig,
                _ => continue,
            };
            node_sigs.insert(sig);
        }
        p.sigs
            .keys()
            .filter(|id| !sugar.contains(id))
            .filter(|id| !tool_sigs.contains(id) || node_sigs.contains(id))
            .collect()
    }

    fn sig_field(&mut self, side: &str, field: &FieldDef) {
        let _ = write!(
            self.out,
            "  {side} {}: {}",
            field.name,
            type_text(&field.ty)
        );
        if field.lm_name != field.name {
            let _ = write!(self.out, " alias {}", json_str(&field.lm_name));
        }
        if let Some(docs) = &field.docs {
            let _ = write!(self.out, " {}", json_str(docs));
        }
        for c in field.constraints.iter() {
            self.out.push(' ');
            self.out.push_str(&constraint_def_text(c));
        }
        if let Some(meta) = render_text(&field.render) {
            let _ = write!(self.out, " {meta}");
        }
        self.out.push('\n');
    }

    fn code_fence(&mut self, source: &str) {
        // Fence longer than any backtick run in the source, minimum 3.
        let mut longest = 0usize;
        let mut run = 0usize;
        for ch in source.chars() {
            if ch == '`' {
                run += 1;
                longest = longest.max(run);
            } else {
                run = 0;
            }
        }
        let ticks = "`".repeat((longest + 1).max(3));
        let _ = write!(self.out, "{ticks}\n{source}\n{ticks}");
    }

    // -- expressions --------------------------------------------------------

    fn indent(&mut self, level: usize) {
        for _ in 0..level {
            self.out.push_str("  ");
        }
    }

    /// Prints an expression in a *target* position (route arm, retry child,
    /// refine body/judge): leaves are always named, containers only when
    /// referenced.
    fn target(&mut self, id: NodeId, level: usize) {
        match self.p.nodes[id].leaf_name() {
            Some(sym) => {
                let name = self.p.syms.get(sym).to_string();
                let _ = write!(self.out, "{name} = ");
                self.expr(id, level);
            }
            None => {
                if self.referenced.contains(&id) {
                    let name = self.name_container(id);
                    let _ = write!(self.out, "{name} = ");
                }
                self.expr(id, level);
            }
        }
    }

    /// Prints a step (`name = expr`) in a seq/fork position.
    fn step(&mut self, id: NodeId, level: usize) {
        self.indent(level);
        let name = match self.p.nodes[id].leaf_name() {
            Some(sym) => self.p.syms.get(sym).to_string(),
            None => self.name_container(id),
        };
        let _ = write!(self.out, "{name} = ");
        self.expr(id, level);
        self.out.push('\n');
    }

    fn expr(&mut self, id: NodeId, level: usize) {
        // Clone the node handle so `self.out` stays mutably borrowable.
        let node = self.p.nodes[id].clone();
        match &node {
            Node::Predict(n) => self.predict(id, n),
            Node::AgentLoop(n) => self.agent(n, level),
            Node::Hole(n) => self.hole(n),
            Node::Seq(n) => {
                self.out.push_str("seq {\n");
                for &child in n.body.iter() {
                    self.step(child, level + 1);
                }
                if !n.out.is_empty() {
                    self.indent(level + 1);
                    self.out.push_str("out ");
                    self.bindmap(&n.out);
                    self.out.push('\n');
                }
                self.indent(level);
                self.out.push('}');
            }
            Node::ForkJoin(n) => {
                self.out.push_str("fork {\n");
                for &branch in n.branches.iter() {
                    self.step(branch, level + 1);
                }
                self.indent(level);
                self.out.push_str("} join ");
                self.bindmap(&n.join);
            }
            Node::Route(n) => {
                self.out.push_str("route ");
                self.port(&n.on);
                self.out.push_str(" {\n");
                for (variant, arm) in n.arms.iter() {
                    self.indent(level + 1);
                    let vname = self.p.syms.get(*variant).to_string();
                    let _ = write!(self.out, "{vname} -> ");
                    self.target(*arm, level + 1);
                    self.out.push('\n');
                }
                if let Some(default) = n.default {
                    self.indent(level + 1);
                    self.out.push_str("else -> ");
                    self.target(default, level + 1);
                    self.out.push('\n');
                }
                self.indent(level);
                self.out.push('}');
            }
            Node::Retry(n) => {
                let _ = write!(self.out, "retry (attempts {}", n.max_attempts);
                if n.backoff_ms != 0 {
                    let _ = write!(self.out, " backoff_ms {}", n.backoff_ms);
                }
                if n.feedback {
                    self.out.push_str(" feedback true");
                }
                self.out.push_str(") ");
                self.target(n.child, level);
            }
            Node::Refine(n) => {
                let field = self.p.syms.get(n.feedback_field).to_string();
                let _ = writeln!(
                    self.out,
                    "refine (threshold {} max_rounds {} feedback_field {field}) {{",
                    fmt_f64(n.threshold),
                    n.max_rounds
                );
                self.indent(level + 1);
                self.out.push_str("body = ");
                self.target(n.child, level + 1);
                self.out.push('\n');
                self.indent(level + 1);
                self.out.push_str("judge = ");
                self.target(n.judge, level + 1);
                self.out.push('\n');
                self.indent(level);
                self.out.push('}');
            }
            Node::Loop(n) => {
                let _ = writeln!(self.out, "loop (max_iters {}) {{", n.max_iters);
                let body = self.p.nodes[n.body].clone();
                if let Node::Seq(seq) = &body {
                    for &child in seq.body.iter() {
                        self.step(child, level + 1);
                    }
                    if !seq.out.is_empty() {
                        self.indent(level + 1);
                        self.out.push_str("out ");
                        self.bindmap(&seq.out);
                        self.out.push('\n');
                    }
                } else {
                    self.step(n.body, level + 1);
                }
                if let Some(port) = &n.while_ {
                    self.indent(level + 1);
                    self.out.push_str("while ");
                    self.port(port);
                    self.out.push('\n');
                }
                if !n.carry.is_empty() {
                    self.indent(level + 1);
                    self.out.push_str("carry ");
                    self.bindmap(&n.carry);
                    self.out.push('\n');
                }
                if !n.out.is_empty() {
                    self.indent(level + 1);
                    self.out.push_str("join ");
                    self.bindmap(&n.out);
                    self.out.push('\n');
                }
                self.indent(level);
                self.out.push('}');
            }
        }
    }

    fn predict(&mut self, id: NodeId, n: &PredictNode) {
        let (keyword, sig_id) = match self.cot_bases.get(&id) {
            Some(base) => ("cot", *base),
            None => ("predict", n.sig),
        };
        let sig_name = self.p.sigs[sig_id].name.to_string();
        let _ = write!(self.out, "{keyword} {sig_name}");
        self.modelref(n.model);
        self.args(&n.binding);
        let mut opts: Vec<String> = Vec::new();
        self.instruction_opt(&mut opts, n.instruction, n.sig);
        self.demos_opt(&mut opts, n.demos);
        if !opts.is_empty() {
            let _ = write!(self.out, " {{ {} }}", opts.join(" "));
        }
    }

    fn agent(&mut self, n: &AgentLoopNode, level: usize) {
        let sig_name = self.p.sigs[n.sig].name.to_string();
        let _ = write!(self.out, "agent {sig_name}");
        self.modelref(n.model);
        self.args(&n.binding);
        self.out.push_str(" {\n");

        if !n.tools.is_empty() {
            let names: Vec<&str> = n
                .tools
                .iter()
                .map(|t| self.p.syms.get(self.p.tools[*t].name))
                .collect();
            self.indent(level + 1);
            let _ = writeln!(self.out, "tools [{}]", names.join(" "));
        }
        // The tool_set gene prints only when its default differs from the
        // full declared list — pre-ToolSet programs keep their canonical
        // text (and hash) byte-for-byte.
        if let ParamValue::ToolSet { tools } = &self.p.params[n.tool_set].default
            && tools.as_slice() != &*n.tools
        {
            let names: Vec<&str> = tools
                .iter()
                .map(|t| self.p.syms.get(self.p.tools[*t].name))
                .collect();
            self.indent(level + 1);
            let _ = writeln!(self.out, "tool_set [{}]", names.join(" "));
        }
        if !n.stop.stop_tools.is_empty() {
            let names: Vec<&str> = n
                .stop
                .stop_tools
                .iter()
                .map(|t| self.p.syms.get(self.p.tools[*t].name))
                .collect();
            self.indent(level + 1);
            let _ = writeln!(self.out, "stop_tools [{}]", names.join(" "));
        }
        self.indent(level + 1);
        let _ = writeln!(self.out, "max_turns {}", n.stop.max_turns);
        if !n.stop.until_parse {
            self.indent(level + 1);
            self.out.push_str("until_parse false\n");
        }
        let budget = &n.budget;
        let mut budget_opts: Vec<String> = Vec::new();
        if let Some(calls) = budget.max_lm_calls {
            budget_opts.push(format!("calls {calls}"));
        }
        if let Some(tokens) = budget.max_tokens {
            budget_opts.push(format!("tokens {tokens}"));
        }
        if let Some(deadline) = budget.deadline_ms {
            budget_opts.push(format!("deadline_ms {deadline}"));
        }
        if budget.on_exhausted == crate::ir::graph::BudgetPolicy::Finalize {
            budget_opts.push("on_exhausted finalize".to_string());
        }
        if !budget_opts.is_empty() {
            self.indent(level + 1);
            let _ = writeln!(self.out, "budget {{ {} }}", budget_opts.join(" "));
        }
        if let ParamValue::ContextPolicy { policy } = &self.p.params[n.context_policy].default {
            let mut ctx: Vec<String> = Vec::new();
            let ContextPolicy {
                max_history_turns,
                tool_result_max_bytes,
                playbook,
            } = policy;
            if let Some(turns) = max_history_turns {
                ctx.push(format!("max_history_turns {turns}"));
            }
            if let Some(bytes) = tool_result_max_bytes {
                ctx.push(format!("tool_result_max_bytes {bytes}"));
            }
            if let Some(playbook) = playbook {
                ctx.push(format!("playbook {}", json_str(playbook)));
            }
            if !ctx.is_empty() {
                self.indent(level + 1);
                let _ = writeln!(self.out, "context {{ {} }}", ctx.join(" "));
            }
        }
        let mut opts: Vec<String> = Vec::new();
        self.instruction_opt(&mut opts, n.instruction, n.sig);
        self.demos_opt(&mut opts, n.demos);
        for opt in opts {
            self.indent(level + 1);
            let _ = writeln!(self.out, "{opt}");
        }
        self.indent(level);
        self.out.push('}');
    }

    fn hole(&mut self, n: &HoleNode) {
        let sig_name = self.p.sigs[n.sig].name.to_string();
        let _ = write!(self.out, "hole {sig_name}");
        self.args(&n.binding);
        let caps: Vec<&str> = n.caps.iter().collect();
        match &n.imp {
            HoleImpl::Sandboxed { code } => {
                let _ = write!(self.out, " caps [{}] js", caps.join(" "));
                if let ParamValue::Code { source, .. } = &self.p.params[*code].default {
                    let source = source.clone();
                    self.code_fence(&source);
                }
            }
            HoleImpl::Host { hash } => {
                let _ = write!(self.out, " caps [{}] extern \"{hash:016x}\"", caps.join(" "));
            }
        }
    }

    fn instruction_opt(&mut self, opts: &mut Vec<String>, param: ParamId, sig: SigId) {
        if let ParamValue::Instruction { text } = &self.p.params[param].default
            && text.as_str() != &*self.p.sigs[sig].instruction
        {
            opts.push(format!("instruction {}", json_str(text)));
        }
    }

    fn demos_opt(&mut self, opts: &mut Vec<String>, param: ParamId) {
        if let ParamValue::Demos { rows } = &self.p.params[param].default
            && !rows.is_empty()
        {
            let json = serde_json::to_string(rows).expect("demo rows serialize");
            opts.push(format!("demos {json}"));
        }
    }

    fn modelref(&mut self, param: ParamId) {
        if let ParamValue::ModelRef { model } = &self.p.params[param].default {
            let name = self.p.models[*model].name.to_string();
            let _ = write!(self.out, " @{name}");
        }
    }

    fn args(&mut self, binds: &[Binding]) {
        if binds.is_empty() {
            return;
        }
        self.out.push_str(" (");
        for (i, b) in binds.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let dst = self.p.syms.get(b.dst).to_string();
            let _ = write!(self.out, "{dst} = ");
            self.port(&b.src);
        }
        self.out.push(')');
    }

    fn bindmap(&mut self, binds: &[Binding]) {
        if binds.is_empty() {
            self.out.push_str("{ }");
            return;
        }
        self.out.push_str("{ ");
        for (i, b) in binds.iter().enumerate() {
            if i > 0 {
                self.out.push_str(", ");
            }
            let dst = self.p.syms.get(b.dst).to_string();
            let _ = write!(self.out, "{dst} = ");
            self.port(&b.src);
        }
        self.out.push_str(" }");
    }

    fn port(&mut self, port: &PortRef) {
        match port {
            PortRef::Input(sym) => {
                let _ = write!(self.out, "$.{}", self.p.syms.get(*sym));
            }
            PortRef::Carried(sym) => {
                let _ = write!(self.out, "^{}", self.p.syms.get(*sym));
            }
            PortRef::Lit(value) => {
                let _ = write!(
                    self.out,
                    "{}",
                    serde_json::to_string(value).expect("literal serializes")
                );
            }
            PortRef::Out { node, field } => {
                let name = match self.p.nodes[*node].leaf_name() {
                    Some(sym) => self.p.syms.get(sym).to_string(),
                    None => self
                        .container_names
                        .get(node)
                        .cloned()
                        // Unreachable for validated programs: targets print
                        // before references. Kept total for hash stability.
                        .unwrap_or_else(|| format!("{node}")),
                };
                let _ = write!(self.out, "{name}.{}", self.p.syms.get(*field));
            }
        }
    }
}

/// Detects `cot` sugar: the node's signature is `base.augmented_with([reasoning])`
/// for some *other* arena signature with identical name/instruction/inputs.
fn cot_base(p: &Program, n: &PredictNode) -> Option<SigId> {
    let sig = &p.sigs[n.sig];
    let first = sig.outputs.first()?;
    if *first != cot_reasoning_field() {
        return None;
    }
    p.sigs.iter().find_map(|(id, base)| {
        (id != n.sig
            && base.name == sig.name
            && base.instruction == sig.instruction
            && base.inputs == sig.inputs
            && *base.outputs == sig.outputs[1..])
            .then_some(id)
    })
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// JSON-escaped string literal.
pub(crate) fn json_str(s: &str) -> String {
    serde_json::to_string(s).expect("strings serialize")
}

/// Shortest round-trip float text (`1` for 1.0 parses back exactly).
fn fmt_f64(v: f64) -> String {
    format!("{v}")
}

fn fmt_f32(v: f32) -> String {
    format!("{v}")
}

/// The grammar's type syntax for a [`FieldType`].
pub(crate) fn type_text(ty: &FieldType) -> String {
    fn atom(ty: &FieldType) -> String {
        match ty {
            FieldType::Union(_) => format!("({})", type_text(ty)),
            other => type_text(other),
        }
    }
    match ty {
        FieldType::String => "string".to_string(),
        FieldType::Int => "int".to_string(),
        FieldType::Float => "float".to_string(),
        FieldType::Bool => "bool".to_string(),
        FieldType::Literal(value) => json_str(value),
        FieldType::List(inner) => format!("{}[]", atom(inner)),
        FieldType::Optional(inner) => format!("{}?", atom(inner)),
        // Map keys are string-implied in the text form (the type system
        // rejects non-string keys in signature fields).
        FieldType::Map(_, value) => format!("map<{}>", type_text(value)),
        FieldType::Class(token) | FieldType::Enum(token) => token.clone(),
        FieldType::Union(items) => items.iter().map(atom).collect::<Vec<_>>().join(" | "),
    }
}

fn constraint_def_text(c: &ConstraintDef) -> String {
    constraint_text(
        c.kind == crate::core::ConstraintKind::Check,
        &c.expr,
        &c.label,
    )
}

fn constraint_text(is_check: bool, expr: &str, label: &str) -> String {
    if is_check {
        format!("check({}, {})", json_str(expr), json_str(label))
    } else if label.is_empty() {
        format!("assert({})", json_str(expr))
    } else {
        format!("assert({}, {})", json_str(expr), json_str(label))
    }
}

fn render_text(render: &RenderSpec) -> Option<String> {
    match render {
        RenderSpec::Default => None,
        RenderSpec::Format(value) => Some(format!("format {}", json_str(value))),
        RenderSpec::Jinja(template) => Some(format!("jinja {}", json_str(template))),
    }
}

/// Model option entries that differ from `LMConfig::default()`, fixed order.
fn model_opts(config: &LMConfig) -> Vec<String> {
    let default = LMConfig::default();
    let mut opts = Vec::new();
    if let Some(url) = &config.base_url {
        opts.push(format!("base_url {}", json_str(url)));
    }
    if config.temperature != default.temperature {
        opts.push(format!("temperature {}", fmt_f32(config.temperature)));
    }
    if config.max_tokens != default.max_tokens {
        opts.push(format!("max_tokens {}", config.max_tokens));
    }
    if config.max_tool_iterations != default.max_tool_iterations {
        opts.push(format!(
            "max_tool_iterations {}",
            config.max_tool_iterations
        ));
    }
    if config.max_retries != default.max_retries {
        opts.push(format!("max_retries {}", config.max_retries));
    }
    if config.retry_base_delay_ms != default.retry_base_delay_ms {
        opts.push(format!(
            "retry_base_delay_ms {}",
            config.retry_base_delay_ms
        ));
    }
    if config.cache != default.cache {
        opts.push(format!("cache {}", config.cache));
    }
    opts
}
