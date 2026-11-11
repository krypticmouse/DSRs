//! Validated DAG fusion/splitting and a feedback/measurement-gated compiler.
use super::edit::{ApplyError, for_each_port};
use super::{
    Binding, Edit, Node, NodeId, ParamOwner, ParamSlot, ParamValue, PortRef, PredictNode, Program,
    SeqNode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

fn reject(message: &str) -> ApplyError {
    ApplyError::Transform(message.into())
}

fn parent_seq(p: &Program, leaf: NodeId) -> Option<NodeId> {
    p.nodes.iter().find_map(|(id, node)| match node {
        Node::Seq(seq) if seq.body.contains(&leaf) => Some(id),
        _ => None,
    })
}

fn no_demos(p: &Program, n: &PredictNode) -> bool {
    matches!(&p.params[n.demos].default, ParamValue::Demos { rows } if rows.is_empty())
}

fn tool_fields(signature: &super::SignatureDef) -> bool {
    signature
        .inputs
        .iter()
        .chain(signature.outputs.iter())
        .any(|field| {
            matches!(
                field.special_kind(),
                Some(crate::SpecialField::Tools | crate::SpecialField::ToolCalls)
            )
        })
}

/// Adjacent, independent predictions are eligible only when every input binding
/// and model agrees and output names/aliases are disjoint. Tools are excluded.
pub fn fusion_next(p: &Program, first: NodeId) -> Option<NodeId> {
    let Node::Predict(a) = p.nodes.get(first)? else {
        return None;
    };
    let Node::Seq(seq) = &p.nodes[parent_seq(p, first)?] else {
        return None;
    };
    let second = *seq
        .body
        .get(seq.body.iter().position(|id| *id == first)? + 1)?;
    let Node::Predict(b) = &p.nodes[second] else {
        return None;
    };
    let (sa, sb) = (&p.sigs[a.sig], &p.sigs[b.sig]);
    if sa.inputs != sb.inputs
        || tool_fields(sa)
        || tool_fields(sb)
        || a.binding != b.binding
        || p.params[a.model].default != p.params[b.model].default
        || !no_demos(p, a)
        || !no_demos(p, b)
    {
        return None;
    }
    if sa.outputs.iter().any(|a| {
        sb.outputs
            .iter()
            .any(|b| a.name == b.name || a.lm_name == b.lm_name)
    }) {
        return None;
    }
    Some(second)
}

pub fn can_split(p: &Program, leaf: NodeId) -> bool {
    let Some(Node::Predict(n)) = p.nodes.get(leaf) else {
        return false;
    };
    p.sigs[n.sig].outputs.len() > 1
        && no_demos(p, n)
        && !tool_fields(&p.sigs[n.sig])
        && (leaf == p.root || parent_seq(p, leaf).is_some())
}

fn sig_name(p: &Program, stem: &str) -> Box<str> {
    let mut name = stem.to_string();
    let mut suffix = 2;
    while p.sigs.values().any(|s| &*s.name == name)
        || p.types.classes.contains_key(&name)
        || p.types.enums.contains_key(&name)
        || p.tools.values().any(|tool| p.syms.get(tool.name) == name)
    {
        name = format!("{stem}{suffix}");
        suffix += 1;
    }
    name.into()
}

pub(crate) fn fuse(p: &mut Program, first: NodeId, second: NodeId) -> Result<(), ApplyError> {
    if fusion_next(p, first) != Some(second) {
        return Err(reject(
            "fusion requires adjacent independent predictions, matching inputs/model, no demos and disjoint outputs",
        ));
    }
    let Node::Predict(a) = p.nodes[first].clone() else {
        unreachable!()
    };
    let Node::Predict(b) = p.nodes[second].clone() else {
        unreachable!()
    };
    let instruction = format!(
        "Perform both independent tasks and produce all their output fields.\n\nTask {}:\n{}\n\nTask {}:\n{}",
        p.syms.get(a.name),
        text(p, a.instruction),
        p.syms.get(b.name),
        text(p, b.instruction)
    );
    let mut signature = p.sigs[a.sig].clone();
    signature.name = sig_name(p, &format!("{}_fused", signature.name));
    signature.instruction = instruction.clone().into();
    signature.outputs = signature
        .outputs
        .iter()
        .chain(p.sigs[b.sig].outputs.iter())
        .cloned()
        .collect();
    let sig = p.sigs.push(signature);
    let Node::Predict(left) = &mut p.nodes[first] else {
        unreachable!()
    };
    left.sig = sig;
    p.params[a.instruction].default = ParamValue::Instruction { text: instruction };
    let parent = parent_seq(p, first).expect("fusion checked parent");
    let Node::Seq(seq) = &mut p.nodes[parent] else {
        unreachable!()
    };
    seq.body = seq
        .body
        .iter()
        .copied()
        .filter(|id| *id != second)
        .collect();
    for node in p.nodes.values_mut() {
        for_each_port(node, |port| {
            if let PortRef::Out { node, .. } = port {
                if *node == second {
                    *node = first;
                }
            }
        });
    }
    Ok(())
}

fn text(p: &Program, id: super::ParamId) -> &str {
    match &p.params[id].default {
        ParamValue::Instruction { text } => text,
        _ => unreachable!("validated instruction"),
    }
}

pub(crate) fn split(
    p: &mut Program,
    leaf: NodeId,
    outputs: &[String],
    name: &str,
) -> Result<(), ApplyError> {
    if !can_split(p, leaf) {
        return Err(reject(
            "split requires a multi-output prediction without demos at root or in a sequence",
        ));
    }
    if !name
        .as_bytes()
        .first()
        .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        || p.leaf_id(name).is_some()
    {
        return Err(reject("split leaf name must be a new identifier"));
    }
    let Node::Predict(original) = p.nodes[leaf].clone() else {
        unreachable!()
    };
    let signature = p.sigs[original.sig].clone();
    let selected: HashSet<_> = outputs.iter().map(String::as_str).collect();
    if selected.len() != outputs.len()
        || selected.is_empty()
        || selected.len() >= signature.outputs.len()
        || selected
            .iter()
            .any(|name| !signature.outputs.iter().any(|f| &*f.name == *name))
    {
        return Err(reject(
            "split outputs must be a nonempty proper subset without duplicates",
        ));
    }
    let mut left_sig = signature.clone();
    left_sig.name = sig_name(p, &format!("{}_left", signature.name));
    left_sig.outputs = signature
        .outputs
        .iter()
        .filter(|f| !selected.contains(&*f.name))
        .cloned()
        .collect();
    let left_sig = p.sigs.push(left_sig);
    let mut right_sig = signature.clone();
    right_sig.name = sig_name(p, &format!("{}_right", signature.name));
    right_sig.outputs = signature
        .outputs
        .iter()
        .filter(|f| selected.contains(&*f.name))
        .cloned()
        .collect();
    let right_sig = p.sigs.push(right_sig);
    let right = p.nodes.push(Node::Predict(original.clone()));
    let mut new_node = original.clone();
    new_node.name = p.syms.intern(name);
    new_node.sig = right_sig;
    for id in [
        &mut new_node.instruction,
        &mut new_node.demos,
        &mut new_node.model,
    ] {
        let old = p.params[*id].clone();
        let suffix = old.path.rsplit('.').next().expect("parameter path");
        *id = p.params.push(ParamSlot {
            path: format!("{name}.{suffix}").into(),
            owner: ParamOwner::Node(right),
            kind: old.kind,
            default: old.default,
        });
    }
    p.nodes[right] = Node::Predict(new_node);
    let Node::Predict(left) = &mut p.nodes[leaf] else {
        unreachable!()
    };
    left.sig = left_sig;
    for node in p.nodes.values_mut() {
        for_each_port(node, |port| {
            if let PortRef::Out { node, field } = port {
                if *node == leaf && selected.contains(p.syms.get(*field)) {
                    *node = right;
                }
            }
        });
    }
    if leaf == p.root {
        let out = signature
            .outputs
            .iter()
            .map(|f| {
                let field = p.syms.intern(&f.name);
                Binding {
                    dst: field,
                    src: PortRef::Out {
                        node: if selected.contains(&*f.name) {
                            right
                        } else {
                            leaf
                        },
                        field,
                    },
                }
            })
            .collect();
        p.root = p.nodes.push(Node::Seq(SeqNode {
            body: vec![leaf, right].into_boxed_slice(),
            out,
        }));
    } else {
        let parent = parent_seq(p, leaf).expect("split checked parent");
        let Node::Seq(seq) = &mut p.nodes[parent] else {
            unreachable!()
        };
        let mut body = seq.body.to_vec();
        body.insert(body.iter().position(|id| *id == leaf).unwrap() + 1, right);
        seq.body = body.into_boxed_slice();
    }
    Ok(())
}

/// A deterministic menu for feedback-driven fusion/splitting. Applying any edit
/// still runs the normal program validator and rebuilds hashes/parameter indices.
pub fn transformation_menu(p: &Program) -> Vec<Edit> {
    let mut edits = Vec::new();
    for (id, node) in p.nodes.iter() {
        if let Some(second) = fusion_next(p, id) {
            edits.push(Edit::Fuse { first: id, second });
        }
        if can_split(p, id) {
            let Node::Predict(n) = node else {
                unreachable!()
            };
            let fields = &p.sigs[n.sig].outputs;
            let stem = format!("{}_split", p.syms.get(n.name));
            let mut name = stem.clone();
            let mut i = 2;
            while p.leaf_id(&name).is_some() {
                name = format!("{stem}{i}");
                i += 1;
            }
            edits.push(Edit::Split {
                leaf: id,
                outputs: fields[fields.len() / 2..]
                    .iter()
                    .map(|f| f.name.to_string())
                    .collect(),
                name,
            });
        }
    }
    edits
}

/// Measurements on a fixed, ordered validation set. Quality is higher-is-better;
/// latency includes the complete pipeline. Keep provider/cache/retry policies
/// fixed when evaluating baseline and candidates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompilerMeasurement {
    pub quality: Vec<f64>,
    pub latency_us: Vec<u64>,
    pub lm_calls: u64,
    pub feedback: Vec<String>,
}

impl CompilerMeasurement {
    fn valid(&self, count: Option<usize>) -> bool {
        !self.quality.is_empty()
            && self.quality.len() == self.latency_us.len()
            && count.is_none_or(|n| n == self.quality.len())
            && self.quality.iter().all(|s| s.is_finite())
            && self.latency_us.iter().all(|t| *t > 0)
    }
    fn latency(&self) -> f64 {
        self.latency_us.iter().map(|t| *t as f64).sum::<f64>() / self.latency_us.len() as f64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompilerStep {
    pub parent_hash: u64,
    pub child_hash: Option<u64>,
    pub edit: Edit,
    pub measurement: Option<CompilerMeasurement>,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompilerReport {
    pub settings: Compiler,
    pub program: Program,
    pub baseline: CompilerMeasurement,
    pub final_measurement: CompilerMeasurement,
    pub steps: Vec<CompilerStep>,
}

/// Bounded DAG optimization: validate, enumerate legal fusion/split rewrites,
/// apply/seal candidates, evaluate the unchanged validation workload, gate each
/// quality result and measured latency, and return a loadable program artifact.
/// A rejected/invalid candidate cannot mutate the incumbent. This is empirical
/// optimization of model behavior, not a proof of equivalent LLM generations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Compiler {
    pub max_rounds: usize,
    pub max_evaluations: usize,
    /// Maximum permitted quality decrease on *each* validation example.
    pub max_quality_drop: f64,
    /// Required fractional mean-latency reduction relative to the incumbent.
    pub min_latency_reduction: f64,
}

impl Default for Compiler {
    fn default() -> Self {
        Self {
            max_rounds: 4,
            max_evaluations: 32,
            max_quality_drop: 0.0,
            min_latency_reduction: 0.01,
        }
    }
}

impl Compiler {
    pub async fn compile<F, Fut>(
        &self,
        program: &Program,
        mut evaluate: F,
    ) -> anyhow::Result<CompilerReport>
    where
        F: FnMut(Arc<Program>) -> Fut,
        Fut: Future<Output = anyhow::Result<CompilerMeasurement>>,
    {
        anyhow::ensure!(
            self.max_quality_drop.is_finite() && self.max_quality_drop >= 0.0,
            "invalid quality margin"
        );
        anyhow::ensure!(
            self.min_latency_reduction.is_finite()
                && self.min_latency_reduction > 0.0
                && self.min_latency_reduction < 1.0,
            "latency reduction must be between zero and one"
        );
        anyhow::ensure!(
            self.max_evaluations > 0,
            "evaluation budget must include the baseline"
        );
        program.validate()?;
        let baseline = evaluate(Arc::new(program.clone())).await?;
        anyhow::ensure!(baseline.valid(None), "invalid baseline measurements");
        let mut incumbent = program.clone();
        let mut current = baseline.clone();
        let mut steps = Vec::new();
        let mut seen = HashSet::from([program.meta.program_hash]);
        let mut evaluations = 1;
        for _ in 0..self.max_rounds {
            let mut promoted = None;
            for edit in transformation_menu(&incumbent) {
                if evaluations >= self.max_evaluations {
                    break;
                }
                let parent_hash = incumbent.meta.program_hash;
                let child = match incumbent.edited(std::slice::from_ref(&edit)) {
                    Ok(child) => child,
                    Err(error) => {
                        steps.push(CompilerStep {
                            parent_hash,
                            child_hash: None,
                            edit,
                            measurement: None,
                            accepted: false,
                            reason: error.to_string(),
                        });
                        continue;
                    }
                };
                if !seen.insert(child.meta.program_hash) {
                    continue;
                }
                evaluations += 1;
                let child_hash = Some(child.meta.program_hash);
                let measurement = match evaluate(Arc::new(child.clone())).await {
                    Ok(measurement) => measurement,
                    Err(error) => {
                        steps.push(CompilerStep {
                            parent_hash,
                            child_hash,
                            edit,
                            measurement: None,
                            accepted: false,
                            reason: format!("candidate evaluation failed: {error:#}"),
                        });
                        continue;
                    }
                };
                let valid = measurement.valid(Some(baseline.quality.len()));
                let quality_ok = valid
                    && measurement
                        .quality
                        .iter()
                        .zip(&baseline.quality)
                        .all(|(score, base)| *score + self.max_quality_drop >= *base);
                let faster = valid
                    && measurement.latency()
                        <= current.latency() * (1.0 - self.min_latency_reduction);
                let accepted = quality_ok && faster;
                let reason = if !valid {
                    "invalid or mismatched validation measurements"
                } else if !quality_ok {
                    "validation quality regressed"
                } else if !faster {
                    "required latency improvement not met"
                } else {
                    "per-example quality and latency gates passed"
                }
                .to_string();
                steps.push(CompilerStep {
                    parent_hash,
                    child_hash,
                    edit,
                    measurement: valid.then(|| measurement.clone()),
                    accepted,
                    reason,
                });
                if accepted {
                    promoted = Some((child, measurement));
                    break;
                }
            }
            let Some((child, measurement)) = promoted else {
                break;
            };
            incumbent = child;
            current = measurement;
        }
        incumbent.validate()?;
        let lineage = incumbent.meta.lineage.get_or_insert_with(Default::default);
        lineage.optimizer = "dsrs-compiler".into();
        lineage.budget = format!("{evaluations} evaluations").into();
        Ok(CompilerReport {
            settings: self.clone(),
            program: incumbent,
            baseline,
            final_measurement: current,
            steps,
        })
    }
}
