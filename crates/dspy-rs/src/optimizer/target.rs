//! What an optimizer optimizes: [`OptimizeTarget`], the lane-erased pair of
//! (thing under optimization, evaluation harness).
//!
//! Two lanes, one currency:
//!
//! - [`OptimizeTarget::module`] — a typed [`Module`] (+ [`Predictors`]
//!   discovery), a trainset slice, and a [`TypedMetric`]. Candidate injection
//!   is ambient ([`fx::with_params`](crate::fx::with_params)) — evaluation
//!   never mutates the module; the single mutation is the caller-driven
//!   [`install`](OptimizeTarget::install) of the winner at the end.
//! - [`OptimizeTarget::program`] — an interpreter-loaded IR
//!   [`Program`](crate::ir::Program), labeled [`DemoRow`] examples, and a
//!   [`ProgramMetric`]. Candidates read through
//!   [`ir::Overlay`](crate::ir::Overlay)s at render time; the winner is
//!   retrievable as an overlay ([`winner_overlay`](OptimizeTarget::winner_overlay))
//!   for [`Program::bake`](crate::ir::Program::bake).
//!
//! Construction runs the **naming pass** (module lane): every leaf the module
//! declares via [`Predictors`] is stamped with its declared name
//! ([`PredictorInfo::set_trace_name`]), so trace spans, candidate entries, and
//! persistence all address the same names. The target also snapshots
//! [`LeafInfo`] (schema text, current instruction, demos) — the read surface
//! strategies build candidates from — and computes the run's baseline
//! identity exactly once.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures::future::LocalBoxFuture;
use serde::Serialize;
use serde_json::Value;

use crate::core::{PredictState, PredictorInfo, Predictors, ToInput};
use crate::evaluate::{Eval, TypedMetric, evaluator::rollout_traced};
use crate::ir::interp::{Budget as RunBudget, Interpreter};
use crate::ir::params::{DemoRow, Overlay, ParamValue};
use crate::ir::graph::Node;
use crate::optimizer::engine::{BoundCandidate, Candidate, CandidatePayload, canonical_hash};
use crate::trace::{JsonMap, Trace, TraceMeta, TraceOutcome, capture_with_meta};
use crate::Module;

/// How a program-lane strategy tells the engine what "good" means: score one
/// interpreter output (`JsonMap` of the program's output signature fields)
/// against a labeled example.
///
/// The JSON-native sibling of [`TypedMetric`](crate::evaluate::TypedMetric) —
/// loaded programs have no static output type, so the metric sees the same
/// value model the interpreter produces. The rollout's captured [`Trace`] is
/// always provided; LM-as-judge metrics run outside the capture scope exactly
/// as in the module lane.
#[allow(async_fn_in_trait)]
pub trait ProgramMetric: Send + Sync {
    async fn evaluate(
        &self,
        example: &DemoRow,
        output: &JsonMap,
        trace: Option<&Trace>,
    ) -> Result<Eval>;
}

/// The read surface strategies build candidates from: one optimizable leaf's
/// name, current values, and field contract. Snapshotted at target
/// construction (candidates never mutate the target mid-run, so the snapshot
/// stays valid for the whole optimization).
#[derive(Clone, Debug)]
pub struct LeafInfo {
    /// The leaf's canonical name (trace component / candidate key).
    pub name: String,
    /// Current effective instruction (override or default).
    pub instruction: String,
    /// The signature/default instruction (ignoring overrides).
    pub default_instruction: String,
    /// Current demos as flat JSON rows.
    pub demos: Vec<JsonMap>,
    /// Input fields as `(lm name, docs)` pairs.
    pub input_fields: Vec<(String, String)>,
    /// Output fields as `(lm name, docs)` pairs.
    pub output_fields: Vec<(String, String)>,
}

impl LeafInfo {
    fn from_predictor(name: &str, info: &dyn PredictorInfo) -> Self {
        let schema = info.schema();
        Self {
            name: name.to_string(),
            instruction: info.instruction(),
            default_instruction: info.default_instruction(),
            demos: info.demos_as_json(),
            input_fields: schema
                .input_fields()
                .iter()
                .map(|field| (field.lm_name.to_string(), field.docs.clone()))
                .collect(),
            output_fields: schema
                .output_fields()
                .iter()
                .map(|field| (field.lm_name.to_string(), field.docs.clone()))
                .collect(),
        }
    }

    /// Renders the leaf's input/output contract for reflection prompts
    /// (GEPA/SIMBA format).
    pub fn schema_for_reflection(&self) -> String {
        let mut result = String::new();
        result.push_str("Input fields:\n");
        for (name, docs) in &self.input_fields {
            let docs = if docs.is_empty() { "No description" } else { docs };
            result.push_str(&format!("  - {name}: {docs}\n"));
        }
        result.push_str("Output fields:\n");
        for (name, docs) in &self.output_fields {
            let docs = if docs.is_empty() { "No description" } else { docs };
            result.push_str(&format!("  - {name}: {docs}\n"));
        }
        result
    }
}

/// The lane-erased evaluation surface the engine drives. Object-safe so the
/// [`Optimizer`](crate::optimizer::Optimizer) trait can be too.
pub(crate) trait Lane {
    fn num_examples(&self) -> usize;
    fn example_uid(&self, idx: usize) -> u64;
    fn baseline(&self) -> u64;
    fn bind(&self, payload: &CandidatePayload) -> Result<BoundCandidate>;
    fn run<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
    ) -> LocalBoxFuture<'s, Result<(Eval, Trace)>>;
    fn output<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
    ) -> LocalBoxFuture<'s, Result<Value>>;
    fn install(&mut self, winner: &Candidate) -> Result<()>;
    fn winner_overlay(&self) -> Option<Arc<Overlay>>;
}

// ---------------------------------------------------------------------------
// Module lane
// ---------------------------------------------------------------------------

struct ModuleLane<'a, E, M, MT> {
    module: &'a mut M,
    /// Validation examples first (empty when no valset), trainset after.
    val: &'a [E],
    train: &'a [E],
    metric: &'a MT,
    uids: Vec<u64>,
    baseline: u64,
}

impl<'a, E, M, MT> ModuleLane<'a, E, M, MT>
where
    E: ToInput<M::Input> + Serialize + Sync,
    M: Module + Predictors,
    MT: TypedMetric<E, M>,
{
    fn example(&self, idx: usize) -> &E {
        if idx < self.val.len() {
            &self.val[idx]
        } else {
            &self.train[idx - self.val.len()]
        }
    }
}

/// Content identity of a module's optimizable state: the `predictors()`
/// snapshot hashed as `{name → PredictState}`. Computed once per target.
fn module_baseline(leaves: &[(String, &dyn PredictorInfo)]) -> u64 {
    let states: std::collections::BTreeMap<&str, PredictState> = leaves
        .iter()
        .map(|(name, info)| (name.as_str(), info.dump_state()))
        .collect();
    canonical_hash(&states)
}

impl<'a, E, M, MT> Lane for ModuleLane<'a, E, M, MT>
where
    E: ToInput<M::Input> + Serialize + Sync,
    M: Module + Predictors,
    MT: TypedMetric<E, M>,
{
    fn num_examples(&self) -> usize {
        self.val.len() + self.train.len()
    }

    fn example_uid(&self, idx: usize) -> u64 {
        self.uids[idx]
    }

    fn baseline(&self) -> u64 {
        self.baseline
    }

    fn bind(&self, payload: &CandidatePayload) -> Result<BoundCandidate> {
        match payload {
            CandidatePayload::Params { params, .. } => {
                Ok(BoundCandidate::Params(Arc::clone(params)))
            }
            CandidatePayload::Overlay(_) => Err(anyhow!(
                "an ir::Overlay candidate cannot be evaluated on a module target; \
                 register a Candidate instead"
            )),
        }
    }

    fn run<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
    ) -> LocalBoxFuture<'s, Result<(Eval, Trace)>> {
        Box::pin(async move {
            let BoundCandidate::Params(params) = bound else {
                return Err(anyhow!("module lane received a non-params candidate"));
            };
            let example = self.example(idx);
            let meta = TraceMeta {
                candidate_hash: Some(candidate_hash),
                ..TraceMeta::default()
            };
            let module: &M = &*self.module;
            // The candidate is scoped ambiently around the whole traced
            // rollout: each Predict leaf binds its own entry at call time.
            crate::fx::with_params_shared(
                params,
                rollout_traced(module, example, self.metric, meta),
            )
            .await
        })
    }

    fn output<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
    ) -> LocalBoxFuture<'s, Result<Value>> {
        Box::pin(async move {
            let BoundCandidate::Params(params) = bound else {
                return Err(anyhow!("module lane received a non-params candidate"));
            };
            let example = self.example(idx);
            let input = example.to_input()?;
            let module: &M = &*self.module;
            let predicted = crate::fx::with_params_shared(params, module.call(input))
                .await
                .map_err(|err| anyhow!("{err}"))?;
            Ok(serde_json::to_value(predicted.into_inner()).unwrap_or(Value::Null))
        })
    }

    fn install(&mut self, winner: &Candidate) -> Result<()> {
        let mut leaves = self.module.predictors_mut();
        for (name, slot) in &winner.slots {
            let Some((_, info)) = leaves.iter_mut().find(|(leaf, _)| leaf == name) else {
                return Err(anyhow!("predictor `{name}` not found in the module"));
            };
            let mut state = info.dump_state();
            if slot.clear_instruction {
                state.instruction_override = None;
            } else if let Some(text) = &slot.instruction {
                state.instruction_override = Some(text.clone());
            }
            if let Some(demos) = &slot.demos {
                state.demos = demos.clone();
            }
            info.load_state(state)
                .map_err(|err| anyhow!("failed to install winner on `{name}`: {err}"))?;
        }
        Ok(())
    }

    fn winner_overlay(&self) -> Option<Arc<Overlay>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Program lane
// ---------------------------------------------------------------------------

struct ProgramLane<'a, MT> {
    interp: &'a Interpreter,
    examples: &'a [DemoRow],
    metric: &'a MT,
    uids: Vec<u64>,
    program_tag: String,
    winner: Option<Arc<Overlay>>,
}

impl<'a, MT: ProgramMetric> Lane for ProgramLane<'a, MT> {
    fn num_examples(&self) -> usize {
        self.examples.len()
    }

    fn example_uid(&self, idx: usize) -> u64 {
        self.uids[idx]
    }

    fn baseline(&self) -> u64 {
        self.interp.program().meta.program_hash
    }

    fn bind(&self, payload: &CandidatePayload) -> Result<BoundCandidate> {
        match payload {
            CandidatePayload::Overlay(overlay) => Ok(BoundCandidate::Overlay(Arc::clone(overlay))),
            CandidatePayload::Params { params, .. } => {
                let overlay = params
                    .bind(self.interp.program())
                    .map_err(|err| anyhow!("failed to bind candidate against program: {err}"))?;
                Ok(BoundCandidate::Overlay(Arc::new(overlay)))
            }
        }
    }

    fn run<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
    ) -> LocalBoxFuture<'s, Result<(Eval, Trace)>> {
        Box::pin(async move {
            let BoundCandidate::Overlay(overlay) = bound else {
                return Err(anyhow!("program lane received a non-overlay candidate"));
            };
            let example = &self.examples[idx];
            let meta = TraceMeta {
                candidate_hash: Some(candidate_hash),
                input: Some(example.input.clone()),
                tags: [("program".to_string(), self.program_tag.clone())]
                    .into_iter()
                    .collect(),
                ..TraceMeta::default()
            };
            let started = Instant::now();
            let (result, mut trace) = capture_with_meta(meta, || {
                self.interp
                    .run(example.input.clone(), Some(overlay), RunBudget::unlimited())
            })
            .await;
            let output =
                result.map_err(|err| anyhow!("candidate failed on example {idx}: {err}"))?;
            // Metric runs outside the capture scope so LM-as-judge metrics
            // don't pollute the execution trace.
            let eval = self.metric.evaluate(example, &output, Some(&trace)).await?;
            trace.outcome = Some(TraceOutcome {
                output: Some(output),
                error: None,
                eval: Some(eval.clone()),
                duration_us: started.elapsed().as_micros() as u64,
            });
            Ok((eval, trace))
        })
    }

    fn output<'s>(
        &'s self,
        idx: usize,
        bound: BoundCandidate,
    ) -> LocalBoxFuture<'s, Result<Value>> {
        Box::pin(async move {
            let BoundCandidate::Overlay(overlay) = bound else {
                return Err(anyhow!("program lane received a non-overlay candidate"));
            };
            let example = &self.examples[idx];
            let output = self
                .interp
                .run(example.input.clone(), Some(overlay), RunBudget::unlimited())
                .await
                .map_err(|err| anyhow!("{err}"))?;
            Ok(Value::Object(output))
        })
    }

    fn install(&mut self, winner: &Candidate) -> Result<()> {
        let overlay = winner
            .to_params()
            .bind(self.interp.program())
            .map_err(|err| anyhow!("failed to bind winner against program: {err}"))?;
        self.winner = Some(Arc::new(overlay));
        Ok(())
    }

    fn winner_overlay(&self) -> Option<Arc<Overlay>> {
        self.winner.clone()
    }
}

/// Leaf snapshot for a loaded program: one [`LeafInfo`] per `predict`/`agent`
/// node, current values read from the param slot defaults.
fn program_leaves(interp: &Interpreter) -> Vec<LeafInfo> {
    let program = interp.program();
    let mut leaves = Vec::new();
    for (node_id, node) in program.nodes.iter() {
        let sig = match node {
            Node::Predict(n) => n.sig,
            Node::AgentLoop(n) => n.sig,
            _ => continue,
        };
        let Some(name) = program.leaf_name(node_id) else {
            continue;
        };
        let def = &program.sigs[sig];
        let instruction = program
            .param_id(&format!("{name}.instruction"))
            .and_then(|id| match &program.params[id].default {
                ParamValue::Instruction { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_else(|| def.instruction.to_string());
        let demos = program
            .param_id(&format!("{name}.demos"))
            .and_then(|id| match &program.params[id].default {
                ParamValue::Demos { rows } => Some(
                    rows.iter()
                        .map(|row| {
                            let mut flat = row.input.clone();
                            flat.extend(row.output.iter().map(|(k, v)| (k.clone(), v.clone())));
                            flat
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let field_pairs = |fields: &[crate::ir::sig::FieldDef]| {
            fields
                .iter()
                .map(|field| {
                    (
                        field.lm_name.to_string(),
                        field.docs.as_deref().unwrap_or("").to_string(),
                    )
                })
                .collect()
        };
        leaves.push(LeafInfo {
            name: name.to_string(),
            default_instruction: instruction.clone(),
            instruction,
            demos,
            input_fields: field_pairs(&def.inputs),
            output_fields: field_pairs(&def.outputs),
        });
    }
    leaves
}

// ---------------------------------------------------------------------------
// OptimizeTarget
// ---------------------------------------------------------------------------

/// The thing an [`Optimizer`](crate::optimizer::Optimizer) optimizes: a
/// module or a program, packaged with its example set and metric. See the
/// module docs for the two lanes.
pub struct OptimizeTarget<'a> {
    lane: Box<dyn Lane + 'a>,
    leaves: Vec<LeafInfo>,
    /// Number of leading example columns that are the validation set, when a
    /// separate valset was supplied.
    val_len: Option<usize>,
}

impl<'a> OptimizeTarget<'a> {
    /// A module-lane target: typed module + trainset (by reference) + metric.
    ///
    /// Runs the naming pass: each leaf declared by [`Predictors`] is stamped
    /// with its declared name so traces, candidates, and persistence agree.
    pub fn module<E, M, MT>(module: &'a mut M, trainset: &'a [E], metric: &'a MT) -> Self
    where
        E: ToInput<M::Input> + Serialize + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        Self::module_with_valset(module, trainset, None, metric)
    }

    /// [`module`](Self::module) with an optional validation set.
    ///
    /// When `valset` is `Some`, its examples become the *leading* columns of
    /// the target ([`val_columns`](Self::val_columns)) and the trainset the
    /// trailing ones ([`train_columns`](Self::train_columns)) — the layout
    /// GEPA's Pareto bookkeeping uses. When `None`, both views cover the
    /// whole trainset.
    pub fn module_with_valset<E, M, MT>(
        module: &'a mut M,
        trainset: &'a [E],
        valset: Option<&'a [E]>,
        metric: &'a MT,
    ) -> Self
    where
        E: ToInput<M::Input> + Serialize + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        // Naming pass: stamp each leaf with its declared name (once per run).
        for (name, info) in module.predictors_mut() {
            info.set_trace_name(&name);
        }
        let named = module.predictors();
        let leaves: Vec<LeafInfo> = named
            .iter()
            .map(|(name, info)| LeafInfo::from_predictor(name, *info))
            .collect();
        let baseline = module_baseline(&named);
        drop(named);

        let val_len = valset.map(<[E]>::len);
        let val = valset.unwrap_or(&[]);
        let uids = val
            .iter()
            .chain(trainset.iter())
            .map(canonical_hash)
            .collect();

        Self {
            lane: Box::new(ModuleLane {
                module,
                val,
                train: trainset,
                metric,
                uids,
                baseline,
            }),
            leaves,
            val_len,
        }
    }

    /// A program-lane target: interpreter-loaded program + labeled examples
    /// (by reference) + JSON metric.
    pub fn program<MT>(interp: &'a Interpreter, examples: &'a [DemoRow], metric: &'a MT) -> Self
    where
        MT: ProgramMetric,
    {
        let leaves = program_leaves(interp);
        let uids = examples.iter().map(canonical_hash).collect();
        let program_tag = format!("{:016x}", interp.program().meta.program_hash);
        Self {
            lane: Box::new(ProgramLane {
                interp,
                examples,
                metric,
                uids,
                program_tag,
                winner: None,
            }),
            leaves,
            val_len: None,
        }
    }

    /// The optimizable leaves' read surface (snapshotted at construction).
    pub fn leaves(&self) -> &[LeafInfo] {
        &self.leaves
    }

    pub fn num_examples(&self) -> usize {
        self.lane.num_examples()
    }

    /// Whether this target carries a distinct validation set.
    pub fn has_valset(&self) -> bool {
        self.val_len.is_some()
    }

    /// The scoring columns: the validation prefix when a valset was supplied,
    /// else every example.
    pub fn val_columns(&self) -> Vec<usize> {
        match self.val_len {
            Some(len) => (0..len).collect(),
            None => (0..self.num_examples()).collect(),
        }
    }

    /// The minibatch-sampling pool: the trainset suffix when a valset was
    /// supplied, else every example.
    pub fn train_columns(&self) -> Vec<usize> {
        match self.val_len {
            Some(len) => (len..self.num_examples()).collect(),
            None => (0..self.num_examples()).collect(),
        }
    }

    /// Installs the winning candidate — the **one** mutation of the run.
    ///
    /// Module lane: merges each slot into the named leaf's state through
    /// [`PredictorInfo::load_state`]. Program lane: binds the winner to an
    /// overlay retrievable via [`winner_overlay`](Self::winner_overlay)
    /// (bake it with [`Program::bake`](crate::ir::Program::bake)).
    pub fn install(&mut self, winner: &Candidate) -> Result<()> {
        self.lane.install(winner)
    }

    /// The installed winner as a bound overlay (program lane only).
    pub fn winner_overlay(&self) -> Option<Arc<Overlay>> {
        self.lane.winner_overlay()
    }

    pub(crate) fn baseline(&self) -> u64 {
        self.lane.baseline()
    }

    pub(crate) fn example_uid(&self, idx: usize) -> u64 {
        self.lane.example_uid(idx)
    }

    pub(crate) fn bind(&self, payload: &CandidatePayload) -> Result<BoundCandidate> {
        self.lane.bind(payload)
    }

    pub(crate) async fn run(
        &self,
        idx: usize,
        bound: BoundCandidate,
        candidate_hash: u64,
    ) -> Result<(Eval, Trace)> {
        self.lane.run(idx, bound, candidate_hash).await
    }

    /// Runs the given examples under `candidate` and returns the bare output
    /// values (no metric, no trace capture) — GEPA's best-output collection.
    /// Sequential, in index order.
    pub async fn candidate_outputs(
        &self,
        indices: &[usize],
        candidate: &Candidate,
    ) -> Result<Vec<Value>> {
        let payload = CandidatePayload::Params {
            candidate: candidate.clone(),
            params: Arc::new(candidate.to_params()),
        };
        let bound = self.lane.bind(&payload)?;
        let mut outputs = Vec::with_capacity(indices.len());
        for &idx in indices {
            outputs.push(self.lane.output(idx, bound.clone()).await?);
        }
        Ok(outputs)
    }
}
