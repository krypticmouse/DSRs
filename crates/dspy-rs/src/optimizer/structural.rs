//! Structural: LM-guided hill-climbing over the graph-edit calculus
//! (RFC 0004 §6) — the sixth strategy over the shared [`Engine`].
//!
//! Where the other five strategies tune parameter *values* through overlays,
//! Structural proposes [`ir::Edit`](crate::ir::Edit)s: each generation it
//! gathers the [`Program::legal_edits`] menu, has a reflection LM choose one
//! edit from the serialized menu plus the incumbent's evaluation feedback,
//! applies it via [`Program::edited`], carries the tuned overlay across the
//! structural change with [`migrate_overlay`], and accepts the child through
//! the engine's minibatch gate — parent and child scored on the same shared
//! minibatch, winner kept.

use std::num::NonZeroU32;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use bon::Builder;
use rand::{Rng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

use crate::evaluate::Eval;
use crate::ir::builder::cot_reasoning_field;
use crate::ir::edit::{Edit, EditKind, SwapTarget, migrate_overlay};
use crate::ir::graph::{Node, NodeBudget, NodeId, Program, StopSpec};
use crate::ir::interp::{Interpreter, RuntimeEnv};
use crate::ir::params::{DemoRow, Overlay};
use crate::optimizer::OptimizerCommon;
use crate::optimizer::engine::{Engine, EngineConfig, EvalOutcome, GateOutcome, Spend};
use crate::optimizer::target::{OptimizeTarget, ProgramMetric};
use crate::trace::Trace;
use crate::utils::truncate;
use crate::{Predict, Signature};

/// Choose one structural edit for an LLM-pipeline program.
///
/// You are optimizing the structure of an LLM pipeline program. Study the
/// program source, the menu of legal structural edits, and the per-example
/// scores and feedback from the current program's last evaluation. Choose the
/// single edit most likely to fix the failure modes the feedback names: add
/// reasoning where answers are shallow, add or drop tools where tool use goes
/// wrong, wrap flaky steps in a retry, remove steps that only add noise.
/// Return only the `option` number of the chosen menu entry, with no preamble
/// or commentary.
#[derive(Signature, Clone, Debug)]
// The struct itself is only a schema: the derive generates the
// `ChooseEditInput`/`ChooseEditOutput` types the reflection call reads.
#[allow(dead_code)]
struct ChooseEdit {
    /// The program in canonical `.dsrs` text form.
    #[input]
    program_source: String,

    /// The menu of legal structural edits, one JSON object per line, each
    /// with an `option` number.
    #[input]
    edit_menu: String,

    /// Per-example scores and textual feedback from the last evaluation.
    #[input]
    execution_feedback: String,

    /// The `option` number of the chosen edit.
    #[output]
    chosen_option: String,
}

/// One entry of the proposer menu: a leaf, one [`EditKind`] admissible there,
/// and the node id to materialize the concrete [`Edit`] against.
#[derive(Clone, Debug)]
struct MenuEntry {
    leaf: String,
    node: NodeId,
    kind: EditKind,
}

/// What one Structural generation did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructuralStep {
    /// Generation index (0-based).
    pub generation: usize,
    /// The leaf the chosen edit targets.
    pub leaf: String,
    /// The concrete edit that was proposed (serde data — replayable against
    /// the generation's parent, whose hash is `parent_hash`).
    pub edit: Edit,
    /// `program_hash` of the parent the edit was applied to.
    pub parent_hash: u64,
    /// Parent's mean score on the generation's minibatch (the gate threshold).
    pub parent_minibatch_score: f64,
    /// Child's mean on the same minibatch; `None` when the child never scored
    /// (the edit failed to apply or the child failed to load).
    pub child_minibatch_score: Option<f64>,
    /// Whether the gate promoted the child to the new incumbent.
    pub accepted: bool,
    /// Full-set mean of the child; `Some` only when accepted.
    pub full_score: Option<f64>,
    /// Why the child was never scored (apply/load failure), when it wasn't.
    pub rejection: Option<String>,
}

/// What a [`Structural`] run did. The winner is returned, not installed:
/// bake it (`report.program.bake(&report.overlay, note)`) or save it — the
/// incumbent interpreter passed in is never mutated.
#[derive(Clone, Debug)]
pub struct StructuralReport {
    /// The winning program (the input program when nothing was accepted).
    pub program: Arc<Program>,
    /// The incumbent overlay re-minted against the winner via
    /// [`migrate_overlay`] at every accepted edit (empty when no overlay was
    /// supplied and nothing migrated).
    pub overlay: Overlay,
    /// Mean metric score of the input program (+ overlay) over the examples.
    pub baseline_score: f64,
    /// Full-set mean of the final program (baseline if nothing was accepted).
    pub final_score: f64,
    /// The accepted edits, in order. Each applies to the program whose hash
    /// its step's `parent_hash` records (node ids are per-parent handles, so
    /// the vec is a lineage, not a single batch).
    pub edits: Vec<Edit>,
    /// Per-generation outcomes, in order.
    pub steps: Vec<StructuralStep>,
    /// Generations promoted by the gate.
    pub accepted: usize,
    /// Generations rejected (gate losses, apply failures, load failures).
    pub rejected: usize,
    /// Engine spend for the whole run (reflection calls included).
    pub spend: Spend,
}

/// Structural optimizer over the graph-edit calculus (RFC 0004 §6).
///
/// Structural is a thin strategy over the shared [`Engine`], but it searches
/// program *structure* instead of parameter values: candidates are whole
/// programs minted by [`Program::edited`], and the tuned overlay follows the
/// incumbent across every accepted edit via [`migrate_overlay`]. It runs on
/// the program lane only — the module lane has no skeleton to edit.
///
/// Each generation:
///
/// 1. **Sample** a shared minibatch (seeded RNG, indices sorted). The
///    incumbent's minibatch mean is the gate threshold; it is served from the
///    rollout cache (the incumbent always has full coverage), so re-scoring
///    the parent costs nothing.
/// 2. **Menu** — gather [`Program::legal_edits`] for every leaf and keep the
///    kinds Structural can materialize without free text: `AugmentSig` (the
///    CoT move), `SwapToAgent`/`SwapToPredict`, `WrapRetry`, `Remove`, and
///    per-tool `AddTool`/`RemoveTool`. `SetStop` and `SetInstructionDefault`
///    are left to value-level optimizers.
/// 3. **Choose** — a reflection LM (`prompt_model`) reads the program's
///    canonical `.dsrs` text, the serialized menu, and the incumbent's
///    per-example feedback, and returns one option number. Without a
///    `prompt_model` (or when the reply doesn't parse) the choice degrades to
///    a seeded-uniform pick from the menu.
/// 4. **Apply** — [`Program::edited`] mints the child; [`migrate_overlay`]
///    re-mints the incumbent overlay against it; the child is loaded through
///    the caller-supplied [`RuntimeEnv`] factory. An edit that fails to
///    apply, a child that fails validation, or a child that fails to load is
///    recorded and skipped — never a panic, never an abort.
/// 5. **Gate** — the child is accepted through the engine's minibatch gate:
///    only if its mean on the shared minibatch strictly beats the parent's
///    does it promote to a full-set evaluation and become the new incumbent.
///
/// Every child is a fresh program that must be re-scored from scratch (its
/// hash keys its own rollout-cache rows), so the budget caps are the real
/// control surface: `max_rollouts` / `max_lm_calls` stop the run cleanly
/// when the next batch wouldn't fit.
///
/// # Hyperparameters
///
/// - **`num_iterations`** (default: 8) — structural generations to attempt.
/// - **`minibatch_size`** (default: 8) — examples in the shared minibatch
///   parent and child are compared on.
/// - **`prompt_model`** — reflection LM that chooses edits from the menu.
///   Strongly recommended; without it the choice is a seeded-uniform pick.
/// - **`max_rollouts`** / **`max_lm_calls`** — hard budget caps. Every
///   accepted child costs a full-set evaluation on top of its minibatch.
/// - **`eval_concurrency`** (default: 16) — rollouts in flight during
///   evaluation.
/// - **`seed`** — fixes minibatch sampling and the fallback edit choice.
///
/// # Cost
///
/// `examples.len()` for the baseline pass, then per generation:
/// `minibatch_size` rollouts for the gate, plus the remaining
/// `examples.len() - minibatch_size` only on promotion, plus one reflection
/// call when a `prompt_model` is set. Rejected children never pay for a full
/// pass.
///
/// ```ignore
/// let structural = Structural::builder()
///     .num_iterations(8)
///     .max_rollouts(Some(400))
///     .prompt_model(reflection_lm)
///     .build();
/// let report = structural
///     .compile_program(&interp, &examples, &metric, || {
///         RuntimeEnv::new().bind_model("m", lm.clone())
///     })
///     .await?;
/// println!("{:.3} -> {:.3}", report.baseline_score, report.final_score);
/// let baked = report.program.bake(&report.overlay, Lineage::default())?;
/// ```
#[derive(Builder)]
pub struct Structural {
    /// Structural generations to attempt (one proposed edit each).
    #[builder(default = 8)]
    pub num_iterations: usize,

    /// Examples in the shared minibatch parent and child are compared on.
    #[builder(default = 8)]
    pub minibatch_size: usize,

    /// Reflection LM that chooses an edit from the serialized menu. Without
    /// it, the choice degrades to a seeded-uniform pick.
    pub prompt_model: Option<crate::LM>,

    /// Hard cap on evaluation rollouts. `None` = unlimited.
    pub max_rollouts: Option<usize>,
    /// Hard cap on LM call units (rollouts + reflection). `None` = unlimited.
    pub max_lm_calls: Option<usize>,

    /// Concurrent rollouts in flight during evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,

    /// Seed for minibatch sampling and the fallback edit choice. `None` uses
    /// a nondeterministic seed.
    pub seed: Option<u64>,
}

/// Per-example bookkeeping for the incumbent: metric result plus the trace
/// when the engine ran it fresh (cache-served cells carry no trace).
type RolloutStore = Vec<Option<(Eval, Option<Trace>)>>;

impl Structural {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            max_metric_calls: self.max_rollouts,
            max_lm_calls: self.max_lm_calls,
            seed: self.seed,
            ..OptimizerCommon::default()
        }
    }

    /// The engine configuration `compile_program` runs with.
    pub fn engine_config(&self) -> EngineConfig {
        self.common().engine_config()
    }

    /// The proposer menu: every leaf's [`Program::legal_edits`], filtered to
    /// the kinds Structural can materialize without free text. `AugmentSig`
    /// is dropped when the leaf's signature already carries the reasoning
    /// field (the edit would only fail with `DuplicateField`).
    fn menu(program: &Program) -> Vec<MenuEntry> {
        let reasoning = cot_reasoning_field();
        let mut menu = Vec::new();
        for (id, node) in program.nodes.iter() {
            let Some(leaf) = program.leaf_name(id) else {
                continue;
            };
            let sig = match node {
                Node::Predict(n) => Some(n.sig),
                Node::AgentLoop(n) => Some(n.sig),
                _ => None,
            };
            for kind in program.legal_edits(id) {
                match kind {
                    EditKind::SetStop | EditKind::SetInstructionDefault => continue,
                    EditKind::AugmentSig => {
                        let taken = sig.is_some_and(|sig| {
                            let def = &program.sigs[sig];
                            def.inputs
                                .iter()
                                .chain(def.outputs.iter())
                                .any(|f| f.name == reasoning.name)
                        });
                        if taken {
                            continue;
                        }
                    }
                    _ => {}
                }
                menu.push(MenuEntry {
                    leaf: leaf.to_string(),
                    node: id,
                    kind,
                });
            }
        }
        menu
    }

    /// One human-readable line per menu entry, for the reflection prompt.
    fn render_menu(program: &Program, menu: &[MenuEntry]) -> String {
        let tool_name = |tool| program.syms.get(program.tools[tool].name);
        let all_tools = || {
            program
                .tools
                .values()
                .map(|t| program.syms.get(t.name))
                .collect::<Vec<_>>()
                .join(", ")
        };
        menu.iter()
            .enumerate()
            .map(|(option, entry)| {
                let note = match entry.kind {
                    EditKind::AugmentSig => {
                        "prepend a chain-of-thought `reasoning` output field".to_string()
                    }
                    EditKind::SwapToAgent => format!(
                        "swap this `predict` leaf into a tool-using `agent` loop (tools: [{}])",
                        all_tools()
                    ),
                    EditKind::SwapToPredict => {
                        "swap this `agent` leaf back into a plain `predict`".to_string()
                    }
                    EditKind::WrapRetry => {
                        "wrap this node in a retry (2 attempts, feedback on)".to_string()
                    }
                    EditKind::Remove => "remove this step from its `seq`".to_string(),
                    EditKind::AddTool { tool } => {
                        format!("declare tool `{}` on this agent", tool_name(tool))
                    }
                    EditKind::RemoveTool { tool } => {
                        format!("undeclare tool `{}` from this agent", tool_name(tool))
                    }
                    EditKind::SetStop | EditKind::SetInstructionDefault => {
                        unreachable!("filtered out of the menu")
                    }
                };
                serde_json::json!({
                    "option": option,
                    "leaf": entry.leaf,
                    "edit": entry.kind,
                    "note": note,
                })
                .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Materializes a chosen menu entry into a concrete [`Edit`] with fixed,
    /// conservative parameters (the CoT reasoning field, default stop/budget,
    /// 2 retry attempts).
    fn materialize(program: &Program, entry: &MenuEntry) -> Edit {
        match entry.kind {
            EditKind::AugmentSig => Edit::AugmentSig {
                leaf: entry.node,
                prepend: cot_reasoning_field(),
            },
            EditKind::SwapToAgent => Edit::SwapLeaf {
                leaf: entry.node,
                to: SwapTarget::Agent {
                    tools: program.tools.keys().collect(),
                    stop: StopSpec::default(),
                    budget: NodeBudget::default(),
                },
            },
            EditKind::SwapToPredict => Edit::SwapLeaf {
                leaf: entry.node,
                to: SwapTarget::Predict,
            },
            EditKind::WrapRetry => Edit::WrapRetry {
                node: entry.node,
                max_attempts: NonZeroU32::new(2).expect("2 is nonzero"),
                backoff_ms: 0,
                feedback: true,
            },
            EditKind::Remove => Edit::Remove { node: entry.node },
            EditKind::AddTool { tool } => Edit::AddTool {
                agent: entry.node,
                tool,
            },
            EditKind::RemoveTool { tool } => Edit::RemoveTool {
                agent: entry.node,
                tool,
            },
            EditKind::SetStop | EditKind::SetInstructionDefault => {
                unreachable!("filtered out of the menu")
            }
        }
    }

    /// Extracts an in-range option number from the reflection LM's reply:
    /// the first contiguous digit run (so "Option 3" parses as 3). `None`
    /// when nothing parses or the number is out of range.
    fn parse_choice(reply: &str, menu_len: usize) -> Option<usize> {
        let digits: String = reply
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(char::is_ascii_digit)
            .collect();
        let option: usize = digits.parse().ok()?;
        (option < menu_len).then_some(option)
    }

    /// Formats the incumbent's minibatch scores/feedback plus any errored
    /// spans from stored traces — what the reflection LM sees.
    fn summarize_feedback(store: &RolloutStore, minibatch: &[usize]) -> String {
        use std::fmt::Write as _;

        let mut text = String::new();
        for &idx in minibatch {
            let Some((eval, trace)) = &store[idx] else {
                continue;
            };
            let _ = writeln!(
                text,
                "{}: score={:.3}; {}",
                idx + 1,
                eval.score,
                eval.feedback.as_deref().unwrap_or("-")
            );
            let Some(trace) = trace else {
                continue;
            };
            for span in &trace.spans {
                if let Some(error) = &span.error {
                    let _ = writeln!(
                        text,
                        "  {} call {}: <{}: {}>",
                        trace.component_name(span.component),
                        span.seq,
                        error.kind.as_str(),
                        truncate(span.raw_output.as_deref().unwrap_or(&error.message), 500)
                    );
                }
            }
        }
        text.trim_end().to_string()
    }

    /// Chooses a menu option, preferring LM reflection when a `prompt_model`
    /// is configured. Returns the option index and the number of reflection
    /// LM calls consumed (0 or 1). Reflection failures degrade to the
    /// seeded-uniform fallback with a warning rather than aborting the run.
    async fn choose(
        &self,
        program: &Program,
        menu: &[MenuEntry],
        execution_feedback: &str,
        generation: usize,
        reflector: Option<&Predict<ChooseEdit>>,
        rng: &mut StdRng,
    ) -> (usize, usize) {
        let fallback = |rng: &mut StdRng| rng.gen_range(0..menu.len());

        let Some(reflector) = reflector else {
            return (fallback(rng), 0);
        };

        let input = ChooseEditInput {
            program_source: program.to_dsrs(),
            edit_menu: Self::render_menu(program, menu),
            execution_feedback: execution_feedback.to_string(),
        };

        match reflector.call(input).await {
            Ok(predicted) => match Self::parse_choice(&predicted.chosen_option, menu.len()) {
                Some(option) => return (option, 1),
                None => {
                    tracing::warn!(
                        generation,
                        reply = %truncate(&predicted.chosen_option, 200),
                        "reflection LM reply is not a menu option; picking uniformly"
                    );
                }
            },
            Err(err) => {
                tracing::warn!(
                    generation,
                    error = %err,
                    "reflection LM call failed; picking uniformly"
                );
            }
        }

        (fallback(rng), 1)
    }

    /// Runs the structural search over a loaded program.
    ///
    /// `interp` is the incumbent (never mutated); `env` supplies a fresh
    /// [`RuntimeEnv`] every time a child program needs loading — the same
    /// model/tool/sandbox bindings the incumbent was loaded with. The winner
    /// comes back in the report as a program plus a migrated overlay.
    pub async fn compile_program<M, F>(
        &self,
        interp: &Interpreter,
        examples: &[DemoRow],
        metric: &M,
        env: F,
    ) -> Result<StructuralReport>
    where
        M: ProgramMetric,
        F: Fn() -> RuntimeEnv,
    {
        self.compile_program_with_overlay(interp, None, examples, metric, env)
            .await
    }

    /// [`compile_program`](Self::compile_program) with an incumbent overlay —
    /// tuned slot values from a prior value-level optimizer, minted against
    /// `interp`'s program. The overlay is applied during every incumbent
    /// evaluation and carried across each accepted edit with
    /// [`migrate_overlay`].
    pub async fn compile_program_with_overlay<M, F>(
        &self,
        interp: &Interpreter,
        overlay: Option<Overlay>,
        examples: &[DemoRow],
        metric: &M,
        env: F,
    ) -> Result<StructuralReport>
    where
        M: ProgramMetric,
        F: Fn() -> RuntimeEnv,
    {
        if examples.is_empty() {
            return Err(anyhow!("no examples to optimize over"));
        }
        let mut program = Arc::clone(interp.program());
        let mut incumbent = overlay.unwrap_or_else(|| Overlay::new(&program));
        if incumbent.base != program.meta.program_hash {
            return Err(anyhow!(
                "overlay minted against program {:016x}, expected {:016x}",
                incumbent.base,
                program.meta.program_hash
            ));
        }

        let mut engine = Engine::new(self.engine_config());
        let reflector = self
            .prompt_model
            .as_ref()
            .map(|lm| Predict::<ChooseEdit>::builder().lm(lm.clone()).build());
        let mut rng = self.common().rng();

        let num_examples = examples.len();
        let all_indices: Vec<usize> = (0..num_examples).collect();

        // Baseline: the incumbent (+ overlay) over the full example set,
        // traced. Seeds the rollout store and the cache — every later parent
        // minibatch read is served for free.
        let mut incumbent_row = engine.register_overlay(incumbent.clone());
        let baseline_eval = {
            let target = OptimizeTarget::program(interp, examples, metric);
            match engine.evaluate(&target, incumbent_row, None).await? {
                EvalOutcome::Complete(eval) => eval,
                EvalOutcome::BudgetExhausted { needed } => {
                    return Err(anyhow!(
                        "budget too small for the baseline pass ({needed} rollouts needed)"
                    ));
                }
            }
        };
        let baseline_score = baseline_eval.mean();
        let mut final_score = baseline_score;

        let mut store: RolloutStore = vec![None; num_examples];
        for rollout in &baseline_eval.rollouts {
            store[rollout.example] = Some((rollout.eval.clone(), rollout.trace.clone()));
        }

        // The incumbent interpreter: the caller's until an edit is accepted.
        let mut owned: Option<Interpreter> = None;

        let mut edits = Vec::new();
        let mut steps = Vec::new();
        let mut accepted = 0usize;
        let mut rejected = 0usize;

        for generation in 0..self.num_iterations {
            // 1. Shared minibatch; sorted for deterministic evaluation order.
            let minibatch_size = num_examples.min(self.minibatch_size.max(1));
            let mut minibatch: Vec<usize> = all_indices
                .choose_multiple(&mut rng, minibatch_size)
                .copied()
                .collect();
            minibatch.sort_unstable();

            // Don't spend a reflection call on a child we can't afford to
            // score on the minibatch.
            if !engine.budget_allows(minibatch.len()) {
                break;
            }

            // 2. The menu against the current incumbent.
            let menu = Self::menu(&program);
            if menu.is_empty() {
                tracing::warn!(
                    generation,
                    "no legal edits at any leaf; stopping the structural search"
                );
                break;
            }

            // Parent's mean on the shared minibatch — the gate threshold.
            // Cache-served (the incumbent always has full coverage).
            let threshold = {
                let cur = owned.as_ref().unwrap_or(interp);
                let target = OptimizeTarget::program(cur, examples, metric);
                match engine
                    .evaluate(&target, incumbent_row, Some(&minibatch))
                    .await?
                {
                    EvalOutcome::Complete(eval) => eval.mean(),
                    EvalOutcome::BudgetExhausted { .. } => break,
                }
            };

            // 3. Choose one edit from the menu.
            let execution_feedback = format!(
                "Incumbent minibatch mean: {threshold:.3}\n{}",
                Self::summarize_feedback(&store, &minibatch)
            );
            let (option, reflection_calls) = self
                .choose(
                    &program,
                    &menu,
                    &execution_feedback,
                    generation,
                    reflector.as_ref(),
                    &mut rng,
                )
                .await;
            engine.charge(0, reflection_calls);
            let entry = &menu[option];
            let edit = Self::materialize(&program, entry);

            let mut step = StructuralStep {
                generation,
                leaf: entry.leaf.clone(),
                edit: edit.clone(),
                parent_hash: program.meta.program_hash,
                parent_minibatch_score: threshold,
                child_minibatch_score: None,
                accepted: false,
                full_score: None,
                rejection: None,
            };

            // 4. Apply, migrate, load — every failure skips the generation.
            let child_program = match program.edited(std::slice::from_ref(&edit)) {
                Ok(child) => child,
                Err(err) => {
                    tracing::warn!(generation, error = %err, "edit failed to apply; skipping");
                    step.rejection = Some(format!("edit failed: {err}"));
                    rejected += 1;
                    steps.push(step);
                    continue;
                }
            };
            let child_interp = match Interpreter::load(child_program, env()).await {
                Ok(interp) => interp,
                Err(err) => {
                    tracing::warn!(generation, error = %err, "edited child failed to load; skipping");
                    step.rejection = Some(format!("load failed: {err}"));
                    rejected += 1;
                    steps.push(step);
                    continue;
                }
            };
            let migrated = migrate_overlay(&program, &incumbent, child_interp.program());
            let child_row = engine.register_overlay(migrated.clone());

            // 5. The gate: child vs parent on the shared minibatch; only a
            // strict win promotes to the full set.
            let gate = {
                let child_target = OptimizeTarget::program(&child_interp, examples, metric);
                engine
                    .evaluate_gated(&child_target, child_row, &minibatch, threshold)
                    .await?
            };
            match gate {
                GateOutcome::BudgetExhausted { .. } => {
                    steps.push(step);
                    break;
                }
                GateOutcome::Rejected { minibatch: mb_eval } => {
                    step.child_minibatch_score = Some(mb_eval.mean());
                    rejected += 1;
                    steps.push(step);
                }
                GateOutcome::Promoted {
                    minibatch: mb_eval,
                    full,
                } => {
                    // The child is the new incumbent. Refresh the store from
                    // the full pass, preferring fresh minibatch traces over
                    // cache-served cells.
                    for rollout in &full.rollouts {
                        store[rollout.example] =
                            Some((rollout.eval.clone(), rollout.trace.clone()));
                    }
                    for rollout in &mb_eval.rollouts {
                        if rollout.trace.is_some() {
                            store[rollout.example] =
                                Some((rollout.eval.clone(), rollout.trace.clone()));
                        }
                    }
                    final_score = full.mean();
                    step.child_minibatch_score = Some(mb_eval.mean());
                    step.accepted = true;
                    step.full_score = Some(final_score);
                    accepted += 1;
                    steps.push(step);
                    edits.push(edit);

                    program = Arc::clone(child_interp.program());
                    incumbent = migrated;
                    incumbent_row = child_row;
                    owned = Some(child_interp);
                }
            }
        }

        Ok(StructuralReport {
            program,
            overlay: incumbent,
            baseline_score,
            final_score,
            edits,
            steps,
            accepted,
            rejected,
            spend: *engine.spend(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LMConfig;
    use crate::ir::{self, FieldType as T, ProgramBuilder, SignatureDef};

    fn qa_program() -> Program {
        let mut b = ProgramBuilder::new("structural_unit");
        b.model(
            "m",
            LMConfig {
                model: "openai:gpt-4o-mini".to_string(),
                ..LMConfig::default()
            },
        );
        let qa = b.sig(
            SignatureDef::build("QA")
                .instruction("Answer the question.")
                .input("question", T::String)
                .output("answer", T::String)
                .finish()
                .unwrap(),
        );
        let node = ir::predict("answerer", qa).bind("question", ir::input("question"));
        b.main(
            qa,
            ir::seq([node]).out("answer", ir::out("answerer", "answer")),
        )
        .unwrap()
    }

    #[test]
    fn menu_excludes_free_text_kinds_and_taken_reasoning_fields() {
        let program = qa_program();
        let menu = Structural::menu(&program);
        assert!(!menu.is_empty());
        assert!(menu.iter().all(|entry| !matches!(
            entry.kind,
            EditKind::SetStop | EditKind::SetInstructionDefault
        )));
        assert!(
            menu.iter()
                .any(|entry| entry.kind == EditKind::AugmentSig && entry.leaf == "answerer")
        );

        // Once the reasoning field is on the leaf, AugmentSig leaves the menu.
        let leaf = program.leaf_id("answerer").unwrap();
        let child = program
            .edited(&[Edit::AugmentSig {
                leaf,
                prepend: cot_reasoning_field(),
            }])
            .unwrap();
        let child_menu = Structural::menu(&child);
        assert!(
            child_menu
                .iter()
                .all(|entry| entry.kind != EditKind::AugmentSig)
        );
    }

    #[test]
    fn parse_choice_extracts_in_range_options() {
        assert_eq!(Structural::parse_choice("3", 5), Some(3));
        assert_eq!(Structural::parse_choice(" Option 2.", 5), Some(2));
        assert_eq!(Structural::parse_choice("option 12 of 20", 20), Some(12));
        assert_eq!(Structural::parse_choice("9", 5), None);
        assert_eq!(Structural::parse_choice("none of these", 5), None);
        assert_eq!(Structural::parse_choice("", 5), None);
    }

    #[test]
    fn materialized_menu_entries_apply() {
        let program = qa_program();
        for entry in Structural::menu(&program) {
            let edit = Structural::materialize(&program, &entry);
            // Remove orphans the out binding — validate.rs's call, surfaced
            // as an EditError, which the loop records and skips.
            let result = program.edited(std::slice::from_ref(&edit));
            if entry.kind == EditKind::Remove {
                assert!(result.is_err(), "removing the only producer must fail");
            } else {
                assert!(
                    result.is_ok(),
                    "{:?} should apply: {:?}",
                    entry.kind,
                    result.err()
                );
            }
        }
    }
}
