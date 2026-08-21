use anyhow::{Result, anyhow};
use bon::Builder;
use rand::seq::SliceRandom;
use tracing::debug;

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{Candidate, Engine, EngineConfig, EvalOutcome};
use crate::optimizer::harvest::{collect_demo_candidates, select_demos};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::trace::Trace;
use crate::{Module, Predictors};

/// The whole-program score recorded on a rollout trace, if a metric ran.
fn trace_score(trace: &Trace) -> Option<f64> {
    trace
        .outcome
        .as_ref()
        .and_then(|outcome| outcome.eval.as_ref())
        .map(|eval| eval.score)
}

/// Library of general prompting best practices used to seed candidate generation.
///
/// These tips are appended to candidate instructions during [`MIPROv2`] optimization
/// to introduce diversity. Each candidate gets a different tip from the rotation.
pub struct PromptingTips {
    pub tips: Vec<String>,
}

impl PromptingTips {
    pub fn default_tips() -> Self {
        Self {
            tips: vec![
                "Use clear and specific language".to_string(),
                "Provide context about the task domain".to_string(),
                "Specify the desired output format".to_string(),
                "Use chain-of-thought reasoning for complex tasks".to_string(),
                "Include few-shot examples when helpful".to_string(),
                "Break down complex instructions into steps".to_string(),
                "Use role-playing (e.g., 'You are an expert...') when appropriate".to_string(),
                "Specify constraints and edge cases".to_string(),
                "Request explanations or reasoning when needed".to_string(),
                "Use structured output formats (JSON, lists, etc.) when applicable".to_string(),
                "Consider the model's strengths and limitations".to_string(),
                "Be explicit about what to avoid or exclude".to_string(),
                "Use positive framing (what to do vs. what not to do)".to_string(),
                "Provide examples of both correct and incorrect outputs when useful".to_string(),
                "Use delimiters or markers to separate different sections".to_string(),
            ],
        }
    }

    pub fn format_for_prompt(&self) -> String {
        self.tips
            .iter()
            .enumerate()
            .map(|(i, tip)| format!("{}. {}", i + 1, tip))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Renders a leaf's field contract in MIPRO's program-description format.
fn format_leaf_fields(leaf: &LeafInfo) -> String {
    let mut result = String::new();

    result.push_str("Input Fields:\n");
    for (name, docs) in &leaf.input_fields {
        let desc = if docs.is_empty() { "No description" } else { docs };
        result.push_str(&format!("  - {name}: {desc}\n"));
    }

    result.push_str("\nOutput Fields:\n");
    for (name, docs) in &leaf.output_fields {
        let desc = if docs.is_empty() { "No description" } else { docs };
        result.push_str(&format!("  - {name}: {desc}\n"));
    }

    result
}

/// Trace-guided instruction and demo optimizer.
///
/// MIPROv2 (Multi-prompt Instruction PRoposal Optimizer v2) is a thin strategy
/// over the shared [`Engine`], working in four phases:
///
/// 1. **Trace collection** — one traced teacher pass over the trainset (the
///    engine's baseline candidate), collecting whole-program scores plus
///    per-`Predict` input/output spans
/// 2. **Demo bootstrapping** — successful spans scoring at least
///    `min_demo_score` (their own span eval when the metric attached one,
///    the rollout score otherwise) become few-shot demos on the predictor
///    that produced them via the trace name-join (top
///    `max_bootstrapped_demos` by score, deduplicated on inputs), folded
///    into the accumulating winner candidate
/// 3. **Candidate generation** — uses the traces and prompting tips to generate
///    `num_candidates` instruction variants per predictor
/// 4. **Trial evaluation** — evaluates up to `num_trials` candidates on a sampled
///    minibatch through the engine (cached, budget-metered fan-out), keeps the best
///
/// The accumulated winner (demos + best instructions) is installed once at
/// the end through [`OptimizeTarget::install`].
///
/// Unlike [`GEPA`](crate::GEPA), MIPROv2 does not require feedback — only numerical scores.
/// Unlike [`COPRO`](crate::COPRO), it uses execution traces to inform candidate generation
/// rather than blind search.
///
/// # Hyperparameters
///
/// - **`num_candidates`** (default: 10) — instruction variants generated per predictor.
/// - **`num_trials`** (default: 20) — maximum candidates evaluated per predictor.
///   If `num_trials` < `num_candidates`, only the first `num_trials` are evaluated.
/// - **`minibatch_size`** (default: 25) — examples per candidate evaluation.
/// - **`max_bootstrapped_demos`** (default: 4) — demos installed per predictor.
/// - **`min_demo_score`** (default: 0.0) — score gate for demo-eligible spans
///   (span eval when present, rollout score otherwise).
/// - **`eval_concurrency`** (default: 16) — LM calls in flight during evaluation.
/// - **`seed`** — fixes minibatch sampling for reproducible runs.
///
/// # Errors
///
/// Any LM-call or metric failure during the teacher pass or a trial evaluation
/// propagates and aborts the run (the engine's evaluation contract).
///
/// # Cost
///
/// Roughly `num_predictors × (trainset_size + num_trials × minibatch_size)` LM calls,
/// minus whatever the engine's rollout cache serves for free (duplicate candidates
/// deduplicate by content hash and re-seen examples cost nothing).
///
/// ```ignore
/// let mipro = MIPROv2::builder()
///     .num_candidates(10)
///     .num_trials(20)
///     .build();
/// mipro.compile_module(&mut module, &trainset, &metric).await?;
/// ```
#[derive(Builder)]
pub struct MIPROv2 {
    /// Instruction variants generated per predictor.
    #[builder(default = 10)]
    pub num_candidates: usize,

    /// Maximum candidates evaluated per predictor.
    #[builder(default = 20)]
    pub num_trials: usize,

    /// Examples per candidate evaluation.
    #[builder(default = 25)]
    pub minibatch_size: usize,

    /// Maximum demos bootstrapped per predictor from successful traces.
    #[builder(default = 4)]
    pub max_bootstrapped_demos: usize,

    /// Minimum score a span needs to qualify as a bootstrapped demo: its own
    /// eval when the metric attached one
    /// ([`TypedMetric::evaluate_spans`](crate::evaluate::TypedMetric::evaluate_spans)),
    /// the whole-program score otherwise.
    #[builder(default = 0.0)]
    pub min_demo_score: f64,

    /// Concurrent LM calls in flight during candidate evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,

    /// Seed for minibatch sampling. `None` uses a nondeterministic seed.
    pub seed: Option<u64>,
}

impl MIPROv2 {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            seed: self.seed,
            ..OptimizerCommon::default()
        }
    }

    fn generate_candidate_instructions(
        &self,
        program_description: &str,
        traces: &[Trace],
        num_candidates: usize,
    ) -> Vec<String> {
        let tips = PromptingTips::default_tips();
        let score_hint = traces.iter().filter_map(trace_score).fold(0.0f64, f64::max);

        (0..num_candidates)
            .map(|idx| {
                let tip = &tips.tips[idx % tips.tips.len()];
                format!(
                    "{program_description}\n\nOptimization candidate {}:\n- {}\n- Target score >= {:.3}",
                    idx + 1,
                    tip,
                    score_hint
                )
            })
            .collect()
    }

    /// Convenience: optimizes a typed module over a trainset with this
    /// optimizer's default engine, installing demos + winning instructions.
    pub async fn compile_module<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        metric: &MT,
    ) -> Result<()>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        let mut target = OptimizeTarget::module(module, trainset, metric);
        let mut engine = Engine::new(Optimizer::engine_config(self));
        Optimizer::compile(self, &mut target, &mut engine).await?;
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl Optimizer for MIPROv2 {
    fn engine_config(&self) -> EngineConfig {
        self.common().engine_config()
    }

    async fn compile(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report> {
        let leaves = target.leaves().to_vec();
        if leaves.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }

        let mut rng = self.common().rng();

        // Phase 1: one traced teacher pass over the trainset — the engine's
        // baseline candidate. Whole-program traces feed candidate generation;
        // per-predictor spans feed demo bootstrapping via the trace name-join
        // (spans record the names stamped by the target's naming pass).
        let baseline = engine.register(Candidate::new());
        let baseline_eval = match engine.evaluate(target, baseline, None).await? {
            EvalOutcome::Complete(eval) => eval,
            EvalOutcome::BudgetExhausted { needed } => {
                return Err(anyhow!(
                    "budget exhausted ({needed} rollouts needed) during the teacher pass"
                ));
            }
        };
        let scored_traces: Vec<(f64, &Trace)> = baseline_eval
            .rollouts
            .iter()
            .filter_map(|rollout| {
                rollout
                    .trace
                    .as_ref()
                    .map(|trace| (rollout.eval.score, trace))
            })
            .collect();

        // Phase 2: bootstrap demos from successful spans of well-scored
        // rollouts. They fold into the accumulating winner so instruction
        // candidates are scored against the demos they will ship with.
        let mut current = Candidate::new();
        let bootstrapped_demos = select_demos(
            collect_demo_candidates(scored_traces, self.min_demo_score),
            self.max_bootstrapped_demos,
        );
        for (predictor_name, demos) in bootstrapped_demos {
            debug!(
                predictor = %predictor_name,
                demo_count = demos.len(),
                "bootstrapping demos into the candidate"
            );
            current.set_demos(predictor_name, demos);
        }

        let traces: Vec<Trace> = baseline_eval
            .rollouts
            .into_iter()
            .filter_map(|rollout| rollout.trace)
            .collect();

        // Phase 3: per-predictor instruction search on a sampled minibatch —
        // one minibatch per predictor round so all candidates score on the
        // same examples and remain comparable.
        let all_indices: Vec<usize> = (0..target.num_examples()).collect();
        for leaf in &leaves {
            let signature_desc = format_leaf_fields(leaf);
            let instructions =
                self.generate_candidate_instructions(&signature_desc, &traces, self.num_candidates);

            let minibatch_size = all_indices.len().min(self.minibatch_size.max(1));
            let minibatch: Vec<usize> = all_indices
                .choose_multiple(&mut rng, minibatch_size)
                .copied()
                .collect();

            let mut best: Option<(f64, String)> = None;
            for instruction in instructions.into_iter().take(self.num_trials.max(1)) {
                let mut candidate = current.clone();
                candidate.set_instruction(&leaf.name, &instruction);
                let row = engine.register(candidate);
                let eval = match engine.evaluate(target, row, Some(&minibatch)).await? {
                    EvalOutcome::Complete(eval) => eval,
                    EvalOutcome::BudgetExhausted { needed } => {
                        return Err(anyhow!(
                            "budget exhausted ({needed} rollouts needed) during trial evaluation"
                        ));
                    }
                };
                let score = eval.mean();
                if best.as_ref().is_none_or(|(top, _)| score > *top) {
                    best = Some((score, instruction));
                }
            }

            let (_, best_instruction) =
                best.ok_or_else(|| anyhow!("no candidates to evaluate"))?;
            current.set_instruction(&leaf.name, best_instruction);
        }

        // The one mutation of the run: install demos + winning instructions.
        target.install(&current)?;

        Ok(Report::None)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::{Result, anyhow};

    use super::*;
    use crate::evaluate::{Eval, TypedMetric};
    use crate::{CallMetadata, Predict, PredictError, Predicted, PredictorInfo, Signature};

    #[derive(Signature, Clone, Debug)]
    struct MiproStateSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    struct MiproStateModule {
        predictor: Predict<MiproStateSig>,
    }

    crate::predictors!(MiproStateModule { predictor });

    impl Module for MiproStateModule {
        type Input = MiproStateSigInput;
        type Output = MiproStateSigOutput;

        async fn forward(
            &self,
            input: MiproStateSigInput,
        ) -> Result<Predicted<MiproStateSigOutput>, PredictError> {
            Ok(Predicted::new(
                MiproStateSigOutput {
                    answer: input.prompt,
                },
                CallMetadata::default(),
            ))
        }
    }

    struct AlwaysFailMetric;

    type MiproRow = (MiproStateSigInput, MiproStateSigOutput);

    impl TypedMetric<MiproRow, MiproStateModule> for AlwaysFailMetric {
        async fn evaluate(
            &self,
            _example: &MiproRow,
            _prediction: &Predicted<MiproStateSigOutput>,
            _trace: Option<&Trace>,
        ) -> Result<Eval> {
            Err(anyhow!("metric failure"))
        }
    }

    fn trainset() -> Vec<MiproRow> {
        vec![(
            MiproStateSigInput {
                prompt: "one".to_string(),
            },
            MiproStateSigOutput {
                answer: "one".to_string(),
            },
        )]
    }

    #[tokio::test]
    async fn compile_leaves_state_untouched_when_metric_errors() {
        let optimizer = MIPROv2::builder()
            .num_candidates(2)
            .num_trials(1)
            .minibatch_size(1)
            .build();
        let mut module = MiproStateModule {
            predictor: Predict::<MiproStateSig>::builder()
                .instruction("seed-instruction")
                .build(),
        };

        let err = optimizer
            .compile_module(&mut module, &trainset(), &AlwaysFailMetric)
            .await
            .expect_err("candidate evaluation should propagate metric failure");
        assert!(err.to_string().contains("metric failure"));

        assert_eq!(
            PredictorInfo::instruction(&module.predictor),
            "seed-instruction"
        );
    }
}
