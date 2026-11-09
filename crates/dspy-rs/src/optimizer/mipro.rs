use anyhow::{Result, anyhow};
use bon::Builder;
use rand::{Rng, seq::SliceRandom};
use tracing::debug;

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{Candidate, Engine, EngineConfig, EvalOutcome};
use crate::optimizer::harvest::{collect_demo_candidates, select_demos};
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::trace::Trace;
use crate::{Module, Predictors};

fn set_instruction_if_changed(
    candidate: &mut Candidate,
    leaf: &crate::optimizer::target::LeafInfo,
    instruction: &str,
) {
    if instruction == leaf.instruction {
        let remove = if let Some(slot) = candidate.slots.get_mut(&leaf.name) {
            slot.instruction = None;
            slot.demos.is_none() && !slot.clear_instruction
        } else {
            false
        };
        if remove {
            candidate.slots.remove(&leaf.name);
        }
    } else {
        candidate.set_instruction(&leaf.name, instruction);
    }
}

fn set_demos_if_changed(
    candidate: &mut Candidate,
    leaf: &crate::optimizer::target::LeafInfo,
    demos: Vec<crate::trace::JsonMap>,
) {
    if demos == leaf.demos {
        let remove = if let Some(slot) = candidate.slots.get_mut(&leaf.name) {
            slot.demos = None;
            slot.instruction.is_none() && !slot.clear_instruction
        } else {
            false
        };
        if remove {
            candidate.slots.remove(&leaf.name);
        }
    } else {
        candidate.set_demos(&leaf.name, demos);
    }
}

/// Library of general prompting best practices used to seed candidate generation.
///
/// Available to applications composing their own proposal prompts.
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
///    `max_bootstrapped_demos` by score, deduplicated on inputs). They are
///    sampled alongside incumbent and empty demo sets during joint trials.
/// 3. **Candidate generation** — a proposal LM reads the configured rollout
///    summaries, task contract, and incumbent instruction; the incumbent remains
///    a candidate.
/// 4. **Joint search** — seeded random trials combine instruction and demo
///    choices across all predictors. The best minibatch trial per interval
///    competes against the baseline on the full validation set. This search
///    does not implement upstream's Optuna/TPE sampler.
///
/// The accumulated winner (demos + best instructions) is installed once at
/// the end through [`OptimizeTarget::install`].
///
/// Unlike [`GEPA`](crate::GEPA), MIPROv2 does not require feedback — only numerical scores.
/// Unlike [`COPRO`](crate::COPRO), it uses safe rollout summaries to inform
/// candidate generation rather than blind search.
/// By default, rollout evidence contains only component names and numerical
/// scores. Set `share_training_examples(true)` to also send parsed training
/// inputs, outputs, and metric feedback to the proposal LM. Raw prompts, tool
/// events, and unparsed model output are always excluded unless
/// `share_full_training_traces(true)` is set, which sends the complete raw trace
/// including tool arguments, results, and model outputs.
///
/// # Hyperparameters
///
/// - **`num_candidates`** (default: 10) — instruction variants generated per predictor.
/// - **`num_trials`** (default: 20) — joint instruction/demo combinations evaluated.
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

    /// Joint instruction/demo combinations evaluated.
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
    /// LM used for grounded proposals; defaults to the configured global LM.
    pub prompt_model: Option<crate::LM>,
    /// Share parsed training inputs, outputs, and metric feedback with the
    /// proposal LM. Raw trace content remains excluded.
    #[builder(default = false)]
    pub share_training_examples: bool,
    /// Send complete raw training traces, including tool arguments/results
    /// and unparsed model output, to the proposal LM. Takes precedence over
    /// `share_training_examples`.
    #[builder(default = false)]
    pub share_full_training_traces: bool,
    #[builder(default = 1.0)]
    pub temperature: f32,
    /// Re-evaluate the best minibatch candidate on all validation examples.
    #[builder(default = 5)]
    pub full_eval_steps: usize,
}

impl MIPROv2 {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            seed: self.seed,
            ..OptimizerCommon::default()
        }
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
        anyhow::ensure!(
            self.num_candidates > 0 && self.num_trials > 0 && self.full_eval_steps > 0,
            "candidate, trial, and full evaluation counts must be positive"
        );
        let train = target.train_columns();
        let validation = target.val_columns();
        anyhow::ensure!(
            !train.is_empty() && !validation.is_empty(),
            "training and validation sets must be nonempty"
        );

        // Phase 1: one traced teacher pass over the trainset — the engine's
        // baseline candidate. Whole-program traces feed candidate generation;
        // per-predictor spans feed demo bootstrapping via the trace name-join
        // (spans record the names stamped by the target's naming pass).
        let baseline = engine.register(Candidate::new());
        let baseline_eval = match engine.evaluate(target, baseline, Some(&train)).await? {
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

        // Phase 2: bootstrap demo options from successful spans of well-scored
        // training rollouts. Joint trials sample these alongside incumbent and
        // empty demo sets; only a validation improvement can install them.
        let bootstrapped_demos = select_demos(
            collect_demo_candidates(scored_traces, self.min_demo_score),
            self.max_bootstrapped_demos,
        );
        // Only training traces enter the proposal prompt or demo pool.
        let observations = if self.share_full_training_traces {
            baseline_eval
                .rollouts
                .iter()
                .filter_map(|r| r.trace.as_ref().map(Trace::to_jsonl))
                .take(8)
                .collect::<Result<Vec<_>>>()?
                .join("\n")
        } else {
            let observations = baseline_eval
                .rollouts
                .iter()
                .filter_map(|r| {
                    r.trace.as_ref().map(|trace| {
                        super::observation::proposal_observation(
                            trace,
                            &r.eval,
                            self.share_training_examples,
                            false,
                        )
                    })
                })
                .take(8)
                .collect::<Result<Vec<_>>>()?;
            serde_json::to_string(&observations)?
        };
        let observations = crate::utils::truncate(&observations, 24000);
        let mut instruction_sets = Vec::new();
        for leaf in &leaves {
            let mut instructions = vec![leaf.instruction.clone()];
            instructions.extend(
                super::propose::propose(
                    self.prompt_model.as_ref(),
                    self.temperature,
                    leaf,
                    &leaf.instruction,
                    observations.to_string(),
                    self.num_candidates.saturating_sub(1),
                    engine,
                )
                .await?,
            );
            instruction_sets.push(instructions);
        }
        let score = |outcome: EvalOutcome| -> Result<f64> {
            match outcome {
                EvalOutcome::Complete(eval) => Ok(eval.mean()),
                EvalOutcome::BudgetExhausted { needed } => {
                    Err(anyhow!("budget exhausted ({needed} rollouts needed)"))
                }
            }
        };
        let mut best_score = score(engine.evaluate(target, baseline, Some(&validation)).await?)?;
        let mut winner = Candidate::new();
        // Use one fixed minibatch so trial scores are directly comparable.
        let minibatch: Vec<_> = validation
            .choose_multiple(&mut rng, validation.len().min(self.minibatch_size.max(1)))
            .copied()
            .collect();
        let mut pending: Option<(f64, usize, Candidate)> = None;
        for trial in 0..self.num_trials {
            let mut candidate = Candidate::new();
            for (leaf, instructions) in leaves.iter().zip(&instruction_sets) {
                let instruction = &instructions[rng.gen_range(0..instructions.len())];
                set_instruction_if_changed(&mut candidate, leaf, instruction);
                if let Some(demos) = bootstrapped_demos.get(&leaf.name) {
                    // Jointly search incumbent demos, no demos, and sampled
                    // bootstrap subsets instead of fixing demos for every trial.
                    match rng.gen_range(0..3) {
                        0 => {}
                        1 => {
                            set_demos_if_changed(&mut candidate, leaf, Vec::new());
                        }
                        _ => {
                            let count = rng.gen_range(1..=demos.len());
                            set_demos_if_changed(
                                &mut candidate,
                                leaf,
                                demos.choose_multiple(&mut rng, count).cloned().collect(),
                            );
                        }
                    }
                }
            }
            let row = engine.register(candidate.clone());
            let minibatch_score = score(engine.evaluate(target, row, Some(&minibatch)).await?)?;
            if pending
                .as_ref()
                .is_none_or(|(s, _, _)| minibatch_score > *s)
            {
                pending = Some((minibatch_score, row, candidate));
            }
            if (trial + 1) % self.full_eval_steps == 0 || trial + 1 == self.num_trials {
                let (_, row, candidate) = pending.take().expect("at least one trial per interval");
                let full_score = score(engine.evaluate(target, row, Some(&validation)).await?)?;
                debug!(trial, full_score, best_score, "MIPRO validation");
                if full_score > best_score {
                    best_score = full_score;
                    winner = candidate;
                }
            }
        }

        // The one mutation of the run: install demos + winning instructions.
        target.install(&winner)?;

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
