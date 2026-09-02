//! GRPO end-to-end on a stochastic LM-free module: the loop samples groups,
//! filters flat ones, distills rules from the fallback (metric feedback),
//! gates the child, and installs the winner exactly once.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use dspy_rs::{
    CallMetadata, Eval, GRPO, Module, Predict, PredictError, Predicted, PredictorInfo, Signature,
    Trace, TypedMetric,
};

#[derive(Signature, Clone, Debug)]
struct EchoSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

type Row = (EchoSigInput, EchoSigOutput);

/// Without rules: alternates good/bad outputs call by call (`Alternate`) or
/// is always bad (`AlwaysBad`). With rules: always good if a rule mentions
/// "suffix", always bad otherwise.
#[derive(Clone, Copy)]
enum Mode {
    Alternate,
    AlwaysBad,
}

struct StochasticModule {
    predictor: Predict<EchoSig>,
    calls: AtomicUsize,
    mode: Mode,
}

dspy_rs::predictors!(StochasticModule { predictor });

impl Module for StochasticModule {
    type Input = EchoSigInput;
    type Output = EchoSigOutput;

    async fn forward(&self, input: EchoSigInput) -> Result<Predicted<EchoSigOutput>, PredictError> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let instruction = dspy_rs::fx::current_instruction("predictor").unwrap_or_default();
        let suffix = if instruction.contains("suffix") {
            0
        } else if instruction.contains("Rules:") {
            1
        } else {
            match self.mode {
                Mode::Alternate => n % 2,
                Mode::AlwaysBad => 1,
            }
        };
        Ok(Predicted::new(
            EchoSigOutput {
                answer: format!("{}#{suffix}", input.prompt),
            },
            CallMetadata::default(),
        ))
    }
}

/// Feedback that names the fix ("suffix"): the fallback teacher turns it
/// into a rule the module then obeys.
struct HelpfulMetric;

impl TypedMetric<Row, StochasticModule> for HelpfulMetric {
    async fn evaluate(
        &self,
        _example: &Row,
        prediction: &Predicted<EchoSigOutput>,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        if prediction.answer.ends_with("#0") {
            Ok(Eval::with_feedback(1.0, "kept the #0 suffix"))
        } else {
            Ok(Eval::with_feedback(0.0, "always end with the #0 suffix"))
        }
    }
}

/// Feedback that doesn't name the fix: the resulting rule makes the module
/// strictly worse, so the gate must reject it.
struct UnhelpfulMetric;

impl TypedMetric<Row, StochasticModule> for UnhelpfulMetric {
    async fn evaluate(
        &self,
        _example: &Row,
        prediction: &Predicted<EchoSigOutput>,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        if prediction.answer.ends_with("#0") {
            Ok(Eval::with_feedback(1.0, "good"))
        } else {
            Ok(Eval::with_feedback(0.0, "bad sample"))
        }
    }
}

fn trainset(n: usize) -> Vec<Row> {
    (0..n)
        .map(|i| {
            (
                EchoSigInput {
                    prompt: format!("q{i}"),
                },
                EchoSigOutput {
                    answer: String::new(),
                },
            )
        })
        .collect()
}

fn module() -> StochasticModule {
    module_with(Mode::Alternate)
}

fn module_with(mode: Mode) -> StochasticModule {
    StochasticModule {
        predictor: Predict::<EchoSig>::builder()
            .instruction("seed-instruction")
            .build(),
        calls: AtomicUsize::new(0),
        mode,
    }
}

#[tokio::test]
async fn learns_a_rule_from_group_contrast_and_installs_it() {
    let grpo = GRPO::builder()
        .steps(2)
        .batch_size(2)
        .group_size(4)
        .seed(1)
        .eval_concurrency(1)
        .build();
    let mut module = module();
    let report = grpo
        .compile_module(&mut module, &trainset(4), &HelpfulMetric)
        .await
        .expect("compile succeeds");

    assert_eq!(
        report.accepted, 1,
        "first step's rule fixes the module: {report:?}"
    );
    assert_eq!(report.final_score, 1.0);
    assert!(report.baseline_score < 1.0);
    let first = &report.steps[0];
    assert_eq!(first.informative_groups, 2);
    assert_eq!(first.flat_groups, 0);
    assert_eq!(first.spreads, vec![1.0, 1.0]);
    assert!(first.accepted);
    // Once the rule is installed the module is deterministic: later groups
    // are flat AND solved, so the reference fallback leaves them alone.
    let second = &report.steps[1];
    assert_eq!(second.flat_groups, 2);
    assert_eq!(second.reference_contrasts, 0);
    assert_eq!(second.spreads, vec![0.0, 0.0]);

    let rules = &report.rules["predictor"];
    assert_eq!(rules, &vec!["always end with the #0 suffix".to_string()]);
    let installed = PredictorInfo::instruction(&module.predictor);
    assert!(installed.starts_with("seed-instruction"));
    assert!(installed.contains("Rules:\n- always end with the #0 suffix"));
    // Spend: baseline 4 + step-0 groups 8 + incumbent redraw 4 + child 4 +
    // step-1 groups 8 (no insights, so no gate).
    assert_eq!(report.spend.metric_calls, 28);
    // The incumbent's running mean after the redraw is what the gate saw.
    assert_eq!(first.parent_score, 0.5);
}

#[tokio::test]
async fn gate_rejects_a_rule_that_makes_the_module_worse() {
    let grpo = GRPO::builder()
        .steps(1)
        .batch_size(2)
        .group_size(4)
        .seed(1)
        .eval_concurrency(1)
        .build();
    let mut module = module();
    let report = grpo
        .compile_module(&mut module, &trainset(4), &UnhelpfulMetric)
        .await
        .expect("compile succeeds");

    assert_eq!(report.rejected, 1);
    assert_eq!(report.accepted, 0);
    assert_eq!(report.steps[0].child_score, Some(0.0));
    assert!(report.rules.is_empty());
    assert_eq!(report.final_score, report.baseline_score);
    assert_eq!(
        PredictorInfo::instruction(&module.predictor),
        "seed-instruction",
        "a rejected child must not touch the module"
    );
}

#[tokio::test]
async fn ungated_run_accepts_every_step() {
    let grpo = GRPO::builder()
        .steps(1)
        .batch_size(2)
        .group_size(4)
        .gate(false)
        .seed(1)
        .eval_concurrency(1)
        .build();
    let mut module = module();
    let report = grpo
        .compile_module(&mut module, &trainset(4), &UnhelpfulMetric)
        .await
        .expect("compile succeeds");
    assert_eq!(report.accepted, 1);
    assert_eq!(report.final_score, 0.0);
    assert!(PredictorInfo::instruction(&module.predictor).contains("bad sample"));
}

#[tokio::test]
async fn group_size_below_two_is_rejected() {
    let grpo = GRPO::builder().group_size(1).build();
    let mut module = module();
    let err = grpo
        .compile_module(&mut module, &trainset(2), &HelpfulMetric)
        .await
        .expect_err("group_size 1 has no contrast");
    assert!(err.to_string().contains("group_size"));
}

#[tokio::test]
async fn flat_groups_fall_back_to_reference_contrast() {
    // Always-bad module: no within-group spread anywhere. The reference
    // fallback contrasts the flat groups anyway (fallback teacher = worst
    // feedback), and the resulting rule fixes the module.
    let grpo = GRPO::builder()
        .steps(1)
        .batch_size(2)
        .group_size(3)
        .seed(1)
        .eval_concurrency(1)
        .build();
    let mut module = module_with(Mode::AlwaysBad);
    let report = grpo
        .compile_module(&mut module, &trainset(4), &HelpfulMetric)
        .await
        .expect("compile succeeds");

    assert_eq!(report.baseline_score, 0.0);
    let step = &report.steps[0];
    assert_eq!(step.informative_groups, 0);
    assert_eq!(step.flat_groups, 2);
    assert_eq!(step.reference_contrasts, 2);
    assert_eq!(step.insights, 2);
    assert!(step.accepted);
    assert_eq!(report.final_score, 1.0);
    assert!(PredictorInfo::instruction(&module.predictor).contains("#0 suffix"));
}

#[tokio::test]
async fn pure_group_contrast_skips_flat_groups() {
    let grpo = GRPO::builder()
        .steps(1)
        .batch_size(2)
        .group_size(3)
        .reference_fallback(false)
        .seed(1)
        .eval_concurrency(1)
        .build();
    let mut module = module_with(Mode::AlwaysBad);
    let report = grpo
        .compile_module(&mut module, &trainset(4), &HelpfulMetric)
        .await
        .expect("compile succeeds");

    let step = &report.steps[0];
    assert_eq!(step.flat_groups, 2);
    assert_eq!(step.reference_contrasts, 0);
    assert_eq!(step.insights, 0);
    assert!(step.child_score.is_none());
    assert!(report.rules.is_empty());
    assert_eq!(
        PredictorInfo::instruction(&module.predictor),
        "seed-instruction"
    );
}
