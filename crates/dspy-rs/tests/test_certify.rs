//! Engine-lane coverage for the standalone ship certification API
//! (`optimizer::certify::certify_candidate`): the pawl driving paired
//! rollouts through `Engine::evaluate_many` over an `OptimizeTarget`, no
//! LM in the loop. The closure lane (`certify_paired`) is covered by the
//! module's own unit tests; this file proves the target/engine wrapper —
//! registration, pairing, budget exhaustion, and the multi-pass refusal.

use anyhow::Result;
use dspy_rs::{
    BetPrior, Budget, Candidate, CertifyConfig, CallMetadata, Engine, EngineConfig, Eval, Module,
    OptimizeTarget, Predict, PredictError, Predicted, Signature, TypedMetric, certify_candidate,
};

#[derive(Signature, Clone, Debug)]
/// Answer the prompt.
struct CertSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

/// No-LM module; the metric reads the candidate hash off the trace, so
/// `forward` itself can stay an echo.
struct CertModule {
    predictor: Predict<CertSig>,
}

dspy_rs::predictors!(CertModule { predictor });

impl Module for CertModule {
    type Input = CertSigInput;
    type Output = CertSigOutput;

    async fn forward(&self, input: CertSigInput) -> Result<Predicted<CertSigOutput>, PredictError> {
        let _ = &self.predictor;
        Ok(Predicted::new(
            CertSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

fn trainset(n: usize) -> Vec<(CertSigInput, CertSigOutput)> {
    (0..n)
        .map(|idx| {
            (
                CertSigInput {
                    prompt: idx.to_string(),
                },
                CertSigOutput {
                    answer: idx.to_string(),
                },
            )
        })
        .collect()
}

/// Deterministic judge noise per (question, candidate-hash): FNV mapped to
/// uniform ±spread — different arms redraw, the paired-noise structure of
/// a real judge.
fn judge_noise(qid: u64, hash: u64, spread: f64) -> f64 {
    let mut acc = 1469598103934665603u64;
    for byte in [qid, hash, 0x9e3779b97f4a7c15] {
        acc ^= byte;
        acc = acc.wrapping_mul(1099511628211);
    }
    ((acc % 10_001) as f64 / 10_000.0 - 0.5) * 2.0 * spread
}

/// Scores by candidate hash (read from the trace meta): `effect` on top of
/// a noisy base for the good candidate, noisy base alone for everyone else.
struct KeyedMetric {
    good_hash: u64,
    effect: f64,
}

impl<M> TypedMetric<(CertSigInput, CertSigOutput), M> for KeyedMetric
where
    M: Module<Input = CertSigInput, Output = CertSigOutput>,
{
    async fn evaluate(
        &self,
        example: &(CertSigInput, CertSigOutput),
        _prediction: &Predicted<CertSigOutput>,
        trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let hash = trace
            .and_then(|trace| trace.meta.candidate_hash)
            .unwrap_or_default();
        let qid: u64 = example.0.prompt.parse()?;
        let lift = if hash == self.good_hash { self.effect } else { 0.0 };
        let score = (0.35 + lift + judge_noise(qid, hash, 0.15)).clamp(0.0, 1.0);
        Ok(Eval::with_feedback(score, "keyed"))
    }
}

fn arms() -> (Candidate, Candidate) {
    let candidate = Candidate::with_instruction("predictor", "the good instruction");
    let incumbent = Candidate::with_instruction("predictor", "the seed instruction");
    (candidate, incumbent)
}

#[tokio::test]
async fn true_effect_certifies_through_the_engine() {
    let (candidate, incumbent) = arms();
    let metric = KeyedMetric {
        good_hash: candidate.stable_hash(),
        effect: 0.3,
    };
    let mut module = CertModule {
        predictor: Predict::<CertSig>::builder().build(),
    };
    let examples = trainset(200);
    let target = OptimizeTarget::module(&mut module, &examples, &metric);
    let mut engine = Engine::new(EngineConfig::default());

    let verdict = certify_candidate(
        &target,
        &mut engine,
        &candidate,
        &incumbent,
        &CertifyConfig::default(),
    )
    .await
    .unwrap();

    assert_eq!(verdict.certified, Some(true));
    assert!(verdict.ship_candidate());
    assert!(verdict.cols < 100, "a +.3 effect must stop early, used {}", verdict.cols);
    assert!(verdict.mean_delta > 0.2);
    assert!(verdict.e_gain >= 10.0);
    // Every column is one PAIRED evaluation: two fresh rollouts.
    assert_eq!(engine.spend().metric_calls, 2 * verdict.cols);
    assert_eq!(engine.spend().cache_hits, 0, "certification rollouts are fresh");
}

#[tokio::test]
async fn null_effect_refuses_and_a_prior_cannot_flip_it() {
    let (candidate, incumbent) = arms();
    let metric = KeyedMetric {
        good_hash: candidate.stable_hash(),
        effect: 0.0,
    };
    let mut module = CertModule {
        predictor: Predict::<CertSig>::builder().build(),
    };
    let examples = trainset(120);
    let target = OptimizeTarget::module(&mut module, &examples, &metric);
    let mut engine = Engine::new(EngineConfig::default());

    let config = CertifyConfig {
        prior: Some(BetPrior { mean: 0.5, sd: 0.05, weight: 100.0 }),
        ..CertifyConfig::default()
    };
    let verdict = certify_candidate(&target, &mut engine, &candidate, &incumbent, &config)
        .await
        .unwrap();

    assert_ne!(verdict.certified, Some(true), "priors shape bets, never wealth");
    assert!(verdict.cols > 0, "a refusal still reports its open evidence");
}

#[tokio::test]
async fn budget_exhaustion_is_an_honest_refusal() {
    let (candidate, incumbent) = arms();
    let metric = KeyedMetric {
        good_hash: candidate.stable_hash(),
        effect: 0.02, // real but far below what 20 columns can certify
    };
    let mut module = CertModule {
        predictor: Predict::<CertSig>::builder().build(),
    };
    let examples = trainset(200);
    let target = OptimizeTarget::module(&mut module, &examples, &metric);
    let mut engine = Engine::new(EngineConfig {
        budget: Budget {
            max_metric_calls: Some(40), // 20 paired columns
            ..Budget::unlimited()
        },
        ..EngineConfig::default()
    });

    let verdict = certify_candidate(
        &target,
        &mut engine,
        &candidate,
        &incumbent,
        &CertifyConfig::default(),
    )
    .await
    .unwrap();

    assert_eq!(verdict.certified, None);
    assert!(verdict.cols <= 20, "stopped by budget, used {}", verdict.cols);
    assert!(engine.spend().metric_calls <= 40);
}

#[tokio::test]
async fn identical_arms_and_multipass_are_refused_loudly() {
    let (candidate, _) = arms();
    let metric = KeyedMetric {
        good_hash: candidate.stable_hash(),
        effect: 0.3,
    };
    let mut module = CertModule {
        predictor: Predict::<CertSig>::builder().build(),
    };
    let examples = trainset(10);
    let target = OptimizeTarget::module(&mut module, &examples, &metric);
    let mut engine = Engine::new(EngineConfig::default());

    let same = certify_candidate(
        &target,
        &mut engine,
        &candidate,
        &candidate.clone(),
        &CertifyConfig::default(),
    )
    .await;
    assert!(same.is_err(), "register dedupes by content hash — same arm twice is an error");

    let multipass = certify_candidate(
        &target,
        &mut engine,
        &candidate,
        &Candidate::with_instruction("predictor", "other"),
        &CertifyConfig { passes: 2, ..CertifyConfig::default() },
    )
    .await;
    assert!(multipass.is_err(), "the engine lane must refuse multi-pass (cached rollouts)");
}
