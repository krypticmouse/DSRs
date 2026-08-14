//! Shared evaluation engine (vision §5.4) coverage: bounded-concurrency
//! fan-out, rollout caching, budget metering, minibatch gating, matrix/Pareto
//! bookkeeping, checkpoint/resume, and the candidate apply/restore seam.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use dspy_rs::{
    Budget, CallMetadata, Candidate, EngineConfig, Eval, EvalEngine, EvalOutcome, Example,
    GateOutcome, LM, LMClient, Module, ModuleState, Predict, PredictError, Predicted, Signature,
    TestCompletionModel, TypedMetric, apply_candidate, restore_candidate,
};
use rig::completion::AssistantContent;
use rig::message::Text;

#[derive(Signature, Clone, Debug)]
/// Answer the prompt.
struct EngSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

fn trainset(n: usize) -> Vec<Example<EngSig>> {
    (0..n)
        .map(|idx| {
            Example::new(
                EngSigInput {
                    prompt: idx.to_string(),
                },
                EngSigOutput {
                    answer: idx.to_string(),
                },
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

/// No-LM module: echoes the prompt back as the answer.
#[derive(facet::Facet)]
#[facet(crate = facet)]
struct EchoModule {
    predictor: Predict<EngSig>,
}

impl EchoModule {
    fn new() -> Self {
        Self {
            predictor: Predict::<EngSig>::builder().instruction("seed").build(),
        }
    }
}

impl Module for EchoModule {
    type Input = EngSigInput;
    type Output = EngSigOutput;

    async fn forward(&self, input: EngSigInput) -> Result<Predicted<EngSigOutput>, PredictError> {
        let _ = &self.predictor;
        Ok(Predicted::new(
            EngSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

/// Echo module that blocks each rollout on a barrier: the test only completes
/// if all N rollouts are genuinely in flight at once.
#[derive(facet::Facet)]
#[facet(crate = facet)]
struct BarrierModule {
    predictor: Predict<EngSig>,
    #[facet(opaque, skip)]
    barrier: Arc<tokio::sync::Barrier>,
}

impl Module for BarrierModule {
    type Input = EngSigInput;
    type Output = EngSigOutput;

    async fn forward(&self, input: EngSigInput) -> Result<Predicted<EngSigOutput>, PredictError> {
        self.barrier.wait().await;
        Ok(Predicted::new(
            EngSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

/// Echo module that gauges how many rollouts are inside `forward` at once.
/// The pair barrier forces at least two to overlap, so a sequential engine
/// deadlocks; the gauge proves the bound is never exceeded.
#[derive(facet::Facet)]
#[facet(crate = facet)]
struct GaugeModule {
    predictor: Predict<EngSig>,
    #[facet(opaque, skip)]
    in_flight: Arc<AtomicUsize>,
    #[facet(opaque, skip)]
    max_in_flight: Arc<AtomicUsize>,
    #[facet(opaque, skip)]
    pair_barrier: Arc<tokio::sync::Barrier>,
}

impl Module for GaugeModule {
    type Input = EngSigInput;
    type Output = EngSigOutput;

    async fn forward(&self, input: EngSigInput) -> Result<Predicted<EngSigOutput>, PredictError> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        self.pair_barrier.wait().await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        Ok(Predicted::new(
            EngSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

/// Real `Predict` leaf backed by [`TestCompletionModel`] — every rollout is an
/// actual LM call against the canned response queue.
#[derive(facet::Facet)]
#[facet(crate = facet)]
struct LmModule {
    predictor: Predict<EngSig>,
}

impl Module for LmModule {
    type Input = EngSigInput;
    type Output = EngSigOutput;

    async fn forward(&self, input: EngSigInput) -> Result<Predicted<EngSigOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

fn answer_response(text: &str) -> AssistantContent {
    AssistantContent::Text(Text {
        text: format!("[[ ## answer ## ]]\n{text}\n\n[[ ## completed ## ]]\n"),
    })
}

async fn make_test_lm(client: TestCompletionModel) -> LM {
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Test(client))
    .await
    .unwrap()
}

async fn lm_module(client: TestCompletionModel) -> LmModule {
    LmModule {
        predictor: Predict::<EngSig>::builder()
            .named("predictor")
            .lm(make_test_lm(client).await)
            .build(),
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Deterministic per-example score: the prompt parsed as an index, over 10.
struct IndexMetric;

impl<M> TypedMetric<EngSig, M> for IndexMetric
where
    M: Module<Input = EngSigInput, Output = EngSigOutput>,
{
    async fn evaluate(
        &self,
        example: &Example<EngSig>,
        _prediction: &Predicted<EngSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let idx: f64 = example.input.prompt.parse().unwrap_or(0.0);
        Ok(Eval::with_feedback(idx / 10.0, format!("idx={idx}")))
    }
}

/// Exact-match against the expected answer — for LM-backed modules.
struct ExactMatch;

impl<M> TypedMetric<EngSig, M> for ExactMatch
where
    M: Module<Input = EngSigInput, Output = EngSigOutput>,
{
    async fn evaluate(
        &self,
        example: &Example<EngSig>,
        prediction: &Predicted<EngSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let score = (prediction.answer == example.output.answer) as u8 as f64;
        Ok(Eval::with_feedback(score, "exact-match"))
    }
}

/// Scores by which candidate produced the rollout, read from the trace meta's
/// `candidate_hash` — lets tests script per-candidate quality with no LM.
struct HashKeyedMetric {
    scores: HashMap<u64, f64>,
}

impl<M> TypedMetric<EngSig, M> for HashKeyedMetric
where
    M: Module<Input = EngSigInput, Output = EngSigOutput>,
{
    async fn evaluate(
        &self,
        _example: &Example<EngSig>,
        _prediction: &Predicted<EngSigOutput>,
        trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let hash = trace
            .and_then(|trace| trace.meta.candidate_hash)
            .unwrap_or_default();
        let score = self.scores.get(&hash).copied().unwrap_or(0.0);
        Ok(Eval::with_feedback(score, "hash-keyed"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fan_out_runs_examples_concurrently_with_correct_results() {
    const N: usize = 8;
    let mut module = BarrierModule {
        predictor: Predict::<EngSig>::builder().instruction("seed").build(),
        barrier: Arc::new(tokio::sync::Barrier::new(N)),
    };
    let metric = IndexMetric;
    let mut engine = EvalEngine::new(
        trainset(N),
        &metric,
        EngineConfig {
            concurrency: N,
            ..EngineConfig::default()
        },
    );

    let candidate = engine.register(Candidate::new());
    // The barrier only opens once all N rollouts are in flight simultaneously;
    // a sequential engine would deadlock and trip the timeout.
    let outcome = tokio::time::timeout(
        Duration::from_secs(10),
        engine.evaluate(&mut module, candidate, None),
    )
    .await
    .expect("fan-out must run all rollouts concurrently")
    .expect("evaluation should succeed");

    let eval = outcome.completed().expect("budget is unlimited");
    assert_eq!(eval.rollouts.len(), N);
    for (idx, rollout) in eval.rollouts.iter().enumerate() {
        assert_eq!(rollout.example, idx, "results come back in request order");
        assert!((rollout.eval.score - idx as f64 / 10.0).abs() < 1e-9);
        assert!(rollout.trace.is_some(), "fresh rollouts carry traces");
    }

    assert_eq!(engine.spend().metric_calls, N);
    assert_eq!(engine.spend().lm_calls, N);
    assert_eq!(engine.spend().cache_hits, 0);
    for idx in 0..N {
        assert!(engine.matrix().score(candidate, idx).is_some());
    }
    assert!((engine.matrix().mean(candidate).unwrap() - eval.mean()).abs() < 1e-9);
}

#[tokio::test]
async fn subset_evaluation_respects_order_and_matrix_cells() {
    let mut module = EchoModule::new();
    let metric = IndexMetric;
    let mut engine = EvalEngine::new(trainset(4), &metric, EngineConfig::default());

    let candidate = engine.register(Candidate::new());
    let eval = engine
        .evaluate(&mut module, candidate, Some(&[3, 1, 2]))
        .await
        .unwrap()
        .completed()
        .unwrap();

    let order: Vec<usize> = eval.rollouts.iter().map(|r| r.example).collect();
    assert_eq!(order, vec![3, 1, 2]);
    assert!(engine.matrix().score(candidate, 0).is_none());
    for idx in [1, 2, 3] {
        assert!(engine.matrix().score(candidate, idx).is_some());
    }
}

#[tokio::test]
async fn rollout_cache_serves_repeats_without_lm_calls() {
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
    ]);
    let mut module = lm_module(client.clone()).await;
    let metric = ExactMatch;
    let mut engine = EvalEngine::new(trainset(3), &metric, EngineConfig::default());

    let candidate = engine.register(Candidate::new());
    let first = engine
        .evaluate(&mut module, candidate, None)
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert!((first.mean() - 1.0).abs() < 1e-9);
    assert_eq!(engine.spend().lm_calls, 3);
    assert_eq!(engine.spend().lm_spans, 3, "each rollout recorded one span");

    // Re-evaluation: the response queue is now EMPTY, so any LM call would
    // error. The cache must serve all three rollouts.
    let second = engine
        .evaluate(&mut module, candidate, None)
        .await
        .expect("cached re-evaluation must not touch the LM")
        .completed()
        .unwrap();
    assert!((second.mean() - 1.0).abs() < 1e-9);
    assert!(second.rollouts.iter().all(|r| r.trace.is_none()));
    assert_eq!(engine.spend().lm_calls, 3, "LM call count must not grow");
    assert_eq!(engine.spend().metric_calls, 3, "metric must not re-run");
    assert_eq!(engine.spend().cache_hits, 3);

    // A *different* candidate is a cache miss: it runs fresh LM calls.
    client.push_response(answer_response("0"));
    client.push_response(answer_response("wrong"));
    client.push_response(answer_response("2"));
    let other = engine.register(Candidate::with_instruction("predictor", "be brief"));
    let third = engine
        .evaluate(&mut module, other, None)
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert!((third.mean() - 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(engine.spend().lm_calls, 6);

    // Candidate evaluation restored the module: no instruction override left.
    let state = ModuleState::from_module(&mut module).unwrap();
    assert_eq!(state.predictors["predictor"].instruction_override, None);
}

#[tokio::test]
async fn budget_stops_cleanly_and_reports_spend() {
    let mut module = EchoModule::new();
    let metric = IndexMetric;
    let mut engine = EvalEngine::new(
        trainset(3),
        &metric,
        EngineConfig {
            budget: Budget {
                max_metric_calls: Some(4),
                ..Budget::unlimited()
            },
            ..EngineConfig::default()
        },
    );

    let base = engine.register(Candidate::new());
    let better = engine.register(Candidate::with_instruction("predictor", "improved"));

    assert!(engine.budget_allows(3));
    engine
        .evaluate(&mut module, base, None)
        .await
        .unwrap()
        .completed()
        .expect("first full eval fits the budget");
    assert_eq!(engine.spend().metric_calls, 3);

    // A second full eval needs 3 more rollouts; only 1 remains.
    match engine.evaluate(&mut module, better, None).await.unwrap() {
        EvalOutcome::BudgetExhausted { needed } => assert_eq!(needed, 3),
        EvalOutcome::Complete(_) => panic!("engine must stop at the budget"),
    }
    assert_eq!(engine.spend().metric_calls, 3, "nothing ran past the budget");

    // Cache-served batches are free even at the budget edge.
    let replay = engine
        .evaluate(&mut module, base, None)
        .await
        .unwrap()
        .completed()
        .expect("cached batch consumes no budget");
    assert_eq!(replay.rollouts.len(), 3);

    // The final budget unit still fits a single-example batch.
    engine
        .evaluate(&mut module, better, Some(&[0]))
        .await
        .unwrap()
        .completed()
        .expect("one rollout fits the remaining budget");
    assert_eq!(engine.spend().metric_calls, 4);
    assert!(!engine.budget_allows(1));
}

#[tokio::test]
async fn minibatch_gate_promotes_only_above_threshold() {
    let mut module = EchoModule::new();
    let strong = Candidate::with_instruction("predictor", "strong");
    let weak = Candidate::with_instruction("predictor", "weak");
    let metric = HashKeyedMetric {
        scores: HashMap::from([(strong.stable_hash(), 0.9), (weak.stable_hash(), 0.1)]),
    };
    let mut engine = EvalEngine::new(trainset(6), &metric, EngineConfig::default());

    let strong = engine.register(strong);
    let weak = engine.register(weak);

    match engine
        .evaluate_gated(&mut module, weak, &[0, 1], 0.5)
        .await
        .unwrap()
    {
        GateOutcome::Rejected { minibatch } => {
            assert_eq!(minibatch.rollouts.len(), 2);
            assert!((minibatch.mean() - 0.1).abs() < 1e-9);
        }
        other => panic!("weak candidate must be rejected, got {other:?}"),
    }
    // Rejection never ran the full set: only the minibatch cells are filled.
    assert_eq!(engine.spend().metric_calls, 2);
    assert!(engine.matrix().score(weak, 2).is_none());

    match engine
        .evaluate_gated(&mut module, strong, &[0, 1], 0.5)
        .await
        .unwrap()
    {
        GateOutcome::Promoted { minibatch, full } => {
            assert!((minibatch.mean() - 0.9).abs() < 1e-9);
            assert_eq!(full.rollouts.len(), 6);
            // The minibatch rollouts are cache hits inside the full pass.
            assert_eq!(
                full.rollouts.iter().filter(|r| r.trace.is_none()).count(),
                2
            );
        }
        other => panic!("strong candidate must be promoted, got {other:?}"),
    }
    assert_eq!(engine.spend().metric_calls, 2 + 6);

    // Pareto bookkeeping over the shared matrix: strong wins everywhere it ran.
    let pareto = engine.pareto();
    assert_eq!(pareto.wins(strong), 6);
    assert_eq!(pareto.wins(weak), 0);
    assert_eq!(pareto.frontier(), vec![strong]);
    assert_eq!(engine.matrix().best_by_mean(), Some(strong));
}

#[tokio::test]
async fn checkpoint_resume_skips_completed_rollouts() {
    let client = TestCompletionModel::new([
        answer_response("0"),
        answer_response("1"),
        answer_response("2"),
    ]);
    let metric = ExactMatch;
    let candidate = Candidate::with_instruction("predictor", "resume-me");

    let checkpoint = {
        let mut module = lm_module(client.clone()).await;
        let mut engine = EvalEngine::new(trainset(3), &metric, EngineConfig::default());
        let idx = engine.register(candidate.clone());
        let eval = engine
            .evaluate(&mut module, idx, None)
            .await
            .unwrap()
            .completed()
            .unwrap();
        assert!((eval.mean() - 1.0).abs() < 1e-9);
        engine.checkpoint().unwrap()
    };

    // Fresh process: new module, EMPTY response queue — any LM call errors.
    let mut module = lm_module(TestCompletionModel::new([])).await;
    let mut engine =
        EvalEngine::resume(trainset(3), &metric, EngineConfig::default(), &checkpoint).unwrap();
    assert_eq!(engine.num_candidates(), 1);
    assert_eq!(engine.spend().metric_calls, 3, "spend carries over");

    let idx = engine.register(candidate);
    assert_eq!(idx, 0, "re-registering dedups by content hash");

    let eval = engine
        .evaluate(&mut module, idx, None)
        .await
        .expect("resumed run must serve completed rollouts from cache")
        .completed()
        .unwrap();
    assert!((eval.mean() - 1.0).abs() < 1e-9);
    assert!(eval.rollouts.iter().all(|r| r.trace.is_none()));
    assert_eq!(engine.spend().metric_calls, 3, "no new metric calls");
    assert_eq!(engine.spend().cache_hits, 3);
    assert_eq!(engine.matrix().mean(idx), Some(1.0));

    // A checkpoint against a different example set is rejected.
    match EvalEngine::<EngSig, ExactMatch>::resume(
        trainset(4),
        &metric,
        EngineConfig::default(),
        &checkpoint,
    ) {
        Err(err) => assert!(err.to_string().contains("does not match")),
        Ok(_) => panic!("mismatched examples must fail resume"),
    }
}

#[tokio::test]
async fn fan_out_never_exceeds_the_concurrency_bound() {
    const N: usize = 6;
    const BOUND: usize = 2;
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let mut module = GaugeModule {
        predictor: Predict::<EngSig>::builder().instruction("seed").build(),
        in_flight: Arc::new(AtomicUsize::new(0)),
        max_in_flight: Arc::clone(&max_in_flight),
        // Pairs must overlap to release the barrier: a concurrency-1 engine
        // deadlocks here and trips the timeout.
        pair_barrier: Arc::new(tokio::sync::Barrier::new(BOUND)),
    };
    let metric = IndexMetric;
    let mut engine = EvalEngine::new(
        trainset(N),
        &metric,
        EngineConfig {
            concurrency: BOUND,
            ..EngineConfig::default()
        },
    );

    let candidate = engine.register(Candidate::new());
    let eval = tokio::time::timeout(
        Duration::from_secs(10),
        engine.evaluate(&mut module, candidate, None),
    )
    .await
    .expect("bounded fan-out must still overlap rollouts")
    .expect("evaluation should succeed")
    .completed()
    .unwrap();

    assert_eq!(eval.rollouts.len(), N);
    assert_eq!(
        max_in_flight.load(Ordering::SeqCst),
        BOUND,
        "the engine must saturate but never exceed EngineConfig::concurrency"
    );
}

#[tokio::test]
async fn gate_reports_budget_exhaustion_for_minibatch_and_promotion() {
    let metric = IndexMetric;

    // Minibatch itself doesn't fit: nothing runs, spend unchanged.
    let mut module = EchoModule::new();
    let mut engine = EvalEngine::new(
        trainset(4),
        &metric,
        EngineConfig {
            budget: Budget {
                max_metric_calls: Some(1),
                ..Budget::unlimited()
            },
            ..EngineConfig::default()
        },
    );
    let candidate = engine.register(Candidate::new());
    match engine
        .evaluate_gated(&mut module, candidate, &[0, 1], 0.0)
        .await
        .unwrap()
    {
        GateOutcome::BudgetExhausted { needed } => assert_eq!(needed, 2),
        other => panic!("minibatch must not fit a 1-rollout budget, got {other:?}"),
    }
    assert_eq!(engine.spend().metric_calls, 0);

    // Minibatch fits and passes the gate, but the full-set promotion doesn't.
    let mut engine = EvalEngine::new(
        trainset(4),
        &metric,
        EngineConfig {
            budget: Budget {
                max_metric_calls: Some(3),
                ..Budget::unlimited()
            },
            ..EngineConfig::default()
        },
    );
    let candidate = engine.register(Candidate::new());
    match engine
        .evaluate_gated(&mut module, candidate, &[2, 3], 0.0)
        .await
        .unwrap()
    {
        // Full set needs 2 uncached rollouts (examples 0 and 1); 1 remains.
        GateOutcome::BudgetExhausted { needed } => assert_eq!(needed, 2),
        other => panic!("promotion must not fit the remaining budget, got {other:?}"),
    }
    assert_eq!(engine.spend().metric_calls, 2, "only the minibatch ran");
    assert!(engine.matrix().score(candidate, 2).is_some());
    assert!(engine.matrix().score(candidate, 0).is_none());
}

#[tokio::test]
async fn auxiliary_charges_count_against_the_budget() {
    let mut module = EchoModule::new();
    let metric = IndexMetric;
    let mut engine = EvalEngine::new(
        trainset(3),
        &metric,
        EngineConfig {
            budget: Budget {
                max_lm_calls: Some(5),
                ..Budget::unlimited()
            },
            ..EngineConfig::default()
        },
    );

    let candidate = engine.register(Candidate::new());
    engine
        .evaluate(&mut module, candidate, None)
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert_eq!(engine.spend().lm_calls, 3);
    assert!(engine.budget_allows(2));

    // A strategy-side reflection call spends budget the engine didn't run.
    engine.charge(0, 2);
    assert_eq!(engine.spend().lm_calls, 5);
    assert!(!engine.budget_allows(1));

    // Charged spend survives checkpoint/resume.
    let checkpoint = engine.checkpoint().unwrap();
    let resumed =
        EvalEngine::<EngSig, IndexMetric>::resume(trainset(3), &metric, *engine.config(), &checkpoint)
            .unwrap();
    assert_eq!(resumed.spend().lm_calls, 5);
    assert_eq!(resumed.spend().metric_calls, 3);
}

#[tokio::test]
async fn checkpoint_with_unknown_version_is_rejected() {
    let metric = IndexMetric;
    let engine = EvalEngine::new(trainset(2), &metric, EngineConfig::default());
    let checkpoint = engine.checkpoint().unwrap();

    let mut doctored: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
    doctored["version"] = serde_json::json!(99);
    let doctored = serde_json::to_string(&doctored).unwrap();

    match EvalEngine::<EngSig, IndexMetric>::resume(
        trainset(2),
        &metric,
        EngineConfig::default(),
        &doctored,
    ) {
        Err(err) => assert!(err.to_string().contains("version")),
        Ok(_) => panic!("unknown checkpoint versions must fail resume"),
    }
}

#[tokio::test]
async fn permanent_install_invalidates_the_cache_via_baseline_hash() {
    let client = TestCompletionModel::new([answer_response("0"), answer_response("1")]);
    let mut module = lm_module(client.clone()).await;
    let metric = ExactMatch;
    let mut engine = EvalEngine::new(trainset(2), &metric, EngineConfig::default());

    let candidate = engine.register(Candidate::new());
    engine
        .evaluate(&mut module, candidate, None)
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert_eq!(engine.spend().lm_calls, 2);

    // Permanently install a winner mid-run (the COPRO-between-rounds shape):
    // the module skeleton changed, so cached entries for the old baseline
    // must NOT be served for the same candidate on the new baseline.
    apply_candidate(
        &mut module,
        &Candidate::with_instruction("predictor", "installed"),
    )
    .unwrap();

    client.push_response(answer_response("0"));
    client.push_response(answer_response("1"));
    let eval = engine
        .evaluate(&mut module, candidate, None)
        .await
        .unwrap()
        .completed()
        .unwrap();
    assert_eq!(engine.spend().lm_calls, 4, "new baseline means fresh rollouts");
    assert_eq!(engine.spend().cache_hits, 0);
    assert!(eval.rollouts.iter().all(|r| r.trace.is_some()));
}

#[tokio::test]
async fn cache_salt_partitions_the_rollout_cache() {
    let client = TestCompletionModel::new([answer_response("0"), answer_response("1")]);
    let metric = ExactMatch;
    let candidate = Candidate::with_instruction("predictor", "salted");

    let checkpoint = {
        let mut module = lm_module(client.clone()).await;
        let mut engine = EvalEngine::new(trainset(2), &metric, EngineConfig::default());
        let idx = engine.register(candidate.clone());
        engine
            .evaluate(&mut module, idx, None)
            .await
            .unwrap()
            .completed()
            .unwrap();
        engine.checkpoint().unwrap()
    };

    // Same checkpoint, bumped salt (the sampling-params seam): every rollout
    // is a cache miss and needs fresh LM responses.
    let mut module = lm_module(client.clone()).await;
    let mut engine = EvalEngine::resume(
        trainset(2),
        &metric,
        EngineConfig {
            cache_salt: 1,
            ..EngineConfig::default()
        },
        &checkpoint,
    )
    .unwrap();
    client.push_response(answer_response("0"));
    client.push_response(answer_response("1"));
    let idx = engine.register(candidate.clone());
    engine
        .evaluate(&mut module, idx, None)
        .await
        .expect("salted evaluation must run fresh rollouts")
        .completed()
        .unwrap();
    assert_eq!(engine.spend().cache_hits, 0, "bumped salt never hits the cache");

    // Salt 0 again: the checkpointed entries are served with no LM calls.
    let mut module = lm_module(TestCompletionModel::new([])).await;
    let mut engine =
        EvalEngine::resume(trainset(2), &metric, EngineConfig::default(), &checkpoint).unwrap();
    let idx = engine.register(candidate);
    let eval = engine
        .evaluate(&mut module, idx, None)
        .await
        .expect("original salt must serve from the checkpointed cache")
        .completed()
        .unwrap();
    assert_eq!(engine.spend().cache_hits, 2);
    assert!(eval.rollouts.iter().all(|r| r.trace.is_none()));
}

#[tokio::test]
async fn apply_and_restore_are_the_single_candidate_seam() {
    let mut module = EchoModule::new();
    let before = ModuleState::from_module(&mut module).unwrap();
    assert_eq!(
        before.predictors["predictor"].instruction_override.as_deref(),
        Some("seed")
    );
    assert!(before.predictors["predictor"].demos.is_empty());

    let mut candidate = Candidate::new();
    candidate.set_instruction("predictor", "overlaid");
    candidate.set_demos(
        "predictor",
        vec![
            serde_json::json!({"prompt": "demo-q", "answer": "demo-a"})
                .as_object()
                .cloned()
                .unwrap(),
        ],
    );

    let undo = apply_candidate(&mut module, &candidate).unwrap();
    let applied = ModuleState::from_module(&mut module).unwrap();
    assert_eq!(
        applied.predictors["predictor"].instruction_override.as_deref(),
        Some("overlaid")
    );
    assert_eq!(applied.predictors["predictor"].demos.len(), 1);

    restore_candidate(&mut module, undo).unwrap();
    let after = ModuleState::from_module(&mut module).unwrap();
    assert_eq!(
        after.predictors["predictor"].instruction_override.as_deref(),
        Some("seed")
    );
    assert!(after.predictors["predictor"].demos.is_empty());

    // Unknown predictor: error, and no partial application sticks.
    let bad = Candidate::with_instruction("missing", "nope");
    let err = apply_candidate(&mut module, &bad).expect_err("unknown predictor must fail");
    assert!(err.to_string().contains("missing"));
    let unchanged = ModuleState::from_module(&mut module).unwrap();
    assert_eq!(
        unchanged.predictors["predictor"].instruction_override.as_deref(),
        Some("seed")
    );
}
