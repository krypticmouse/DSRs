use anyhow::{Result, anyhow};
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::core::{Module, ToInput};
use crate::trace::{Trace, TraceMeta, TraceOutcome, capture_with_meta};
use crate::Predicted;

pub use crate::trace::Eval;

/// Default number of examples evaluated concurrently by [`evaluate_trainset`].
pub const DEFAULT_EVAL_CONCURRENCY: usize = 16;

/// How you tell the optimizer what "good" means.
///
/// Implement this to score a module's prediction against a ground-truth example.
/// The trait is generic over `E` (the trainset row — any struct you like) and
/// `M` (module), so your metric sees fully typed data: the row with *all* its
/// fields (gold labels, metric-only context the module never sees), and the
/// [`Predicted<M::Output>`](crate::Predicted) which may be augmented (e.g.
/// `WithReasoning<QAOutput>` for `ChainOfThought`).
///
/// The third argument is the rollout's execution [`Trace`] when the caller
/// captured one (the evaluation loop always does; direct callers may pass
/// `None`). Metrics that inspect intermediate steps slice it by component:
/// `trace.for_component("retriever")`.
///
/// Return [`Eval::score()`] for a numerical score (0.0–1.0 by convention).
/// Return [`Eval::with_feedback()`] to explain *why* — [`GEPA`](crate::GEPA)
/// requires the feedback, other optimizers ignore it.
///
/// # Example
///
/// ```ignore
/// struct HotpotRow { question: String, answer: String, supporting_facts: Vec<String> }
///
/// struct ExactMatch;
///
/// impl TypedMetric<HotpotRow, Predict<QA>> for ExactMatch {
///     async fn evaluate(
///         &self,
///         example: &HotpotRow,
///         prediction: &Predicted<QAOutput>,
///         _trace: Option<&Trace>,
///     ) -> Result<Eval> {
///         let score = if prediction.answer == example.answer { 1.0 } else { 0.0 };
///         Ok(Eval::score(score))
///     }
/// }
/// ```
#[allow(async_fn_in_trait)]
pub trait TypedMetric<E, M>: Send + Sync
where
    M: Module,
{
    async fn evaluate(
        &self,
        example: &E,
        prediction: &Predicted<M::Output>,
        trace: Option<&Trace>,
    ) -> Result<Eval>;
}

/// Runs a module on every example in a trainset and scores each with a metric.
///
/// The trainset is a slice of any row type `E` that projects into the module's
/// input via [`ToInput`] — a `#[derive(Example)]` struct, an
/// `(S::Input, S::Output)` tuple, or a hand-written impl.
///
/// Returns one [`Eval`] per example, in trainset order. Individual LM call
/// failures are propagated (not swallowed) — if any call fails, the whole evaluation
/// fails. For fault-tolerant batching, use [`forward_all`](crate::forward_all) instead.
///
/// Each example runs inside a trace [`capture`](crate::trace::capture) scope and
/// the metric receives the rollout's `Trace`; metric evaluation itself happens
/// outside the scope, so LM-as-judge metrics don't pollute the execution trace.
///
/// Examples are evaluated concurrently ([`DEFAULT_EVAL_CONCURRENCY`] at a time); use
/// [`evaluate_trainset_with_concurrency`] to tune the level. Optimizers call this
/// internally; you can also use it directly to benchmark your module:
///
/// ```ignore
/// let evals = evaluate_trainset(&module, &trainset, &metric).await?;
/// println!("Average: {:.3}", average_score(&evals));
/// ```
///
/// # Errors
///
/// - Any [`Module::call`] failure propagates immediately
/// - Any [`TypedMetric::evaluate`] failure propagates immediately
pub async fn evaluate_trainset<E, M, MT>(
    module: &M,
    trainset: &[E],
    metric: &MT,
) -> Result<Vec<Eval>>
where
    E: ToInput<M::Input> + Sync,
    M: Module,
    MT: TypedMetric<E, M>,
{
    evaluate_trainset_with_concurrency(module, trainset, metric, DEFAULT_EVAL_CONCURRENCY).await
}

/// [`evaluate_trainset`] with an explicit concurrency level.
///
/// `max_concurrency` LM calls run in flight at once; results come back in trainset
/// order. Use `1` for strictly sequential evaluation (e.g. rate-limited providers).
pub async fn evaluate_trainset_with_concurrency<E, M, MT>(
    module: &M,
    trainset: &[E],
    metric: &MT,
    max_concurrency: usize,
) -> Result<Vec<Eval>>
where
    E: ToInput<M::Input> + Sync,
    M: Module,
    MT: TypedMetric<E, M>,
{
    evaluate_examples(module, trainset, metric, max_concurrency).await
}

/// One evaluated rollout: the metric result plus the execution trace that
/// produced it (with [`Trace::outcome`] filled in).
pub type Rollout = (Eval, Trace);

/// Concurrency core shared by the public entry points and optimizers: runs each
/// example under a capture scope, scores it with the metric (outside the scope),
/// and records the eval into the trace outcome.
pub(crate) async fn evaluate_examples_traced<'a, E, M, MT, I>(
    module: &M,
    examples: I,
    metric: &MT,
    max_concurrency: usize,
) -> Result<Vec<Rollout>>
where
    E: ToInput<M::Input> + Sync + 'a,
    M: Module,
    MT: TypedMetric<E, M>,
    I: IntoIterator<Item = &'a E>,
{
    stream::iter(examples.into_iter().map(|example| async move {
        let input = example.to_input();
        let meta = TraceMeta {
            input: serde_json::to_value(&input)
                .ok()
                .and_then(|value| match value {
                    serde_json::Value::Object(map) => Some(map),
                    _ => None,
                }),
            ..TraceMeta::default()
        };
        let started = std::time::Instant::now();
        let (result, mut trace) = capture_with_meta(meta, || module.call(input)).await;
        let predicted = result.map_err(|err| anyhow!("{err}"))?;
        let eval = metric.evaluate(example, &predicted, Some(&trace)).await?;
        trace.outcome = Some(TraceOutcome {
            output: serde_json::to_value(&*predicted)
                .ok()
                .and_then(|value| match value {
                    serde_json::Value::Object(map) => Some(map),
                    _ => None,
                }),
            error: None,
            eval: Some(eval.clone()),
            duration_us: started.elapsed().as_micros() as u64,
        });
        Ok::<_, anyhow::Error>((eval, trace))
    }))
    .buffered(max_concurrency.max(1))
    .try_collect()
    .await
}

/// [`evaluate_examples_traced`] projected down to the metric results.
pub(crate) async fn evaluate_examples<'a, E, M, MT, I>(
    module: &M,
    examples: I,
    metric: &MT,
    max_concurrency: usize,
) -> Result<Vec<Eval>>
where
    E: ToInput<M::Input> + Sync + 'a,
    M: Module,
    MT: TypedMetric<E, M>,
    I: IntoIterator<Item = &'a E>,
{
    Ok(
        evaluate_examples_traced(module, examples, metric, max_concurrency)
            .await?
            .into_iter()
            .map(|(eval, _)| eval)
            .collect(),
    )
}

/// Arithmetic mean of scores from a slice of [`Eval`]s.
///
/// Returns `0.0` for an empty slice.
pub fn average_score(evals: &[Eval]) -> f64 {
    if evals.is_empty() {
        return 0.0;
    }

    evals.iter().map(|e| e.score).sum::<f64>() / evals.len() as f64
}
