use anyhow::{Result, anyhow};
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::core::Module;
use crate::predictors::Example;
use crate::{Predicted, Signature};

use super::FeedbackMetric;

/// Default number of examples evaluated concurrently by [`evaluate_trainset`].
pub const DEFAULT_EVAL_CONCURRENCY: usize = 16;

/// Result of evaluating a single example: a score and optional textual feedback.
///
/// Score-only metrics use [`MetricOutcome::score()`]. Feedback-aware metrics (required
/// by [`GEPA`](crate::GEPA)) use [`MetricOutcome::with_feedback()`] to include a [`FeedbackMetric`]
/// explaining *why* the example scored the way it did.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricOutcome {
    pub score: f32,
    pub feedback: Option<FeedbackMetric>,
}

impl MetricOutcome {
    /// Creates an outcome with only a numerical score.
    ///
    /// Sufficient for [`COPRO`](crate::COPRO) and [`MIPROv2`](crate::MIPROv2).
    /// [`GEPA`](crate::GEPA) will error if it receives outcomes without feedback.
    pub fn score(score: f32) -> Self {
        Self {
            score,
            feedback: None,
        }
    }

    /// Creates an outcome with a score and textual feedback.
    ///
    /// Required by [`GEPA`](crate::GEPA), which appends the feedback text to candidate
    /// instructions during evolutionary mutation.
    pub fn with_feedback(score: f32, feedback: FeedbackMetric) -> Self {
        Self {
            score,
            feedback: Some(feedback),
        }
    }
}

/// How you tell the optimizer what "good" means.
///
/// Implement this to score a module's prediction against a ground-truth example.
/// The trait is generic over `S` (signature) and `M` (module) so your metric sees
/// fully typed data: the [`Example<S>`](crate::predictors::Example) with its typed
/// input and expected output, and the [`Predicted<M::Output>`](crate::Predicted) which
/// may be augmented (e.g. `WithReasoning<QAOutput>` for `ChainOfThought`).
///
/// Return [`MetricOutcome::score()`] for a numerical score (0.0–1.0 by convention).
/// Return [`MetricOutcome::with_feedback()`] to include textual feedback explaining
/// *why* — [`GEPA`](crate::GEPA) uses this to guide its search, other optimizers ignore it.
///
/// # Example
///
/// ```ignore
/// struct ExactMatch;
///
/// impl TypedMetric<QA, Predict<QA>> for ExactMatch {
///     async fn evaluate(
///         &self,
///         example: &Example<QA>,
///         prediction: &Predicted<QAOutput>,
///     ) -> Result<MetricOutcome> {
///         let score = if prediction.answer == example.output.answer { 1.0 } else { 0.0 };
///         Ok(MetricOutcome::score(score))
///     }
/// }
/// ```
#[allow(async_fn_in_trait)]
pub trait TypedMetric<S, M>: Send + Sync
where
    S: Signature,
    M: Module<Input = S::Input>,
{
    async fn evaluate(
        &self,
        example: &Example<S>,
        prediction: &Predicted<M::Output>,
    ) -> Result<MetricOutcome>;
}

/// Runs a module on every example in a trainset and scores each with a metric.
///
/// Returns one [`MetricOutcome`] per example, in trainset order. Individual LM call
/// failures are propagated (not swallowed) — if any call fails, the whole evaluation
/// fails. For fault-tolerant batching, use [`forward_all`](crate::forward_all) instead.
///
/// Examples are evaluated concurrently ([`DEFAULT_EVAL_CONCURRENCY`] at a time); use
/// [`evaluate_trainset_with_concurrency`] to tune the level. Optimizers call this
/// internally; you can also use it directly to benchmark your module:
///
/// ```ignore
/// let outcomes = evaluate_trainset(&module, &trainset, &metric).await?;
/// println!("Average: {:.3}", average_score(&outcomes));
/// ```
///
/// # Errors
///
/// - Any [`Module::call`] failure propagates immediately
/// - Any [`TypedMetric::evaluate`] failure propagates immediately
pub async fn evaluate_trainset<S, M, MT>(
    module: &M,
    trainset: &[Example<S>],
    metric: &MT,
) -> Result<Vec<MetricOutcome>>
where
    S: Signature,
    S::Input: Clone,
    M: Module<Input = S::Input>,
    MT: TypedMetric<S, M>,
{
    evaluate_trainset_with_concurrency(module, trainset, metric, DEFAULT_EVAL_CONCURRENCY).await
}

/// [`evaluate_trainset`] with an explicit concurrency level.
///
/// `max_concurrency` LM calls run in flight at once; results come back in trainset
/// order. Use `1` for strictly sequential evaluation (e.g. rate-limited providers).
pub async fn evaluate_trainset_with_concurrency<S, M, MT>(
    module: &M,
    trainset: &[Example<S>],
    metric: &MT,
    max_concurrency: usize,
) -> Result<Vec<MetricOutcome>>
where
    S: Signature,
    S::Input: Clone,
    M: Module<Input = S::Input>,
    MT: TypedMetric<S, M>,
{
    evaluate_examples(module, trainset, metric, max_concurrency).await
}

/// Concurrency core shared by the public entry points and optimizers, generic over
/// borrowed examples so callers can evaluate sampled subsets without cloning rows.
pub(crate) async fn evaluate_examples<'a, S, M, MT, I>(
    module: &M,
    examples: I,
    metric: &MT,
    max_concurrency: usize,
) -> Result<Vec<MetricOutcome>>
where
    S: Signature,
    S::Input: Clone,
    M: Module<Input = S::Input>,
    MT: TypedMetric<S, M>,
    I: IntoIterator<Item = &'a Example<S>>,
{
    stream::iter(examples.into_iter().map(|example| async move {
        let input = example.input.clone();
        let predicted = module.call(input).await.map_err(|err| anyhow!("{err}"))?;
        metric.evaluate(example, &predicted).await
    }))
    .buffered(max_concurrency.max(1))
    .try_collect()
    .await
}

/// Arithmetic mean of scores from a slice of [`MetricOutcome`]s.
///
/// Returns `0.0` for an empty slice.
pub fn average_score(outcomes: &[MetricOutcome]) -> f32 {
    if outcomes.is_empty() {
        return 0.0;
    }

    outcomes.iter().map(|o| o.score).sum::<f32>() / outcomes.len() as f32
}
