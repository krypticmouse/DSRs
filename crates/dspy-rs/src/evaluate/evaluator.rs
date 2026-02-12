use anyhow::{Result, anyhow};

use crate::core::Module;
use crate::predictors::Example;
use crate::{Predicted, Signature};

use super::FeedbackMetric;

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
/// This runs sequentially (one example at a time). Optimizers call this internally;
/// you can also use it directly to benchmark your module:
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
    let mut outcomes = Vec::with_capacity(trainset.len());

    for example in trainset {
        let input = example.input.clone();
        let predicted = module.call(input).await.map_err(|err| anyhow!("{err}"))?;
        outcomes.push(metric.evaluate(example, &predicted).await?);
    }

    Ok(outcomes)
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
