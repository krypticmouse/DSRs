use futures::stream::{self, StreamExt};
use kdam::{BarExt, tqdm};
use tracing::debug;

use crate::{BamlType, Facet, PredictError, Predicted};

type IndexedForwardResult<T> = (usize, Result<Predicted<T>, PredictError>);

/// Strategy-swapping interface for prompting modules.
///
/// Everything in dsrs is a Module — a bare LM call ([`crate::Predict`]),
/// chain-of-thought reasoning, a multi-step retrieval pipeline. The trait's purpose
/// is composition through types: swap `Predict<QA>` for `ChainOfThought<QA>` and the
/// compiler catches every downstream change. That's the design.
///
/// Two methods: [`call`](Module::call) for callers, [`forward`](Module::forward) for
/// implementors. `call` currently just delegates to `forward` — the split exists so we
/// can add hooks or tracing around `call` without breaking module implementations.
///
/// # Two kinds of output data
///
/// Every call returns [`Predicted<Output>`](crate::Predicted), which carries:
/// - **`Output`** — what the LM was asked to produce. Shaped by your signature and any
///   augmentations. Accessible directly via `Deref`: `result.answer`, `result.reasoning`.
/// - **[`CallMetadata`](crate::CallMetadata)** — what the runtime observed. Token counts,
///   raw response, constraint results. Never enters a prompt. Via `result.metadata()`.
///
/// This drives the type system: [`ChainOfThought`](crate::ChainOfThought) changes `Output`
/// because it modifies the prompt (adds a `reasoning` field). A wrapper like `BestOfN` keeps
/// the same `Output` — same prompt, just picks the best result.
///
/// # Implementing `Module`
///
/// Implement [`forward`](Module::forward). Derive `Facet` on your struct so the
/// optimizer's walker can find your [`Predict`](crate::Predict) leaves automatically.
///
/// ```ignore
/// #[derive(Facet)]
/// struct TwoStepQA {
///     retrieve: Predict<RetrieveSig>,
///     answer: ChainOfThought<AnswerSig>,
/// }
///
/// impl Module for TwoStepQA {
///     type Input = RetrieveInput;
///     type Output = WithReasoning<AnswerOutput>;
///
///     async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
///         let ctx = self.retrieve.call(input).await?;
///         self.answer.call(AnswerInput { context: ctx.passages.clone() }).await
///     }
/// }
/// ```
///
/// Does not handle batching (use [`forward_all`]), retries, or rate limiting.
#[allow(async_fn_in_trait)]
pub trait Module: Send + Sync {
    /// What the module receives. Usually a `Signature`'s generated input struct.
    type Input: BamlType + for<'a> Facet<'a> + Send + Sync;

    /// What the LM is asked to produce.
    ///
    /// Augmented modules change this (e.g. [`crate::ChainOfThought`] wraps it with
    /// `WithReasoning<_>` because the LM now generates a reasoning field). Wrapper modules
    /// that don't modify the prompt keep the inner module's output — their bookkeeping
    /// lives on [`crate::CallMetadata`], not here.
    type Output: BamlType + for<'a> Facet<'a> + Send + Sync;

    /// The implementation hook. Module authors put their execution logic here.
    ///
    /// Callers should use [`call`](Module::call) instead.
    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError>;

    /// Runs the module. This is what you call.
    ///
    /// Delegates to [`forward`](Module::forward). The split exists for future
    /// hooks/tracing/middleware.
    async fn call(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        self.forward(input).await
    }
}

/// Runs a module on many inputs concurrently.
///
/// Returns `Vec<Result<...>>`, not `Result<Vec<...>>` — individual failures don't
/// abort the batch. Results preserve input order regardless of completion order.
///
/// Shows a progress bar on stderr. Use [`forward_all_with_progress`] to disable it.
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// let predict = Predict::<QA>::new();
/// let inputs = vec![
///     QAInput { question: "What is 2+2?".into() },
///     QAInput { question: "What is 3+3?".into() },
/// ];
/// let results = forward_all(&predict, inputs, 5).await;
/// for result in results {
///     match result {
///         Ok(predicted) => println!("{}", predicted.answer),
///         Err(e) => eprintln!("failed: {e}"),
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[tracing::instrument(
    name = "dsrs.forward_all",
    level = "debug",
    skip(module, inputs),
    fields(total_inputs = inputs.len(), max_concurrency)
)]
pub async fn forward_all<M>(
    module: &M,
    inputs: Vec<M::Input>,
    max_concurrency: usize,
) -> Vec<Result<Predicted<M::Output>, PredictError>>
where
    M: Module + ?Sized,
{
    forward_all_with_progress(module, inputs, max_concurrency, true).await
}

/// Like [`forward_all`], but with explicit control over the progress bar.
#[tracing::instrument(
    name = "dsrs.forward_all_with_progress",
    level = "debug",
    skip(module, inputs),
    fields(total_inputs = inputs.len(), max_concurrency, display_progress)
)]
pub async fn forward_all_with_progress<M>(
    module: &M,
    inputs: Vec<M::Input>,
    max_concurrency: usize,
    display_progress: bool,
) -> Vec<Result<Predicted<M::Output>, PredictError>>
where
    M: Module + ?Sized,
{
    let total = inputs.len();
    let mut pb = if display_progress {
        Some(tqdm!(total = total, desc = "Processing"))
    } else {
        None
    };

    let mut indexed_results: Vec<IndexedForwardResult<M::Output>> =
        stream::iter(inputs.into_iter().enumerate())
            .map(|(idx, input)| async move { (idx, module.call(input).await) })
            .buffer_unordered(max_concurrency)
            .inspect(|_| {
                if let Some(ref mut progress) = pb {
                    let _ = progress.update(1);
                }
            })
            .collect()
            .await;

    indexed_results.sort_by_key(|(idx, _)| *idx);

    let mut outcomes = Vec::with_capacity(indexed_results.len());
    for (_, outcome) in indexed_results {
        outcomes.push(outcome);
    }
    debug!(outcomes = outcomes.len(), "forward_all completed");
    outcomes
}
