use anyhow::Result as AnyResult;
use futures::stream::{self, StreamExt};
use tracing::debug;

use crate::core::PredictState;
use crate::trace::JsonMap;
use crate::{Facet, PredictError, Predicted, Schema, SignatureSchema};

type IndexedForwardResult<T> = (usize, Result<Predicted<T>, PredictError>);

/// What optimizers read from — and, at explicit boundaries, write to — a
/// [`Predict`](crate::Predict) leaf.
///
/// This is the typed, object-safe view a module exposes through
/// [`Predictors::predictors`]. Optimizers only ever *read* through it during a
/// run (schema for reflection prompts, current instruction, demos as JSON);
/// the two mutating methods are boundary operations:
///
/// - [`set_trace_name`](PredictorInfo::set_trace_name) — the naming pass, run
///   once per optimization run (and by [`ModuleState`](crate::ModuleState))
///   so trace spans record the same name the module declared for the leaf;
/// - [`load_state`](PredictorInfo::load_state) — the install seam, called by
///   `ModuleState::apply` and by the optimizer's final one-shot install of the
///   winning candidate. Candidate *evaluation* never calls it: candidates are
///   injected ambiently per call tree, not written into the module.
pub trait PredictorInfo: Send + Sync {
    /// The [`SignatureSchema`] for this predictor's signature — field names
    /// and docs for optimizer reflection prompts.
    fn schema(&self) -> &'static SignatureSchema;

    /// The current effective instruction (override if set, else the
    /// signature's default).
    fn instruction(&self) -> String;

    /// The signature's default instruction (ignoring any override).
    fn default_instruction(&self) -> String;

    /// Current demos as flat JSON rows (input and output fields merged into
    /// one object per row).
    fn demos_as_json(&self) -> Vec<JsonMap>;

    /// Snapshot of the optimizable state (instruction override + demos).
    fn dump_state(&self) -> PredictState;

    /// Restores optimizable state from a snapshot — the one mutation seam.
    ///
    /// This is a *full* overwrite: `instruction_override: None` clears the
    /// override, `demos` replaces the demo set.
    ///
    /// # Errors
    ///
    /// Returns an error if the demos can't be converted to the predictor's
    /// typed `Demo<S>` form (schema mismatch).
    fn load_state(&mut self, state: PredictState) -> AnyResult<()>;

    /// Assigns the component name this predictor records on trace spans.
    ///
    /// Part of the trace-name contract (see [`Predictors`]): the optimizer
    /// stamps each leaf with the name the module declared before any traced
    /// pass, so spans join back to the same names candidates address.
    fn set_trace_name(&mut self, name: &str);
}

/// Explicit predictor-leaf discovery: a module *names* its optimizable
/// [`Predict`](crate::Predict) leaves.
///
/// This replaces the old reflection walker. A module that wants to be
/// optimizable (or persistable via [`ModuleState`](crate::ModuleState))
/// declares its leaves explicitly — no derive magic, no pointer casts:
///
/// ```ignore
/// struct TwoStepQA {
///     retrieve: Predict<RetrieveSig>,
///     answer: ChainOfThought<AnswerSig>,
/// }
///
/// dspy_rs::predictors!(TwoStepQA { retrieve, answer });
/// // or by hand:
/// impl Predictors for TwoStepQA {
///     fn predictors(&self) -> Vec<(String, &dyn PredictorInfo)> {
///         vec![("retrieve".into(), &self.retrieve), ("answer".into(), &self.answer)]
///     }
///     fn predictors_mut(&mut self) -> Vec<(String, &mut dyn PredictorInfo)> {
///         vec![("retrieve".into(), &mut self.retrieve), ("answer".into(), &mut self.answer)]
///     }
/// }
/// ```
///
/// # The trace-name contract
///
/// The names returned here are the *canonical identity* of each leaf:
///
/// 1. they become the leaf's trace-span component name (the optimizer stamps
///    them via [`PredictorInfo::set_trace_name`] once per run);
/// 2. optimizer candidates address leaves by these names (ambient
///    [`fx::Params`](crate::fx::Params) entries bind per leaf at call time);
/// 3. [`ModuleState`](crate::ModuleState) persists per-leaf state under them.
///
/// Names must be unique within a module and stable across
/// [`predictors`](Predictors::predictors) / [`predictors_mut`](Predictors::predictors_mut).
pub trait Predictors {
    /// The module's predictor leaves, `(name, read view)` in declaration order.
    fn predictors(&self) -> Vec<(String, &dyn PredictorInfo)>;

    /// The module's predictor leaves, `(name, mutable view)` in declaration
    /// order. Only the boundary operations (naming pass, state install) go
    /// through this.
    fn predictors_mut(&mut self) -> Vec<(String, &mut dyn PredictorInfo)>;
}

/// Implements [`Predictors`] for a module struct from a list of predictor
/// fields, using each field's identifier as its leaf name.
///
/// ```ignore
/// struct Pipeline { draft: Predict<Draft>, refine: ChainOfThought<Refine> }
/// dspy_rs::predictors!(Pipeline { draft, refine });
/// ```
#[macro_export]
macro_rules! predictors {
    ($ty:ty { $($field:ident),* $(,)? }) => {
        impl $crate::Predictors for $ty {
            fn predictors(&self) -> ::std::vec::Vec<(::std::string::String, &dyn $crate::PredictorInfo)> {
                ::std::vec![$(
                    (
                        ::std::string::String::from(::core::stringify!($field)),
                        &self.$field as &dyn $crate::PredictorInfo,
                    )
                ),*]
            }

            fn predictors_mut(&mut self) -> ::std::vec::Vec<(::std::string::String, &mut dyn $crate::PredictorInfo)> {
                ::std::vec![$(
                    (
                        ::std::string::String::from(::core::stringify!($field)),
                        &mut self.$field as &mut dyn $crate::PredictorInfo,
                    )
                ),*]
            }
        }
    };
}

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
    type Input: Schema + for<'a> Facet<'a> + Send + Sync;

    /// What the LM is asked to produce.
    ///
    /// Augmented modules change this (e.g. [`crate::ChainOfThought`] wraps it with
    /// `WithReasoning<_>` because the LM now generates a reasoning field). Wrapper modules
    /// that don't modify the prompt keep the inner module's output — their bookkeeping
    /// lives on [`crate::CallMetadata`], not here.
    type Output: Schema + for<'a> Facet<'a> + Send + Sync;

    /// The implementation hook. Module authors put their execution logic here.
    ///
    /// Callers should use [`call`](Module::call) instead.
    ///
    /// # Why `input` is taken by value
    ///
    /// A deliberate API decision: by-value input lets pipeline authors *move*
    /// fields into sub-module inputs with no clones (`AnswerInput { context:
    /// ctx.passages }`), which is the common composition pattern. The cost is
    /// one input clone per example in evaluation loops that reuse a trainset —
    /// noise next to the LM calls those loops make. Taking `&Input` would invert
    /// the trade: every intermediate hand-off inside `forward` would clone instead.
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
    let mut indexed_results: Vec<IndexedForwardResult<M::Output>> =
        stream::iter(inputs.into_iter().enumerate())
            .map(|(idx, input)| async move { (idx, module.call(input).await) })
            .buffer_unordered(max_concurrency)
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
