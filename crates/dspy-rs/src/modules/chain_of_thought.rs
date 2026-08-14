use crate::Augmentation;
use crate::augmentation::Augmented;
use crate::core::Signature;
use crate::predictors::Predict;

/// Augmentation that prepends a `reasoning: String` field to a signature's output.
///
/// The "think step by step" primitive. The LM sees `reasoning` as the *first* output
/// field and generates it before the actual answer — this matters because the reasoning
/// text is in the context window when the LM produces subsequent fields, so it literally
/// has its own chain of thought to draw on. Used by [`ChainOfThought`].
#[derive(Augmentation, Clone, Debug)]
#[augment(output, prepend)]
pub struct Reasoning {
    #[output]
    pub reasoning: String,
}

/// Convenience alias for `ChainOfThought`'s output type.
pub type ChainOfThoughtOutput<S> = WithReasoning<<S as Signature>::Output>;

/// Asks the LM to reason step-by-step before producing the answer.
///
/// The simplest strategy upgrade from bare [`Predict`]. This is pure sugar:
/// `ChainOfThought<S>` *is* `Predict<Augmented<S, Reasoning>>` — the augmentation is
/// the one chain-of-thought mechanism, and this alias just names the common case.
/// The prompt includes a `reasoning` field before the regular output fields, and the
/// LM fills it in. The reasoning text is a real output field, not hidden metadata.
///
/// ```no_run
/// # async fn example() -> Result<(), dspy_rs::PredictError> {
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// let cot = ChainOfThought::<QA>::new();
/// let result = cot.call(QAInput { question: "What is 2+2?".into() }).await?;
/// println!("{}", result.reasoning);  // the LM's chain of thought
/// println!("{}", result.answer);     // the actual answer, via Deref
/// # Ok(())
/// # }
/// ```
///
/// Configure demos, instruction, and tools through the regular
/// [`Predict::builder`] — demos are `Example<Augmented<S, Reasoning>>`, so they
/// must include reasoning, showing the LM what good chain-of-thought looks like.
///
/// Swapping `Predict<QA>` → `ChainOfThought<QA>` changes the output type from
/// `QAOutput` to [`WithReasoning<QAOutput>`]. The compiler catches every downstream
/// site that needs updating — that's the strategy swap working as designed.
///
/// If you're using a reasoning model (o1, o3, DeepSeek-R1, etc.), you probably don't
/// want this — the model already thinks internally before answering. Adding an explicit
/// `reasoning` output field on top of that is redundant and can hurt quality. Use bare
/// [`Predict`] instead.
///
/// This is not multi-turn conversation. Reasoning and answer are produced in a single
/// LM call. The LM is simply asked to show its work before answering.
pub type ChainOfThought<S> = Predict<Augmented<S, Reasoning>>;
