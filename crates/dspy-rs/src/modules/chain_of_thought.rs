use crate::Augmentation;
use crate::augmentation::Augmented;
use crate::core::{Module, Signature};
use crate::predictors::{Demo, Predict, PredictBuilder};
use crate::{BamlType, PredictError, Predicted};

/// Augmentation that prepends a `reasoning: String` field to a signature's output.
///
/// This is the "think step by step" primitive. The LM sees the field in its output
/// format and generates reasoning text before answering. Used by [`ChainOfThought`].
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
/// The simplest strategy upgrade from bare [`Predict`]. Internally
/// just `Predict<Augmented<S, Reasoning>>` — the prompt includes a `reasoning` field
/// before the regular output fields, and the LM fills it in. The reasoning text is a
/// real output field, not hidden metadata.
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
/// Swapping `Predict<QA>` → `ChainOfThought<QA>` changes the output type from
/// `QAOutput` to [`WithReasoning<QAOutput>`]. The compiler catches every downstream
/// site that needs updating — that's the strategy swap working as designed.
///
/// This is not multi-turn conversation. Reasoning and answer are produced in a single
/// LM call. The LM is simply asked to show its work before answering.
#[derive(Default, facet::Facet)]
#[facet(crate = facet)]
pub struct ChainOfThought<S: Signature> {
    predictor: Predict<Augmented<S, Reasoning>>,
}

impl<S: Signature> ChainOfThought<S> {
    pub fn new() -> Self {
        Self {
            predictor: Predict::<Augmented<S, Reasoning>>::new(),
        }
    }

    pub fn with_predict(predictor: Predict<Augmented<S, Reasoning>>) -> Self {
        Self { predictor }
    }

    pub fn builder() -> ChainOfThoughtBuilder<S> {
        ChainOfThoughtBuilder::new()
    }

    pub async fn call(
        &self,
        input: S::Input,
    ) -> Result<Predicted<WithReasoning<S::Output>>, PredictError>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        self.forward(input).await
    }

    pub async fn forward(
        &self,
        input: S::Input,
    ) -> Result<Predicted<WithReasoning<S::Output>>, PredictError>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        self.predictor.call(input).await
    }
}

impl<S> Module for ChainOfThought<S>
where
    S: Signature + Clone,
    S::Input: BamlType,
    S::Output: BamlType,
{
    type Input = S::Input;
    type Output = WithReasoning<S::Output>;

    async fn forward(
        &self,
        input: S::Input,
    ) -> Result<Predicted<WithReasoning<S::Output>>, PredictError> {
        ChainOfThought::forward(self, input).await
    }
}

/// Builder for [`ChainOfThought`] with demos, tools, and instruction override.
pub struct ChainOfThoughtBuilder<S: Signature> {
    inner: PredictBuilder<Augmented<S, Reasoning>>,
}

impl<S: Signature> ChainOfThoughtBuilder<S> {
    fn new() -> Self {
        Self {
            inner: Predict::builder(),
        }
    }

    pub fn demo(mut self, demo: Demo<Augmented<S, Reasoning>>) -> Self {
        self.inner = self.inner.demo(demo);
        self
    }

    pub fn with_demos(
        mut self,
        demos: impl IntoIterator<Item = Demo<Augmented<S, Reasoning>>>,
    ) -> Self {
        self.inner = self.inner.with_demos(demos);
        self
    }

    pub fn add_tool(mut self, tool: impl rig::tool::ToolDyn + 'static) -> Self {
        self.inner = self.inner.add_tool(tool);
        self
    }

    pub fn with_tools(
        mut self,
        tools: impl IntoIterator<Item = std::sync::Arc<dyn rig::tool::ToolDyn>>,
    ) -> Self {
        self.inner = self.inner.with_tools(tools);
        self
    }

    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.inner = self.inner.instruction(instruction);
        self
    }

    pub fn build(self) -> ChainOfThought<S> {
        ChainOfThought::with_predict(self.inner.build())
    }
}
