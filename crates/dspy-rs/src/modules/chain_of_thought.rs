use crate::Augmentation;
use crate::augmentation::Augmented;
use crate::core::{Module, Signature};
use crate::predictors::{Demo, Predict, PredictBuilder};
use crate::{BamlType, PredictError, Predicted};

#[derive(Augmentation, Clone, Debug)]
#[augment(output, prepend)]
pub struct Reasoning {
    #[output]
    pub reasoning: String,
}

pub type ChainOfThoughtOutput<S> = WithReasoning<<S as Signature>::Output>;

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
