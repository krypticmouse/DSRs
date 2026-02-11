use anyhow::Result;
use facet;
use dspy_rs::{
    COPRO, CallMetadata, Example, MetricOutcome, Module, Optimizer, Predict, PredictError,
    Predicted, Signature, TypedMetric,
};

#[derive(Signature, Clone, Debug)]
struct OptimizerSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct InstructionEchoModule {
    predictor: Predict<OptimizerSig>,
}

impl Module for InstructionEchoModule {
    type Input = OptimizerSigInput;
    type Output = OptimizerSigOutput;

    async fn forward(
        &self,
        input: OptimizerSigInput,
    ) -> Result<Predicted<OptimizerSigOutput>, PredictError> {
        let _ = &self.predictor;
        Ok(Predicted::new(
            OptimizerSigOutput {
                answer: input.prompt,
            },
            CallMetadata::default(),
        ))
    }
}

struct InstructionLengthMetric;

impl TypedMetric<OptimizerSig, InstructionEchoModule> for InstructionLengthMetric {
    async fn evaluate(
        &self,
        _example: &Example<OptimizerSig>,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(prediction.answer.len() as f32))
    }
}

fn trainset() -> Vec<Example<OptimizerSig>> {
    vec![
        Example::new(
            OptimizerSigInput {
                prompt: "one".to_string(),
            },
            OptimizerSigOutput {
                answer: "one".to_string(),
            },
        ),
        Example::new(
            OptimizerSigInput {
                prompt: "two".to_string(),
            },
            OptimizerSigOutput {
                answer: "two".to_string(),
            },
        ),
    ]
}

#[tokio::test]
async fn optimizer_compile_succeeds_without_public_named_parameter_access() {
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = COPRO::builder().breadth(4).depth(1).build();
    optimizer
        .compile::<OptimizerSig, _, _>(&mut module, trainset(), &InstructionLengthMetric)
        .await
        .expect("COPRO compile should succeed with internal predictor discovery");
}
