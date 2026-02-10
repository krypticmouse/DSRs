use anyhow::Result;
use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    COPRO, CallMetadata, DynPredictor, Example, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric, named_parameters_ref,
};
use serde_json::json;
use std::collections::HashMap;

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
        _input: OptimizerSigInput,
    ) -> Result<Predicted<OptimizerSigOutput>, PredictError> {
        let answer = <Predict<OptimizerSig> as DynPredictor>::instruction(&self.predictor);
        Ok(Predicted::new(
            OptimizerSigOutput { answer },
            CallMetadata::default(),
        ))
    }
}

struct InstructionLengthMetric;

impl TypedMetric<InstructionEchoModule> for InstructionLengthMetric {
    async fn evaluate(
        &self,
        _example: &Example,
        prediction: &Predicted<OptimizerSigOutput>,
    ) -> Result<MetricOutcome> {
        Ok(MetricOutcome::score(prediction.answer.len() as f32))
    }
}

fn trainset() -> Vec<Example> {
    vec![
        Example::new(
            HashMap::from([("prompt".to_string(), json!("one"))]),
            vec!["prompt".to_string()],
            vec![],
        ),
        Example::new(
            HashMap::from([("prompt".to_string(), json!("two"))]),
            vec!["prompt".to_string()],
            vec![],
        ),
    ]
}

#[tokio::test]
async fn optimizer_mutates_predictor_instruction_via_named_parameters() {
    let mut module = InstructionEchoModule {
        predictor: Predict::<OptimizerSig>::builder().instruction("seed").build(),
    };

    let optimizer = COPRO::builder().breadth(4).depth(1).build();
    optimizer
        .compile(&mut module, trainset(), &InstructionLengthMetric)
        .await
        .expect("COPRO compile should succeed");

    let params = named_parameters_ref(&module).expect("predictor should be discoverable");
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].0, "predictor");

    let instruction = params[0].1.instruction();
    assert_ne!(instruction, "seed");
    assert!(instruction.contains("Optimization hint"));
}
