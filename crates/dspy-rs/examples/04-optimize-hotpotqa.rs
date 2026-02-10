/*
Script to optimize a typed QA module for a HotpotQA subset with COPRO.

Run with:
```
cargo run --example 04-optimize-hotpotqa --features dataloaders
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    COPRO, ChatAdapter, DataLoader, Example, LM, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric, average_score, configure, evaluate_trainset,
    init_tracing, named_parameters_ref,
};

#[derive(Signature, Clone, Debug)]
struct QA {
    /// Concisely answer the question, but be accurate.

    #[input]
    question: String,

    #[output(desc = "Answer in less than 5 words.")]
    answer: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct QAModule {
    #[builder(default = Predict::<QA>::builder().instruction("Answer clearly and briefly.").build())]
    answerer: Predict<QA>,
}

impl Module for QAModule {
    type Input = QAInput;
    type Output = QAOutput;

    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.answerer.call(input).await
    }
}

struct ExactMatchMetric;

impl TypedMetric<QAModule> for ExactMatchMetric {
    async fn evaluate(
        &self,
        example: &Example,
        prediction: &Predicted<QAOutput>,
    ) -> Result<MetricOutcome> {
        let expected = example
            .data
            .get("answer")
            .and_then(|value| value.as_str())
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();
        Ok(MetricOutcome::score((expected == actual) as u8 as f32))
    }
}

fn answerer_instruction(module: &QAModule) -> Result<String> {
    let params = named_parameters_ref(module)?;
    let (_, predictor) = params
        .iter()
        .find(|(path, _)| path == "answerer")
        .ok_or_else(|| anyhow::anyhow!("answerer predictor not found"))?;
    Ok(predictor.instruction())
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let examples = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "validation",
        true,
    )?[..10]
        .to_vec();

    let metric = ExactMatchMetric;
    let mut module = QAModule::builder().build();

    let baseline = average_score(&evaluate_trainset(&module, &examples, &metric).await?);
    println!("baseline score: {baseline:.3}");
    println!("baseline instruction: {}", answerer_instruction(&module)?);

    let optimizer = COPRO::builder().breadth(10).depth(1).build();
    optimizer
        .compile(&mut module, examples.clone(), &metric)
        .await?;

    let optimized = average_score(&evaluate_trainset(&module, &examples, &metric).await?);
    println!("optimized score: {optimized:.3}");
    println!("optimized instruction: {}", answerer_instruction(&module)?);

    Ok(())
}
