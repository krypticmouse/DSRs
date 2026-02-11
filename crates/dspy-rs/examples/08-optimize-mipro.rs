/*
Example: optimize a typed QA module using MIPROv2.

Run with:
```
cargo run --example 08-optimize-mipro --features dataloaders
```
*/

use anyhow::Result;
use bon::Builder;
use facet;
use dspy_rs::{
    ChatAdapter, DataLoader, Example, LM, MIPROv2, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric, average_score, configure,
    evaluate_trainset, init_tracing,
};
use dspy_rs::data::RawExample;

#[derive(Signature, Clone, Debug)]
struct QuestionAnswering {
    /// Answer the question accurately and concisely.

    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct SimpleQA {
    #[builder(default = Predict::<QuestionAnswering>::builder().instruction("Answer clearly.").build())]
    answerer: Predict<QuestionAnswering>,
}

impl Module for SimpleQA {
    type Input = QuestionAnsweringInput;
    type Output = QuestionAnsweringOutput;

    async fn forward(
        &self,
        input: QuestionAnsweringInput,
    ) -> Result<Predicted<QuestionAnsweringOutput>, PredictError> {
        self.answerer.call(input).await
    }
}

struct ExactMatchMetric;

impl TypedMetric<QuestionAnswering, SimpleQA> for ExactMatchMetric {
    async fn evaluate(
        &self,
        example: &Example<QuestionAnswering>,
        prediction: &Predicted<QuestionAnsweringOutput>,
    ) -> Result<MetricOutcome> {
        let expected = example.output.answer.trim().to_lowercase();
        let actual = prediction.answer.trim().to_lowercase();

        let score = if expected == actual {
            1.0
        } else if expected.contains(&actual) || actual.contains(&expected) {
            0.5
        } else {
            0.0
        };

        Ok(MetricOutcome::score(score))
    }
}

fn typed_hotpot_examples(raw_examples: Vec<RawExample>) -> Vec<Example<QuestionAnswering>> {
    raw_examples
        .into_iter()
        .filter_map(|example| {
            let question = example
                .data
                .get("question")
                .and_then(|value| value.as_str())?
                .to_string();
            let answer = example
                .data
                .get("answer")
                .and_then(|value| value.as_str())?
                .to_string();
            Some(Example::new(
                QuestionAnsweringInput { question },
                QuestionAnsweringOutput { answer },
            ))
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    println!("=== MIPROv2 Optimizer Example ===\n");

    configure(LM::default(), ChatAdapter);

    println!("Loading training data from HuggingFace...");
    let raw_train_examples = DataLoader::load_hf(
        "hotpotqa/hotpot_qa",
        vec!["question".to_string()],
        vec!["answer".to_string()],
        "fullwiki",
        "validation",
        true,
    )?;
    let train_examples = typed_hotpot_examples(raw_train_examples);

    let train_subset = train_examples[..15].to_vec();
    println!("Using {} training examples\n", train_subset.len());

    let metric = ExactMatchMetric;
    let mut qa_module = SimpleQA::builder().build();

    println!("Evaluating baseline performance...");
    let baseline_score = average_score(&evaluate_trainset(&qa_module, &train_subset[..5], &metric).await?);
    println!("Baseline score: {:.3}\n", baseline_score);

    let optimizer = MIPROv2::builder()
        .num_candidates(8)
        .num_trials(15)
        .minibatch_size(10)
        .build();

    println!("Starting MIPROv2 optimization...");
    optimizer
        .compile(&mut qa_module, train_subset.clone(), &metric)
        .await?;

    println!("Evaluating optimized performance...");
    let optimized_score = average_score(&evaluate_trainset(&qa_module, &train_subset[..5], &metric).await?);
    println!("Optimized score: {:.3}", optimized_score);

    let improvement = ((optimized_score - baseline_score) / baseline_score.max(1e-6)) * 100.0;
    println!(
        "\nImprovement: {:.1}% ({:.3} -> {:.3})",
        improvement, baseline_score, optimized_score
    );

    let result = qa_module
        .call(QuestionAnsweringInput {
            question: "What is the capital of France?".to_string(),
        })
        .await?
        .into_inner();
    println!("Question: What is the capital of France?");
    println!("Answer: {}", result.answer);

    Ok(())
}
