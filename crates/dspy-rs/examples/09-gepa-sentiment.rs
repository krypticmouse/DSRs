/*
Example: using GEPA to optimize a typed sentiment module.

Run with:
```
OPENAI_API_KEY=your_key cargo run --example 09-gepa-sentiment
```
*/

use anyhow::Result;
use bon::Builder;
use facet;
use dspy_rs::{
    ChatAdapter, Example, FeedbackMetric, GEPA, LM, MetricOutcome, Module, Optimizer, Predict,
    PredictError, Predicted, Signature, TypedMetric, average_score, configure, evaluate_trainset,
    init_tracing,
};

#[derive(Signature, Clone, Debug)]
struct SentimentSignature {
    /// Analyze the sentiment and classify as positive, negative, or neutral.

    #[input]
    text: String,

    #[output]
    sentiment: String,

    #[output]
    reasoning: String,
}

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct SentimentAnalyzer {
    #[builder(default = Predict::<SentimentSignature>::new())]
    predictor: Predict<SentimentSignature>,
}

impl Module for SentimentAnalyzer {
    type Input = SentimentSignatureInput;
    type Output = SentimentSignatureOutput;

    async fn forward(
        &self,
        input: SentimentSignatureInput,
    ) -> Result<Predicted<SentimentSignatureOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

struct SentimentMetric;

impl TypedMetric<SentimentSignature, SentimentAnalyzer> for SentimentMetric {
    async fn evaluate(
        &self,
        example: &Example<SentimentSignature>,
        prediction: &Predicted<SentimentSignatureOutput>,
    ) -> Result<MetricOutcome> {
        let predicted = prediction.sentiment.trim().to_lowercase();
        let expected = example.output.sentiment.trim().to_lowercase();

        let score = (predicted == expected) as u8 as f32;
        let feedback = FeedbackMetric::new(
            score,
            format!(
                "expected={expected}; predicted={predicted}; reasoning={}",
                prediction.reasoning
            ),
        );

        Ok(MetricOutcome::with_feedback(score, feedback))
    }
}

fn sentiment_example(text: &str, expected: &str) -> Example<SentimentSignature> {
    Example::new(
        SentimentSignatureInput {
            text: text.to_string(),
        },
        SentimentSignatureOutput {
            sentiment: expected.to_string(),
            reasoning: String::new(),
        },
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(
        LM::builder().temperature(0.7).build().await?,
        ChatAdapter,
    );

    let trainset = vec![
        sentiment_example(
            "This movie was absolutely fantastic! I loved every minute of it.",
            "positive",
        ),
        sentiment_example("Terrible service, will never come back again.", "negative"),
        sentiment_example("The weather is okay, nothing special.", "neutral"),
        sentiment_example(
            "Despite some minor issues, I'm quite happy with the purchase.",
            "positive",
        ),
        sentiment_example("I have mixed feelings about this product.", "neutral"),
        sentiment_example("This is the worst experience I've ever had!", "negative"),
    ];

    let metric = SentimentMetric;
    let mut module = SentimentAnalyzer::builder().build();

    let baseline = average_score(&evaluate_trainset(&module, &trainset, &metric).await?);
    println!("Baseline score: {baseline:.3}");

    let gepa = GEPA::builder()
        .num_iterations(5)
        .minibatch_size(4)
        .num_trials(3)
        .temperature(0.9)
        .track_stats(true)
        .build();

    let result = gepa.compile(&mut module, trainset.clone(), &metric).await?;

    println!("Best average score: {:.3}", result.best_candidate.average_score());
    println!("Total rollouts: {}", result.total_rollouts);
    println!("Total LM calls: {}", result.total_lm_calls);
    println!("Best instruction: {}", result.best_candidate.instruction);

    let test_example = sentiment_example(
        "This product changed my life! Absolutely amazing!",
        "positive",
    );
    let test_prediction = module
        .call(SentimentSignatureInput {
            text: "This product changed my life! Absolutely amazing!".to_string(),
        })
        .await?;
    let test_feedback = metric.evaluate(&test_example, &test_prediction).await?;

    println!("Test prediction: {}", test_prediction.sentiment);
    println!("Test score: {:.3}", test_feedback.score);
    if let Some(feedback) = test_feedback.feedback {
        println!("Feedback: {}", feedback.feedback);
    }

    Ok(())
}
