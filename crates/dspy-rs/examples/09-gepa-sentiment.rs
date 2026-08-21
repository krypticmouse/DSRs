/*
Example: using GEPA to optimize a typed sentiment module.

Run with:
```
OPENAI_API_KEY=your_key cargo run --example 09-gepa-sentiment
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    Eval, GEPA, LM, Module, ModuleState, Optimizer,
    Predict, PredictError, Predicted, Signature, TypedMetric, average_score, configure,
    evaluate_trainset, init_tracing,
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

/// A labeled trainset row: the input plus the gold output. Tuples implement
/// `ToInput`/`ToOutput` out of the box, so a small inline trainset needs no
/// dedicated row struct.
type SentimentRow = (SentimentSignatureInput, SentimentSignatureOutput);

#[derive(Builder, facet::Facet)]
#[facet(crate = facet)]
struct SentimentAnalyzer {
    #[builder(default = Predict::<SentimentSignature>::new())]
    predictor: Predict<SentimentSignature>,
}

dspy_rs::predictors!(SentimentAnalyzer { predictor });

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

impl TypedMetric<SentimentRow, SentimentAnalyzer> for SentimentMetric {
    async fn evaluate(
        &self,
        example: &SentimentRow,
        prediction: &Predicted<SentimentSignatureOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let predicted = prediction.sentiment.trim().to_lowercase();
        let expected = example.1.sentiment.trim().to_lowercase();

        let score = (predicted == expected) as u8 as f64;
        Ok(Eval::with_feedback(
            score,
            format!(
                "expected={expected}; predicted={predicted}; reasoning={}",
                prediction.reasoning
            ),
        ))
    }
}

fn sentiment_example(text: &str, expected: &str) -> SentimentRow {
    (
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

    configure(LM::builder().temperature(0.7).build().await?);

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

    // A reflection LM turns GEPA's mutation step into a real rewrite: it reads
    // the current instruction plus per-example feedback and proposes an improved
    // instruction each generation. Without `prompt_model`, GEPA falls back to
    // deterministic feedback concatenation.
    let reflection_lm = LM::builder().temperature(1.0).build().await?;

    let gepa = GEPA::builder()
        .num_iterations(5)
        .minibatch_size(4)
        .prompt_model(reflection_lm)
        .seed(42) // reproducible minibatch sampling
        .eval_concurrency(8) // LM calls in flight during candidate evaluation
        .track_stats(true)
        .build();

    let result = gepa.compile_module(&mut module, &trainset, &metric).await?;

    println!(
        "Best average score: {:.3}",
        result.best_candidate.average_score()
    );
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
    let test_feedback = metric.evaluate(&test_example, &test_prediction, None).await?;

    println!("Test prediction: {}", test_prediction.sentiment);
    println!("Test score: {:.3}", test_feedback.score);
    if let Some(feedback) = test_feedback.feedback {
        println!("Feedback: {feedback}");
    }

    // Persist the optimized instructions/demos so production can reload them
    // with `ModuleState::load(...)?.apply(&mut module)?` — no re-optimization.
    ModuleState::from_module(&module)?.save("optimized-sentiment.json")?;
    println!("Saved optimized module state to optimized-sentiment.json");

    Ok(())
}
