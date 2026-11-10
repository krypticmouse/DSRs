//! Bounded cross-framework assessment: live, warm response cache, mock CPU.
use dspy_rs::{LM, LMClient, Predict, Signature, TestCompletionModel};
use rig::completion::AssistantContent;
use serde_json::json;
use std::time::Instant;

// Compatibility shim for pre-scalar revisions. Candidate inherent methods take
// precedence; the baseline keeps its existing object schema. This lets both
// revisions compile the byte-identical typed-prediction workload below.
#[allow(dead_code)]
trait ScalarSchemaBaseline: Sized {
    fn with_scalar_output_schema(self, _: bool) -> Self {
        self
    }
    fn supports_scalar_output_schema(&self) -> bool {
        false
    }
}
impl ScalarSchemaBaseline for LM {}

/// Answer the geography question with only the city name in the answer field.
#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    /// The geography question.
    question: String,
    #[output]
    /// The city name, without extra commentary.
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "offline".into());
    anyhow::ensure!(
        ["live", "cached", "offline"].contains(&mode.as_str()),
        "unknown mode"
    );
    let iterations = if mode == "live" { 1 } else { 1000 };
    let warmup = if mode == "live" { 1 } else { 100 };
    let mut lm = LM::builder()
        .base_url(std::env::var("OLLAMA_BASE_URL")?)
        .model("llama3.2".into())
        .temperature(0.0)
        .max_tokens(64)
        .max_retries(0)
        .cache(mode == "cached")
        .build()
        .await?
        .with_custom_output_schema(true)
        .with_scalar_output_schema(true);
    if mode == "offline" {
        let response = if lm.supports_scalar_output_schema() {
            r#""Paris""#
        } else {
            r#"{"answer":"Paris"}"#
        };
        lm = lm
            .with_client(LMClient::Test(TestCompletionModel::new(
                (0..iterations + warmup).map(|_| AssistantContent::text(response)),
            )))
            .await?;
    }
    dspy_rs::set_history_capacity(0);
    lm.history.set_capacity(0);
    let predictor = Predict::<QA>::builder().lm(lm).build();
    predictor.set_history_capacity(0);
    let input = || QAInput {
        question: "What is the capital of France?".into(),
    };
    let prompt = predictor.build_chat(&input()).await?.to_json().to_string();
    for expected in [
        "Answer the geography question with only the city name in the answer field.",
        "The geography question.",
        "The city name, without extra commentary.",
    ] {
        anyhow::ensure!(
            prompt.contains(expected),
            "missing benchmark instruction: {expected}"
        );
    }
    for _ in 0..warmup {
        assert_eq!(predictor.call(input()).await?.answer, "Paris");
    }
    let mut samples = Vec::new();
    let mut tokens = Vec::new();
    for _ in 0..iterations {
        let started = Instant::now();
        let output = predictor.call(input()).await?;
        samples.push(started.elapsed().as_nanos() as u64);
        assert_eq!(output.answer, "Paris");
        tokens.push(output.metadata().lm_usage.total_tokens);
    }
    println!(
        "{}",
        json!({"framework":"dsrs","mode":mode,"iterations":iterations,
        "warmup":warmup,"samples_ns":samples,"reported_tokens":tokens,"answer":"Paris"})
    );
    Ok(())
}
