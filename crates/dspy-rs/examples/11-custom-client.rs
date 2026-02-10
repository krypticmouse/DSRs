/*
Example demonstrating LMClient::from_custom() with a typed predictor.

Run with:
```
cargo run --example 11-custom-client
```
*/

use anyhow::Result;
use dspy_rs::{ChatAdapter, LM, LMClient, Predict, Signature, configure, init_tracing};
use rig::providers::azure;
use std::env;

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    let api_key = env::var("AZURE_OPENAI_API_KEY").unwrap_or_else(|_| "dummy-key".to_string());
    let endpoint = env::var("AZURE_OPENAI_ENDPOINT")
        .unwrap_or_else(|_| "https://your-resource.openai.azure.com".to_string());

    let azure_client = azure::Client::builder()
        .api_key(api_key)
        .azure_endpoint(endpoint)
        .build()?;
    let azure_model = azure::CompletionModel::new(azure_client, "gpt-4o-mini");

    let custom_lm_client: LMClient = azure_model.into();
    let lm = LM::builder()
        .build()
        .await?
        .with_client(custom_lm_client)
        .await?;

    configure(lm, ChatAdapter);

    let predictor = Predict::<QA>::new();
    let prediction = predictor
        .call(QAInput {
            question: "What is the capital of France?".to_string(),
        })
        .await?;

    println!("answer: {}", prediction.answer);
    Ok(())
}
