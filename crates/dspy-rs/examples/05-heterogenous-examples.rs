/*
Script to run a typed predictor from a heterogeneous `Example` payload.

Run with:
```
cargo run --example 05-heterogenous-examples
```
*/

use anyhow::Result;
use dspy_rs::{
    ChatAdapter, Example, LM, Predict, Signature, configure, init_tracing, input_from_example,
};
use serde_json::json;
use std::collections::HashMap;

#[derive(Signature, Clone, Debug)]
struct NumberSignature {
    #[input]
    number: i32,

    #[output]
    number_squared: i32,

    #[output]
    number_cubed: i32,
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

    let heterogeneous = Example::new(
        HashMap::from([
            ("number".to_string(), json!(10)),
            ("debug_note".to_string(), json!("metadata not used by the signature")),
            ("tags".to_string(), json!(["math", "demo"])),
        ]),
        vec!["number".to_string()],
        vec![],
    );

    let input: NumberSignatureInput = input_from_example(&heterogeneous)?;
    let predictor = Predict::<NumberSignature>::new();
    let prediction = predictor.call(input).await?.into_inner();

    println!(
        "squared={}, cubed={}",
        prediction.number_squared, prediction.number_cubed
    );
    Ok(())
}
