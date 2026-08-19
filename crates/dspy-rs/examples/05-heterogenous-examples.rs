/*
Script to run a typed predictor from a heterogeneous JSON payload.

Source rows often carry more fields than a signature consumes (metadata,
tags, debug notes). The serde boundary handles this directly: deserialize
the signature's generated input struct straight from the JSON row — extra
fields are ignored, missing or mistyped ones are loud errors.

Run with:
```
cargo run --example 05-heterogenous-examples
```
*/

use anyhow::Result;
use dspy_rs::{LM, Predict, Signature, configure, init_tracing};
use serde_json::json;

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

    configure(LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?);

    // A heterogeneous source row: only `number` is part of the signature.
    let row = json!({
        "number": 10,
        "debug_note": "metadata not used by the signature",
        "tags": ["math", "demo"],
    });

    // The serde boundary: typed input straight from the JSON row. Unknown
    // fields are ignored; a missing/mistyped `number` is a deserialize error.
    let input: NumberSignatureInput = serde_json::from_value(row)?;

    let predictor = Predict::<NumberSignature>::new();
    let prediction = predictor.call(input).await?.into_inner();

    println!(
        "squared={}, cubed={}",
        prediction.number_squared, prediction.number_cubed
    );
    Ok(())
}
