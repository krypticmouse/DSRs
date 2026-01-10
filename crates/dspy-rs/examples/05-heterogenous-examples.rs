/*
Script to run a heterogenous example.

Run with:
```
cargo run --example 05-heterogenous-examples
```
*/

#![allow(deprecated)]

use dspy_rs::{
    ChatAdapter, LM, LegacyPredict, Predictor, LegacySignature, configure, example,
};

#[LegacySignature]
struct NumberSignature {
    #[input]
    number: i32,
    #[output]
    number_squared: i32,
    #[output]
    number_cubed: i32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure(
        LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await
            .unwrap(),
        ChatAdapter {},
    );

    let exp = example! {
        "number": "input" => 10,
    };
    let predict = LegacyPredict::new(NumberSignature::new());

    let prediction = predict.forward(exp).await?;
    println!("{prediction:?}");

    Ok(())
}
