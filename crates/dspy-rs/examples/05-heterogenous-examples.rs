/*
Script to run a heterogenous example.

Run with:
```
cargo run --example 05-heterogenous-examples
```
*/

use dspy_rs::{ChatAdapter, LM, Predict, Predictor, configure, example, sign};
use secrecy::SecretString;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure(
        LM::builder()
            .api_key(SecretString::from(std::env::var("OPENAI_API_KEY")?))
            .build()
            .await,
        ChatAdapter {},
    );

    let exp = example! {
        "number": "input" => 10,
    };
    let predict = Predict::new(sign! {
        (number: i32) -> number_squared: i32, number_cubed: i32
    });

    let prediction = predict.forward(exp).await?;
    println!("{prediction:?}");

    Ok(())
}
