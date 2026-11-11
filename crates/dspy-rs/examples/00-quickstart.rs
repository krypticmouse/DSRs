//! Run with OPENAI_API_KEY set, or use --inspect-only to print the prompt offline.
use dspy_rs::{LM, Predict, Signature};

/// Answer questions accurately and concisely.
#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let inspect_only = std::env::args().any(|arg| arg == "--inspect-only");
    let api_key = if inspect_only {
        "unused-for-prompt-inspection".to_string()
    } else {
        std::env::var("OPENAI_API_KEY")?
    };
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .api_key(api_key)
        .build()
        .await?;
    let predict = Predict::<QA>::builder().lm(lm.clone()).build();
    let input = QAInput {
        question: "What is the capital of France?".into(),
    };

    // Building a chat formats the prompt without sending a provider request.
    let chat = predict.build_chat(&input).await?;
    println!("{}", serde_json::to_string_pretty(&chat.to_json())?);
    if inspect_only {
        return Ok(());
    }

    let (prediction, _conversation) = predict.call_and_parse(chat).await?;
    println!("Answer: {}", prediction.answer);
    for entry in lm.inspect_history(1).await {
        println!("Raw output: {:?}", entry.raw_output);
        println!("Parsed output: {:?}", entry.output);
    }
    Ok(())
}
