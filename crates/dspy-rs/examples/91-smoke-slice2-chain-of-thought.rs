use anyhow::{Result, bail};
use dspy_rs::{ChainOfThought, ChatAdapter, LM, PredictError, Signature, configure};

#[derive(Signature, Clone, Debug)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 2 ChainOfThought
    configure(
        LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let module = ChainOfThought::<SmokeSig>::new();
    let input = SmokeSigInput {
        prompt: "Reply with exactly: smoke-ok".to_string(),
    };

    let output = module
        .call(input)
        .await
        .map_err(|err| {
            eprintln!("smoke call failed: {err}");
            if let PredictError::Parse { raw_response, .. } = &err {
                eprintln!("raw_response: {:?}", raw_response);
            }
            anyhow::anyhow!("slice2 smoke failed")
        })?
        .into_inner();

    println!("reasoning: {}", output.reasoning);
    println!("answer: {}", output.answer);

    if !output.answer.to_ascii_lowercase().contains("smoke-ok") {
        bail!("unexpected answer content: {}", output.answer);
    }

    Ok(())
}
