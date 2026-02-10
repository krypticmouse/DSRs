use anyhow::{Result, bail};
use dspy_rs::{ChatAdapter, LM, Predict, PredictError, Signature, configure};

#[derive(Signature, Clone, Debug)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 1 Typed Predict
    configure(
        LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let module = Predict::<SmokeSig>::new();
    let input = SmokeSigInput {
        prompt: "Reply with exactly: smoke-ok".to_string(),
    };

    let output = module.call(input).await.map_err(|err| {
        eprintln!("smoke call failed: {err}");
        if let PredictError::Parse { raw_response, .. } = &err {
            eprintln!("raw_response: {:?}", raw_response);
        }
        anyhow::anyhow!("slice1 smoke failed")
    })?
    .into_inner();

    println!("answer: {}", output.answer);

    if !output.answer.to_ascii_lowercase().contains("smoke-ok") {
        bail!("unexpected answer content: {}", output.answer);
    }

    Ok(())
}
