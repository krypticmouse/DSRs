use anyhow::{Result, bail};
use dspy_rs::{ChatAdapter, LM, ReAct, Signature, configure, forward_all};

#[derive(Signature, Clone, Debug)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 4 ReAct + Operational
    configure(
        LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let module = ReAct::<SmokeSig>::builder()
        .max_steps(3)
        .tool("echo", "Echoes tool arguments", |args| async move {
            format!("echo-result: {args}")
        })
        .build();

    let input = SmokeSigInput {
        prompt: "Use the echo tool once if needed, then reply with exactly smoke-ok in the answer field."
            .to_string(),
    };

    let mut outcomes = forward_all(&module, vec![input], 1).await.into_iter();
    let outcome = outcomes.next().expect("expected one batch outcome");
    let (result, metadata) = outcome.into_parts();

    let output = result.map_err(|err| {
        eprintln!("smoke call failed: {}", err);
        eprintln!("raw_response: {:?}", metadata.raw_response);
        anyhow::anyhow!("slice4 smoke failed")
    })?;

    println!("tool_executions: {}", metadata.tool_executions.len());
    println!("answer: {}", output.answer);

    if !output.answer.to_ascii_lowercase().contains("smoke-ok") {
        bail!("unexpected answer content: {}", output.answer);
    }

    Ok(())
}
