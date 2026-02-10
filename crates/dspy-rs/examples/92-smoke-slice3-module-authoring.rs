use anyhow::{Result, bail};
use dspy_rs::{ChatAdapter, LM, Module, Predict, PredictError, Predicted, Signature, configure};

#[derive(Signature, Clone, Debug)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

struct SmokeModule {
    inner: Predict<SmokeSig>,
}

impl SmokeModule {
    fn new() -> Self {
        Self {
            inner: Predict::<SmokeSig>::new(),
        }
    }
}

impl Module for SmokeModule {
    type Input = <SmokeSig as Signature>::Input;
    type Output = <SmokeSig as Signature>::Output;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        self.inner.call(input).await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 3 Module Authoring
    configure(
        LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let module = SmokeModule::new();
    let input = SmokeSigInput {
        prompt: "Reply with exactly: smoke-ok".to_string(),
    };

    let output = module.call(input).await.map_err(|err| {
        eprintln!("smoke call failed: {err}");
        if let PredictError::Parse { raw_response, .. } = &err {
            eprintln!("raw_response: {:?}", raw_response);
        }
        anyhow::anyhow!("slice3 smoke failed")
    })?
    .into_inner();

    println!("answer: {}", output.answer);

    if !output.answer.to_ascii_lowercase().contains("smoke-ok") {
        bail!("unexpected answer content: {}", output.answer);
    }

    Ok(())
}
