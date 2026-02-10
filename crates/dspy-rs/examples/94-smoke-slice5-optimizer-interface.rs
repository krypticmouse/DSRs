use anyhow::{Result, bail};
use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{
    ChainOfThought, ChatAdapter, LM, PredictError, Signature, configure, named_parameters,
};

#[derive(Signature, Clone, Debug, facet::Facet)]
#[facet(crate = facet)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 5 Optimizer Interface
    configure(
        LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let mut module = ChainOfThought::<SmokeSig>::new();
    {
        let mut params = named_parameters(&mut module)?;
        let paths: Vec<String> = params.iter().map(|(path, _)| path.clone()).collect();
        println!("named_parameters: {:?}", paths);

        let (_, predictor) = params
            .iter_mut()
            .find(|(path, _)| path == "predictor")
            .ok_or_else(|| anyhow::anyhow!("expected `predictor` path"))?;
        predictor.set_instruction("Reply with exactly: smoke-ok".to_string());
    }

    let output = module
        .call(SmokeSigInput {
            prompt: "Return exactly smoke-ok.".to_string(),
        })
        .await
        .map_err(|err| {
            eprintln!("slice5 smoke call failed: {err}");
            if let PredictError::Parse { raw_response, .. } = &err {
                eprintln!("raw_response: {:?}", raw_response);
            }
            anyhow::anyhow!("slice5 smoke failed")
        })?
        .into_inner();

    println!("reasoning: {}", output.reasoning);
    println!("answer: {}", output.answer);

    if !output.answer.to_ascii_lowercase().contains("smoke-ok") {
        bail!("unexpected answer content: {}", output.answer);
    }

    Ok(())
}
