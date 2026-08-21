use anyhow::{Result, bail};
use dspy_rs::{
    COPRO, ChainOfThought, LM, Eval, Optimizer, Predicted,
    Signature, TypedMetric, WithReasoning, configure,
};

#[derive(Signature, Clone, Debug, facet::Facet)]
#[facet(crate = facet)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

struct SmokeMetric;

impl TypedMetric<(SmokeSigInput, SmokeSigOutput), ChainOfThought<SmokeSig>> for SmokeMetric {
    async fn evaluate(
        &self,
        _example: &(SmokeSigInput, SmokeSigOutput),
        prediction: &Predicted<WithReasoning<SmokeSigOutput>>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let answer = prediction.answer.to_ascii_lowercase();
        Ok(Eval::score(
            (answer.contains("smoke") || answer.contains("ok")) as u8 as f64,
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 5 Optimizer Interface
    configure(LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?);

    let mut module = ChainOfThought::<SmokeSig>::new();
    let trainset = vec![(
        SmokeSigInput {
            prompt: "Return exactly smoke-ok.".to_string(),
        },
        SmokeSigOutput {
            answer: "smoke-ok".to_string(),
        },
    )];

    let optimizer = COPRO::builder().breadth(4).depth(1).build();
    optimizer
        .compile_module(&mut module, &trainset, &SmokeMetric)
        .await?;

    let output = module
        .call(SmokeSigInput {
            prompt: "Return exactly smoke-ok.".to_string(),
        })
        .await?
        .into_inner();

    println!("reasoning: {}", output.reasoning);
    println!("answer: {}", output.answer);

    if output.answer.trim().is_empty() {
        bail!("unexpected empty answer");
    }

    Ok(())
}
