use anyhow::{Result, bail};
use dspy_rs::{
    COPRO, ChainOfThought, ChatAdapter, Example, LM, MetricOutcome, Optimizer, Predicted,
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

impl TypedMetric<SmokeSig, ChainOfThought<SmokeSig>> for SmokeMetric {
    async fn evaluate(
        &self,
        _example: &Example<SmokeSig>,
        prediction: &Predicted<WithReasoning<SmokeSigOutput>>,
    ) -> Result<MetricOutcome> {
        let answer = prediction.answer.to_ascii_lowercase();
        Ok(MetricOutcome::score(
            (answer.contains("smoke") || answer.contains("ok")) as u8 as f32,
        ))
    }
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
    let trainset = vec![Example::new(
        SmokeSigInput {
            prompt: "Return exactly smoke-ok.".to_string(),
        },
        SmokeSigOutput {
            answer: "smoke-ok".to_string(),
        },
    )];

    let optimizer = COPRO::builder().breadth(4).depth(1).build();
    optimizer
        .compile(&mut module, trainset, &SmokeMetric)
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
