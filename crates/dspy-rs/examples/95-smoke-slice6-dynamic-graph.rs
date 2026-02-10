use anyhow::{Result, anyhow, bail};
use dspy_rs::__macro_support::indexmap::IndexMap;
use dspy_rs::{
    BamlValue, ChatAdapter, LM, ProgramGraph, Signature, SignatureSchema, configure, registry,
};

#[derive(Signature, Clone, Debug)]
struct SmokeSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Smoke Label: Slice 6 Dynamic Graph
    configure(
        LM::builder()
            .model("openai:gpt-5.2".to_string())
            .build()
            .await?,
        ChatAdapter,
    );

    let schema = SignatureSchema::of::<SmokeSig>();
    let mut graph = ProgramGraph::new();
    graph.add_node(
        "predict",
        registry::create("predict", schema, serde_json::json!({}))?,
    )?;
    graph.connect("input", "question", "predict", "question")?;

    {
        let predict_node = graph
            .nodes_mut()
            .get_mut("predict")
            .ok_or_else(|| anyhow!("missing `predict` node"))?;
        let mut predictors = predict_node.module.predictors_mut();
        let (_, predictor) = predictors
            .iter_mut()
            .find(|(name, _)| *name == "predictor")
            .ok_or_else(|| anyhow!("missing `predictor` leaf on dynamic node"))?;
        predictor.set_instruction("Reply with exactly: smoke-ok".to_string());
    }

    let input = BamlValue::Class(
        "SmokeSigInput".to_string(),
        IndexMap::from([(
            "question".to_string(),
            BamlValue::String("Return exactly smoke-ok.".to_string()),
        )]),
    );
    let output = graph.execute(input).await?;

    let answer_field = schema
        .output_field_by_rust("answer")
        .ok_or_else(|| anyhow!("missing `answer` field in smoke schema"))?;
    let answer = match schema.navigate_field(answer_field.path(), &output) {
        Some(BamlValue::String(answer)) => answer.clone(),
        Some(other) => {
            bail!("unexpected answer type: {other:?}");
        }
        None => {
            bail!("missing answer in graph output");
        }
    };
    println!("answer: {}", answer);

    if !answer.to_ascii_lowercase().contains("smoke-ok") {
        bail!("unexpected answer content: {}", answer);
    }

    Ok(())
}
