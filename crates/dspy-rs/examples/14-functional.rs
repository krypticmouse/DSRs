/*
Example: functional DSRs (`fx`) — harnesses as plain async functions.

Instead of a struct with `Predict` fields and a `Module::forward` impl, the
pipeline below is just an async function calling `fx::predict` atoms. The
optimizable state (instructions, demos) lives OUTSIDE the function in an
`fx::Params` pytree, injected ambiently with `fx::with_params` — same function,
different candidate, nothing mutated. Struct-based modules remain fully
supported; the two styles share the trace format and the `ModuleState`
persistence format.

Runs offline with the in-process test client — no API key needed.

Run with:
```
cargo run --example 14-functional
```
*/

use anyhow::Result;
use dspy_rs::{
    LM, LMClient, PredictError, Predicted, Signature, TestCompletionModel, configure,
    fx,
};
use rig::completion::AssistantContent;
use rig::message::Text;

#[derive(Signature, Clone, Debug)]
/// Draft an answer to the question.
struct Draft {
    #[input]
    question: String,

    #[output]
    draft: String,
}

#[derive(Signature, Clone, Debug)]
/// Refine the draft into a final answer.
struct Refine {
    #[input]
    draft: String,

    #[output]
    answer: String,
}

/// The whole harness: a plain async function. No struct, no trait impl.
async fn pipeline(question: String) -> Result<Predicted<RefineOutput>, PredictError> {
    let draft = fx::predict::<Draft>("drafter", DraftInput { question }).await?;
    fx::predict::<Refine>(
        "refiner",
        RefineInput {
            draft: draft.draft.clone(),
        },
    )
    .await
}

// Or skip signatures entirely: the function IS the signature. Parameters are
// inputs, the return type is an output field named after the function, the doc
// comment is the instruction, and the fn name is the params/trace slot.
#[dspy_rs::predict]
/// Give a one-word answer.
fn quick_answer(question: String) -> String;

fn canned(fields: &[(&str, &str)]) -> AssistantContent {
    let mut text = String::new();
    for (name, value) in fields {
        text.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    text.push_str("[[ ## completed ## ]]\n");
    AssistantContent::Text(Text { text })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Offline LM: canned responses for two full pipeline runs.
    let client = TestCompletionModel::new([
        canned(&[("draft", "Paris, probably.")]),
        canned(&[("answer", "Paris is the capital of France.")]),
        canned(&[("draft", "Paris — the capital since 987 AD.")]),
        canned(&[("answer", "The capital of France is Paris (since 987 AD).")]),
        canned(&[("quick_answer", "Paris")]),
    ]);
    let lm = temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("offline"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await?
    .with_client(LMClient::Test(client))
    .await?;
    configure(lm);

    let question = "What is the capital of France?".to_string();

    // 1. Eager run with default params.
    let out = pipeline(question.clone()).await?;
    println!("default params  -> {}", out.answer);

    // 2. Same function under a capture scope with candidate params injected.
    let mut candidate = fx::Params::new();
    candidate.set_instruction("drafter", "Draft thoroughly; include one supporting fact.");
    candidate.set_instruction("refiner", "Refine into one precise sentence, keep the fact.");

    let (result, trace) =
        dspy_rs::trace::capture(|| fx::with_params(candidate.clone(), pipeline(question))).await;
    println!("candidate params -> {}", result?.answer);

    // The trace addresses spans by the SAME names the params use — the slot
    // names above are the component names below, so a sub-trace of one
    // parameter is one `for_component` call.
    for span in &trace.spans {
        println!(
            "  span {} = {:?} seq={} (links: {:?})",
            span.id.0,
            trace.component_name(span.component),
            span.seq,
            span.links
        );
    }
    let refiner_calls = trace.for_component("refiner").count();
    println!("  refiner sub-trace: {refiner_calls} invocation(s)");

    // 3. The #[predict] macro form: a bodyless fn, called like any function.
    let quick = quick_answer("Capital of France?".to_string()).await?;
    println!("\n#[predict] fn -> {}", quick.quick_answer);

    // 4. Params share the persistence format with struct-based modules.
    println!(
        "\ncandidate as ModuleState JSON:\n{}",
        candidate.to_module_state().to_json()?
    );

    Ok(())
}
