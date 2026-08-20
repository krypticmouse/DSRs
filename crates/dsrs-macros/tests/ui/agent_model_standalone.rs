//! `#[agent(model = "…")]` cannot be honored on the standalone call path —
//! model refs bind only inside a `#[module]` program — so no standalone fn is
//! generated and calling one is a compile error, not a silent fallback to the
//! globally configured LM.
use dsrs_macros::{agent, tool};

/// Uppercase text.
#[tool]
fn shout(text: String) -> String {
    text.to_uppercase()
}

/// Research the question with the fast model.
#[agent(model = "@fast", tools(shout))]
fn research(question: String) -> String;

async fn call_it() {
    let _ = research("hi".to_string()).await;
}

fn main() {}
