//! `max_turns = 0` contradicts the IR's mandatory-and-bounded loop contract
//! and is rejected at macro expansion.
use dsrs_macros::{agent, tool};

/// Uppercase text.
#[tool]
fn shout(text: String) -> String {
    text.to_uppercase()
}

/// Research the question.
#[agent(tools(shout), max_turns = 0)]
fn research(question: String) -> String;

fn main() {}
