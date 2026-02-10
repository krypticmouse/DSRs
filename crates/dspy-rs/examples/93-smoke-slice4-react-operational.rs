use anyhow::{Result, bail};
use dspy_rs::{ChatAdapter, LM, PredictError, ReAct, Signature, configure, forward_all};
use serde_json::Value;

#[derive(Signature, Clone, Debug)]
struct SmokeSig {
    #[input]
    prompt: String,

    #[output]
    answer: String,
}

fn parse_binary_args(args: &str) -> Result<(i64, i64)> {
    let value: Value = serde_json::from_str(args)?;
    let a = value.get("a").and_then(Value::as_i64).unwrap_or(0);
    let b = value.get("b").and_then(Value::as_i64).unwrap_or(0);
    Ok((a, b))
}

fn extract_first_integer(text: &str) -> Option<i64> {
    let mut token = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || (token.is_empty() && ch == '-') {
            token.push(ch);
            continue;
        }
        if !token.is_empty() {
            break;
        }
    }
    token.parse::<i64>().ok()
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
        .max_steps(6)
        .tool("add", "Add two integers. Args JSON: {\"a\":int,\"b\":int}", |args| async move {
            match parse_binary_args(&args) {
                Ok((a, b)) => (a + b).to_string(),
                Err(err) => format!("calculator_error: {err}"),
            }
        })
        .tool(
            "multiply",
            "Multiply two integers. Args JSON: {\"a\":int,\"b\":int}",
            |args| async move {
                match parse_binary_args(&args) {
                    Ok((a, b)) => (a * b).to_string(),
                    Err(err) => format!("calculator_error: {err}"),
                }
            },
        )
        .action_instruction(
            "You are a strict ReAct planner. Choose exactly one tool each step, and use tool names exactly as declared.",
        )
        .extract_instruction(
            "Read trajectory and return only the final integer in output.answer.",
        )
        .build();

    let input = SmokeSigInput {
        prompt: "Use tools to compute ((17 + 5) * 3) + 4. You MUST call add, then multiply, then add again, then finish. Return only the final integer string."
            .to_string(),
    };

    let mut outcomes = forward_all(&module, vec![input], 1).await.into_iter();
    let outcome = outcomes.next().expect("expected one batch outcome");
    let predicted = outcome.map_err(|err| {
        eprintln!("smoke call failed: {err}");
        if let PredictError::Parse { raw_response, .. } = &err {
            eprintln!("raw_response: {:?}", raw_response);
        }
        anyhow::anyhow!("slice4 smoke failed")
    })?;
    let (output, metadata) = predicted.into_parts();

    println!("tool_calls: {}", metadata.tool_calls.len());
    println!("tool_executions: {}", metadata.tool_executions.len());
    println!("trajectory:");
    for entry in &metadata.tool_executions {
        if entry.trim().is_empty() {
            continue;
        }
        println!("{entry}");
        println!("---");
    }
    println!("answer: {}", output.answer);

    let called_tools: Vec<String> = metadata
        .tool_calls
        .iter()
        .map(|call| call.function.name.to_ascii_lowercase())
        .collect();
    let add_calls = called_tools
        .iter()
        .filter(|name| name.as_str() == "add")
        .count();
    let multiply_calls = called_tools
        .iter()
        .filter(|name| name.as_str() == "multiply")
        .count();

    if add_calls < 2 || multiply_calls < 1 {
        bail!(
            "expected multi-tool trajectory with add x2 and multiply x1, got {:?}",
            called_tools
        );
    }

    let answer_value = extract_first_integer(&output.answer)
        .ok_or_else(|| anyhow::anyhow!("answer did not contain integer: {}", output.answer))?;
    if answer_value != 70 {
        bail!(
            "unexpected calculator result: expected 70, got {} (raw answer: {})",
            answer_value,
            output.answer
        );
    }

    Ok(())
}
