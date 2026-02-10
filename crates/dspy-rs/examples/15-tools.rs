/*
Example: using tools with a typed predictor.

Run with:
```
cargo run --example 15-tools
```
*/

use anyhow::Result;
use dspy_rs::{ChatAdapter, LM, Predict, Signature, configure, init_tracing};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

#[derive(Debug, Deserialize, Serialize)]
struct CalculatorArgs {
    operation: String,
    a: f64,
    b: f64,
}

#[derive(Clone)]
struct CalculatorTool;

#[derive(Debug)]
struct CalculatorError(String);

impl fmt::Display for CalculatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Calculator error: {}", self.0)
    }
}

impl Error for CalculatorError {}

impl Tool for CalculatorTool {
    const NAME: &'static str = "calculator";

    type Error = CalculatorError;
    type Args = CalculatorArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "A calculator for add/subtract/multiply/divide/power".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "power"]
                    },
                    "a": { "type": "number" },
                    "b": { "type": "number" }
                },
                "required": ["operation", "a", "b"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = match args.operation.as_str() {
            "add" => args.a + args.b,
            "subtract" => args.a - args.b,
            "multiply" => args.a * args.b,
            "divide" => {
                if args.b == 0.0 {
                    return Err(CalculatorError("division by zero".to_string()));
                }
                args.a / args.b
            }
            "power" => args.a.powf(args.b),
            other => return Err(CalculatorError(format!("unknown operation: {other}"))),
        };

        Ok(result.to_string())
    }
}

#[derive(Signature, Clone, Debug)]
struct MathQuestionSignature {
    /// Use the calculator tool for arithmetic.

    #[input]
    question: String,

    #[output]
    answer: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    let lm = LM::builder()
        .model("groq:openai/gpt-oss-120b".to_string())
        .build()
        .await?;
    configure(lm, ChatAdapter);

    let predictor = Predict::<MathQuestionSignature>::builder()
        .instruction("You must call the calculator tool for arithmetic.")
        .add_tool(CalculatorTool)
        .build();

    let predicted = predictor
        .call(MathQuestionSignatureInput {
            question: "Calculate 15 multiplied by 23 using the calculator tool.".to_string(),
        })
        .await?;

    println!("answer: {}", predicted.answer);

    let metadata = predicted.metadata();
    println!("tool calls: {}", metadata.tool_calls.len());
    for (idx, call) in metadata.tool_calls.iter().enumerate() {
        println!("  {}. {}", idx + 1, call.function.name);
    }

    println!("tool executions: {}", metadata.tool_executions.len());
    for (idx, exec) in metadata.tool_executions.iter().enumerate() {
        println!("  {}. {}", idx + 1, exec);
    }

    Ok(())
}
