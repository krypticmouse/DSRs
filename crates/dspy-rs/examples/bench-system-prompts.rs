//! Steady-state prompt construction; no HTTP requests, inference or output parsing.
use dspy_rs::ir::{self, Interpreter, ProgramBuilder, RuntimeEnv, SignatureDef};
use dspy_rs::{ChatAdapter, FieldType, LM, LMConfig};
use serde_json::json;
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iterations: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(2000);
    anyhow::ensure!(iterations > 0, "iterations must be positive");
    let dump = args
        .iter()
        .position(|s| s == "--dump")
        .map(|i| {
            args.get(i + 1)
                .ok_or_else(|| anyhow::anyhow!("--dump requires a directory"))
        })
        .transpose()?;
    if let Some(path) = dump {
        std::fs::create_dir_all(path)?;
    }
    let lm = std::sync::Arc::new(
        LM::builder()
            .base_url("http://localhost:1/v1".into())
            .model("not-called".into())
            .cache(false)
            .build()
            .await?,
    );
    let mut records = Vec::new();
    for width in [1, 8, 16] {
        let mut signature =
            SignatureDef::build("SystemBench").instruction("Answer using the supplied context.");
        let mut input = dspy_rs::JsonMap::new();
        for i in 0..width {
            let name = format!("input_{i}");
            signature = signature.input(&name, FieldType::String);
            input.insert(name, json!("fixed context"));
        }
        let signature = signature.output("answer", FieldType::String).finish()?;
        let oracle = ChatAdapter.build_system_def(
            &signature,
            &Default::default(),
            Some(&signature.instruction),
        );
        let mut builder = ProgramBuilder::new("system-bench");
        let model = builder.model("default", LMConfig::default());
        let sig = builder.sig(signature.clone());
        let mut leaf = ir::predict("qa", sig).model(model);
        for field in &signature.inputs {
            leaf = leaf.bind(&field.name, ir::input(&field.name));
        }
        let program = builder.main(sig, ir::seq([leaf]).out("answer", ir::out("qa", "answer")))?;
        let interpreter =
            Interpreter::load(program, RuntimeEnv::new().bind_model("default", lm.clone())).await?;
        let chat = interpreter.conversation_opening(&input, None)?;
        assert_eq!(chat.messages[0].content(), oracle);
        let expected = serde_json::to_string(&chat.to_json())?;
        for _ in 0..100 {
            std::hint::black_box(interpreter.conversation_opening(&input, None)?);
        }
        let started = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(interpreter.conversation_opening(&input, None)?);
        }
        let elapsed = started.elapsed();
        if let Some(path) = dump {
            std::fs::write(
                std::path::Path::new(path).join(format!("system-{width}.json")),
                &expected,
            )?;
        }
        records.push(json!({"case":format!("system-{width}"),"iterations":iterations,"warmup":100,
            "ns_per_op":elapsed.as_nanos() as f64 / iterations as f64,"output_bytes":expected.len()}));
    }
    println!("{}", serde_json::to_string(&records)?);
    Ok(())
}
