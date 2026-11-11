//! Run a measured DAG compiler with either a delayed mock or --live BASE_URL MODEL.
use anyhow::Result;
use dspy_rs::ir::compiler::{Compiler, CompilerMeasurement};
use dspy_rs::ir::{
    self, FieldType as T, Interpreter, Node, Program, ProgramBuilder, RuntimeEnv, SignatureDef,
};
use dspy_rs::{CompletionProvider, LM, LMClient, LMConfig, TestCompletionModel};
use rig::completion::{AssistantContent, CompletionError, CompletionRequest, CompletionResponse};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

struct CountedProvider {
    inner: LMClient,
    calls: Arc<AtomicU64>,
    delay: bool,
}
impl CompletionProvider for CountedProvider {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.delay {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        self.inner.completion(request).await
    }
}

fn program() -> Program {
    let mut b = ProgramBuilder::new("country");
    b.model("m", LMConfig::default());
    let main = b.sig(
        SignatureDef::build("Country")
            .input("country", T::String)
            .output("capital", T::String)
            .output("continent", T::String)
            .finish()
            .unwrap(),
    );
    let capital = b.sig(
        SignatureDef::build("Capital")
            .instruction("Give only the name of the country's capital city.")
            .input("country", T::String)
            .output("capital", T::String)
            .finish()
            .unwrap(),
    );
    let continent = b.sig(
        SignatureDef::build("Continent")
            .instruction("Give only the name of the country's continent.")
            .input("country", T::String)
            .output("continent", T::String)
            .finish()
            .unwrap(),
    );
    b.main(
        main,
        ir::seq([
            ir::predict("capital", capital).bind("country", ir::input("country")),
            ir::predict("continent", continent).bind("country", ir::input("country")),
        ])
        .out("capital", ir::out("capital", "capital"))
        .out("continent", ir::out("continent", "continent")),
    )
    .unwrap()
}

async fn evaluate(
    program: Arc<Program>,
    live: Option<(String, String)>,
) -> Result<CompilerMeasurement> {
    let cases = [("France", "Paris"), ("Germany", "Berlin")];
    let calls = Arc::new(AtomicU64::new(0));
    let fused = program
        .nodes
        .values()
        .filter(|n| matches!(n, Node::Predict(_)))
        .count()
        == 1;
    let mut replies = Vec::new();
    // One warmup case followed by two repeated passes over the same two cases.
    for capital in ["Paris", "Paris", "Berlin", "Paris", "Berlin"] {
        if fused {
            replies.push(AssistantContent::text(format!("[[ ## capital ## ]]\n{capital}\n[[ ## continent ## ]]\nEurope\n[[ ## completed ## ]]")));
        } else {
            replies.push(AssistantContent::text(format!(
                "[[ ## capital ## ]]\n{capital}\n[[ ## completed ## ]]"
            )));
            replies.push(AssistantContent::text(
                "[[ ## continent ## ]]\nEurope\n[[ ## completed ## ]]",
            ));
        }
    }
    let (url, model) = live
        .clone()
        .unwrap_or_else(|| ("http://127.0.0.1:1/v1".into(), "mock".into()));
    let inner = if live.is_some() {
        LMClient::from_local(&url, &model)?
    } else {
        LMClient::Test(TestCompletionModel::new(replies))
    };
    let lm = LM::builder()
        .base_url(url)
        .model(model)
        .cache(false)
        .temperature(0.0)
        .max_tokens(128)
        .max_retries(0)
        .build()
        .await?
        .with_client(LMClient::custom(CountedProvider {
            inner,
            calls: calls.clone(),
            delay: live.is_none(),
        }))
        .await?;
    lm.history.set_capacity(0);
    let interpreter =
        Interpreter::load_shared(program, RuntimeEnv::new().bind_model("m", Arc::new(lm))).await?;
    let input = |country: &str| [("country".into(), country.into())].into_iter().collect();
    interpreter
        .run(input("France"), None, ir::Budget::unlimited())
        .await?;
    calls.store(0, Ordering::Relaxed);
    let mut result = CompilerMeasurement {
        quality: Vec::new(),
        latency_us: Vec::new(),
        lm_calls: 0,
        feedback: Vec::new(),
    };
    for (country, capital) in cases.into_iter().cycle().take(4) {
        let start = Instant::now();
        let output = interpreter
            .run(input(country), None, ir::Budget::unlimited())
            .await?;
        result.latency_us.push(start.elapsed().as_micros() as u64);
        result.quality.push(
            if output["capital"] == capital && output["continent"] == "Europe" {
                1.0
            } else {
                0.0
            },
        );
        result.feedback.push(
            serde_json::json!({"country":country,"expected_capital":capital,"output":output})
                .to_string(),
        );
    }
    result.lm_calls = calls.load(Ordering::Relaxed);
    Ok(result)
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    dspy_rs::set_history_capacity(0);
    let args: Vec<_> = std::env::args().skip(1).collect();
    let live = if args.first().is_some_and(|s| s == "--live") {
        anyhow::ensure!(args.len() == 3, "usage: --live BASE_URL MODEL");
        Some((args[1].clone(), args[2].clone()))
    } else {
        anyhow::ensure!(args.is_empty(), "usage: [--live BASE_URL MODEL]");
        None
    };
    let compiler = Compiler {
        max_rounds: 1,
        max_evaluations: 2,
        ..Compiler::default()
    };
    let report = compiler
        .compile(&program(), |program| evaluate(program, live.clone()))
        .await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
