use std::time::{Duration, Instant};

use anyhow::Result;
use dspy_rs::optimizer::engine::BatchEvalOutcome;
use dspy_rs::{
    Budget, CallMetadata, Candidate, Engine, EngineConfig, Eval, Module, OptimizeTarget, Predict,
    PredictError, Predicted, Signature, TypedMetric, evaluate_trainset_with_concurrency,
};

#[derive(Clone, Debug, Signature)]
struct Probe {
    #[input]
    index: u32,
    #[output]
    number: u32,
}

struct Echo {
    predictor: Predict<Probe>,
    skewed: bool,
}

dspy_rs::predictors!(Echo { predictor });

impl Module for Echo {
    type Input = ProbeInput;
    type Output = ProbeOutput;

    async fn forward(&self, input: ProbeInput) -> Result<Predicted<ProbeOutput>, PredictError> {
        let delay = if self.skewed {
            if input.index % 8 == 0 { 40 } else { 2 }
        } else {
            5
        };
        tokio::time::sleep(Duration::from_millis(delay)).await;
        Ok(Predicted::new(
            ProbeOutput { number: input.index },
            CallMetadata::default(),
        ))
    }
}

struct Metric;

impl TypedMetric<(ProbeInput, ProbeOutput), Echo> for Metric {
    async fn evaluate(
        &self,
        row: &(ProbeInput, ProbeOutput),
        output: &Predicted<ProbeOutput>,
        trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        assert!(trace.is_some());
        assert_eq!(row.1.number, output.number);
        Ok(Eval::score(output.number as f64 / 64.0))
    }
}

fn rows(n: u32) -> Vec<(ProbeInput, ProbeOutput)> {
    (0..n)
        .map(|index| (ProbeInput { index }, ProbeOutput { number: index }))
        .collect()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let workload = std::env::args().nth(1).expect("workload argument");
    let mut module = Echo {
        predictor: Predict::<Probe>::builder().instruction("echo").build(),
        skewed: workload == "skewed",
    };

    match workload.as_str() {
        "skewed" | "uniform" => {
            let data = rows(64);
            let start = Instant::now();
            let results = evaluate_trainset_with_concurrency(&module, &data, &Metric, 8).await?;
            let elapsed = start.elapsed();
            assert_eq!(results.len(), data.len());
            for (index, result) in results.iter().enumerate() {
                assert_eq!(result.score, index as f64 / 64.0);
            }
            println!("{workload},64,{}", elapsed.as_micros());
        }
        "preflight" => {
            for n in [512, 4096] {
                let data = rows(n);
                let target = OptimizeTarget::module(&mut module, &data, &Metric);
                let mut engine = Engine::new(EngineConfig {
                    budget: Budget {
                        max_metric_calls: Some(0),
                        ..Budget::default()
                    },
                    ..EngineConfig::default()
                });
                let ids: Vec<_> = (0..8)
                    .map(|id| engine.register(Candidate::with_instruction("predictor", id.to_string())))
                    .collect();
                let needed = data.len() * ids.len();
                let start = Instant::now();
                let response = engine.evaluate_many(&target, &ids, None).await?;
                let elapsed = start.elapsed();
                assert!(matches!(response, BatchEvalOutcome::BudgetExhausted { needed: count } if count == needed));
                assert_eq!(engine.spend().metric_calls, 0);
                assert_eq!(engine.spend().lm_calls, 0);
                println!("{workload},{needed},{}", elapsed.as_micros());
            }
        }
        _ => anyhow::bail!("unknown workload {workload}"),
    }
    Ok(())
}
