use std::io::{self, Write as _};
use std::time::Instant;

use anyhow::Result;
use dspy_rs::optimizer::{BatchEvalOutcome, RolloutCache};
use dspy_rs::{
    Budget, CallMetadata, Candidate, DataLoadError, DataLoader, Engine, EngineConfig, Eval, Module,
    OptimizeTarget, Predict, PredictError, Predicted, Signature, Trace, TypedLoadOptions,
    TypedMetric,
};

#[derive(Signature, Clone, Debug)]
struct Probe {
    #[input]
    index: u32,
    #[output]
    number: u32,
}

struct Echo {
    predictor: Predict<Probe>,
}
dspy_rs::predictors!(Echo { predictor });

impl Module for Echo {
    type Input = ProbeInput;
    type Output = ProbeOutput;

    async fn forward(&self, input: ProbeInput) -> Result<Predicted<ProbeOutput>, PredictError> {
        Ok(Predicted::new(
            ProbeOutput {
                number: input.index,
            },
            CallMetadata::default(),
        ))
    }
}

struct Metric;

fn score(index: usize) -> f64 {
    (index % 16) as f64 / 16.0
}

impl TypedMetric<(ProbeInput, ProbeOutput), Echo> for Metric {
    async fn evaluate(
        &self,
        row: &(ProbeInput, ProbeOutput),
        out: &Predicted<ProbeOutput>,
        trace: Option<&Trace>,
    ) -> Result<Eval> {
        assert!(trace.is_some());
        assert_eq!(out.number, row.1.number);
        Ok(Eval::score(score(row.0.index as usize)))
    }
}

fn rows(size: usize) -> Vec<(ProbeInput, ProbeOutput)> {
    (0..size as u32)
        .map(|index| (ProbeInput { index }, ProbeOutput { number: index }))
        .collect()
}

fn echo() -> Echo {
    Echo {
        predictor: Predict::new(),
    }
}

fn candidates(engine: &mut Engine, count: usize) -> Vec<usize> {
    (0..count)
        .map(|i| {
            engine.register(Candidate::with_instruction(
                "predictor",
                format!("candidate {i}"),
            ))
        })
        .collect()
}

async fn cache() -> Result<()> {
    const BATCHES: usize = 512;
    let mut module = echo();
    let data = rows(2048);
    let target = OptimizeTarget::module(&mut module, &data, &Metric);
    let mut engine = Engine::new(EngineConfig::default());
    let candidates = candidates(&mut engine, 32);
    for &candidate in &candidates {
        let full = engine
            .evaluate(&target, candidate, None)
            .await?
            .completed()
            .unwrap();
        assert_eq!(full.rollouts.len(), data.len());
        assert!(
            full.rollouts
                .iter()
                .all(|r| r.trace.as_ref().unwrap().outcome.is_some())
        );
    }
    assert_eq!(engine.cache().len(), 32 * 2048);
    let subset: Vec<_> = (0..16).map(|i| i * 127).rev().collect();
    let started = Instant::now();
    for batch in 0..BATCHES {
        let candidate = candidates[batch % candidates.len()];
        let result = engine
            .evaluate(&target, candidate, Some(&subset))
            .await?
            .completed()
            .unwrap();
        assert_eq!(result.candidate, candidate);
        assert_eq!(result.rollouts.len(), subset.len());
        for (rollout, &example) in result.rollouts.iter().zip(&subset) {
            assert_eq!(
                (rollout.example, rollout.eval.score),
                (example, score(example))
            );
            assert!(rollout.trace.is_none());
        }
    }
    let elapsed = started.elapsed();
    assert_eq!(engine.spend().cache_hits, BATCHES * subset.len());
    assert_eq!(engine.spend().metric_calls, 32 * 2048);
    println!("cache,{},{}", BATCHES * subset.len(), elapsed.as_nanos());
    Ok(())
}

async fn preflight() -> Result<()> {
    let mut module = echo();
    let data = rows(4096);
    let target = OptimizeTarget::module(&mut module, &data, &Metric);
    let mut engine = Engine::new(EngineConfig {
        budget: Budget {
            max_metric_calls: Some(0),
            ..Budget::default()
        },
        ..EngineConfig::default()
    });
    let mut candidates = candidates(&mut engine, 8);
    candidates.push(candidates[0]);
    let mut subset: Vec<_> = (0..data.len()).rev().collect();
    subset.push(0);
    let started = Instant::now();
    let result = engine
        .evaluate_many(&target, &candidates, Some(&subset))
        .await?;
    let elapsed = started.elapsed();
    assert!(matches!(
        result,
        BatchEvalOutcome::BudgetExhausted { needed: 32768 }
    ));
    assert!(engine.cache().is_empty());
    assert_eq!(engine.spend().metric_calls, 0);
    println!("preflight,32768,{}", elapsed.as_nanos());
    Ok(())
}

#[derive(Debug, facet::Facet, serde::Deserialize)]
struct DatasetRow {
    id: u32,
    text: String,
}

fn load_jsonl(path: &str) -> Result<()> {
    let started = Instant::now();
    let records = DataLoader::load_json::<DatasetRow>(path, true, TypedLoadOptions::default())?;
    let elapsed = started.elapsed();
    assert_eq!(records.len(), 8192);
    for (i, row) in records.iter().enumerate() {
        assert_eq!(row.id as usize, i);
        assert_eq!(row.text.len(), 16384);
        assert!(row.text.starts_with('x') && row.text.ends_with('x'));
    }
    println!("jsonl,{},{}", records.len(), elapsed.as_nanos());
    io::stdout().flush()?;
    let mut ack = String::new();
    io::stdin().read_line(&mut ack)?;
    assert_eq!(ack.trim(), "done");
    Ok(())
}

fn contracts() -> Result<()> {
    let mut cache = RolloutCache::default();
    let keys = (u64::MAX, 0, 0x0123456789abcdef, 0x9);
    cache.insert(keys.0, keys.1, keys.2, keys.3, Eval::score(0.5));
    let wire = serde_json::to_value(&cache)?;
    let key = "ffffffffffffffff:0000000000000000:0123456789abcdef:0000000000000009";
    assert_eq!(wire["entries"][key]["score"], 0.5);
    let restored: RolloutCache = serde_json::from_value(wire)?;
    assert_eq!(
        restored.get(keys.0, keys.1, keys.2, keys.3).unwrap().score,
        0.5
    );
    assert!(restored.get(keys.0, keys.1, keys.2, keys.3 + 1).is_none());

    let tmp = tempfile::tempdir()?;
    let file = |name: &str, bytes: &[u8]| -> Result<String> {
        let path = tmp.path().join(name);
        std::fs::write(&path, bytes)?;
        Ok(path.into_os_string().into_string().unwrap())
    };
    let mixed = file("mixed.jsonl", b"broken JSON\n\xff\n")?;
    let error =
        DataLoader::load_json::<DatasetRow>(&mixed, true, TypedLoadOptions::default()).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<DataLoadError>(),
        Some(DataLoadError::Io(_))
    ));
    let blank = file(
        "blank.jsonl",
        b"\r\n{\"id\":1,\"text\":\"first\"}\r\n  \n{\"id\":2,\"text\":\"last\"}",
    )?;
    let rows = DataLoader::load_json::<DatasetRow>(&blank, true, TypedLoadOptions::default())?;
    assert_eq!((rows[0].id, rows[0].text.as_str()), (1, "first"));
    assert_eq!((rows[1].id, rows[1].text.as_str()), (2, "last"));
    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("cache") => cache().await,
        Some("preflight") => preflight().await,
        Some("jsonl") => load_jsonl(args.next().as_deref().expect("JSONL fixture")),
        Some("contracts") => contracts(),
        _ => anyhow::bail!("unknown mode"),
    }
}
