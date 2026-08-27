//! RATCHET arm of the optimizer benchmark: optimize a HotpotQA instruction
//! with the RATCHET optimizer under the exact conditions of the GEPA arms
//! (benchmarks/ratchet-hotpot/run_python_gepa.py and example 29 with
//! HOTPOT_DIR): same student, same bare rendering, same splits, same budget,
//! same teacher.
//!
//! Env: HOTPOT_DIR (default benchmarks/ratchet-hotpot), HOTPOT_SEED,
//! HOTPOT_SEED_PROMPT, HOTPOT_STUDENT, HOTPOT_ITERS, HOTPOT_MINIBATCH,
//! HOTPOT_BUDGET, HOTPOT_OUT_PREFIX (default ratchet_winner).
//!
//! Run from the repo root:
//!     set -a; source .env; set +a
//!     HOTPOT_SEED_PROMPT="Answer the question." cargo run --release --example 30-ratchet-hotpot

use anyhow::{Context, Result};
use dspy_rs::ir::RenderMode;
use dspy_rs::{
    Eval, LM, Module, Predict, PredictError, Predicted, Ratchet, Signature, TypedMetric,
    configure, init_tracing,
};
use serde::{Deserialize, Serialize};

fn student_model() -> String {
    match std::env::var("HOTPOT_STUDENT") {
        Ok(m) => format!("openrouter:{m}"),
        Err(_) => "openrouter:meta-llama/llama-3.2-3b-instruct".to_string(),
    }
}
const TEACHER_MODEL: &str = "openrouter:openai/gpt-5.6-sol";

/// Answer the question using the provided context paragraphs. Give only the
/// exact answer — a few words, no explanation.
#[derive(Signature, Clone, Debug)]
struct HotpotQA {
    #[input]
    context: String,

    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct HotpotRow {
    context: String,
    question: String,
    answer: String,
}

/// Retrying student, mirroring the other arms' 3-attempt `call_student`.
struct Student {
    answerer: Predict<HotpotQA>,
}

dspy_rs::predictors!(Student { answerer });

impl Module for Student {
    type Input = HotpotQAInput;
    type Output = HotpotQAOutput;

    async fn forward(&self, input: HotpotQAInput) -> Result<Predicted<HotpotQAOutput>, PredictError> {
        let mut last = None;
        for attempt in 0..5u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(3u64 << (attempt - 1))).await;
            }
            match self.answerer.call(input.clone()).await {
                Ok(out) => return Ok(out),
                Err(e) => last = Some(e),
            }
        }
        eprintln!("rollout failed after retries, scoring 0: {:?}", last);
        Ok(Predicted::new(
            HotpotQAOutput {
                answer: String::new(),
            },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

fn normalize(text: &str) -> String {
    let lowered: String = text
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .collect();
    lowered
        .split_whitespace()
        .filter(|t| !matches!(*t, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_f1(prediction: &str, gold: &str) -> f64 {
    let p = normalize(prediction);
    let g = normalize(gold);
    let p_tokens: Vec<&str> = p.split_whitespace().collect();
    let g_tokens: Vec<&str> = g.split_whitespace().collect();
    if p_tokens.is_empty() || g_tokens.is_empty() {
        return if p == g { 1.0 } else { 0.0 };
    }
    let mut gold_counts = std::collections::HashMap::new();
    for t in &g_tokens {
        *gold_counts.entry(*t).or_insert(0usize) += 1;
    }
    let mut overlap = 0usize;
    for t in &p_tokens {
        if let Some(c) = gold_counts.get_mut(*t)
            && *c > 0
        {
            *c -= 1;
            overlap += 1;
        }
    }
    if overlap == 0 {
        return 0.0;
    }
    let precision = overlap as f64 / p_tokens.len() as f64;
    let recall = overlap as f64 / g_tokens.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

struct F1Metric;

impl TypedMetric<HotpotRow, Student> for F1Metric {
    async fn evaluate(
        &self,
        example: &HotpotRow,
        prediction: &dspy_rs::Predicted<HotpotQAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let answer = prediction.answer.trim();
        let f1 = token_f1(answer, &example.answer);
        Ok(Eval::with_feedback(
            f1,
            format!(
                "Predicted: {answer:?}. Gold answer: {:?}. Token F1 = {f1:.2}. \
                 The score is exact-span token overlap after lowercasing and \
                 dropping punctuation/articles — extra words beyond the answer \
                 span reduce it.",
                example.answer
            ),
        ))
    }
}

fn load_rows(path: &str) -> Result<Vec<HotpotRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run from the repo root"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<HotpotRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    configure(LM::builder().model(TEACHER_MODEL.to_string()).build().await?);

    let dir =
        std::env::var("HOTPOT_DIR").unwrap_or_else(|_| "benchmarks/ratchet-hotpot".to_string());
    let train = load_rows(&format!("{dir}/train.jsonl"))?;
    let val_file =
        std::env::var("HOTPOT_VAL_FILE").unwrap_or_else(|_| "val.jsonl".to_string());
    let val = load_rows(&format!("{dir}/{val_file}"))?;
    println!("{dir}: {} train / {} val", train.len(), val.len());

    let student_lm = LM::builder()
        .model(student_model())
        .temperature(0.2)
        .max_tokens(256)
        .build()
        .await?;
    let answerer = Predict::<HotpotQA>::builder()
        .named("hotpot_answerer")
        .render_mode(RenderMode::Bare)
        .lm(student_lm);
    let answerer = match std::env::var("HOTPOT_SEED_PROMPT") {
        Ok(prompt) => answerer.instruction(prompt).build(),
        Err(_) => answerer.build(),
    };
    let mut student = Student { answerer };

    let reflection_lm = LM::builder()
        .model(TEACHER_MODEL.to_string())
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;

    let seed: u64 = env_usize("HOTPOT_SEED", 0) as u64;
    let iters = env_usize("HOTPOT_ITERS", 60);
    let minibatch = env_usize("HOTPOT_MINIBATCH", 5);
    let budget = env_usize("HOTPOT_BUDGET", 600);
    println!("ratchet seed: {seed}, iterations: {iters}, minibatch: {minibatch}, budget: {budget}");

    let champion = env_usize("HOTPOT_CHAMPION", 40);
    let ratchet = Ratchet::builder()
        .num_iterations(iters)
        .minibatch_size(minibatch)
        .race_block(8)
        .champion_fresh(champion)
        .max_rollouts(budget)
        .prompt_model(reflection_lm)
        .seed(seed)
        .eval_concurrency(8)
        .track_stats(true)
        .build();

    let result = ratchet
        .compile_module_with_valset(&mut student, &train, Some(&val), &F1Metric)
        .await?;

    println!("\n== RATCHET result ==");
    println!(
        "rollouts: {}  lm_calls: {}  promoted: {}",
        result.total_rollouts,
        result.total_lm_calls,
        result.all_candidates.len()
    );
    println!(
        "gate_rejected: {}  race_killed: {}  skipped: {}  leak_retries: {}  leak_skipped: {}",
        result.gate_rejected,
        result.race_killed,
        result.skipped,
        result.leak_retries,
        result.leak_skipped
    );
    println!("operator stats (name, attempts, wins): {:?}", result.operator_stats);
    println!("fresh_selection_cols: {}", result.fresh_selection_cols);
    println!("evolution: {:?}", result.evolution_history);
    println!("\n== trial ledger ==");
    for trial in &result.trials {
        println!(
            "[gen {:>2} | {:>8} | parent {}] {:?} :: {}",
            trial.generation,
            trial.operator.name(),
            trial.parent_id,
            trial.outcome,
            trial.hypothesis
        );
    }
    println!(
        "\nchosen candidate (gen {}, val avg {:.3}):\n---\n{}\n---",
        result.best_candidate.generation,
        result.best_candidate.average_score(),
        result.best_candidate.instruction
    );
    if let Some(argmax) = &result.best_by_mean {
        println!(
            "argmax-by-val-mean differs (gen {}, val avg {:.3}):\n---\n{}\n---",
            argmax.generation,
            argmax.average_score(),
            argmax.instruction
        );
    }

    let prefix =
        std::env::var("HOTPOT_OUT_PREFIX").unwrap_or_else(|_| "ratchet_winner".to_string());
    let out = format!("{dir}/{prefix}_s{seed}.txt");
    std::fs::write(&out, &result.best_candidate.instruction)?;
    let argmax_instruction = result
        .best_by_mean
        .as_ref()
        .map(|c| c.instruction.as_str())
        .unwrap_or(&result.best_candidate.instruction);
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.txt");
    std::fs::write(&out_argmax, argmax_instruction)?;
    println!("\nchosen instruction written to {out}\nargmax instruction written to {out_argmax}");
    Ok(())
}
