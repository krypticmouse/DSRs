//! Dogfood: optimize the DSRs docs-chat assistant with DSRs' own GEPA.
//!
//! Student: Mixedbread toast-1 grounded in the dsrs-docs + dsrs-code stores
//! (store_search / store_grep), called over raw HTTP. The prompt is rendered
//! through `Predict::build_chat`, which resolves the ambient candidate
//! overlay — this is the only public path on which a hand-rolled HTTP module
//! observes the instruction GEPA is currently testing.
//! Judge + reflection: GPT-5.6-sol via OpenRouter.
//!
//! Evalset: identical to the Python-gepa baseline run (see
//! docs-chat-worker/gepa-optimization.md) — same 33/11 train/val split
//! (docs-chat-worker/gepa/train.jsonl + val.jsonl) and the same 6-question
//! holdout the optimizer never sees. Only the iteration count is reduced.
//!
//! Run from the repo root:
//!     set -a; source .env; set +a
//!     cargo run --release --example 27-gepa-docs-chat
//!
//! Env: MXBAI_API_KEY, OPENROUTER_API_KEY
//! Optional overrides: DOCSQA_TRAIN / DOCSQA_VAL / DOCSQA_HOLDOUT (data paths),
//! DOCSQA_ITERS / DOCSQA_BUDGET (GEPA iterations / max rollouts),
//! DOCSQA_TEACHER (reflection model), DOCSQA_OUT (best-instruction path).

use anyhow::{Context, Result};
use dspy_rs::{
    CallMetadata, Eval, GEPA, LM, Module, Predict, PredictError, Predicted,
    Role, Signature, TypedMetric, average_score, configure, evaluate_trainset, init_tracing,
};
use serde::{Deserialize, Serialize};

const MXBAI_URL: &str = "https://api.mixedbread.com/v1/chat/completions";
const STORES: [&str; 2] = ["dsrs-docs", "dsrs-code"];
const JUDGE_MODEL: &str = "openrouter:openai/gpt-5.6-sol";

/// You are the DSRs docs assistant. DSRs (dspy-rs) is a Rust framework for
/// building and optimizing LM pipelines. Answer only from the attached
/// stores: dsrs-docs (published documentation) and dsrs-code (the Rust
/// sources under crates/). Search before answering; use grep for exact
/// symbol or signature lookups. When an answer touches implementation,
/// quote the real code with its file path. If the stores don't cover the
/// question, say so instead of guessing. Answer in concise markdown.
#[derive(Signature, Clone, Debug)]
struct DocsQA {
    /// The user's question about DSRs.
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// Grade an answer from a documentation assistant for DSRs (dspy-rs, a Rust
/// framework for LM pipelines). Grade strictly: "covered" means the fact is
/// stated correctly, not merely alluded to; "fluff" means spans carrying no
/// key point (preamble, restating the question, hedging, generic filler).
#[derive(Signature, Clone, Debug)]
struct JudgeAnswer {
    #[input]
    question: String,

    #[input]
    answer: String,

    /// Key points a correct answer must contain, one per line, numbered.
    #[input]
    key_points: String,

    /// File paths the answer should cite; "(none required)" if empty.
    #[input]
    required_citations: String,

    /// One boolean per key point, in the listed order: stated correctly?
    #[output]
    covered: Vec<bool>,

    /// Key points absent or wrong, restated briefly.
    #[output]
    missed: Vec<String>,

    /// Verbatim spans from the answer that carry no key point.
    #[output]
    fluff: Vec<String>,

    /// Paths from the required list that the answer mentions.
    #[output]
    citations_present: Vec<String>,

    /// Claims that contradict the key points, if any.
    #[output]
    wrong_claims: Vec<String>,

    /// 2-4 sentences of direct advice: what to add, cut, or restructure.
    #[output]
    advice: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct DocsRow {
    question: String,
    key_points: Vec<String>,
    #[serde(default)]
    citations: Vec<String>,
}

// ---------------------------------------------------------------------------
// Student: toast-1 over raw HTTP, prompt rendered by the answerer leaf
// ---------------------------------------------------------------------------

struct DocsAgent {
    answerer: Predict<DocsQA>,
    http: reqwest::Client,
    api_key: String,
}

dspy_rs::predictors!(DocsAgent { answerer });

impl DocsAgent {
    fn new(api_key: String) -> Self {
        Self {
            answerer: Predict::<DocsQA>::new(),
            http: reqwest::Client::new(),
            api_key,
        }
    }
}

fn http_err(msg: impl Into<String>) -> PredictError {
    PredictError::Params {
        name: "docs-agent-http".into(),
        source: Box::<dyn std::error::Error + Send + Sync>::from(msg.into()),
    }
}

fn role_str(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

/// toast-1 follows the ChatAdapter marker protocol; strip it back to text.
fn extract_answer(content: &str) -> String {
    let s = content
        .split("[[ ## answer ## ]]")
        .nth(1)
        .unwrap_or(content);
    let s = s.split("[[ ## completed ## ]]").next().unwrap_or(s);
    s.trim().to_string()
}

impl Module for DocsAgent {
    type Input = DocsQAInput;
    type Output = DocsQAOutput;

    async fn forward(&self, input: DocsQAInput) -> Result<Predicted<DocsQAOutput>, PredictError> {
        // build_chat resolves instance state ⊕ the ambient GEPA candidate,
        // so the rendered system prompt carries the instruction under test.
        let chat = self.answerer.build_chat(&input).await?;
        let messages: Vec<serde_json::Value> = chat
            .messages
            .iter()
            .map(|m| serde_json::json!({"role": role_str(&m.role), "content": m.text_content()}))
            .collect();
        let body = serde_json::json!({
            "model": "toast-1",
            "stream": false,
            "max_tokens": 2048,
            "messages": messages,
            "tools": [
                {"type": "store_search", "store_identifiers": STORES},
                {"type": "store_grep", "store_identifiers": STORES},
            ],
        });

        let mut last_err = String::new();
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(5 * attempt as u64)).await;
            }
            let resp = match self
                .http
                .post(MXBAI_URL)
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = e.to_string();
                    continue;
                }
            };
            let status = resp.status();
            let value: serde_json::Value = match resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    last_err = e.to_string();
                    continue;
                }
            };
            if !status.is_success() {
                last_err = format!("mixedbread {status}: {:.300}", value);
                continue;
            }
            let content = value["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or_default();
            let answer = extract_answer(content);
            return Ok(Predicted::new(
                DocsQAOutput { answer },
                CallMetadata::default(),
            ));
        }
        Err(http_err(format!("toast-1 call failed after retries: {last_err}")))
    }
}

// ---------------------------------------------------------------------------
// Metric: coverage-dominant density score + textual feedback for reflection
// ---------------------------------------------------------------------------

struct DensityMetric {
    judge: Predict<JudgeAnswer>,
}

impl TypedMetric<DocsRow, DocsAgent> for DensityMetric {
    async fn evaluate(
        &self,
        example: &DocsRow,
        prediction: &Predicted<DocsQAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let answer = prediction.answer.trim().to_string();
        if answer.is_empty() {
            return Ok(Eval::with_feedback(0.0, "Empty answer."));
        }
        let key_points = example
            .key_points
            .iter()
            .enumerate()
            .map(|(i, k)| format!("{}. {k}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let required = if example.citations.is_empty() {
            "(none required)".to_string()
        } else {
            example.citations.join("\n")
        };
        let judged = self
            .judge
            .call(JudgeAnswerInput {
                question: example.question.clone(),
                answer: answer.clone(),
                key_points,
                required_citations: required,
            })
            .await;
        let j = match judged {
            Ok(j) => j,
            // a judge failure must not abort the whole compile; score it
            // zero with the error visible to reflection
            Err(e) => return Ok(Eval::with_feedback(0.0, format!("Judge error: {e}"))),
        };

        let n = example.key_points.len().max(1) as f64;
        let coverage = j.covered.iter().filter(|c| **c).count() as f64 / n;
        let cite = if example.citations.is_empty() {
            1.0
        } else {
            (j.citations_present.len() as f64 / example.citations.len() as f64).min(1.0)
        };
        let fluff_chars: usize = j.fluff.iter().map(|s| s.len()).sum();
        let noise = (fluff_chars as f64 / answer.len().max(1) as f64).min(1.0);
        let wrong_penalty = (0.15 * j.wrong_claims.len() as f64).min(0.30);
        let score =
            (0.70 * coverage + 0.10 * cite + 0.20 * (1.0 - noise) - wrong_penalty).clamp(0.0, 1.0);

        let mut fb = format!(
            "Score {score:.2} (coverage {:.0}%, noise {:.0}% of answer).",
            coverage * 100.0,
            noise * 100.0
        );
        if !j.missed.is_empty() {
            fb.push_str("\nMissing or wrong:");
            for m in &j.missed {
                fb.push_str(&format!("\n  - {m}"));
            }
        }
        if !j.wrong_claims.is_empty() {
            fb.push_str("\nContradicts the docs:");
            for w in &j.wrong_claims {
                fb.push_str(&format!("\n  - {w}"));
            }
        }
        if !j.fluff.is_empty() {
            fb.push_str("\nFluff to cut:");
            for f in j.fluff.iter().take(5) {
                fb.push_str(&format!("\n  - \"{f}\""));
            }
        }
        if !j.advice.trim().is_empty() {
            fb.push_str(&format!("\nAdvice: {}", j.advice));
        }
        Ok(Eval::with_feedback(score, fb))
    }
}

// ---------------------------------------------------------------------------

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_or_usize(key: &str, default: usize) -> Result<usize> {
    match std::env::var(key) {
        Ok(v) => v.parse().with_context(|| format!("{key}={v} is not a number")),
        Err(_) => Ok(default),
    }
}

fn load_rows(path: &str) -> Result<Vec<DocsRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run from the repo root"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<DocsRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    let mxbai_key = std::env::var("MXBAI_API_KEY").context("MXBAI_API_KEY not set")?;
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    // Global LM is only used to render prompts (build_chat) — never called.
    configure(
        LM::builder()
            .model(JUDGE_MODEL.to_string())
            .build()
            .await?,
    );

    let train = load_rows(&env_or("DOCSQA_TRAIN", "docs-chat-worker/gepa/train.jsonl"))?;
    let val = load_rows(&env_or("DOCSQA_VAL", "docs-chat-worker/gepa/val.jsonl"))?;
    let holdout = load_rows(&env_or("DOCSQA_HOLDOUT", "docs-chat-worker/gepa/holdout.jsonl"))?;
    println!(
        "{} train / {} val / {} holdout",
        train.len(),
        val.len(),
        holdout.len()
    );

    let judge_lm = LM::builder()
        .model(JUDGE_MODEL.to_string())
        .temperature(1.0)
        .max_tokens(3000)
        .build()
        .await?;
    let metric = DensityMetric {
        judge: Predict::<JudgeAnswer>::builder().lm(judge_lm).build(),
    };

    let mut module = DocsAgent::new(mxbai_key.clone());

    println!("== baseline on holdout ==");
    let baseline_evals = evaluate_trainset(&module, &holdout, &metric).await?;
    let baseline = average_score(&baseline_evals);
    println!("baseline holdout score: {baseline:.3}");

    // DOCSQA_TEACHER overrides the reflection model only; judge stays put.
    let reflection_lm = LM::builder()
        .model(env_or("DOCSQA_TEACHER", JUDGE_MODEL))
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;

    let gepa = GEPA::builder()
        .num_iterations(env_or_usize("DOCSQA_ITERS", 5)?)
        .minibatch_size(3)
        .max_rollouts(env_or_usize("DOCSQA_BUDGET", 120)?)
        .prompt_model(reflection_lm)
        .seed(0)
        .eval_concurrency(4)
        .track_stats(true)
        .track_best_outputs(true)
        .build();

    let result = gepa
        .compile_module_with_valset(&mut module, &train, Some(&val), &metric)
        .await?;

    println!("\n== GEPA result ==");
    println!(
        "rollouts: {}  lm_calls: {}  candidates: {}",
        result.total_rollouts,
        result.total_lm_calls,
        result.all_candidates.len()
    );
    println!("evolution: {:?}", result.evolution_history);
    println!(
        "best candidate (gen {}, avg {:.3}):\n---\n{}\n---",
        result.best_candidate.generation,
        result.best_candidate.average_score(),
        result.best_candidate.instruction
    );

    // compile_module installs the winner into `module`; a fresh agent is the seed
    println!("\n== holdout comparison (never seen by GEPA) ==");
    let optimized_evals = evaluate_trainset(&module, &holdout, &metric).await?;
    let optimized = average_score(&optimized_evals);
    for (row, (b, o)) in holdout
        .iter()
        .zip(baseline_evals.iter().zip(optimized_evals.iter()))
    {
        println!(
            "  {:<62} seed {:.2} -> gepa {:.2}",
            row.question.chars().take(60).collect::<String>(),
            b.score,
            o.score
        );
    }
    println!("mean: seed {baseline:.3} -> gepa {optimized:.3} (delta {:+.3})", optimized - baseline);

    let out_path = env_or("DOCSQA_OUT", "docs-chat-worker/gepa/dsrs_best_instruction.txt");
    std::fs::write(&out_path, &result.best_candidate.instruction)?;
    println!("\nbest instruction written to {out_path}");
    Ok(())
}
