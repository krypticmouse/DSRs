//! DocsQA showdown rig: the DSRs docs assistant (toast-1 + hosted store
//! tools) optimized by RATCHET / merged GEPA, with a serve mode so the
//! reference pip-gepa arm drives BYTE-IDENTICAL rollouts (HoVer pattern,
//! benchmarks/hover/run_python_gepa.py).
//!
//! Student + judge + metric are lifted verbatim from 27-gepa-docs-chat.rs
//! (the deployed docs-chat worker's semantics): toast-1 over raw HTTP with
//! server-side store_search/store_grep over dsrs-docs + dsrs-code; judge =
//! JudgeAnswer on gpt-5.6-sol; score = 0.70*coverage + 0.10*citations +
//! 0.20*(1-fluff) - wrong_claims penalty.
//!
//! ONE optimizable slot: `answerer` (the system prompt). Winner JSON:
//! {"answerer": <text>}.
//!
//! Modes:
//! - DOCSQA_MODE=optimize (default): DOCSQA_OPT=ratchet|gepa. RATCHET takes
//!   P×N parallel proposals via DOCSQA_PARENTS / DOCSQA_PROPOSALS
//!   (default 1×1 = sequential v2; 2×2 = the ComBEE-style breadth arm).
//! - DOCSQA_MODE=serve: JSONL rollout server on stdin/stdout.
//!   Request:  {"id":1,"instructions":{"answerer":"..."|null}|null,
//!              "split":"val","idx":3}
//!   Response: {"id":1,"score":0.83,"feedback":"...","answer":"..."}
//!
//! Env: DOCSQA_DIR (benchmarks/docsqa), DOCSQA_MODE, DOCSQA_OPT,
//! DOCSQA_SEED, DOCSQA_BUDGET (600), DOCSQA_ITERS, DOCSQA_MINIBATCH,
//! DOCSQA_CHAMPION, DOCSQA_CONCURRENCY (6), DOCSQA_TRAIN_FILE, DOCSQA_VAL_FILE,
//! DOCSQA_OUT_PREFIX, DOCSQA_TEACHER, DOCSQA_PARENTS, DOCSQA_PROPOSALS.
//! v6.4 (compound/harvest): DOCSQA_PRUNE=1 turns on post-certification Occam
//! pruning by lineage credit, DOCSQA_PRUNE_BUDGET=<n> its probe budget (120).
//!
//! Run from the repo root: set -a; source .env; set +a
//!     DOCSQA_OPT=ratchet cargo run --release --example 36-docsqa

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use dspy_rs::{
    CallMetadata, Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Predictors, Ratchet,
    Role, Signature, TypedMetric, compound, configure, harvest, init_tracing,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MXBAI_URL: &str = "https://api.mixedbread.com/v1/chat/completions";
const STORES: [&str; 2] = ["dsrs-docs", "dsrs-code"];

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Signatures (verbatim from 27-gepa-docs-chat.rs)
// ---------------------------------------------------------------------------

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

fn load_rows(path: &str) -> Result<Vec<DocsRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run make_splits.py first"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<DocsRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// Student: toast-1 over raw HTTP (verbatim from 27)
// ---------------------------------------------------------------------------

struct DocsAgent {
    answerer: Predict<DocsQA>,
    http: reqwest::Client,
    api_key: String,
}

dspy_rs::predictors!(DocsAgent { answerer });

/// One shared connection pool for every agent (2026-08-30 audit S3: a
/// fresh `Client` per rollout rebuilt the pool + TLS each time, and the
/// missing request timeout let a hung connection pin a serve permit
/// indefinitely). 90s/attempt × 3 attempts + backoff stays under the eval
/// driver's 300s row timeout, so a degraded endpoint frees its permit
/// before the driver gives up on the row.
fn shared_http() -> reqwest::Client {
    static HTTP: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    HTTP.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(90))
            .build()
            .expect("reqwest client builds")
    })
    .clone()
}

impl DocsAgent {
    fn new(api_key: String, instruction: Option<String>) -> Self {
        let answerer = match instruction {
            Some(text) => Predict::<DocsQA>::builder().named("answerer").instruction(text).build(),
            None => Predict::<DocsQA>::builder().named("answerer").build(),
        };
        Self { answerer, http: shared_http(), api_key }
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
    let s = content.split("[[ ## answer ## ]]").nth(1).unwrap_or(content);
    let s = s.split("[[ ## completed ## ]]").next().unwrap_or(s);
    s.trim().to_string()
}

impl Module for DocsAgent {
    type Input = DocsQAInput;
    type Output = DocsQAOutput;

    async fn forward(&self, input: DocsQAInput) -> Result<Predicted<DocsQAOutput>, PredictError> {
        // build_chat resolves instance state ⊕ the ambient candidate, so the
        // rendered system prompt carries the instruction under test.
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
            let content = value["choices"][0]["message"]["content"].as_str().unwrap_or_default();
            let answer = extract_answer(content);
            return Ok(Predicted::new(DocsQAOutput { answer }, CallMetadata::default()));
        }
        // Never Err: in optimize mode a forward error aborts the entire
        // compile. A transport failure is an empty answer scored 0 — the
        // same convention the serve path (and thus the reference arm) sees.
        eprintln!("toast-1 call failed after retries (scored 0): {last_err}");
        Ok(Predicted::new(DocsQAOutput { answer: String::new() }, CallMetadata::default()))
    }
}

// ---------------------------------------------------------------------------
// Metric (verbatim from 27): coverage-dominant density score
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
// Serve mode (HoVer protocol, one module)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    /// {"answerer": "..." | null} — null / missing = seed.
    instructions: Option<HashMap<String, Option<String>>>,
    split: String,
    idx: usize,
}

#[derive(Serialize)]
struct ServeResponse {
    id: u64,
    score: f64,
    feedback: String,
    answer: String,
}

async fn serve(dir: &str, api_key: String, metric: Arc<DensityMetric>) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let mut splits: HashMap<String, Vec<DocsRow>> = HashMap::new();
    for entry in std::fs::read_dir(format!("{dir}/data"))? {
        let path = entry?.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if name.starts_with("pool") || name.starts_with("gen_") {
            continue; // raw pools are not eval splits
        }
        splits.insert(name.to_string(), load_rows(&path.to_string_lossy())?);
    }
    eprintln!(
        "serve: splits loaded: {:?}",
        splits.iter().map(|(k, v)| (k.clone(), v.len())).collect::<Vec<_>>()
    );

    let concurrency = env_usize("DOCSQA_CONCURRENCY", 6);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let stdout = Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
    let splits = Arc::new(splits);
    let api_key = Arc::new(api_key);

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut tasks = Vec::new();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request: ServeRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("bad request line ({e}): {line:.120}");
                continue;
            }
        };
        let (splits, stdout, semaphore, api_key, metric) = (
            splits.clone(),
            stdout.clone(),
            semaphore.clone(),
            api_key.clone(),
            metric.clone(),
        );
        tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("semaphore open");
            let Some(row) = splits.get(&request.split).and_then(|s| s.get(request.idx)) else {
                eprintln!("unknown split/idx: {}/{}", request.split, request.idx);
                return;
            };
            let instruction = request
                .instructions
                .as_ref()
                .and_then(|m| m.get("answerer").cloned())
                .flatten();
            let agent = DocsAgent::new((*api_key).clone(), instruction);
            let (score, feedback, answer) =
                match agent.forward(DocsQAInput { question: row.question.clone() }).await {
                    Ok(prediction) => {
                        let eval = metric
                            .evaluate(row, &prediction, None)
                            .await
                            .unwrap_or_else(|e| {
                                Eval::with_feedback(0.0, format!("metric error: {e}"))
                            });
                        (eval.score as f64, eval.feedback.unwrap_or_default(), prediction.answer.clone())
                    }
                    Err(e) => (0.0, format!("ROLLOUT ERROR (scored 0): {e}"), String::new()),
                };
            let response = ServeResponse { id: request.id, score, feedback, answer };
            let mut line = serde_json::to_string(&response).expect("serializable");
            line.push('\n');
            let mut out = stdout.lock().await;
            let _ = out.write_all(line.as_bytes()).await;
            let _ = out.flush().await;
        }));
    }
    for task in tasks {
        let _ = task.await;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Optimize mode
// ---------------------------------------------------------------------------

fn write_winner(path: &str, instruction: &str) -> Result<()> {
    let map = BTreeMap::from([("answerer".to_string(), instruction.to_string())]);
    std::fs::write(path, serde_json::to_string_pretty(&map)?)?;
    Ok(())
}

async fn optimize(
    dir: &str,
    api_key: String,
    metric: &DensityMetric,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("DOCSQA_OPT", "ratchet");
    let train = load_rows(&format!("{dir}/data/{}", env_str("DOCSQA_TRAIN_FILE", "train.jsonl")))?;
    let val = load_rows(&format!("{dir}/data/{}", env_str("DOCSQA_VAL_FILE", "val.jsonl")))?;
    println!("docsqa: {} train / {} val, arm {opt_name}", train.len(), val.len());

    let seed = env_usize("DOCSQA_SEED", 0) as u64;
    let budget = env_usize("DOCSQA_BUDGET", 600);
    let concurrency = env_usize("DOCSQA_CONCURRENCY", 6);
    let mut agent = DocsAgent::new(api_key, None);

    let prefix = env_str("DOCSQA_OUT_PREFIX", &format!("{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.json");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.json");

    if opt_name == "gepa" {
        let iters = env_usize("DOCSQA_ITERS", 40);
        let minibatch = env_usize("DOCSQA_MINIBATCH", 3);
        println!("gepa seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}");
        let optimizer = GEPA::builder()
            .num_iterations(iters)
            .minibatch_size(minibatch)
            .max_rollouts(budget)
            .prompt_model(reflection_lm)
            .seed(seed)
            .eval_concurrency(concurrency)
            .track_stats(true)
            .build();
        let result = optimizer
            .compile_module_with_valset(&mut agent, &train, Some(&val), metric)
            .await?;
        println!(
            "rollouts: {} lm_calls: {} candidates: {} gate_rejected: {} leak: {}/{}",
            result.total_rollouts,
            result.total_lm_calls,
            result.all_candidates.len(),
            result.gate_rejected,
            result.leak_retries,
            result.leak_skipped
        );
        println!(
            "\nchosen (gen {}, val avg {:.3}):\n{}",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.instruction
        );
        write_winner(&out, &result.best_candidate.instruction)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| c.instruction.clone())
            .unwrap_or_else(|| result.best_candidate.instruction.clone());
        write_winner(&out_argmax, &argmax)?;
    } else if opt_name == "ratchet" {
        let iters = env_usize("DOCSQA_ITERS", 40);
        let minibatch = env_usize("DOCSQA_MINIBATCH", 3);
        let champion = env_usize("DOCSQA_CHAMPION", 20);
        let parents = env_usize("DOCSQA_PARENTS", 1);
        let proposals = env_usize("DOCSQA_PROPOSALS", 1);
        println!(
            "ratchet seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, \
             champion {champion}, P×N {parents}×{proposals}"
        );
        let optimizer = {
            let mut optimizer = Ratchet::builder().prompt_model(reflection_lm).build();
            optimizer.num_iterations = iters;
            optimizer.minibatch_size = minibatch;
            optimizer.champion_fresh = champion;
            optimizer.max_rollouts = Some(budget);
            optimizer.seed = Some(seed);
            optimizer.eval_concurrency = concurrency;
            optimizer.track_stats = true;
            optimizer.parents_per_step = parents;
            optimizer.proposals_per_parent = proposals;
            optimizer
        };
        let result = optimizer
            .compile_module_with_valset(&mut agent, &train, Some(&val), metric)
            .await?;
        println!(
            "rollouts: {} lm_calls: {} promoted: {}",
            result.total_rollouts,
            result.total_lm_calls,
            result.all_candidates.len()
        );
        println!(
            "gate_rejected: {} race_killed: {} skipped: {} leak: {}/{} length: {}/{}",
            result.gate_rejected,
            result.race_killed,
            result.skipped,
            result.leak_retries,
            result.leak_skipped,
            result.length_retries,
            result.length_skipped
        );
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
            "\nchosen (gen {}, val avg {:.3}):\n{}",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.instruction
        );
        write_winner(&out, &result.best_candidate.instruction)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| c.instruction.clone())
            .unwrap_or_else(|| result.best_candidate.instruction.clone());
        write_winner(&out_argmax, &argmax)?;
    } else if opt_name == "compound" || opt_name == "harvest" {
        // RATCHET v4 COMPOUND (docs/ratchet-v4-design.md): e-process racing
        // + sequential paired ship certification vs the seed on reserved
        // never-touched train columns. train120 carve: champion 24 +
        // certify 60 leaves 36 batch columns; 60 paired certification
        // columns certify a ≥ +.055-class bundle at alpha 0.1 under this
        // rig's judge noise (the prereg carries the arithmetic).
        let iters = env_usize("DOCSQA_ITERS", 150);
        let minibatch = env_usize("DOCSQA_MINIBATCH", 3);
        let champion = env_usize("DOCSQA_CHAMPION", 24);
        let certify_reserve = env_usize("DOCSQA_CERTIFY_RESERVE", 60);
        let certify_budget = env_usize("DOCSQA_CERTIFY_BUDGET", 150);
        let parents = env_usize("DOCSQA_PARENTS", 2);
        let proposals = env_usize("DOCSQA_PROPOSALS", 2);
        println!(
            "compound seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, \
             champion {champion}, certify {certify_reserve} cols / {certify_budget} rollouts, \
             P×N {parents}×{proposals}"
        );
        // Harness-supplied certification split (port spec v2 §2): rows are
        // appended to the trainset and named as explicit certify columns —
        // holdout-class ship-gate power the train carve can't provide.
        let mut train_ext = train.clone();
        let mut certify_idx: Vec<usize> = Vec::new();
        let certify_file = env_str("DOCSQA_CERTIFY_FILE", "");
        if !certify_file.is_empty() {
            let extra = load_rows(&format!("{dir}/data/{certify_file}"))?;
            certify_idx = (train_ext.len()..train_ext.len() + extra.len()).collect();
            println!("certify split: {} rows from {certify_file}", extra.len());
            train_ext.extend(extra);
        }
        let optimizer = {
            let mut optimizer = if opt_name == "harvest" { harvest() } else { compound() };
            optimizer.prompt_model = Some(reflection_lm);
            optimizer.num_iterations = iters;
            // The preset's minibatch (harvest: 5) stands unless DOCSQA_MINIBATCH is set.
            optimizer.minibatch_size = env_usize("DOCSQA_MINIBATCH", optimizer.minibatch_size);
            optimizer.champion_fresh = champion;
            optimizer.certify_reserve = certify_reserve;
            optimizer.certify_budget = certify_budget;
            optimizer.max_rollouts = Some(budget);
            optimizer.seed = Some(seed);
            optimizer.eval_concurrency = concurrency;
            optimizer.track_stats = true;
            optimizer.parents_per_step = parents;
            optimizer.proposals_per_parent = proposals;
            // Corpus-fact task-config (port spec v2 §3): DOCSQA_ALLOW_MEM=1
            // turns the leak guard off so feedback facts can be pinned —
            // certification still guards the ship, which pip-gepa lacks.
            if env_str("DOCSQA_ALLOW_MEM", "0") == "1" {
                optimizer.allow_memorization = true;
                println!("leak guard OFF (corpus-fact task-config)");
            }
            // Accretion economics: DOCSQA_ACCEPT=gate skips the per-child
            // race (gate + scout slice only); certification carries all
            // the verification.
            if env_str("DOCSQA_ACCEPT", "race") == "gate" {
                optimizer.accept_mode = dspy_rs::AcceptMode::Gate;
                optimizer.scout_size = env_usize("DOCSQA_SCOUT", optimizer.scout_size);
                println!("accept mode: GATE (scout {})", optimizer.scout_size);
            }
            optimizer.champion_k = env_usize("DOCSQA_CHAMPION_K", 2);
            // E5a band mixture (escrow, ratchet-v5-escrow-design.md):
            // "horizon" calibrates fixed Kelly bets to this attempt's
            // actual certify column count at the Bonferroni-worst level
            // (alpha/2, two challengers) — conservative sizing costs
            // growth only, never validity. "" (default) = v4 wallet.
            let certify_cols_total = certify_reserve + certify_idx.len();
            match env_str("DOCSQA_CERTIFY_BAND", "").as_str() {
                "default" => {
                    optimizer.certify_band = Some(dspy_rs::BandBets::default());
                    println!("certify band: DEFAULT (.02/.035/.06 @ sd .2)");
                }
                "horizon" => {
                    let band = dspy_rs::BandBets::for_horizon(
                        certify_cols_total.max(1),
                        optimizer.certify_alpha / 2.0,
                        0.2,
                    );
                    println!(
                        "certify band: HORIZON {} cols -> effects {:?}",
                        certify_cols_total, band.effects
                    );
                    optimizer.certify_band = Some(band);
                }
                _ => {}
            }
            optimizer.certify_cols = certify_idx;
            // v6 HARVEST knobs (optional; the `harvest()` preset sets the
            // defaults, and none of these change `compound` unless set).
            match env_str("DOCSQA_SCREEN", "").as_str() {
                "a" => optimizer.screen_rule = dspy_rs::ScreenRule::OnTargetA,
                "b" => optimizer.screen_rule = dspy_rs::ScreenRule::HeldOutB,
                _ => {}
            }
            if let Ok(v) = std::env::var("DOCSQA_REWRITE_ONLY") {
                optimizer.rewrite_only = v == "1";
            }
            if let Ok(v) = std::env::var("DOCSQA_POLICY") {
                optimizer.policy_channel = v == "1";
            }
            if let Ok(v) = std::env::var("DOCSQA_POLICY_SHARE") {
                optimizer.policy_share = v.parse().unwrap_or(optimizer.policy_share);
            }
            if let Ok(v) = std::env::var("DOCSQA_LEDGER_GATE_KILLS") {
                optimizer.ledger_gate_kills = v == "1";
            }
            if let Ok(v) = std::env::var("DOCSQA_PARENT_GREED") {
                optimizer.parent_greed = v.parse().unwrap_or(optimizer.parent_greed);
            }
            if let Ok(v) = std::env::var("DOCSQA_GATE_SIZE") {
                optimizer.gate_size = v.parse().ok();
            }
            // v6.4 Occam pruning: DOCSQA_PRUNE=1 bundle-removes the certified
            // champion's zero-evidence clauses (lineage-diff credit, zero
            // extra rollouts to compute) on the selection columns and
            // re-certifies the pruned prompt vs the seed with the
            // certification budget the first pass left. Probe budget
            // (default 120 = 3 × champion_fresh) via DOCSQA_PRUNE_BUDGET.
            if env_str("DOCSQA_PRUNE", "0") == "1" {
                optimizer.prune_after_certify = true;
                optimizer.prune_budget = env_usize("DOCSQA_PRUNE_BUDGET", optimizer.prune_budget);
                println!("occam prune after certify ON (probe budget {})", optimizer.prune_budget);
            }
            println!(
                "screen {:?} accept {:?} scout {} rewrite_only {} policy {} (share {}) ledger_gate_kills {} \
                 greed {} gate_size {:?} minibatch {} P×N {}×{}",
                optimizer.screen_rule,
                optimizer.accept_mode,
                optimizer.scout_size,
                optimizer.rewrite_only,
                optimizer.policy_channel,
                optimizer.policy_share,
                optimizer.ledger_gate_kills,
                optimizer.parent_greed,
                optimizer.gate_size,
                optimizer.minibatch_size,
                optimizer.parents_per_step,
                optimizer.proposals_per_parent
            );
            // FORAGE (v5): DOCSQA_TEACHER_CONTEXT=digest|tools gives the
            // Rewrite teacher the whole-ledger contrastive digest (and, for
            // tools, read-only drill-down tools). The check extractor
            // parses this rig's judge feedback ("Missing or wrong:"
            // bullets) into the discrete key points a rollout failed —
            // the harvest frontier, rendered as OPEN checks.
            match env_str("DOCSQA_TEACHER_CONTEXT", "minibatch").as_str() {
                "digest" => {
                    optimizer.teacher_context = dspy_rs::TeacherContext::Digest;
                    println!("teacher context: FORAGE DIGEST");
                }
                "tools" => {
                    optimizer.teacher_context = dspy_rs::TeacherContext::Tools;
                    println!("teacher context: FORAGE TOOLS");
                }
                _ => {}
            }
            // FORAGE v5.1: DOCSQA_ANCHOR=1 freezes the seed base — every
            // proposal must reproduce it verbatim and only append facts
            // (style contamination was the first live arm's regression
            // channel; Simplify goes ineligible under anchoring).
            if env_str("DOCSQA_ANCHOR", "0") == "1" {
                optimizer.anchored_accretion = true;
                println!("anchored accretion ON (base frozen, facts-only edits)");
            }
            // FORAGE v5.2: DOCSQA_ANCHOR_RULES=1 re-opens the behavior-rule
            // channel (directed at judge-aligned verification/refusal
            // rules) inside the anchored appended section.
            if env_str("DOCSQA_ANCHOR_RULES", "0") == "1" {
                optimizer.anchor_allow_rules = true;
                println!("anchored RULES channel ON (verification/refusal class)");
            }
            // ASSAY (v5.3) Stage 1: DOCSQA_REAUDIT_FRAC re-invests that
            // fraction of each step's spend in re-auditing pool members on
            // never-yet-drawn val columns; ranking and champion screening
            // switch to cumulative paired-vs-seed evidence.
            let reaudit_frac: f64 = env_str("DOCSQA_REAUDIT_FRAC", "0")
                .parse()
                .unwrap_or(0.0);
            if reaudit_frac > 0.0 {
                optimizer.reaudit_budget_frac = reaudit_frac;
                optimizer.reaudit_block = env_usize("DOCSQA_REAUDIT_BLOCK", 8);
                println!(
                    "ASSAY stage-1 re-audit ON (frac {}, block {})",
                    optimizer.reaudit_budget_frac, optimizer.reaudit_block
                );
            }
            // ASSAY (v5.3) Stage 2: DOCSQA_PIN_ASSAY=1 turns on the endgame
            // per-pin ablation probes + stopped e-BH composition; the
            // frozen certification still takes the ship decision.
            if env_str("DOCSQA_PIN_ASSAY", "0") == "1" {
                optimizer.pin_assay = true;
                optimizer.pin_probe_cols = env_usize("DOCSQA_PIN_PROBE_COLS", 20);
                optimizer.pin_probe_budget = env_usize("DOCSQA_PIN_PROBE_BUDGET", 400);
                optimizer.pin_fdr_alpha = env_str("DOCSQA_PIN_FDR", "0.1")
                    .parse()
                    .unwrap_or(0.1);
                optimizer.pin_probe_repeats = env_usize("DOCSQA_PIN_REPEATS", 1);
                println!(
                    "ASSAY stage-2 pin assay ON (cols {}, repeats {}, budget {}, FDR {})",
                    optimizer.pin_probe_cols,
                    optimizer.pin_probe_repeats,
                    optimizer.pin_probe_budget,
                    optimizer.pin_fdr_alpha
                );
                // E3 subset screen (escrow design): replaces the per-pin
                // probes with joint subset-lasso credit assignment.
                optimizer.pin_screen_draws = env_usize("DOCSQA_PIN_SCREEN_DRAWS", 0);
                optimizer.pin_screen_cols = env_usize("DOCSQA_PIN_SCREEN_COLS", 20);
                optimizer.pin_screen_min_keep = env_usize("DOCSQA_PIN_SCREEN_MIN_KEEP", 0);
                if optimizer.pin_screen_draws > 0 {
                    println!(
                        "E3 SUBSET SCREEN ON (draws {}, slice {}, min_keep {}, budget {})",
                        optimizer.pin_screen_draws,
                        optimizer.pin_screen_cols,
                        optimizer.pin_screen_min_keep,
                        optimizer.pin_probe_budget
                    );
                }
            }
            if !matches!(optimizer.teacher_context, dspy_rs::TeacherContext::Minibatch) {
                optimizer.forage_chars = env_usize("DOCSQA_FORAGE_CHARS", 15_000);
                optimizer.forage_open_cap = env_usize("DOCSQA_FORAGE_OPEN", 80);
                optimizer.forage_contrast_cap = env_usize("DOCSQA_FORAGE_CONTRAST", 12);
                optimizer.check_extractor = Some(std::sync::Arc::new(|feedback: &str| {
                    let mut checks = Vec::new();
                    let mut in_missing = false;
                    for line in feedback.lines() {
                        if line.starts_with("Missing or wrong:") {
                            in_missing = true;
                            continue;
                        }
                        if in_missing {
                            if let Some(item) = line.strip_prefix("  - ") {
                                checks.push(item.trim().to_string());
                            } else {
                                in_missing = false;
                            }
                        }
                    }
                    checks
                }));
            }
            optimizer
        };
        let result = optimizer
            .compile_module_with_valset(&mut agent, &train_ext, Some(&val), metric)
            .await?;
        println!(
            "rollouts: {} lm_calls: {} promoted: {}",
            result.total_rollouts,
            result.total_lm_calls,
            result.all_candidates.len()
        );
        println!(
            "gate_rejected: {} race_killed: {} faded: {} skipped: {} leak: {}/{} length: {}/{} anchor: {}/{}",
            result.gate_rejected,
            result.race_killed,
            result.faded,
            result.skipped,
            result.leak_retries,
            result.leak_skipped,
            result.length_retries,
            result.length_skipped,
            result.anchor_retries,
            result.anchor_skipped
        );
        println!("operator stats: {:?}", result.operator_stats);
        if result.audit_rollouts > 0 || result.pins_tested > 0 {
            println!(
                "ASSAY: audit {} evicted / {} rollouts; pins {}/{} kept ({} probe rollouts)",
                result.audit_evicted,
                result.audit_rollouts,
                result.pins_kept,
                result.pins_tested,
                result.pin_probe_rollouts
            );
            for (pin, cols, mean, e_gain) in &result.pin_assay_report {
                let excerpt: String = pin.chars().take(110).collect();
                println!("  pin [{cols} cols, mean {mean:+.4}, e {e_gain:.2}] {excerpt}");
            }
        }
        println!(
            "CERTIFICATION: {:?} after {} cols, mean paired delta {:+.4}, e-value {:.2}",
            result.certified,
            result.certify_cols_used,
            result.certify_mean_delta,
            result.certify_e_value
        );
        if result.prune_rollouts > 0 || result.prune_removed > 0 {
            println!(
                "OCCAM PRUNE: {} -> {} chars, {} clauses removed, {} rollouts, re-certified {:?}",
                result.prune_chars_before,
                result.prune_chars_after,
                result.prune_removed,
                result.prune_rollouts,
                result.prune_certified
            );
        }
        let credit = result.clause_credit_report();
        if !credit.is_empty() {
            println!("\n== lineage credit ==\n{credit}");
        }
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
            "\nchosen (gen {}, val avg {:.3}):\n{}",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.instruction
        );
        write_winner(&out, &result.best_candidate.instruction)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| c.instruction.clone())
            .unwrap_or_else(|| result.best_candidate.instruction.clone());
        write_winner(&out_argmax, &argmax)?;
    } else {
        bail!("unknown DOCSQA_OPT {opt_name:?} (ratchet|gepa|compound|harvest)");
    }
    println!("\nwinner written to {out}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = env_str("DOCSQA_MODE", "optimize");
    if mode != "serve" {
        init_tracing()?;
    }
    let api_key = std::env::var("MXBAI_API_KEY").context("MXBAI_API_KEY not set")?;
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    let dir = env_str("DOCSQA_DIR", "benchmarks/docsqa");

    let judge_lm = LM::builder()
        .model(env_str("DOCSQA_JUDGE", "openrouter:openai/gpt-5.6-sol"))
        .temperature(0.0)
        .max_tokens(2000)
        .build()
        .await?;
    let metric = DensityMetric {
        judge: Predict::<JudgeAnswer>::builder().lm(judge_lm.clone()).build(),
    };

    let reflection_lm = LM::builder()
        .model(env_str("DOCSQA_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(8000)
        .build()
        .await?;
    configure(judge_lm);

    // Dump the exact seed instruction so the reference pip-gepa arm starts
    // from a byte-identical seed candidate (common.py reads this file).
    {
        let probe = DocsAgent::new(api_key.clone(), None);
        let named = probe.predictors();
        if let Some((_, info)) = named.iter().find(|(n, _)| n == "answerer") {
            std::fs::write(format!("{dir}/seed_instruction.txt"), info.instruction())?;
        }
    }

    match mode.as_str() {
        "serve" => serve(&dir, api_key, Arc::new(metric)).await,
        "optimize" => optimize(&dir, api_key, &metric, reflection_lm).await,
        other => bail!("unknown DOCSQA_MODE {other:?} (optimize|serve)"),
    }
}
