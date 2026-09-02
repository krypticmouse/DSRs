//! DocsQA with Qwen 3.6 as the student — the co-evolution control arm.
//!
//! The toast showdown (36-docsqa, docs/handoff-docsqa-showdown.md) found the
//! deployed prompt near-unimprovable — toast-1 was RL-trained with it. This
//! rig asks the control question: on the SAME dataset, SAME judge, SAME
//! protocol, does prompt optimization work against a student that never saw
//! the prompt? Student: openrouter qwen/qwen3.6-35b-a3b — the same base
//! family as toast's grpo40 lineage, minus the RL.
//!
//! Because Qwen has no server-side store grounding, the student is a
//! TWO-stage DSRs-native pipeline (client-side retrieval):
//!
//!   queries  = searcher(question)                  # optimizable slot 1
//!   excerpts = mixedbread stores/search per query  # dsrs-docs + dsrs-code
//!   answer   = answerer(question, excerpts)        # optimizable slot 2
//!
//! Both stages are real Predicts -> real traces -> **stage-repair
//! (SlotCredit::Repair, "mode 2") runs natively**: DOCSQA_REPAIR=<probes>
//! (0/unset = round-robin), DOCSQA_REPAIR_ATTEMPTS (default 2).
//!
//! Modes mirror 36-docsqa:
//! - DOCSQA_MODE=optimize (default), DOCSQA_OPT=ratchet|gepa
//! - DOCSQA_MODE=serve: {"id":1,"instructions":{"searcher":...,"answerer":...}|null,
//!   "split":"val","idx":3} -> {"id":1,"score":..,"feedback":"..","answer":"..",
//!   "queries":[..]}
//!
//! Seeds are dumped to seed_searcher.txt / seed_answerer.txt for the python
//! reference arm. Winner JSON: {"searcher": text, "answerer": text}.
//!
//! Env: as 36-docsqa plus DOCSQA_STUDENT (openrouter:qwen/qwen3.6-35b-a3b),
//! DOCSQA_STUDENT_TEMP (0.2), DOCSQA_TOPK (6/query), DOCSQA_REPAIR,
//! DOCSQA_REPAIR_ATTEMPTS. Out prefix default qwen_<opt>_winner.
//!
//! Run: set -a; source .env; set +a
//!     DOCSQA_OPT=ratchet DOCSQA_REPAIR=2 cargo run --release --example 37-docsqa-qwen

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use dspy_rs::{
    CallMetadata, Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Predictors, Ratchet,
    Signature, SlotCredit, TypedMetric, configure, init_tracing,
};
use serde::{Deserialize, Serialize};

const STORES: [&str; 2] = ["dsrs-docs", "dsrs-code"];
const MODULE_NAMES: [&str; 2] = ["searcher", "answerer"];

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------

/// Write up to three search queries over the DSRs documentation and source
/// stores that would retrieve the material needed to answer the user's
/// question. DSRs (dspy-rs) is a Rust framework for building and optimizing
/// LM pipelines. Prefer exact type, method, trait, and flag names from the
/// question; add one broader conceptual query when the question spans
/// topics. One query per line, nothing else.
#[derive(Signature, Clone, Debug)]
struct WriteQueries {
    /// The user's question about DSRs.
    #[input]
    question: String,

    /// Search queries, one per line (at most three).
    #[output]
    queries: String,
}

/// You are the DSRs docs assistant. DSRs (dspy-rs) is a Rust framework for
/// building and optimizing LM pipelines. Answer ONLY from the retrieved
/// excerpts below: they come from dsrs-docs (published documentation) and
/// dsrs-code (the Rust sources under crates/). When an answer touches
/// implementation, quote the real code with its file path. If the excerpts
/// don't cover the question, say so instead of guessing. Answer in concise
/// markdown.
#[derive(Signature, Clone, Debug)]
struct AnswerFromDocs {
    /// The user's question about DSRs.
    #[input]
    question: String,

    /// Retrieved excerpts, each headed by its source path.
    #[input]
    excerpts: String,

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
// Retrieval: mixedbread stores/search, client-side
// ---------------------------------------------------------------------------

struct StoreClient {
    http: reqwest::Client,
    api_key: String,
    top_k: usize,
}

#[derive(Clone)]
struct Chunk {
    path: String,
    text: String,
}

impl StoreClient {
    async fn search(&self, query: &str) -> Vec<Chunk> {
        let body = serde_json::json!({
            "query": query,
            "store_identifiers": STORES,
            "top_k": self.top_k,
        });
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(3 * attempt as u64)).await;
            }
            let resp = self
                .http
                .post("https://api.mixedbread.com/v1/stores/search")
                .bearer_auth(&self.api_key)
                .json(&body)
                .send()
                .await;
            let Ok(resp) = resp else { continue };
            let Ok(value) = resp.json::<serde_json::Value>().await else { continue };
            let Some(data) = value["data"].as_array() else { continue };
            return data
                .iter()
                .map(|c| Chunk {
                    path: c["filename"]
                        .as_str()
                        .or_else(|| c["file_id"].as_str())
                        .unwrap_or("?")
                        .to_string(),
                    text: c["text"].as_str().unwrap_or_default().to_string(),
                })
                .collect();
        }
        Vec::new()
    }
}

/// open-instruct's remove_thinking_section (ported from 34-hover): reasoning
/// students wrap or precede the real output with think/answer tags — strip
/// back to what follows them. Without this, a failed marker parse turns a
/// chain-of-thought dump into "search queries" (the bug that voided the
/// first qwen cell — see prereg deviation log).
fn strip_thinking(response: &str) -> String {
    let response = response.replace("<|assistant|>", "");
    let response = response.rsplit("</think>").next().unwrap_or(&response).to_string();
    response.replace("<answer>", "").replace("</answer>", "").trim().to_string()
}

/// A line is a plausible search query, not reasoning debris.
fn query_like(line: &str) -> bool {
    let l = line.trim();
    !l.is_empty()
        && l.len() <= 160
        && !l.contains("**")
        && !l.ends_with(':')
        && !l.to_lowercase().starts_with("the user")
        && !l.to_lowercase().starts_with("i need")
}

fn truncate_words(text: &str, words: usize) -> String {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() <= words {
        return text.to_string();
    }
    format!("{} ...", parts[..words].join(" "))
}

// ---------------------------------------------------------------------------
// The student pipeline
// ---------------------------------------------------------------------------

#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct QwenInput {
    question: String,
}

#[dspy_rs::Schema]
#[derive(Clone, Debug, Default)]
struct QwenOutput {
    answer: String,
    /// The executed search queries, for serve traces / reflection records.
    queries: Vec<String>,
}

struct QwenDocsAgent {
    searcher: Predict<WriteQueries>,
    answerer: Predict<AnswerFromDocs>,
    store: Arc<StoreClient>,
}

dspy_rs::predictors!(QwenDocsAgent { searcher, answerer });

impl Module for QwenDocsAgent {
    type Input = QwenInput;
    type Output = QwenOutput;

    async fn forward(&self, input: QwenInput) -> Result<Predicted<QwenOutput>, PredictError> {
        let question = input.question;

        // Stage 1: query writing. Degrades to the raw question.
        // Mine query lines ONLY from a successful field parse. A failed
        // parse (thinking dumps, free-form code — both observed) falls back
        // to the question itself: an honest seed, and marker compliance
        // becomes optimizable headroom instead of silent garbage retrieval.
        let mut queries: Vec<String> = match self
            .searcher
            .call(WriteQueriesInput { question: question.clone() })
            .await
        {
            Ok(out) => strip_thinking(&out.queries)
                .lines()
                .map(|l| l.trim().trim_start_matches(['-', '*', ' ']).to_string())
                .filter(|l| query_like(l))
                .take(3)
                .collect(),
            Err(_) => Vec::new(),
        };
        if queries.is_empty() {
            queries.push(question.clone());
        }

        // Retrieval: merged, deduped by path+prefix, capped.
        let mut excerpts = String::new();
        let mut seen: Vec<String> = Vec::new();
        let mut n_chunks = 0usize;
        for query in &queries {
            for chunk in self.store.search(query).await {
                let key = format!("{}:{:.60}", chunk.path, chunk.text);
                if seen.contains(&key) || n_chunks >= 10 {
                    continue;
                }
                seen.push(key);
                n_chunks += 1;
                excerpts.push_str(&format!(
                    "[{}]\n{}\n\n",
                    chunk.path,
                    truncate_words(&chunk.text, 300)
                ));
            }
        }
        if excerpts.is_empty() {
            excerpts = "(no results retrieved)".to_string();
        }

        // Stage 2: answer. Degrades to raw response on parse failure, empty
        // (scored 0) on hard failure — never Err (G1: an optimize-mode
        // forward Err aborts the compile).
        let answer = match self
            .answerer
            .call(AnswerFromDocsInput { question, excerpts })
            .await
        {
            Ok(out) => strip_thinking(&out.answer),
            Err(PredictError::Parse { raw_response, .. }) => strip_thinking(&raw_response),
            Err(e) => {
                eprintln!("answerer failed (scored 0): {e}");
                String::new()
            }
        };
        Ok(Predicted::new(QwenOutput { answer, queries }, CallMetadata::default()))
    }
}

// ---------------------------------------------------------------------------
// Metric (same density score as 36-docsqa)
// ---------------------------------------------------------------------------

struct DensityMetric {
    judge: Predict<JudgeAnswer>,
}

impl TypedMetric<DocsRow, QwenDocsAgent> for DensityMetric {
    async fn evaluate(
        &self,
        example: &DocsRow,
        prediction: &Predicted<QwenOutput>,
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
            "Score {score:.2} (coverage {:.0}%, noise {:.0}% of answer).\nSearch queries used: {}",
            coverage * 100.0,
            noise * 100.0,
            prediction.queries.join(" | ")
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
// Serve mode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
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
    queries: Vec<String>,
}

struct Factory {
    student_lm: LM,
    store: Arc<StoreClient>,
}

impl Factory {
    fn build(&self, instructions: Option<&HashMap<String, Option<String>>>) -> QwenDocsAgent {
        let get = |name: &str| -> Option<String> {
            instructions.and_then(|m| m.get(name).cloned()).flatten()
        };
        let searcher = match get("searcher") {
            Some(text) => Predict::<WriteQueries>::builder()
                .named("searcher")
                .lm(self.student_lm.clone())
                .instruction(text)
                .build(),
            None => Predict::<WriteQueries>::builder()
                .named("searcher")
                .lm(self.student_lm.clone())
                .build(),
        };
        let answerer = match get("answerer") {
            Some(text) => Predict::<AnswerFromDocs>::builder()
                .named("answerer")
                .lm(self.student_lm.clone())
                .instruction(text)
                .build(),
            None => Predict::<AnswerFromDocs>::builder()
                .named("answerer")
                .lm(self.student_lm.clone())
                .build(),
        };
        QwenDocsAgent { searcher, answerer, store: self.store.clone() }
    }
}

async fn serve(dir: &str, factory: Arc<Factory>, metric: Arc<DensityMetric>) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let mut splits: HashMap<String, Vec<DocsRow>> = HashMap::new();
    for entry in std::fs::read_dir(format!("{dir}/data"))? {
        let path = entry?.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if name.starts_with("pool") || name.starts_with("gen_") {
            continue;
        }
        splits.insert(name.to_string(), load_rows(&path.to_string_lossy())?);
    }
    eprintln!(
        "serve(qwen): splits loaded: {:?}",
        splits.iter().map(|(k, v)| (k.clone(), v.len())).collect::<Vec<_>>()
    );

    let concurrency = env_usize("DOCSQA_CONCURRENCY", 6);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let stdout = Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
    let splits = Arc::new(splits);

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
        let (splits, stdout, semaphore, factory, metric) = (
            splits.clone(),
            stdout.clone(),
            semaphore.clone(),
            factory.clone(),
            metric.clone(),
        );
        tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("semaphore open");
            let Some(row) = splits.get(&request.split).and_then(|s| s.get(request.idx)) else {
                eprintln!("unknown split/idx: {}/{}", request.split, request.idx);
                return;
            };
            let agent = factory.build(request.instructions.as_ref());
            let (score, feedback, answer, queries) = match agent
                .forward(QwenInput { question: row.question.clone() })
                .await
            {
                Ok(prediction) => {
                    let eval = metric.evaluate(row, &prediction, None).await.unwrap_or_else(
                        |e| Eval::with_feedback(0.0, format!("metric error: {e}")),
                    );
                    (
                        eval.score as f64,
                        eval.feedback.unwrap_or_default(),
                        prediction.answer.clone(),
                        prediction.queries.clone(),
                    )
                }
                Err(e) => (0.0, format!("ROLLOUT ERROR (scored 0): {e}"), String::new(), vec![]),
            };
            let response = ServeResponse { id: request.id, score, feedback, answer, queries };
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
// Optimize
// ---------------------------------------------------------------------------

fn winner_map(
    module: Option<&QwenDocsAgent>,
    seeds: &BTreeMap<String, String>,
    slots: &BTreeMap<String, String>,
    module_name: &str,
    instruction: &str,
) -> BTreeMap<String, String> {
    let mut map = seeds.clone();
    if let Some(module) = module {
        for (name, info) in module.predictors() {
            map.insert(name, info.instruction());
        }
    }
    for (name, text) in slots {
        if MODULE_NAMES.contains(&name.as_str()) {
            map.insert(name.clone(), text.clone());
        }
    }
    if MODULE_NAMES.contains(&module_name) {
        map.insert(module_name.to_string(), instruction.to_string());
    }
    map
}

fn write_winner(path: &str, map: &BTreeMap<String, String>) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(map)?)?;
    Ok(())
}

async fn optimize(
    dir: &str,
    factory: &Factory,
    seeds: &BTreeMap<String, String>,
    metric: &DensityMetric,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("DOCSQA_OPT", "ratchet");
    let train = load_rows(&format!("{dir}/data/{}", env_str("DOCSQA_TRAIN_FILE", "train.jsonl")))?;
    let val = load_rows(&format!("{dir}/data/{}", env_str("DOCSQA_VAL_FILE", "val.jsonl")))?;
    println!("docsqa-qwen: {} train / {} val, arm {opt_name}, 2 slots", train.len(), val.len());

    let seed = env_usize("DOCSQA_SEED", 0) as u64;
    let budget = env_usize("DOCSQA_BUDGET", 600);
    let concurrency = env_usize("DOCSQA_CONCURRENCY", 6);
    let mut agent = factory.build(None);

    let prefix = env_str("DOCSQA_OUT_PREFIX", &format!("qwen_{opt_name}_winner"));
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
            "rollouts: {} lm_calls: {} candidates: {}",
            result.total_rollouts,
            result.total_lm_calls,
            result.all_candidates.len()
        );
        let winner = winner_map(
            Some(&agent),
            seeds,
            &result.best_candidate.slots,
            &result.best_candidate.module_name,
            &result.best_candidate.instruction,
        );
        println!(
            "\nchosen (gen {}, val avg {:.3}, touched {}):",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.module_name
        );
        write_winner(&out, &winner)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| winner_map(None, seeds, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    } else if opt_name == "ratchet" {
        let iters = env_usize("DOCSQA_ITERS", 40);
        let minibatch = env_usize("DOCSQA_MINIBATCH", 3);
        let champion = env_usize("DOCSQA_CHAMPION", 20);
        println!(
            "ratchet seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, champion {champion}"
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
            // Mode 2: counterfactual stage-repair credit assignment.
            let repair_probes = env_usize("DOCSQA_REPAIR", 0);
            if repair_probes > 0 {
                optimizer.slot_credit = SlotCredit::Repair;
                optimizer.repair_probes = repair_probes;
                optimizer.repair_attempts = env_usize("DOCSQA_REPAIR_ATTEMPTS", 2);
                println!(
                    "stage-repair ON ({repair_probes} probes/gen, {} attempts)",
                    optimizer.repair_attempts
                );
            }
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
        if result.repair_probes_run > 0 {
            println!(
                "repair: {} probes, {} recovered, per-slot {:?}",
                result.repair_probes_run, result.repair_recovered, result.repair_slot_stats
            );
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
        let winner = winner_map(
            Some(&agent),
            seeds,
            &result.best_candidate.slots,
            &result.best_candidate.module_name,
            &result.best_candidate.instruction,
        );
        println!(
            "\nchosen (gen {}, val avg {:.3}, touched {}):",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.module_name
        );
        write_winner(&out, &winner)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| winner_map(None, seeds, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    } else {
        bail!("unknown DOCSQA_OPT {opt_name:?} (ratchet|gepa)");
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

    let student_temp: f32 = env_str("DOCSQA_STUDENT_TEMP", "0.2").parse().unwrap_or(0.2);
    // "mxbai:<model>" routes to the Mixedbread OpenAI-compatible endpoint
    // (e.g. mxbai:toast-1 — the two-stage toast repair arm); anything else
    // goes through the model string as-is (openrouter:...).
    let student_spec = env_str("DOCSQA_STUDENT", "openrouter:qwen/qwen3.6-35b-a3b");
    let student_lm = if let Some(model) = student_spec.strip_prefix("mxbai:") {
        LM::builder()
            .base_url("https://api.mixedbread.com/v1".to_string())
            .api_key(std::env::var("MXBAI_API_KEY").context("MXBAI_API_KEY not set")?)
            .model(model.to_string())
            .temperature(student_temp)
            .max_tokens(2048)
            .build()
            .await?
    } else {
        LM::builder()
            .model(student_spec)
            .temperature(student_temp)
            .max_tokens(2048)
            .build()
            .await?
    };
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

    let store = Arc::new(StoreClient {
        http: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?,
        api_key,
        top_k: env_usize("DOCSQA_TOPK", 6),
    });
    let factory = Factory { student_lm, store };

    // Seed dump for the python reference arm (byte-identical seeds).
    let probe = factory.build(None);
    let mut seeds = BTreeMap::new();
    for (name, info) in probe.predictors() {
        std::fs::write(format!("{dir}/seed_{name}.txt"), info.instruction())?;
        seeds.insert(name, info.instruction());
    }

    match mode.as_str() {
        "serve" => serve(&dir, Arc::new(factory), Arc::new(metric)).await,
        "optimize" => optimize(&dir, &factory, &seeds, &metric, reflection_lm).await,
        other => bail!("unknown DOCSQA_MODE {other:?} (optimize|serve)"),
    }
}
