//! BrowseComp-Plus optimizer rig (design: docs/temper-design.md §3).
//!
//! One binary, two modes:
//!
//! - `BCPLUS_MODE=optimize` (default): optimize the search agent's
//!   instruction with `BCPLUS_OPT=temper|ratchet|hybrid|gepa` and write the
//!   winner (`hybrid` = RATCHET search + TEMPER's variant-champion endgame).
//! - `BCPLUS_MODE=serve`: JSONL rollout server on stdin/stdout — the Python
//!   reference arm (`benchmarks/bcplus/run_python_gepa.py`) and every eval
//!   script drive rollouts through this, so all arms share byte-identical
//!   agent, retrieval, and judging.
//!   Request:  {"id":1,"instruction":null|"...","split":"val","idx":3}
//!   Response: {"id":1,"score":1.0,"correct":true,"answer":"...",
//!              "feedback":"...","tool_calls":4,"found_gold":1,"gold_total":2}
//!
//! The agent is a single tooled Predict (`search` + `open_doc` against the
//! local BM25 server, `benchmarks/bcplus/search_server.py`) — the optimized
//! surface is its instruction. The metric is the official BrowseComp judge
//! (fixed model), shaped for the optimizer: correct → 1.0, else
//! 0.3 × gold-doc recall from the trajectory (found-the-evidence ≠
//! never-found-it). Eval scripts report the binary `correct` only.
//!
//! Env: BCPLUS_DIR (benchmarks/bcplus), BCPLUS_MODE, BCPLUS_OPT,
//! BCPLUS_SEED, BCPLUS_SEED_PROMPT, BCPLUS_STUDENT, BCPLUS_JUDGE,
//! BCPLUS_ITERS, BCPLUS_MINIBATCH, BCPLUS_BUDGET, BCPLUS_CHAMPION,
//! BCPLUS_MAX_TURNS, BCPLUS_VAL_FILE, BCPLUS_OUT_PREFIX, BCPLUS_SEARCH,
//! BCPLUS_CONCURRENCY.
//!
//! Run from the repo root (search server must be up):
//!     set -a; source .env; set +a
//!     cd benchmarks/bcplus && .venv/bin/python search_server.py &
//!     BCPLUS_OPT=temper cargo run --release --example 31-bcplus

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dspy_rs::{
    AgentLoopSpec, Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Ratchet, Signature,
    TypedMetric, configure, init_tracing, temper,
};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

const DEFAULT_SEED_PROMPT: &str =
    "Answer the question. Use the search tool to find evidence in the corpus.";

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct BcRow {
    query_id: String,
    #[serde(alias = "query")]
    question: String,
    answer: String,
    gold_docids: Vec<String>,
}

fn load_rows(path: &str) -> Result<Vec<BcRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run fetch_data.py first, from the repo root"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<BcRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// Tools against the local BM25 search server
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ToolFailure(String);

impl std::fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ToolFailure {}

fn percent_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

async fn get_json(client: &reqwest::Client, url: &str, params: &[(&str, &str)]) -> Result<serde_json::Value, ToolFailure> {
    let query: Vec<String> = params
        .iter()
        .map(|(k, v)| format!("{k}={}", percent_encode(v)))
        .collect();
    let url = format!("{url}?{}", query.join("&"));
    let mut last = String::new();
    for attempt in 0..3u32 {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(1 << (attempt - 1))).await;
        }
        match client.get(&url).send().await {
            Ok(resp) => match resp.json::<serde_json::Value>().await {
                Ok(value) => return Ok(value),
                Err(e) => last = e.to_string(),
            },
            Err(e) => last = e.to_string(),
        }
    }
    Err(ToolFailure(format!("search service unreachable: {last}")))
}

#[derive(Debug, Deserialize, Serialize)]
struct SearchArgs {
    query: String,
}

#[derive(Clone)]
struct SearchTool {
    base: String,
    client: reqwest::Client,
}

impl Tool for SearchTool {
    const NAME: &'static str = "search";
    type Error = ToolFailure;
    type Args = SearchArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "BM25 keyword search over a fixed web-page corpus. Returns the \
                          top 5 documents with a 512-token snippet each. Results match \
                          keywords, not meaning — search distinctive terms."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "keyword search query" }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let url = format!("{}/search", self.base);
        let value = get_json(&self.client, &url, &[("q", args.query.as_str()), ("k", "5")]).await?;
        let results = value["results"].as_array().cloned().unwrap_or_default();
        if results.is_empty() {
            return Ok("no results".to_string());
        }
        // Students with tighter contexts (toast-1) overflow on 512-word
        // snippets by turn ~11; keep the cap env-tunable per arm matrix.
        let snippet_words = env_usize("BCPLUS_SNIPPET_WORDS", 512);
        let mut out = String::new();
        for r in results {
            let snippet = r["snippet"].as_str().unwrap_or("");
            let snippet: String = snippet
                .split_whitespace()
                .take(snippet_words)
                .collect::<Vec<_>>()
                .join(" ");
            out.push_str(&format!(
                "[docid:{}] {}\n{}\n\n",
                r["docid"].as_str().unwrap_or("?"),
                r["url"].as_str().unwrap_or(""),
                snippet,
            ));
        }
        Ok(out.trim_end().to_string())
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct SubmitAnswerArgs {
    answer: String,
}

/// Stop tool for students trained to act through tools (toast-1 never emits
/// a final text turn — it searches until the budget dies). Calling this ends
/// the loop; the args coerce directly into the typed output.
#[derive(Clone)]
struct SubmitAnswerTool;

impl Tool for SubmitAnswerTool {
    const NAME: &'static str = "final_answer";
    type Error = ToolFailure;
    type Args = SubmitAnswerArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Submit your final answer and finish. Call this exactly once, \
                          when you have verified the answer against the corpus."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string", "description": "the exact final answer, a few words" }
                },
                "required": ["answer"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.answer)
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct OpenDocArgs {
    docid: String,
}

#[derive(Clone)]
struct OpenDocTool {
    base: String,
    client: reqwest::Client,
}

impl Tool for OpenDocTool {
    const NAME: &'static str = "open_doc";
    type Error = ToolFailure;
    type Args = OpenDocArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Open a document by its docid (from search results) and read \
                          up to 4000 words of its full text."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "docid": { "type": "string", "description": "docid from a search result" }
                },
                "required": ["docid"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let url = format!("{}/doc", self.base);
        let value = get_json(&self.client, &url, &[("docid", args.docid.as_str())]).await?;
        if let Some(err) = value["error"].as_str() {
            return Ok(format!("error: {err}"));
        }
        Ok(format!(
            "[docid:{}] {}\n{}",
            value["docid"].as_str().unwrap_or("?"),
            value["url"].as_str().unwrap_or(""),
            value["text"].as_str().unwrap_or(""),
        ))
    }
}

// ---------------------------------------------------------------------------
// The agent under optimization
// ---------------------------------------------------------------------------

/// Answer the question by searching the corpus for evidence.
#[derive(Signature, Clone, Debug)]
struct BrowseQA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

struct Agent {
    answerer: Predict<BrowseQA>,
}

dspy_rs::predictors!(Agent { answerer });

impl Module for Agent {
    type Input = BrowseQAInput;
    type Output = BrowseQAOutput;

    async fn forward(&self, input: BrowseQAInput) -> Result<Predicted<BrowseQAOutput>, PredictError> {
        let mut last = None;
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(3u64 << (attempt - 1))).await;
            }
            match self.answerer.call(input.clone()).await {
                Ok(out) => return Ok(out),
                // A parse failure still carries the model's final text; the
                // BrowseComp judge extracts answers from free text, so grade
                // it instead of scoring 0. (Some students emit bare-string
                // stop-tool args or skip the field markers.)
                Err(PredictError::Parse { raw_response, .. }) if !raw_response.trim().is_empty() => {
                    return Ok(Predicted::new(
                        BrowseQAOutput { answer: raw_response.trim().to_string() },
                        dspy_rs::CallMetadata::default(),
                    ));
                }
                Err(e) => last = Some(e),
            }
        }
        eprintln!("rollout failed after retries, scoring 0: {:?}", last);
        Ok(Predicted::new(
            BrowseQAOutput { answer: String::new() },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

struct AgentFactory {
    student_lm: LM,
    search: SearchTool,
    open_doc: OpenDocTool,
    max_turns: u32,
}

impl AgentFactory {
    fn build(&self, instruction: &str) -> Agent {
        let stop_tool = env_str("BCPLUS_STOP_TOOL", "0") == "1";
        let builder = Predict::<BrowseQA>::builder()
            .named("bcplus_agent")
            .lm(self.student_lm.clone())
            .add_tool(self.search.clone())
            .add_tool(self.open_doc.clone());
        let builder = if stop_tool {
            builder.add_tool(SubmitAnswerTool).with_agent_spec(AgentLoopSpec {
                max_turns: Some(self.max_turns),
                stop_tools: vec!["final_answer".to_string()],
                ..Default::default()
            })
        } else {
            builder.with_agent_spec(AgentLoopSpec {
                max_turns: Some(self.max_turns),
                ..Default::default()
            })
        };
        Agent { answerer: builder.instruction(instruction.to_string()).build() }
    }
}

// ---------------------------------------------------------------------------
// Metric: official BrowseComp judge, shaped with gold-doc recall
// ---------------------------------------------------------------------------

/// Judge whether the response answers the question correctly, based only on
/// the precise and unambiguous gold answer. Extract the response's final
/// answer, then decide whether it matches the gold answer (within a small
/// margin of error for numerical answers). Focus only on whether the answers
/// match: do not attempt to solve the question, and do not argue for any
/// answer different from the gold answer. Any inconsistency, ambiguity, or
/// non-equivalency means incorrect.
#[derive(Signature, Clone, Debug)]
struct JudgeVerdict {
    #[input]
    question: String,

    #[input]
    gold_answer: String,

    #[input]
    response: String,

    /// The final exact answer extracted from the response ("None" if absent).
    #[output]
    extracted_final_answer: String,

    /// Why the extracted answer does or does not match the gold answer.
    #[output]
    reasoning: String,

    #[output]
    correct: bool,
}

struct BrowseMetric {
    judge: Predict<JudgeVerdict>,
}

/// Everything one rollout produced, for serve-mode responses.
struct Graded {
    score: f64,
    correct: bool,
    feedback: String,
    tool_calls: usize,
    found_gold: usize,
}

impl BrowseMetric {
    async fn grade(&self, example: &BcRow, prediction: &Predicted<BrowseQAOutput>) -> Graded {
        let answer = prediction.answer.trim().to_string();
        let meta = prediction.metadata();
        let tool_calls = meta.tool_calls.len();
        let found_gold = example
            .gold_docids
            .iter()
            .filter(|d| {
                let marker = format!("[docid:{d}]");
                meta.tool_executions.iter().any(|t| t.contains(&marker))
            })
            .count();
        let gold_total = example.gold_docids.len().max(1);
        let recall = found_gold as f64 / gold_total as f64;

        let (correct, verdict_text) = if answer.is_empty() {
            (false, "No final answer was produced.".to_string())
        } else {
            match self
                .judge
                .call(JudgeVerdictInput {
                    question: example.question.clone(),
                    gold_answer: example.answer.clone(),
                    response: answer.clone(),
                })
                .await
            {
                Ok(v) => (
                    v.correct,
                    format!("Judge extracted {:?}: {}", v.extracted_final_answer, v.reasoning),
                ),
                Err(e) => {
                    eprintln!("judge call failed, scoring incorrect: {e:?}");
                    (false, format!("Judge unavailable ({e}); scored incorrect."))
                }
            }
        };

        let score = if correct { 1.0 } else { 0.3 * recall };
        let evidence = format!(
            "Evidence: found {found_gold}/{gold_total} gold documents across {tool_calls} tool calls."
        );
        let feedback = if correct {
            format!("Correct. {verdict_text} {evidence}")
        } else {
            // Fallback-parsed rollouts carry no trajectory metadata; don't
            // fabricate a diagnosis from an empty record.
            let hint = if tool_calls == 0 && !answer.is_empty() {
                ""
            } else if found_gold == 0 {
                " The search never surfaced the evidence documents — the failure is search \
                 strategy (query choice, persistence, decomposition), not answer extraction."
            } else {
                " The evidence was retrieved but the final answer is wrong — the failure is \
                 reading/verification, not search."
            };
            format!(
                "Incorrect. {verdict_text} Gold answer: {:?}. {evidence}{hint}",
                example.answer
            )
        };
        Graded { score, correct, feedback, tool_calls, found_gold }
    }
}

impl TypedMetric<BcRow, Agent> for BrowseMetric {
    async fn evaluate(
        &self,
        example: &BcRow,
        prediction: &Predicted<BrowseQAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let graded = self.grade(example, prediction).await;
        Ok(Eval::with_feedback(graded.score, graded.feedback))
    }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    instruction: Option<String>,
    split: String,
    idx: usize,
}

#[derive(Serialize)]
struct ServeResponse {
    id: u64,
    score: f64,
    correct: bool,
    answer: String,
    feedback: String,
    tool_calls: usize,
    found_gold: usize,
    gold_total: usize,
}

async fn serve(dir: &str, factory: Arc<AgentFactory>, metric: Arc<BrowseMetric>, seed_prompt: String) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    // Every data/*.jsonl except the corpus is addressable as a split, so
    // eval scripts can target paraphrase-variant files without code changes.
    let mut splits: HashMap<String, Vec<BcRow>> = HashMap::new();
    for entry in std::fs::read_dir(format!("{dir}/data"))? {
        let path = entry?.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") || name == "corpus" {
            continue;
        }
        splits.insert(name.to_string(), load_rows(&path.to_string_lossy())?);
    }
    eprintln!(
        "serve: splits loaded: {:?}",
        splits.iter().map(|(k, v)| (k.clone(), v.len())).collect::<Vec<_>>()
    );

    let concurrency = env_usize("BCPLUS_CONCURRENCY", 8);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let stdout = Arc::new(tokio::sync::Mutex::new(tokio::io::stdout()));
    let splits = Arc::new(splits);
    let seed_prompt = Arc::new(seed_prompt);

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
        let (factory, metric, splits, stdout, semaphore, seed_prompt) = (
            factory.clone(),
            metric.clone(),
            splits.clone(),
            stdout.clone(),
            semaphore.clone(),
            seed_prompt.clone(),
        );
        tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("semaphore open");
            let Some(row) = splits.get(&request.split).and_then(|s| s.get(request.idx)) else {
                eprintln!("unknown split/idx: {}/{}", request.split, request.idx);
                return;
            };
            let instruction = request.instruction.as_deref().unwrap_or(&seed_prompt);
            let agent = factory.build(instruction);
            let prediction = agent
                .forward(BrowseQAInput { question: row.question.clone() })
                .await
                .expect("forward returns an empty fallback, never Err");
            let graded = metric.grade(row, &prediction).await;
            let response = ServeResponse {
                id: request.id,
                score: graded.score,
                correct: graded.correct,
                answer: prediction.answer.trim().to_string(),
                feedback: graded.feedback,
                tool_calls: graded.tool_calls,
                found_gold: graded.found_gold,
                gold_total: row.gold_docids.len(),
            };
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

#[allow(clippy::too_many_arguments)]
async fn optimize(
    dir: &str,
    factory: &AgentFactory,
    metric: &BrowseMetric,
    seed_prompt: &str,
) -> Result<()> {
    let opt_name = env_str("BCPLUS_OPT", "temper");
    let train = load_rows(&format!("{dir}/data/train.jsonl"))?;
    let val_file = env_str("BCPLUS_VAL_FILE", "val.jsonl");
    let val = load_rows(&format!("{dir}/data/{val_file}"))?;

    // TEMPER T3: variants are appended to the trainset with an index map;
    // the optimizer keeps them out of every batch and races the champion on
    // them. Other arms never see them.
    let variants_path = format!("{dir}/data/variants.jsonl");
    let (trainset, variant_pairs) = if (opt_name == "temper" || opt_name == "hybrid")
        && std::path::Path::new(&variants_path).exists()
    {
        let variants = load_rows(&variants_path)?;
        let mut rows = train.clone();
        let mut pairs = Vec::new();
        for variant in variants {
            let Some(source) = train.iter().position(|r| r.query_id == variant.query_id) else {
                eprintln!("variant {} has no train source; skipping", variant.query_id);
                continue;
            };
            pairs.push((source, rows.len()));
            rows.push(variant);
        }
        eprintln!("temper: {} variant pairs appended to trainset", pairs.len());
        (rows, pairs)
    } else {
        (train, Vec::new())
    };
    println!(
        "{dir}: {} train (+variants) / {} val, arm {opt_name}",
        trainset.len(),
        val.len()
    );

    let reflection_lm = LM::builder()
        .model(env_str("BCPLUS_TEACHER", "openrouter:openai/gpt-5.6-sol"))
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;

    let seed = env_usize("BCPLUS_SEED", 0) as u64;
    let budget = env_usize("BCPLUS_BUDGET", 600);
    let concurrency = env_usize("BCPLUS_CONCURRENCY", 8);
    let mut agent = factory.build(seed_prompt);

    let prefix = env_str("BCPLUS_OUT_PREFIX", &format!("{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.txt");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.txt");

    if opt_name == "gepa" {
        let iters = env_usize("BCPLUS_ITERS", 40);
        let minibatch = env_usize("BCPLUS_MINIBATCH", 4);
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
            .compile_module_with_valset(&mut agent, &trainset, Some(&val), metric)
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
            "\nchosen (gen {}, val avg {:.3}):\n---\n{}\n---",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.instruction
        );
        std::fs::write(&out, &result.best_candidate.instruction)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| c.instruction.as_str())
            .unwrap_or(&result.best_candidate.instruction);
        std::fs::write(&out_argmax, argmax)?;
    } else {
        let iters = env_usize("BCPLUS_ITERS", 60);
        let minibatch = env_usize("BCPLUS_MINIBATCH", 5);
        let champion = env_usize("BCPLUS_CHAMPION", 30);
        println!("{opt_name} seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, champion {champion}");
        let optimizer = match opt_name.as_str() {
            "temper" => {
                let mut optimizer = temper(variant_pairs);
                optimizer.prompt_model = Some(reflection_lm);
                optimizer
            }
            // HYBRID: RATCHET's search (Pareto parents, epoch batches, no
            // length cap) + TEMPER's variant-tempered champion endgame.
            "hybrid" => Ratchet::builder()
                .variant_pairs(variant_pairs)
                .prompt_model(reflection_lm)
                .build(),
            "ratchet" => Ratchet::builder().prompt_model(reflection_lm).build(),
            other => bail!("unknown BCPLUS_OPT {other:?} (temper|ratchet|hybrid|gepa)"),
        };
        let optimizer = {
            let mut optimizer = optimizer;
            optimizer.num_iterations = iters;
            optimizer.minibatch_size = minibatch;
            optimizer.champion_fresh = champion;
            optimizer.max_rollouts = Some(budget);
            optimizer.seed = Some(seed);
            optimizer.eval_concurrency = concurrency;
            optimizer.track_stats = true;
            optimizer
        };
        let result = optimizer
            .compile_module_with_valset(&mut agent, &trainset, Some(&val), metric)
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
        println!(
            "operator stats: {:?}  fresh cols: {}  variant cols: {}",
            result.operator_stats, result.fresh_selection_cols, result.variant_selection_cols
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
            "\nchosen (gen {}, val avg {:.3}):\n---\n{}\n---",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.instruction
        );
        std::fs::write(&out, &result.best_candidate.instruction)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| c.instruction.as_str())
            .unwrap_or(&result.best_candidate.instruction);
        std::fs::write(&out_argmax, argmax)?;
    }
    println!("\nwinner written to {out}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = env_str("BCPLUS_MODE", "optimize");
    if mode != "serve" {
        init_tracing()?;
    }
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let dir = env_str("BCPLUS_DIR", "benchmarks/bcplus");
    let search_base = env_str("BCPLUS_SEARCH", "http://127.0.0.1:8123");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let health = client
        .get(format!("{search_base}/health"))
        .send()
        .await
        .map_err(|e| anyhow!("search server not reachable at {search_base}: {e} — start benchmarks/bcplus/search_server.py"))?;
    if !health.status().is_success() {
        bail!("search server unhealthy at {search_base}");
    }

    // Student chosen by the Phase-A pilot (docs/temper-design.md §5):
    // gpt-oss-120b always engages the tools (0/15 zero-tool rollouts) and
    // sits in the 10-35% seed-accuracy headroom band; llama-4-scout and
    // qwen3-30b-a3b skip tools on most rollouts. "mxbai:<model>" routes to
    // the Mixedbread OpenAI-compatible endpoint (e.g. mxbai:toast-1), which
    // supports client-side function calling.
    let student_spec = env_str("BCPLUS_STUDENT", "openai/gpt-oss-120b");
    let student_lm = if let Some(model) = student_spec.strip_prefix("mxbai:") {
        LM::builder()
            .base_url(env_str("BCPLUS_MXBAI_URL", "https://api.mixedbread.com/v1"))
            .api_key(std::env::var("MXBAI_API_KEY").context("MXBAI_API_KEY not set")?)
            .model(model.to_string())
            .temperature(0.2)
            .max_tokens(1024)
            .build()
            .await?
    } else {
        LM::builder()
            .model(format!("openrouter:{student_spec}"))
            .temperature(0.2)
            .max_tokens(1024)
            .build()
            .await?
    };
    let judge_lm = LM::builder()
        .model(env_str("BCPLUS_JUDGE", "openrouter:openai/gpt-5.6-sol"))
        .temperature(0.0)
        .max_tokens(2000)
        .build()
        .await?;
    // Global LM only renders reflection prompts; scoring always goes through
    // the leaf/judge LMs above.
    configure(judge_lm.clone());

    let factory = AgentFactory {
        student_lm,
        search: SearchTool { base: search_base.clone(), client: client.clone() },
        open_doc: OpenDocTool { base: search_base.clone(), client },
        max_turns: env_usize("BCPLUS_MAX_TURNS", 12) as u32,
    };
    let metric = BrowseMetric {
        judge: Predict::<JudgeVerdict>::builder().lm(judge_lm).build(),
    };
    let seed_prompt = env_str("BCPLUS_SEED_PROMPT", DEFAULT_SEED_PROMPT);

    match mode.as_str() {
        "serve" => serve(&dir, Arc::new(factory), Arc::new(metric), seed_prompt).await,
        "optimize" => optimize(&dir, &factory, &metric, &seed_prompt).await,
        other => bail!("unknown BCPLUS_MODE {other:?} (optimize|serve)"),
    }
}
