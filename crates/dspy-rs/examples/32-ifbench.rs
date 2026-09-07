//! IFBench optimizer rig (instruction following with verifiable constraints;
//! mirrors the BrowseComp-Plus rig, docs/handoff-temper-bcplus-campaign.md).
//!
//! One binary, two modes:
//!
//! - `IFB_MODE=optimize` (default): optimize the responder's instruction with
//!   `IFB_OPT=temper|ratchet|hybrid|harvest|gepa` and write the winner
//!   (`hybrid` = RATCHET search + TEMPER's variant-champion endgame;
//!   `harvest` = RATCHET v6, docs/ratchet-v6-harvest-design.md, with
//!   sequential paired certification vs the seed on reserved train columns).
//! - `IFB_MODE=serve`: JSONL rollout server on stdin/stdout — the Python
//!   reference arm (`benchmarks/ifbench/run_python_gepa.py`) and every eval
//!   script drive rollouts through this, so all arms share byte-identical
//!   student calls and checker scoring.
//!   Request:  {"id":1,"instruction":null|"...","split":"val","idx":3}
//!   Response: {"id":1,"score":0.5,"correct":false,"answer":"...",
//!              "feedback":"...","n_constraints":2,"n_passed":1}
//!
//! The task is a single untooled Predict (`prompt -> response`) — the
//! optimized surface is its instruction. The metric is the local IFBench
//! checker server (`benchmarks/ifbench/checker_server.py`, deterministic,
//! judge-free, zero API cost): score = frac_pass over the row's constraints,
//! correct = all pass; feedback lists the failed constraint ids with their
//! canonical descriptions (the surface reflective optimizers mutate against).
//!
//! `IFB_STOP_TOOL=1` ports the bcplus stop-tool pattern for students trained
//! to act through tools (mxbai toast-1 never emits a final text turn): a
//! `final_answer` tool becomes the response channel, and a raw-text parse
//! fallback still grades students that emit bare-string tool args.
//!
//! Env: IFB_DIR (benchmarks/ifbench), IFB_MODE, IFB_OPT, IFB_SEED,
//! IFB_SEED_PROMPT, IFB_STUDENT, IFB_TEACHER, IFB_ITERS, IFB_MINIBATCH,
//! IFB_BUDGET, IFB_CHAMPION, IFB_VAL_FILE, IFB_OUT_PREFIX, IFB_CONCURRENCY,
//! IFB_CHECKER, IFB_STOP_TOOL, IFB_MAX_TURNS, IFB_MAX_TOKENS,
//! IFB_CERTIFY_RESERVE (45), IFB_CERTIFY_BUDGET (150).
//! HARVEST ablation knobs (unset = preset; mirror of 33-pupa.rs): IFB_SCREEN (a|b),
//! IFB_REWRITE_ONLY, IFB_POLICY, IFB_POLICY_SHARE, IFB_LEDGER_GATE_KILLS,
//! IFB_PARENT_GREED, IFB_GATE_SIZE, IFB_SCOUT, IFB_PARENTS, IFB_PROPOSALS,
//! IFB_CHAMPION_K, IFB_LENGTH_PRIOR (chars; soft teacher budget, hard guard ×
//! IFB_LENGTH_SLACK 1.25).
//!
//! Run from the repo root (checker server must be up):
//!     set -a; source .env; set +a
//!     cd benchmarks/ifbench && .venv/bin/python checker_server.py &
//!     IFB_OPT=temper cargo run --release --example 32-ifbench

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dspy_rs::{
    AgentLoopSpec, Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Ratchet, Signature,
    TypedMetric, configure, harvest, init_tracing, temper,
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

const DEFAULT_SEED_PROMPT: &str = "Follow the user's instructions exactly.";

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct IfRow {
    id: String,
    prompt: String,
    instruction_id_list: Vec<String>,
    kwargs: Vec<serde_json::Value>,
}

fn load_rows(path: &str) -> Result<Vec<IfRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run fetch_data.py first, from the repo root"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<IfRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// Stop tool (IFB_STOP_TOOL=1): the response channel for students trained to
// act through tools — toast-1 never emits a final text turn.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ToolFailure(String);

impl std::fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ToolFailure {}

#[derive(Debug, Deserialize, Serialize)]
struct FinalAnswerArgs {
    response: String,
}

/// Calling this ends the loop; the args coerce directly into the typed output.
#[derive(Clone)]
struct FinalAnswerTool;

impl Tool for FinalAnswerTool {
    const NAME: &'static str = "final_answer";
    type Error = ToolFailure;
    type Args = FinalAnswerArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Submit your final response and finish. Call this exactly once, \
                          with the complete response that satisfies every constraint."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "response": { "type": "string", "description": "the complete final response" }
                },
                "required": ["response"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.response)
    }
}

// ---------------------------------------------------------------------------
// The responder under optimization
// ---------------------------------------------------------------------------

/// Follow the user's instructions exactly.
#[derive(Signature, Clone, Debug)]
struct IfTask {
    #[input]
    prompt: String,

    #[output]
    response: String,
}

struct Agent {
    responder: Predict<IfTask>,
}

dspy_rs::predictors!(Agent { responder });

impl Module for Agent {
    type Input = IfTaskInput;
    type Output = IfTaskOutput;

    async fn forward(&self, input: IfTaskInput) -> Result<Predicted<IfTaskOutput>, PredictError> {
        let mut last = None;
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(3u64 << (attempt - 1))).await;
            }
            match self.responder.call(input.clone()).await {
                Ok(out) => return Ok(out),
                // A parse failure still carries the model's final text; the
                // constraint checkers grade free text, so grade it instead of
                // scoring 0. (Some students emit bare-string stop-tool args
                // or skip the field markers.)
                Err(PredictError::Parse { raw_response, .. }) if !raw_response.trim().is_empty() => {
                    return Ok(Predicted::new(
                        IfTaskOutput { response: raw_response.trim().to_string() },
                        dspy_rs::CallMetadata::default(),
                    ));
                }
                Err(e) => last = Some(e),
            }
        }
        eprintln!("rollout failed after retries, scoring 0: {:?}", last);
        Ok(Predicted::new(
            IfTaskOutput { response: String::new() },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

struct AgentFactory {
    student_lm: LM,
    max_turns: u32,
}

impl AgentFactory {
    fn build(&self, instruction: &str) -> Agent {
        let stop_tool = env_str("IFB_STOP_TOOL", "0") == "1";
        let builder = Predict::<IfTask>::builder()
            .named("ifbench_agent")
            .lm(self.student_lm.clone());
        let builder = if stop_tool {
            builder.add_tool(FinalAnswerTool).with_agent_spec(AgentLoopSpec {
                max_turns: Some(self.max_turns),
                stop_tools: vec!["final_answer".to_string()],
                ..Default::default()
            })
        } else {
            builder
        };
        Agent { responder: builder.instruction(instruction.to_string()).build() }
    }
}

// ---------------------------------------------------------------------------
// Metric: the local deterministic checker server (judge-free)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CheckResult {
    per_constraint: Vec<bool>,
    follow_all_instructions_loose: bool,
    failed: Vec<String>,
    failed_descriptions: Vec<String>,
}

struct CheckerMetric {
    base: String,
    client: reqwest::Client,
}

/// Everything one rollout produced, for serve-mode responses.
struct Graded {
    score: f64,
    correct: bool,
    feedback: String,
    n_constraints: usize,
    n_passed: usize,
    answer: String,
}

/// open-instruct's remove_thinking_section: reasoning students wrap the real
/// response in think/answer tags; the checkers must grade what follows them.
fn strip_thinking(response: &str) -> String {
    let response = response.replace("<|assistant|>", "");
    let response = response.rsplit("</think>").next().unwrap_or(&response);
    response.replace("<answer>", "").replace("</answer>", "").trim().to_string()
}

impl CheckerMetric {
    async fn grade(&self, example: &IfRow, prediction: &Predicted<IfTaskOutput>) -> Graded {
        let answer = strip_thinking(&prediction.response);
        let n_constraints = example.instruction_id_list.len();
        let body = serde_json::json!({
            "prompt": example.prompt,
            "response": answer,
            "instruction_ids": example.instruction_id_list,
            "kwargs": example.kwargs,
        });
        let url = format!("{}/check", self.base);
        let mut last = String::new();
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(1 << (attempt - 1))).await;
            }
            let sent = self.client.post(&url).json(&body).send().await;
            match sent {
                Ok(resp) => match resp.json::<CheckResult>().await {
                    Ok(result) => {
                        let n_passed = result.per_constraint.iter().filter(|b| **b).count();
                        let correct = result.follow_all_instructions_loose;
                        let feedback = if correct {
                            format!("Correct. All {n_constraints} constraints satisfied.")
                        } else {
                            let failures = result
                                .failed
                                .iter()
                                .zip(result.failed_descriptions.iter())
                                .map(|(id, desc)| format!("- {id}: {desc}"))
                                .collect::<Vec<_>>()
                                .join("\n");
                            format!(
                                "Failed {} of {n_constraints} constraints:\n{failures}",
                                result.failed.len()
                            )
                        };
                        return Graded {
                            score: if correct { 1.0 } else { 0.0 },
                            correct,
                            feedback,
                            n_constraints,
                            n_passed,
                            answer,
                        };
                    }
                    Err(e) => last = e.to_string(),
                },
                Err(e) => last = e.to_string(),
            }
        }
        eprintln!("checker unreachable, scoring 0: {last}");
        Graded {
            score: 0.0,
            correct: false,
            feedback: format!("Checker server unreachable ({last}); scored 0."),
            n_constraints,
            n_passed: 0,
            answer,
        }
    }
}

impl TypedMetric<IfRow, Agent> for CheckerMetric {
    async fn evaluate(
        &self,
        example: &IfRow,
        prediction: &Predicted<IfTaskOutput>,
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
    metric_protocol: &'static str,
    follow_all_instructions_loose: bool,
    id: u64,
    score: f64,
    correct: bool,
    answer: String,
    feedback: String,
    n_constraints: usize,
    n_passed: usize,
}

async fn serve(dir: &str, factory: Arc<AgentFactory>, metric: Arc<CheckerMetric>, seed_prompt: String) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    // Every data/*.jsonl is addressable as a split, so eval scripts can
    // target paraphrase-variant files without code changes.
    let mut splits: HashMap<String, Vec<IfRow>> = HashMap::new();
    for entry in std::fs::read_dir(format!("{dir}/data"))? {
        let path = entry?.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        splits.insert(name.to_string(), load_rows(&path.to_string_lossy())?);
    }
    eprintln!(
        "serve: splits loaded: {:?}",
        splits.iter().map(|(k, v)| (k.clone(), v.len())).collect::<Vec<_>>()
    );

    let concurrency = env_usize("IFB_CONCURRENCY", 8);
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
                .forward(IfTaskInput { prompt: row.prompt.clone() })
                .await
                .expect("forward returns an empty fallback, never Err");
            let graded = metric.grade(row, &prediction).await;
            let response = ServeResponse {
                metric_protocol: "dataset-native-v1",
                follow_all_instructions_loose: graded.correct,
                id: request.id,
                score: graded.score,
                correct: graded.correct,
                answer: graded.answer,
                feedback: graded.feedback,
                n_constraints: graded.n_constraints,
                n_passed: graded.n_passed,
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

async fn optimize(
    dir: &str,
    factory: &AgentFactory,
    metric: &CheckerMetric,
    seed_prompt: &str,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("IFB_OPT", "temper");
    let train = load_rows(&format!("{dir}/data/train.jsonl"))?;
    let val_file = env_str("IFB_VAL_FILE", "val.jsonl");
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
            let Some(source) = train.iter().position(|r| r.id == variant.id) else {
                eprintln!("variant {} has no train source; skipping", variant.id);
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

    let seed = env_usize("IFB_SEED", 0) as u64;
    let budget = env_usize("IFB_BUDGET", 600);
    let concurrency = env_usize("IFB_CONCURRENCY", 8);
    let mut agent = factory.build(seed_prompt);

    let prefix = env_str("IFB_OUT_PREFIX", &format!("{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.txt");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.txt");

    if opt_name == "gepa" {
        let iters = env_usize("IFB_ITERS", 40);
        let minibatch = env_usize("IFB_MINIBATCH", 4);
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
        // HARVEST keeps its preset cycle shape (iters 150 / minibatch 5 /
        // P×N 2×2) unless overridden; the v2-era arms keep theirs.
        let is_harvest = matches!(opt_name.as_str(), "harvest" | "harvest_search");
        let iters = env_usize("IFB_ITERS", if is_harvest { 150 } else { 60 });
        let minibatch = env_usize("IFB_MINIBATCH", 5);
        let champion = env_usize("IFB_CHAMPION", if opt_name == "harvest_search" { 0 } else { 30 });
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
            // HARVEST (RATCHET v6, docs/ratchet-v6-harvest-design.md): on-target
            // screen, full-val ranking, Rewrite + Policy operators, sequential
            // paired certification vs the seed on reserved train columns.
            // train150 carve: champion 30 + certify 45 leaves 75 batch columns
            // (the reserve is carved inside the optimizer; no driver-side split).
            "harvest" | "harvest_search" => {
                let mut optimizer = if opt_name == "harvest_search" {
                    dspy_rs::harvest_search()
                } else { harvest() };
                optimizer.prompt_model = Some(reflection_lm);
                optimizer.certify_reserve = env_usize("IFB_CERTIFY_RESERVE", 45);
                optimizer.certify_budget = env_usize("IFB_CERTIFY_BUDGET", 150);
                // One-factor ablation knobs (mirror of 33-pupa.rs): each
                // overrides the `harvest()` preset only when set; unset keeps
                // the preset value.
                match env_str("IFB_SCREEN", "").as_str() {
                    "a" => optimizer.screen_rule = dspy_rs::ScreenRule::OnTargetA,
                    "b" => optimizer.screen_rule = dspy_rs::ScreenRule::HeldOutB,
                    _ => {}
                }
                if let Ok(v) = std::env::var("IFB_REWRITE_ONLY") {
                    optimizer.rewrite_only = v == "1";
                }
                if let Ok(v) = std::env::var("IFB_POLICY") {
                    optimizer.policy_channel = v == "1";
                }
                if let Ok(v) = std::env::var("IFB_POLICY_SHARE") {
                    optimizer.policy_share = v.parse().unwrap_or(optimizer.policy_share);
                }
                if let Ok(v) = std::env::var("IFB_LEDGER_GATE_KILLS") {
                    optimizer.ledger_gate_kills = v == "1";
                }
                if let Ok(v) = std::env::var("IFB_PARENT_GREED") {
                    optimizer.parent_greed = v.parse().unwrap_or(optimizer.parent_greed);
                }
                if let Ok(v) = std::env::var("IFB_GATE_SIZE") {
                    optimizer.gate_size = v.parse().ok();
                }
                // v6.5 length prior (soft, teacher-facing; unset = off):
                // IFB_LENGTH_PRIOR=N chars, hard guard N × IFB_LENGTH_SLACK
                // (default 1.25) through the length_cap path.
                if let Ok(v) = std::env::var("IFB_LENGTH_PRIOR") {
                    optimizer.length_prior = v.parse().ok().filter(|n: &usize| *n > 0);
                }
                if let Ok(v) = std::env::var("IFB_LENGTH_SLACK") {
                    optimizer.length_prior_slack =
                        v.parse().unwrap_or(optimizer.length_prior_slack);
                }
                if optimizer.length_prior.is_some() {
                    println!(
                        "length prior {:?} chars (slack {}, hard cap {:?})",
                        optimizer.length_prior,
                        optimizer.length_prior_slack,
                        optimizer.effective_length_cap()
                    );
                }
                optimizer.scout_size = env_usize("IFB_SCOUT", optimizer.scout_size);
                optimizer.parents_per_step = env_usize("IFB_PARENTS", optimizer.parents_per_step);
                optimizer.proposals_per_parent =
                    env_usize("IFB_PROPOSALS", optimizer.proposals_per_parent);
                optimizer.champion_k = env_usize("IFB_CHAMPION_K", optimizer.champion_k);
                optimizer
            }
            other => bail!("unknown IFB_OPT {other:?} (temper|ratchet|hybrid|harvest|gepa)"),
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
            if matches!(opt_name.as_str(), "harvest" | "harvest_search") {
                println!(
                    "harvest: screen {:?} accept {:?} scout {} rewrite_only {} policy {} (share {}) \
                     ledger_gate_kills {} greed {} gate_size {:?} champion_k {} \
                     certify reserve {} / budget {} alpha {} iters {} minibatch {} P×N {}×{}",
                    optimizer.screen_rule,
                    optimizer.accept_mode,
                    optimizer.scout_size,
                    optimizer.rewrite_only,
                    optimizer.policy_channel,
                    optimizer.policy_share,
                    optimizer.ledger_gate_kills,
                    optimizer.parent_greed,
                    optimizer.gate_size,
                    optimizer.champion_k,
                    optimizer.certify_reserve,
                    optimizer.certify_budget,
                    optimizer.certify_alpha,
                    optimizer.num_iterations,
                    optimizer.minibatch_size,
                    optimizer.parents_per_step,
                    optimizer.proposals_per_parent
                );
            }
            optimizer
        };
        let result = optimizer
            .compile_module_with_valset(&mut agent, &trainset, Some(&val), metric)
            .await?;
        if let Ok(path) = std::env::var("IFB_REPORT") {
            std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
        }
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
        println!(
            "certification: {:?} cols {} mean {:+.4} e {:.2}",
            result.certified, result.certify_cols_used, result.certify_mean_delta, result.certify_e_value
        );
        if result.certify_cols_used > 0 || result.certified.is_some() {
            println!(
                "CERTIFICATION: {:?} after {} cols, mean paired delta {:+.4}, e-value {:.2}",
                result.certified,
                result.certify_cols_used,
                result.certify_mean_delta,
                result.certify_e_value
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
    let mode = env_str("IFB_MODE", "optimize");
    if mode != "serve" {
        init_tracing()?;
    }
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let dir = env_str("IFB_DIR", "benchmarks/ifbench");
    let checker_base = env_str("IFB_CHECKER", "http://127.0.0.1:8124");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let health = client
        .get(format!("{checker_base}/health"))
        .send()
        .await
        .map_err(|e| anyhow!("checker server not reachable at {checker_base}: {e} — start benchmarks/ifbench/checker_server.py"))?;
    if !health.status().is_success() {
        bail!("checker server unhealthy at {checker_base}");
    }

    // "mxbai:<model>" routes to the Mixedbread OpenAI-compatible endpoint
    // (e.g. mxbai:toast-1, which needs IFB_STOP_TOOL=1); anything else goes
    // through OpenRouter. max_tokens is generous because length constraints
    // in the train pool reach hundreds of words.
    let student_spec = env_str("IFB_STUDENT", "openai/gpt-oss-120b");
    let max_tokens = env_usize("IFB_MAX_TOKENS", 2048) as u32;
    let student_lm = if let Ok(base_url) = std::env::var("IFB_STUDENT_BASE_URL") {
        LM::builder()
            .base_url(base_url)
            .api_key("local-benchmark-gateway".to_string())
            .model(student_spec.clone())
            .temperature(0.2)
            .max_tokens(max_tokens)
            .max_retries(0)
            .request_timeout_secs(130)
            .build().await?
    } else if let Some(model) = student_spec.strip_prefix("mxbai:") {
        LM::builder()
            .base_url(env_str("IFB_MXBAI_URL", "https://api.mixedbread.com/v1"))
            .api_key(std::env::var("MXBAI_API_KEY").context("MXBAI_API_KEY not set")?)
            .model(model.to_string())
            .temperature(0.2)
            .max_tokens(max_tokens)
            .build()
            .await?
    } else {
        LM::builder()
            .model(format!("openrouter:{student_spec}"))
            .temperature(0.2)
            .max_tokens(max_tokens)
            .build()
            .await?
    };
    let reflection_lm = LM::builder()
        .model(env_str("IFB_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;
    // Global LM only renders reflection prompts; scoring never touches an LM
    // (the checker server is the whole metric).
    configure(reflection_lm.clone());

    let factory = AgentFactory {
        student_lm,
        max_turns: env_usize("IFB_MAX_TURNS", 4) as u32,
    };
    let metric = CheckerMetric { base: checker_base, client };
    let seed_prompt = env_str("IFB_SEED_PROMPT", DEFAULT_SEED_PROMPT);

    match mode.as_str() {
        "serve" => serve(&dir, Arc::new(factory), Arc::new(metric), seed_prompt).await,
        "optimize" => optimize(&dir, &factory, &metric, &seed_prompt, reflection_lm).await,
        other => bail!("unknown IFB_MODE {other:?} (optimize|serve)"),
    }
}
