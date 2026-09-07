//! SpreadsheetBench Verified-400 rig (Rig 5, docs/rigs-spreadsheet-automation-design.md)
//! — an EXTERNAL-HARNESS rig on the mxagent seam (examples/35-mxagent.rs):
//! the program under optimization is a four-stage spreadsheet-manipulation
//! harness (inspect → plan → code → docker exec + LibreOffice recalc →
//! verify with one repair round → upstream-faithful scorer) run
//! out-of-process by `benchmarks/spreadsheet/sidecar.py`.
//!
//! How the wrap works:
//! - The module holds one named `Predict` per prompt slot (`inspect`,
//!   `plan`, `code`, `verify`). They are **never called** — `predictors!`
//!   registers them as the optimizer's slots and each carries its seed text
//!   (upstream's single-round prompt split into stages, fetched from the
//!   sidecar's `GET /slots` at startup and snapshotted).
//! - `forward` resolves each slot's effective text itself:
//!   `fx::current_instruction(name)` (the ambient candidate during
//!   rollouts) → the serve-request override → the seed. Only slots that
//!   differ from seed are POSTed, so the seed candidate runs the harness's
//!   byte-identical production prompt.
//! - The sidecar runs one episode with the student (gpt-oss-120b via
//!   openrouter, temperature 0.2) and returns the scorer verdict plus a
//!   teacher-readable trace. Feedback never carries golden cell values
//!   (leak hygiene: address + produced value + golden type/emptiness only).
//!
//! Metric: pass → 1.0; produced-and-executed but wrong → 0.3; else 0.
//! Paper metric: pass rate (`passed`).
//!
//! Modes:
//! - `SB_MODE=optimize` (default): `SB_OPT=harvest|ratchet|gepa`; winner
//!   written as a JSON slot map {"inspect": ..., "plan": ..., "code": ...,
//!   "verify": ...} to `<SB_OUT_PREFIX>_s<seed>.json` (+ `_argmax_`).
//! - `SB_MODE=serve`: JSONL rollout server on stdin/stdout (36-docsqa
//!   pattern) so pip-gepa and every eval drive byte-identical episodes.
//!   Request:  {"id":1,"instruction":{"inspect":"..."|null,...}|null,
//!              "split":"val","idx":3}
//!   Response: {"id":1,"score":1.0,"passed":true,"answer":"<program>",
//!              "feedback":"...","stage":"done","error":null}
//!
//! Env (design "Shared conventions"): SB_DIR (benchmarks/spreadsheet),
//! SB_MODE, SB_OPT, SB_STUDENT (openai/gpt-oss-120b), SB_TEACHER
//! (openrouter:openai/gpt-5.6-terra), SB_SEED, SB_BUDGET (600),
//! SB_CONCURRENCY (6), SB_SIDECAR (http://127.0.0.1:8140), SB_TRAIN_FILE,
//! SB_VAL_FILE, SB_OUT_PREFIX, SB_CERTIFY_RESERVE (45), SB_CERTIFY_BUDGET
//! (150), SB_ITERS, SB_MINIBATCH, SB_CHAMPION, SB_TEMPERATURE (0.2),
//! SB_MAX_TOKENS (8000).
//! One-factor HARVEST knobs (unset = preset): SB_SCREEN (a|b),
//! SB_REWRITE_ONLY, SB_POLICY, SB_POLICY_SHARE, SB_LEDGER_GATE_KILLS,
//! SB_PARENT_GREED, SB_GATE_SIZE, SB_SCOUT, SB_PARENTS, SB_PROPOSALS,
//! SB_CHAMPION_K, SB_LENGTH_PRIOR (chars; hard guard × SB_LENGTH_SLACK 1.25).
//! Leak guard is ON (benchmark semantics; no allow_memorization switch).
//!
//! Run from the repo root (sidecar must be up, see sidecar.py header):
//!     set -a; source .env; set +a
//!     SB_OPT=harvest SB_BUDGET=600 cargo run --release --example 39-spreadsheet

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dspy_rs::{
    Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Predictors, Ratchet, Signature,
    TypedMetric, configure, fx, harvest, init_tracing,
};
use serde::{Deserialize, Serialize};

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// The four slots in harness order — predictor names, sidecar slot keys,
/// serve-override keys and winner-JSON keys. Order is the round-robin
/// mutation order. Must match `predictors!` below and the sidecar's SEEDS.
const SLOTS: [&str; 4] = ["inspect", "plan", "code", "verify"];

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct SbRow {
    id: String,
    instruction: String,
    instruction_type: String,
    answer_position: String,
    #[serde(default)]
    answer_sheet: Option<String>,
    #[serde(default)]
    data_position: Option<String>,
    #[serde(default)]
    folder: String,
}

fn load_rows(path: &str) -> Result<Vec<SbRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run make_splits.py first"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<SbRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// Slot signature (parameter holder — never called as an LM)
// ---------------------------------------------------------------------------

/// One stage's system prompt in the four-stage spreadsheet harness. The
/// instruction IS the stage's prompt text; the episode shows what the whole
/// harness did with it.
#[derive(Signature, Clone, Debug)]
#[allow(dead_code)] // parameter holder: fields are never read, the slot text is what matters
struct HarnessStage {
    /// The spreadsheet manipulation instruction, verbatim.
    #[input]
    instruction: String,

    /// The finished episode this stage prompt took part in: the ranges
    /// previewed, the plan, the program, its execution log, the verify
    /// verdict / repair, and the scorer's verdict (first mismatching cell
    /// address, produced value, golden type — never the golden value).
    #[output]
    episode: String,
}

// ---------------------------------------------------------------------------
// Sidecar client
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SlotsResponse {
    slots: BTreeMap<String, String>,
    #[serde(default)]
    config: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct EpisodeResponse {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    score: f64,
    #[serde(default)]
    passed: bool,
    #[serde(default)]
    produced: bool,
    #[serde(default)]
    executed: bool,
    #[serde(default)]
    feedback: String,
    #[serde(default)]
    stage: String,
    #[serde(default)]
    in_tokens: u64,
    #[serde(default)]
    out_tokens: u64,
    #[serde(default)]
    trace_summary: String,
    #[serde(default)]
    program: String,
    #[serde(default)]
    wall_s: f64,
}

struct SidecarClient {
    base: String,
    student: String,
    temperature: f64,
    max_tokens: usize,
    client: reqwest::Client,
}

impl SidecarClient {
    async fn fetch_slots(&self) -> Result<SlotsResponse> {
        let resp = self.client.get(format!("{}/slots", self.base)).send().await?.error_for_status()?;
        Ok(resp.json().await?)
    }

    async fn episode(
        &self,
        id: &str,
        split: &str,
        changed_slots: &BTreeMap<String, String>,
    ) -> Result<EpisodeResponse> {
        let body = serde_json::json!({
            "id": id,
            "split": split,
            "slots": if changed_slots.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::to_value(changed_slots)?
            },
            "student": self.student,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        });
        let resp = self
            .client
            .post(format!("{}/episode", self.base))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        Ok(resp.json().await?)
    }
}

// ---------------------------------------------------------------------------
// The wrapper module
// ---------------------------------------------------------------------------

#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct SbInput {
    id: String,
    instruction: String,
}

#[dspy_rs::Schema]
#[derive(Clone, Debug, Default)]
struct SbOutput {
    score: f64,
    passed: bool,
    produced: bool,
    executed: bool,
    stage: String,
    feedback: String,
    program: String,
    error: Option<String>,
    in_tokens: u64,
    out_tokens: u64,
    wall_s: f64,
}

struct SbHarness {
    inspect: Predict<HarnessStage>,
    plan: Predict<HarnessStage>,
    code: Predict<HarnessStage>,
    verify: Predict<HarnessStage>,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    /// Serve-mode per-request overrides (None = seed). Optimize-mode
    /// candidates arrive through `fx::current_instruction` instead.
    overrides: BTreeMap<String, String>,
    /// Split label forwarded to the sidecar (logging only).
    split: String,
}

dspy_rs::predictors!(SbHarness { inspect, plan, code, verify });

impl Module for SbHarness {
    type Input = SbInput;
    type Output = SbOutput;

    async fn forward(&self, input: SbInput) -> Result<Predicted<SbOutput>, PredictError> {
        // Resolve each slot: ambient candidate → serve override → seed.
        // Only texts that differ from seed travel; untouched slots keep the
        // harness's production bytes.
        let mut changed: BTreeMap<String, String> = BTreeMap::new();
        for name in SLOTS {
            let seed = self.seeds.get(name).map(String::as_str).unwrap_or("");
            let text = fx::current_instruction(name).or_else(|| self.overrides.get(name).cloned());
            if let Some(text) = text
                && text != seed
            {
                changed.insert(name.to_string(), text);
            }
        }
        let response = self
            .sidecar
            .episode(&input.id, &self.split, &changed)
            .await
            .unwrap_or_else(|e| EpisodeResponse {
                error: Some(format!("sidecar transport error: {e}")),
                stage: "transport".to_string(),
                ..EpisodeResponse::default()
            });
        Ok(Predicted::new(
            SbOutput {
                score: response.score,
                passed: response.passed,
                produced: response.produced,
                executed: response.executed,
                stage: response.stage,
                feedback: if response.feedback.is_empty() {
                    response.trace_summary
                } else {
                    response.feedback
                },
                program: response.program,
                error: response.error,
                in_tokens: response.in_tokens,
                out_tokens: response.out_tokens,
                wall_s: response.wall_s,
            },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Metric: score with the sidecar's leak-safe feedback
// ---------------------------------------------------------------------------

struct SbMetric;

impl SbMetric {
    /// The upstream workbook hard-pass verdict, with no execution bonus.
    fn grade(&self, output: &SbOutput) -> (f64, String) {
        if let Some(error) = &output.error {
            let mut fb = format!("EPISODE FAILED (scored 0) at stage {}: {error}\n", output.stage);
            fb.push_str(&output.feedback);
            return (0.0, fb.trim_end().to_string());
        }
        let score = if output.passed { 1.0 } else { 0.0 };
        if (score - output.score).abs() > 1e-9 {
            eprintln!(
                "score mismatch: driver {score} vs sidecar {} (passed={}, produced={}, executed={})",
                output.score, output.passed, output.produced, output.executed
            );
        }
        (score, output.feedback.trim_end().to_string())
    }
}

impl TypedMetric<SbRow, SbHarness> for SbMetric {
    async fn evaluate(
        &self,
        _example: &SbRow,
        prediction: &Predicted<SbOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let (score, feedback) = self.grade(prediction);
        Ok(Eval::with_feedback(score, feedback))
    }
}

// ---------------------------------------------------------------------------
// Factory / winner assembly
// ---------------------------------------------------------------------------

fn build_module(
    seeds: &Arc<BTreeMap<String, String>>,
    sidecar: &Arc<SidecarClient>,
    overrides: BTreeMap<String, String>,
    split: &str,
) -> SbHarness {
    let seed = |name: &str| seeds.get(name).cloned().unwrap_or_default();
    SbHarness {
        inspect: Predict::<HarnessStage>::builder()
            .named("inspect")
            .instruction(seed("inspect"))
            .build(),
        plan: Predict::<HarnessStage>::builder().named("plan").instruction(seed("plan")).build(),
        code: Predict::<HarnessStage>::builder().named("code").instruction(seed("code")).build(),
        verify: Predict::<HarnessStage>::builder()
            .named("verify")
            .instruction(seed("verify"))
            .build(),
        seeds: seeds.clone(),
        sidecar: sidecar.clone(),
        overrides,
        split: split.to_string(),
    }
}

/// Winner's full slot assignment: seeds, overlaid with post-install module
/// state, overlaid with the candidate's slots, touched slot last (the HoVer
/// / mxagent winner_map contract).
fn winner_map(
    seeds: &BTreeMap<String, String>,
    module: Option<&SbHarness>,
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
        if SLOTS.contains(&name.as_str()) {
            map.insert(name.clone(), text.clone());
        }
    }
    if SLOTS.contains(&module_name) {
        map.insert(module_name.to_string(), instruction.to_string());
    }
    map
}

fn write_winner(path: &str, map: &BTreeMap<String, String>) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(map)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Serve mode (36-docsqa protocol, four slots under the `instruction` key)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    /// {"inspect": "..." | null, ...} — null / missing key = that slot's seed;
    /// null for the whole map = the all-seed harness.
    instruction: Option<HashMap<String, Option<String>>>,
    split: String,
    idx: usize,
}

#[derive(Serialize)]
struct ServeResponse {
    id: u64,
    score: f64,
    passed: bool,
    answer: String,
    feedback: String,
    stage: String,
    error: Option<String>,
    produced: bool,
    executed: bool,
    in_tokens: u64,
    out_tokens: u64,
    wall_s: f64,
}

async fn serve(
    dir: &str,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    metric: Arc<SbMetric>,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    // Every data/*.jsonl is addressable as a split.
    let mut splits: HashMap<String, Vec<SbRow>> = HashMap::new();
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

    let concurrency = env_usize("SB_CONCURRENCY", 6);
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
        let (seeds, sidecar, metric, splits, stdout, semaphore) = (
            seeds.clone(),
            sidecar.clone(),
            metric.clone(),
            splits.clone(),
            stdout.clone(),
            semaphore.clone(),
        );
        tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("semaphore open");
            let Some(row) = splits.get(&request.split).and_then(|s| s.get(request.idx)) else {
                eprintln!("unknown split/idx: {}/{}", request.split, request.idx);
                return;
            };
            let mut overrides = BTreeMap::new();
            if let Some(map) = &request.instruction {
                for (name, value) in map {
                    if !SLOTS.contains(&name.as_str()) {
                        eprintln!("unknown slot in instruction map: {name:?}");
                        continue;
                    }
                    if let Some(text) = value {
                        overrides.insert(name.clone(), text.clone());
                    }
                }
            }
            let module = build_module(&seeds, &sidecar, overrides, &request.split);
            let prediction = module
                .forward(SbInput { id: row.id.clone(), instruction: row.instruction.clone() })
                .await
                .expect("forward degrades transport errors to a scored-0 output, never Err");
            let (score, feedback) = metric.grade(&prediction);
            let response = ServeResponse {
                id: request.id,
                score,
                passed: prediction.passed,
                answer: prediction.program.clone(),
                feedback,
                stage: prediction.stage.clone(),
                error: prediction.error.clone(),
                produced: prediction.produced,
                executed: prediction.executed,
                in_tokens: prediction.in_tokens,
                out_tokens: prediction.out_tokens,
                wall_s: prediction.wall_s,
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

// ---------------------------------------------------------------------------
// Optimize mode
// ---------------------------------------------------------------------------

async fn optimize(
    dir: &str,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    metric: &SbMetric,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("SB_OPT", "harvest");
    let train_file = env_str("SB_TRAIN_FILE", "train.jsonl");
    let val_file = env_str("SB_VAL_FILE", "val.jsonl");
    let trainset = load_rows(&format!("{dir}/data/{train_file}"))?;
    let val = load_rows(&format!("{dir}/data/{val_file}"))?;
    println!(
        "spreadsheet: {} train / {} val, arm {opt_name}, {} slots, student {}",
        trainset.len(),
        val.len(),
        SLOTS.len(),
        sidecar.student
    );

    let seed = env_usize("SB_SEED", 0) as u64;
    let budget = env_usize("SB_BUDGET", 600);
    let concurrency = env_usize("SB_CONCURRENCY", 6);
    let mut module = build_module(&seeds, &sidecar, BTreeMap::new(), "optimize");

    let prefix = env_str("SB_OUT_PREFIX", &format!("{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.json");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.json");

    if opt_name == "gepa" {
        let iters = env_usize("SB_ITERS", 40);
        let minibatch = env_usize("SB_MINIBATCH", 4);
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
            .compile_module_with_valset(&mut module, &trainset, Some(&val), metric)
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
        let winner = winner_map(
            &seeds,
            Some(&module),
            &result.best_candidate.slots,
            &result.best_candidate.module_name,
            &result.best_candidate.instruction,
        );
        println!(
            "\nchosen (gen {}, val avg {:.3}, touched slot {}):",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.module_name
        );
        for (name, text) in &winner {
            println!("--- {name} ---\n{text}\n");
        }
        write_winner(&out, &winner)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| winner_map(&seeds, None, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    } else {
        // HARVEST keeps its preset cycle shape (iters 150 / minibatch 5 /
        // P×N 2×2) unless overridden; ratchet keeps the v2-era shape.
        let is_harvest = matches!(opt_name.as_str(), "harvest" | "harvest_search");
        let iters = env_usize("SB_ITERS", if is_harvest { 150 } else { 60 });
        let minibatch = env_usize("SB_MINIBATCH", 5);
        let champion = env_usize("SB_CHAMPION", if opt_name == "harvest_search" { 0 } else { 30 });
        println!(
            "{opt_name} seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, champion {champion}"
        );
        let optimizer = match opt_name.as_str() {
            "ratchet" => {
                let mut optimizer = Ratchet::builder().prompt_model(reflection_lm).build();
                optimizer.parents_per_step = env_usize("SB_PARENTS", optimizer.parents_per_step);
                optimizer.proposals_per_parent =
                    env_usize("SB_PROPOSALS", optimizer.proposals_per_parent);
                optimizer
            }
            // HARVEST (RATCHET v6, docs/ratchet-v6-harvest-design.md): on-target
            // screen, full-val ranking, Rewrite + Policy operators, sequential
            // paired certification vs the seed on reserved train columns.
            // train120 carve: champion 30 + certify 45 leaves 45 batch columns.
            "harvest" | "harvest_search" => {
                let mut optimizer = if opt_name == "harvest_search" {
                    dspy_rs::harvest_search()
                } else { harvest() };
                optimizer.prompt_model = Some(reflection_lm);
                optimizer.certify_reserve = env_usize("SB_CERTIFY_RESERVE", if opt_name == "harvest_search" { 0 } else { 45 });
                optimizer.certify_budget = env_usize("SB_CERTIFY_BUDGET", if opt_name == "harvest_search" { 0 } else { 150 });
                // One-factor ablation knobs (mirror of 33-pupa.rs / 36-docsqa.rs):
                // each overrides the `harvest()` preset only when set.
                match env_str("SB_SCREEN", "").as_str() {
                    "a" => optimizer.screen_rule = dspy_rs::ScreenRule::OnTargetA,
                    "b" => optimizer.screen_rule = dspy_rs::ScreenRule::HeldOutB,
                    _ => {}
                }
                if let Ok(v) = std::env::var("SB_REWRITE_ONLY") {
                    optimizer.rewrite_only = v == "1";
                }
                if let Ok(v) = std::env::var("SB_POLICY") {
                    optimizer.policy_channel = v == "1";
                }
                if let Ok(v) = std::env::var("SB_POLICY_SHARE") {
                    optimizer.policy_share = v.parse().unwrap_or(optimizer.policy_share);
                }
                if let Ok(v) = std::env::var("SB_LEDGER_GATE_KILLS") {
                    optimizer.ledger_gate_kills = v == "1";
                }
                if let Ok(v) = std::env::var("SB_PARENT_GREED") {
                    optimizer.parent_greed = v.parse().unwrap_or(optimizer.parent_greed);
                }
                if let Ok(v) = std::env::var("SB_GATE_SIZE") {
                    optimizer.gate_size = v.parse().ok();
                }
                // v6.5 length prior (soft, teacher-facing; unset = off):
                // SB_LENGTH_PRIOR=N chars, hard guard N × SB_LENGTH_SLACK
                // (default 1.25) through the length_cap path.
                if let Ok(v) = std::env::var("SB_LENGTH_PRIOR") {
                    optimizer.length_prior = v.parse().ok().filter(|n: &usize| *n > 0);
                }
                if let Ok(v) = std::env::var("SB_LENGTH_SLACK") {
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
                optimizer.scout_size = env_usize("SB_SCOUT", optimizer.scout_size);
                optimizer.parents_per_step = env_usize("SB_PARENTS", optimizer.parents_per_step);
                optimizer.proposals_per_parent =
                    env_usize("SB_PROPOSALS", optimizer.proposals_per_parent);
                optimizer.champion_k = env_usize("SB_CHAMPION_K", optimizer.champion_k);
                optimizer
            }
            other => bail!("unknown SB_OPT {other:?} (harvest|harvest_search|ratchet|gepa)"),
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
            // harvest_search permits training examples; certified harvest
            // retains its instruction-only guard. Stage-repair is not
            // wired: rollouts happen outside DSRs, no Trace to replay —
            // arms run round-robin slot credit over the four slots.
            if is_harvest {
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
            .compile_module_with_valset(&mut module, &trainset, Some(&val), metric)
            .await?;
        if let Ok(path) = std::env::var("SB_REPORT") {
            std::fs::write(path, serde_json::to_string_pretty(&result)?)?;
        }
        println!(
            "rollouts: {} lm_calls: {} promoted: {}",
            result.total_rollouts,
            result.total_lm_calls,
            result.all_candidates.len()
        );
        println!(
            "gate_rejected: {} race_killed: {} faded: {} skipped: {} leak: {}/{} length: {}/{}",
            result.gate_rejected,
            result.race_killed,
            result.faded,
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
        let winner = winner_map(
            &seeds,
            Some(&module),
            &result.best_candidate.slots,
            &result.best_candidate.module_name,
            &result.best_candidate.instruction,
        );
        println!(
            "\nchosen (gen {}, val avg {:.3}, touched slot {}):",
            result.best_candidate.generation,
            result.best_candidate.average_score(),
            result.best_candidate.module_name
        );
        for (name, text) in &winner {
            println!("--- {name} ---\n{text}\n");
        }
        write_winner(&out, &winner)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| winner_map(&seeds, None, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    }
    println!("\nwinner written to {out}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = env_str("SB_MODE", "optimize");
    if mode != "serve" {
        init_tracing()?;
    }
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let dir = env_str("SB_DIR", "benchmarks/spreadsheet");
    let sidecar_base = env_str("SB_SIDECAR", "http://127.0.0.1:8140");

    // An episode is five student calls + two docker execs + two LibreOffice
    // passes; 30 minutes bounds the worst case with room to spare.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1800))
        .build()?;
    let health = client
        .get(format!("{sidecar_base}/health"))
        .send()
        .await
        .map_err(|e| {
            anyhow!(
                "sidecar not reachable at {sidecar_base}: {e} — start \
                 benchmarks/spreadsheet/sidecar.py (see its header)"
            )
        })?;
    if !health.status().is_success() {
        bail!("sidecar unhealthy at {sidecar_base}");
    }

    let sidecar = Arc::new(SidecarClient {
        base: sidecar_base,
        student: env_str("SB_STUDENT", "openai/gpt-oss-120b"),
        temperature: env_f64("SB_TEMPERATURE", 0.2),
        max_tokens: env_usize("SB_MAX_TOKENS", 8000),
        client,
    });

    // Seeds = upstream's prompt split into stages, served by the sidecar.
    // Snapshot them for provenance (and for the pip-gepa reference arm).
    let slots_response = sidecar.fetch_slots().await.context("GET /slots failed")?;
    for name in SLOTS {
        if !slots_response.slots.contains_key(name) {
            bail!("sidecar /slots is missing slot {name:?} — sidecar/example out of sync");
        }
    }
    let seeds: Arc<BTreeMap<String, String>> = Arc::new(
        SLOTS
            .iter()
            .map(|name| (name.to_string(), slots_response.slots[*name].clone()))
            .collect(),
    );
    std::fs::write(
        format!("{dir}/seed_slots_snapshot.json"),
        serde_json::to_string_pretty(&*seeds)?,
    )?;
    if mode != "serve" {
        println!(
            "spreadsheet: {} seed slots from the sidecar, student {}, config {}",
            seeds.len(),
            sidecar.student,
            slots_response.config
        );
    }

    let reflection_lm = LM::builder()
        .maybe_base_url(std::env::var("SB_TEACHER_BASE_URL").ok())
        .model(env_str("SB_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(env_usize("SB_TEACHER_MAX_TOKENS", 8000) as u32)
        .build()
        .await?;
    // The global LM only renders reflection prompts; the student runs inside
    // the sidecar and the metric is judge-free.
    configure(reflection_lm.clone());

    match mode.as_str() {
        "serve" => serve(&dir, seeds, sidecar, Arc::new(SbMetric)).await,
        "optimize" => optimize(&dir, seeds, sidecar, &SbMetric, reflection_lm).await,
        other => bail!("unknown SB_MODE {other:?} (optimize|serve)"),
    }
}


#[cfg(test)]
mod native_metric_tests {
    use super::*;
    #[test]
    fn executable_wrong_workbook_gets_no_credit() {
        let output=SbOutput { produced:true, executed:true, score:0.0, ..Default::default() };
        assert_eq!(SbMetric.grade(&output).0,0.0);
        assert_eq!(SbMetric.grade(&SbOutput { passed:true,score:1.0,..output }).0,1.0);
    }
}
