//! AutomationBench (Zapier) four-slot harness optimization — Rig 6 of
//! docs/rigs-spreadsheet-automation-design.md. Second EXTERNAL-HARNESS rig
//! after mxagent (examples/35-mxagent.rs): the program under optimization is
//! a four-stage agent harness over the benchmark's own simulated-SaaS tool
//! environment, driven out-of-process through the sidecar at
//! `benchmarks/automation/sidecar.py` (port 8141).
//!
//! How the wrap works (the harness-for-harnesses seam):
//! - The module holds one named `Predict` per prompt slot. They are **never
//!   called** — they exist so `predictors!` registers the slots (LeafInfo,
//!   candidate keys, persistence) and so each carries its seed text (fetched
//!   from the sidecar's `/slots` at startup, snapshotted to
//!   `seeds_snapshot.json`).
//! - `forward` resolves each slot's effective text itself:
//!   `fx::current_instruction(name)` (the ambient candidate injected by the
//!   module lane during rollouts) falling back to the Predict's own
//!   instruction (the seed, or the serve request's override). Only slots that
//!   differ from seed are POSTed.
//! - The sidecar runs one episode: plan (one chat call), the benchmark's
//!   tool loop with system prompt = actor + tool_policy + plan (max_turns
//!   25), verify (one chat call, may run a <= 5-turn fix loop in the same
//!   env state), then the rubric on the final world state.
//!
//! Slots: plan, actor, tool_policy, verify (must match the sidecar).
//!
//! Metric: `partial_credit` (fraction of scored assertions) — the dense
//! optimizer signal; `task_completed_correctly` (all assertions) is the
//! paper's primary metric, carried through serve responses as `completed`.
//! Feedback = the sidecar's failed-assertion list + action summary. Errored
//! episodes score 0 (mxagent convention).
//!
//! Modes:
//! - `AB_MODE=optimize` (default): `AB_OPT=harvest|ratchet|gepa`, winner
//!   written as a JSON slot map `<AB_OUT_PREFIX>_s<seed>.json` (+ `_argmax_`).
//! - `AB_MODE=serve`: JSONL rollout server on stdin/stdout (36-docsqa
//!   protocol). Request {"id":1,"instruction":{slot: text|null}|null,
//!   "split":"val","idx":3}; response {"id","score","completed",
//!   "partial_credit","answer","feedback", ...}.
//!
//! Env: AB_DIR (benchmarks/automation), AB_MODE, AB_OPT, AB_STUDENT
//! (openai/gpt-oss-120b, forwarded to the sidecar), AB_TEACHER, AB_SEED,
//! AB_BUDGET (600), AB_CONCURRENCY (6), AB_SIDECAR (http://127.0.0.1:8141),
//! AB_TRAIN_FILE, AB_VAL_FILE, AB_OUT_PREFIX, AB_MAX_TURNS (25), AB_ITERS,
//! AB_MINIBATCH, AB_CHAMPION, AB_PARENTS, AB_PROPOSALS, AB_CERTIFY_RESERVE
//! (45), AB_CERTIFY_BUDGET (150).
//! HARVEST one-factor knobs (unset = preset): AB_SCREEN (a|b),
//! AB_REWRITE_ONLY, AB_POLICY, AB_POLICY_SHARE, AB_LEDGER_GATE_KILLS,
//! AB_PARENT_GREED, AB_GATE_SIZE, AB_SCOUT, AB_PARENTS, AB_PROPOSALS,
//! AB_CHAMPION_K, AB_LENGTH_PRIOR (chars; hard guard × AB_LENGTH_SLACK 1.25).
//! Leak guard is ON (benchmark semantics).
//!
//! Run from the repo root (sidecar must be up, see sidecar.py header):
//!     set -a; source .env; set +a
//!     AB_OPT=harvest AB_BUDGET=600 cargo run --release --example 40-automation

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dspy_rs::{
    Eval, GEPA, LM, Module, Predict, PredictError, Predicted, PredictorInfo, Predictors, Ratchet,
    Signature, TypedMetric, configure, fx, harvest, init_tracing,
};
use serde::{Deserialize, Serialize};

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Slot registry: (module/predictor name, sidecar slot key). Order matters —
/// it is the round-robin mutation order. Must match `predictors!` below and
/// the sidecar's SLOT_NAMES.
const SLOTS: [(&str, &str); 4] = [
    ("plan", "plan"),
    ("actor", "actor"),
    ("tool_policy", "tool_policy"),
    ("verify", "verify"),
];

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/// One task row (make_splits.py): `id` is the package's task_name, `prompt`
/// the trigger text (truncated for display — the sidecar re-reads the full
/// task by id).
#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct AbRow {
    id: String,
    #[serde(default)]
    example_id: u64,
    #[serde(default)]
    domain: String,
    prompt: String,
    #[serde(default)]
    n_assertions: usize,
}

fn load_rows(path: &str) -> Result<Vec<AbRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run make_splits.py first"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<AbRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// Slot signature (parameter holder — never called as an LM)
// ---------------------------------------------------------------------------

/// One prompt slot of the AutomationBench four-stage harness (plan / actor /
/// tool_policy / verify). The instruction IS the slot text; the episode
/// below shows what behavior it produced.
#[derive(Signature, Clone, Debug)]
#[allow(dead_code)]
struct HarnessSlot {
    /// The business request that triggered the workflow, verbatim.
    #[input]
    prompt: String,

    /// The finished episode this slot steered: the plan, the tool calls made
    /// and their results, the verifier's verdict, and the rubric's failed
    /// assertions against the final state of the business apps.
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
#[allow(dead_code)]
struct EpisodeResponse {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    score: f64,
    #[serde(default)]
    task_completed_correctly: bool,
    #[serde(default)]
    partial_credit: f64,
    #[serde(default)]
    feedback: String,
    #[serde(default)]
    answer: String,
    #[serde(default)]
    turns: u32,
    #[serde(default)]
    tool_calls: u32,
    #[serde(default)]
    in_tokens: u64,
    #[serde(default)]
    out_tokens: u64,
    #[serde(default)]
    stage: String,
    #[serde(default)]
    wall_s: f64,
}

struct SidecarClient {
    base: String,
    student: String,
    max_turns: u32,
    client: reqwest::Client,
}

impl SidecarClient {
    async fn fetch_slots(&self) -> Result<SlotsResponse> {
        let url = format!("{}/slots", self.base);
        let resp = self.client.get(&url).send().await?.error_for_status()?;
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
            "max_turns": self.max_turns,
        });
        // One transport retry: the episode itself is retried inside the
        // sidecar's client (40 attempts per turn); this only covers a
        // dropped HTTP connection to the sidecar.
        let mut last = None;
        for attempt in 0..2u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
            match self.client.post(format!("{}/episode", self.base)).json(&body).send().await {
                Ok(resp) => match resp.error_for_status() {
                    Ok(resp) => return Ok(resp.json().await?),
                    Err(e) => last = Some(anyhow!(e)),
                },
                Err(e) => last = Some(anyhow!(e)),
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("sidecar episode failed")))
    }
}

// ---------------------------------------------------------------------------
// The wrapper module
// ---------------------------------------------------------------------------

#[dspy_rs::Schema]
#[derive(Clone, Debug, Default)]
struct AbOutput {
    score: f64,
    completed: bool,
    partial_credit: f64,
    feedback: String,
    answer: String,
    turns: u32,
    tool_calls: u32,
    error: Option<String>,
}

#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct AbInput {
    id: String,
    prompt: String,
}

struct AbHarness {
    plan: Predict<HarnessSlot>,
    actor: Predict<HarnessSlot>,
    tool_policy: Predict<HarnessSlot>,
    verify: Predict<HarnessSlot>,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    /// Informational split label sent with each episode (sidecar log).
    split: String,
}

dspy_rs::predictors!(AbHarness { plan, actor, tool_policy, verify });

impl AbHarness {
    /// The Predict's own instruction for a slot (seed, post-install winner,
    /// or a serve request's override).
    fn slot_instruction(&self, name: &str) -> String {
        match name {
            "plan" => self.plan.instruction(),
            "actor" => self.actor.instruction(),
            "tool_policy" => self.tool_policy.instruction(),
            "verify" => self.verify.instruction(),
            _ => String::new(),
        }
    }

    /// Effective slot texts that differ from seed: ambient candidate wins,
    /// the Predict's own instruction otherwise.
    fn changed_slots(&self) -> BTreeMap<String, String> {
        let mut changed = BTreeMap::new();
        for (name, key) in SLOTS {
            let seed = self.seeds.get(name).map(String::as_str).unwrap_or("");
            let text = fx::current_instruction(name).unwrap_or_else(|| self.slot_instruction(name));
            if text != seed {
                changed.insert(key.to_string(), text);
            }
        }
        changed
    }
}

impl Module for AbHarness {
    type Input = AbInput;
    type Output = AbOutput;

    async fn forward(&self, input: AbInput) -> Result<Predicted<AbOutput>, PredictError> {
        let changed = self.changed_slots();
        let response = self
            .sidecar
            .episode(&input.id, &self.split, &changed)
            .await
            .unwrap_or_else(|e| EpisodeResponse {
                error: Some(format!("sidecar transport error: {e}")),
                stage: "transport".to_string(),
                ..EpisodeResponse::default()
            });
        let _ = &input.prompt;
        Ok(Predicted::new(
            AbOutput {
                score: response.score,
                completed: response.task_completed_correctly,
                partial_credit: response.partial_credit,
                feedback: response.feedback,
                answer: response.answer,
                turns: response.turns,
                tool_calls: response.tool_calls,
                error: response.error.map(|e| format!("[{}] {e}", response.stage)),
            },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Metric: native full-task pass with the sidecar's diagnostic feedback
// ---------------------------------------------------------------------------

struct AbMetric;

impl AbMetric {
    fn grade(&self, _example: &AbRow, output: &AbOutput) -> (f64, String) {
        if let Some(error) = &output.error {
            // The sidecar scores whatever world state exists even when a
            // stage failed; a transport failure (no state at all) scores 0.
            if output.feedback.is_empty() || error.starts_with("[transport]") {
                return (0.0, format!("ROLLOUT FAILED (scored 0):\n{error}"));
            }
        }
        (if output.completed { 1.0 } else { 0.0 }, output.feedback.trim_end().to_string())
    }
}

impl TypedMetric<AbRow, AbHarness> for AbMetric {
    async fn evaluate(
        &self,
        example: &AbRow,
        prediction: &Predicted<AbOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let (score, feedback) = self.grade(example, prediction);
        Ok(Eval::with_feedback(score, feedback))
    }
}

// ---------------------------------------------------------------------------
// Factory / winner assembly
// ---------------------------------------------------------------------------

fn build_module(
    seeds: &Arc<BTreeMap<String, String>>,
    sidecar: &Arc<SidecarClient>,
    overrides: Option<&HashMap<String, Option<String>>>,
    split: &str,
) -> AbHarness {
    let text = |name: &str| -> String {
        overrides
            .and_then(|m| m.get(name).cloned().flatten())
            .unwrap_or_else(|| seeds.get(name).cloned().unwrap_or_default())
    };
    AbHarness {
        plan: Predict::<HarnessSlot>::builder().named("plan").instruction(text("plan")).build(),
        actor: Predict::<HarnessSlot>::builder().named("actor").instruction(text("actor")).build(),
        tool_policy: Predict::<HarnessSlot>::builder()
            .named("tool_policy")
            .instruction(text("tool_policy"))
            .build(),
        verify: Predict::<HarnessSlot>::builder()
            .named("verify")
            .instruction(text("verify"))
            .build(),
        seeds: seeds.clone(),
        sidecar: sidecar.clone(),
        split: split.to_string(),
    }
}

/// Winner's full slot assignment: seeds, overlaid with post-install module
/// state, overlaid with the candidate's slots, touched slot last (same
/// robustness contract as the HoVer / mxagent winner_map).
fn winner_map(
    seeds: &BTreeMap<String, String>,
    module: Option<&AbHarness>,
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
    let known = |n: &str| SLOTS.iter().any(|(name, _)| *name == n);
    for (name, text) in slots {
        if known(name) {
            map.insert(name.clone(), text.clone());
        }
    }
    if known(module_name) {
        map.insert(module_name.to_string(), instruction.to_string());
    }
    map
}

fn write_winner(path: &str, map: &BTreeMap<String, String>) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(map)?)?;
    Ok(())
}

fn print_winner(map: &BTreeMap<String, String>, seeds: &BTreeMap<String, String>) {
    for (name, text) in map {
        let tag = if seeds.get(name) == Some(text) { " (seed)" } else { "" };
        println!("--- {name}{tag} [{} chars] ---\n{text}\n", text.chars().count());
    }
}

// ---------------------------------------------------------------------------
// Serve mode (36-docsqa protocol, slot map)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    /// {slot: text | null} — null / missing = that slot's seed. `null` for
    /// the whole map = the all-seed harness. `instructions` accepted too.
    #[serde(default, alias = "instructions")]
    instruction: Option<HashMap<String, Option<String>>>,
    split: String,
    idx: usize,
}

#[derive(Serialize)]
struct ServeResponse {
    id: u64,
    score: f64,
    completed: bool,
    partial_credit: f64,
    answer: String,
    feedback: String,
    task_id: String,
    turns: u32,
    tool_calls: u32,
    error: Option<String>,
}

async fn serve(
    dir: &str,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    metric: Arc<AbMetric>,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    // Every data/*.jsonl is addressable as a split.
    let mut splits: HashMap<String, Vec<AbRow>> = HashMap::new();
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

    let concurrency = env_usize("AB_CONCURRENCY", 6);
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
            let module = build_module(&seeds, &sidecar, request.instruction.as_ref(), &request.split);
            let prediction = module
                .forward(AbInput { id: row.id.clone(), prompt: row.prompt.clone() })
                .await
                .expect("forward returns an error output, never Err");
            let (score, feedback) = metric.grade(row, &prediction);
            let response = ServeResponse {
                id: request.id,
                score,
                completed: prediction.completed,
                partial_credit: prediction.partial_credit,
                answer: prediction.answer.clone(),
                feedback,
                task_id: row.id.clone(),
                turns: prediction.turns,
                tool_calls: prediction.tool_calls,
                error: prediction.error.clone(),
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
// Optimize
// ---------------------------------------------------------------------------

async fn optimize(
    dir: &str,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    metric: &AbMetric,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("AB_OPT", "harvest");
    let train_file = env_str("AB_TRAIN_FILE", "train.jsonl");
    let val_file = env_str("AB_VAL_FILE", "val.jsonl");
    let trainset = load_rows(&format!("{dir}/data/{train_file}"))?;
    let val = load_rows(&format!("{dir}/data/{val_file}"))?;
    println!(
        "automation: {} train / {} val, arm {opt_name}, {} slots, student {}, max_turns {}",
        trainset.len(),
        val.len(),
        SLOTS.len(),
        sidecar.student,
        sidecar.max_turns
    );

    let seed = env_usize("AB_SEED", 0) as u64;
    let budget = env_usize("AB_BUDGET", 600);
    let concurrency = env_usize("AB_CONCURRENCY", 6);
    let mut module = build_module(&seeds, &sidecar, None, "optimize");

    let prefix = env_str("AB_OUT_PREFIX", &format!("{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.json");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.json");

    if opt_name == "gepa" {
        let iters = env_usize("AB_ITERS", 40);
        let minibatch = env_usize("AB_MINIBATCH", 3);
        println!("gepa seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, leak guard ON");
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
        print_winner(&winner, &seeds);
        write_winner(&out, &winner)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| winner_map(&seeds, None, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    } else if matches!(opt_name.as_str(), "ratchet" | "harvest" | "harvest_search") {
        let is_harvest = matches!(opt_name.as_str(), "harvest" | "harvest_search");
        // train120 carve for harvest: champion 30 + certify 45 leaves 45
        // batch columns (the PUPA carve, docs/ratchet-v6-harvest-design.md).
        let iters = env_usize("AB_ITERS", if is_harvest { 150 } else { 40 });
        let champion = env_usize("AB_CHAMPION", if opt_name == "harvest_search" { 0 } else if is_harvest { 30 } else { 20 });
        let optimizer = if is_harvest {
            let mut optimizer = if opt_name == "harvest_search" {
                dspy_rs::harvest_search()
            } else { harvest() };
            optimizer.prompt_model = Some(reflection_lm);
            optimizer.certify_reserve = env_usize("AB_CERTIFY_RESERVE", if opt_name == "harvest_search" { 0 } else { 45 });
            optimizer.certify_budget = env_usize("AB_CERTIFY_BUDGET", if opt_name == "harvest_search" { 0 } else { 150 });
            // One-factor ablation knobs (mirror of 33-pupa / 36-docsqa):
            // each overrides the `harvest()` preset only when set.
            match env_str("AB_SCREEN", "").as_str() {
                "a" => optimizer.screen_rule = dspy_rs::ScreenRule::OnTargetA,
                "b" => optimizer.screen_rule = dspy_rs::ScreenRule::HeldOutB,
                _ => {}
            }
            if let Ok(v) = std::env::var("AB_REWRITE_ONLY") {
                optimizer.rewrite_only = v == "1";
            }
            if let Ok(v) = std::env::var("AB_POLICY") {
                optimizer.policy_channel = v == "1";
            }
            if let Ok(v) = std::env::var("AB_POLICY_SHARE") {
                optimizer.policy_share = v.parse().unwrap_or(optimizer.policy_share);
            }
            if let Ok(v) = std::env::var("AB_LEDGER_GATE_KILLS") {
                optimizer.ledger_gate_kills = v == "1";
            }
            if let Ok(v) = std::env::var("AB_PARENT_GREED") {
                optimizer.parent_greed = v.parse().unwrap_or(optimizer.parent_greed);
            }
            if let Ok(v) = std::env::var("AB_GATE_SIZE") {
                optimizer.gate_size = v.parse().ok();
            }
            // v6.5 length prior (soft, teacher-facing; unset = off):
            // AB_LENGTH_PRIOR=N chars, hard guard N × AB_LENGTH_SLACK
            // (default 1.25) through the length_cap path.
            if let Ok(v) = std::env::var("AB_LENGTH_PRIOR") {
                optimizer.length_prior = v.parse().ok().filter(|n: &usize| *n > 0);
            }
            if let Ok(v) = std::env::var("AB_LENGTH_SLACK") {
                optimizer.length_prior_slack = v.parse().unwrap_or(optimizer.length_prior_slack);
            }
            if optimizer.length_prior.is_some() {
                println!(
                    "length prior {:?} chars (slack {}, hard cap {:?})",
                    optimizer.length_prior,
                    optimizer.length_prior_slack,
                    optimizer.effective_length_cap()
                );
            }
            optimizer.scout_size = env_usize("AB_SCOUT", optimizer.scout_size);
            optimizer.parents_per_step = env_usize("AB_PARENTS", optimizer.parents_per_step);
            optimizer.proposals_per_parent =
                env_usize("AB_PROPOSALS", optimizer.proposals_per_parent);
            optimizer.champion_k = env_usize("AB_CHAMPION_K", optimizer.champion_k);
            // The preset's minibatch (harvest: 5) stands unless AB_MINIBATCH is set.
            optimizer.minibatch_size = env_usize("AB_MINIBATCH", optimizer.minibatch_size);
            optimizer
        } else {
            let mut optimizer = Ratchet::builder().prompt_model(reflection_lm).build();
            optimizer.minibatch_size = env_usize("AB_MINIBATCH", 3);
            optimizer.parents_per_step = env_usize("AB_PARENTS", 1);
            optimizer.proposals_per_parent = env_usize("AB_PROPOSALS", 1);
            optimizer
        };
        let optimizer = {
            let mut optimizer = optimizer;
            optimizer.num_iterations = iters;
            optimizer.champion_fresh = champion;
            optimizer.max_rollouts = Some(budget);
            optimizer.seed = Some(seed);
            optimizer.eval_concurrency = concurrency;
            optimizer.track_stats = true;
            // The search preset permits training examples independently of
            // the legacy allow_memorization switch; evaluation stays disjoint.
            optimizer.allow_memorization = false;
            println!(
                "{opt_name} seed {seed}, iters {}, minibatch {}, budget {budget}, champion {champion}, \
                 P×N {}×{}, leak guard ON",
                optimizer.num_iterations,
                optimizer.minibatch_size,
                optimizer.parents_per_step,
                optimizer.proposals_per_parent
            );
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
        if let Ok(path) = std::env::var("AB_REPORT") {
            std::fs::write(path, serde_json::to_string_pretty(&result)?)?;
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
        print_winner(&winner, &seeds);
        write_winner(&out, &winner)?;
        let argmax = result
            .best_by_mean
            .as_ref()
            .map(|c| winner_map(&seeds, None, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    } else {
        bail!("unknown AB_OPT {opt_name:?} (harvest|harvest_search|ratchet|gepa)");
    }
    println!("\nwinner written to {out}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = env_str("AB_MODE", "optimize");
    if mode != "serve" {
        init_tracing()?;
    }
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let dir = env_str("AB_DIR", "benchmarks/automation");
    let sidecar_base = env_str("AB_SIDECAR", "http://127.0.0.1:8141");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2400))
        .build()?;
    let health = client
        .get(format!("{sidecar_base}/health"))
        .send()
        .await
        .map_err(|e| {
            anyhow!(
                "sidecar not reachable at {sidecar_base}: {e} — start \
                 benchmarks/automation/sidecar.py (see its header)"
            )
        })?;
    if !health.status().is_success() {
        bail!("sidecar unhealthy at {sidecar_base}");
    }

    let sidecar = Arc::new(SidecarClient {
        base: sidecar_base,
        student: env_str("AB_STUDENT", "openai/gpt-oss-120b"),
        max_turns: env_usize("AB_MAX_TURNS", 50) as u32,
        client,
    });

    // Seeds = the sidecar's slot texts (actor + tool_policy are the
    // production SYSTEM_PROMPT split by sentence). Snapshot for provenance.
    let slots_response = sidecar.fetch_slots().await.context("GET /slots failed")?;
    for (name, key) in SLOTS {
        if !slots_response.slots.contains_key(key) {
            bail!("sidecar /slots is missing slot {name:?} — sidecar/example out of sync");
        }
    }
    let seeds: Arc<BTreeMap<String, String>> = Arc::new(
        SLOTS
            .iter()
            .map(|(name, key)| {
                (name.to_string(), slots_response.slots.get(*key).cloned().unwrap_or_default())
            })
            .collect(),
    );
    std::fs::write(format!("{dir}/seeds_snapshot.json"), serde_json::to_string_pretty(&*seeds)?)?;
    eprintln!(
        "automation: {} seed slots ({}), sidecar config {}",
        seeds.len(),
        seeds.iter().map(|(k, v)| format!("{k} {}ch", v.chars().count())).collect::<Vec<_>>().join(", "),
        slots_response.config
    );

    let reflection_lm = LM::builder()
        .maybe_base_url(std::env::var("AB_TEACHER_BASE_URL").ok())
        .model(env_str("AB_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;
    configure(reflection_lm.clone());

    match mode.as_str() {
        "serve" => serve(&dir, seeds, sidecar, Arc::new(AbMetric)).await,
        "optimize" => optimize(&dir, seeds, sidecar, &AbMetric, reflection_lm).await,
        other => bail!("unknown AB_MODE {other:?} (optimize|serve)"),
    }
}


#[cfg(test)]
mod native_metric_tests {
    use super::*;
    #[test]
    fn partial_assertions_get_no_optimizer_credit() {
        let row: AbRow=serde_json::from_value(serde_json::json!({"id":"test","prompt":"test","domain":"test"})).unwrap();
        let output=AbOutput { partial_credit:0.99, feedback:"one assertion failed".into(), ..Default::default() };
        assert_eq!(AbMetric.grade(&row,&output).0,0.0);
        assert_eq!(AbMetric.grade(&row,&AbOutput {completed:true,partial_credit:1.0,..output}).0,1.0);
    }
}
