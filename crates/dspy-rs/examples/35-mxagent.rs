//! mxagent-harness (toast agentic search) prompt optimization — the first
//! EXTERNAL-HARNESS rig: the optimized program is not a DSRs pipeline but
//! Mixedbread's production search harness (`mxagent-harness-internal`,
//! `run_fast_agentic_search`), driven out-of-process through the sidecar at
//! `benchmarks/mxagent/sidecar.py`.
//!
//! How the wrap works (the harness-for-harnesses seam):
//! - The module holds one named `Predict` per optimizable prompt slot. They
//!   are **never called** — they exist so `predictors!` registers the slots
//!   (LeafInfo, candidate keys, persistence) and so each carries its seed
//!   text (the harness's production prompt fragment, fetched from the
//!   sidecar's `/slots` at startup).
//! - `forward` resolves each slot's effective text itself:
//!   `fx::current_instruction(name)` (the ambient candidate injected by the
//!   module lane during rollouts) falling back to the seed. Only slots that
//!   differ from seed are POSTed, so the seed candidate runs the harness's
//!   byte-identical production prompt.
//! - The sidecar substitutes the texts into the rendered searcher system
//!   prompt (by section header) and tool schemas (by tool name), runs one
//!   full toast episode against the hosted Mixedbread store, and returns the
//!   final ranking plus a teacher-readable trace summary.
//!
//! Slots (must match the sidecar's SYSTEM_SLOTS/TOOL_SLOTS):
//!   task, workflow, retrieval, context_management, examples, final_rules,
//!   tool_search_corpus (-> "tool:search_corpus"),
//!   tool_filter_chunks  (-> "tool:filter_chunks").
//!
//! Metric: binary nDCG@k (default) or recall@k vs the row's relevant_ids —
//! exact port of examples/browsecomp/scoring.py from the harness repo, which
//! is parity-verified against the agent-rl campaign scorer. Errored rollouts
//! score 0 (their convention). Feedback = scores + hits/misses + the
//! sidecar's episode trace.
//!
//! NOTE: stage-repair (SlotCredit::Repair) is NOT wired here — the rollout
//! happens outside DSRs, so there is no Trace to counterfactually replay.
//! Arms run the showdown-validated engine with round-robin slot credit.
//!
//! Env: MXA_DIR (benchmarks/mxagent), MXA_BENCH (longseal|bcp|officeqa),
//! MXA_SIDECAR (http://127.0.0.1:8130), MXA_OPT (ratchet|gepa), MXA_SEED,
//! MXA_BUDGET (300), MXA_ITERS, MXA_MINIBATCH, MXA_CHAMPION,
//! MXA_CONCURRENCY (4), MXA_TRAIN_FILE, MXA_VAL_FILE, MXA_OUT_PREFIX,
//! MXA_SCORE (ndcg|recall), MXA_GEN_SEED (1005), MXA_TEACHER.
//!
//! Run from the repo root (sidecar must be up, see sidecar.py header):
//!     set -a; source .env; set +a
//!     MXA_BENCH=longseal MXA_OPT=ratchet cargo run --release --example 35-mxagent

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dspy_rs::{
    Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Predictors, Ratchet, Signature,
    TypedMetric, configure, fx, init_tracing,
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
/// the sidecar's SYSTEM_SLOTS + TOOL_SLOTS.
const SLOTS: [(&str, &str); 8] = [
    ("task", "task"),
    ("workflow", "workflow"),
    ("retrieval", "retrieval"),
    ("context_management", "context_management"),
    ("examples", "examples"),
    ("final_rules", "final_rules"),
    ("tool_search_corpus", "tool:search_corpus"),
    ("tool_filter_chunks", "tool:filter_chunks"),
];

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct MxaRow {
    query_id: String,
    /// Campaign files use "text" for the query string.
    #[serde(alias = "text")]
    query: String,
    relevant_ids: Vec<String>,
}

fn load_rows(path: &str) -> Result<Vec<MxaRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run make_splits.py first"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<MxaRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// Slot signatures (parameter holders — never called as LMs)
// ---------------------------------------------------------------------------

/// One section of the toast searcher's system prompt. The instruction IS the
/// section text; the episode below shows what behavior it produced.
#[derive(Signature, Clone, Debug)]
struct HarnessSection {
    /// The user's search request, verbatim.
    #[input]
    query: String,

    /// The finished agentic-search episode this prompt section steered: the
    /// search/filter/read tool calls made, the final ranked document ids, and
    /// their retrieval scores against the gold documents.
    #[output]
    episode: String,
}

/// One tool's JSON-schema description in the toast searcher's tool list. The
/// instruction IS the description text the model sees for that tool.
#[derive(Signature, Clone, Debug)]
struct HarnessToolDesc {
    /// The user's search request, verbatim.
    #[input]
    query: String,

    /// The finished agentic-search episode: how (and whether) the agent used
    /// this tool, the final ranked document ids, and their retrieval scores.
    #[output]
    episode: String,
}

// ---------------------------------------------------------------------------
// Sidecar client
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SlotsResponse {
    slots: BTreeMap<String, String>,
    config: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct RolloutResponse {
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    ranked_ids: Vec<String>,
    #[serde(default)]
    trace_summary: String,
    #[serde(default)]
    feedback: String,
    #[serde(default)]
    rounds_executed: Option<u32>,
    #[serde(default)]
    forced_ranking: Option<bool>,
    #[serde(default)]
    wall_s: f64,
}

struct SidecarClient {
    base: String,
    bench: String,
    gen_seed: u64,
    client: reqwest::Client,
}

impl SidecarClient {
    async fn fetch_slots(&self) -> Result<SlotsResponse> {
        let url = format!("{}/slots?bench={}", self.base, self.bench);
        let resp = self.client.get(&url).send().await?.error_for_status()?;
        Ok(resp.json().await?)
    }

    async fn rollout(
        &self,
        query: &str,
        changed_slots: &BTreeMap<String, String>,
    ) -> Result<RolloutResponse> {
        let body = serde_json::json!({
            "bench": self.bench,
            "query": query,
            "slots": if changed_slots.is_empty() {
                serde_json::Value::Null
            } else {
                serde_json::to_value(changed_slots)?
            },
            "seed": self.gen_seed,
        });
        let resp = self
            .client
            .post(format!("{}/rollout", self.base))
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
#[derive(Clone, Debug, Default)]
struct MxaOutput {
    ranked_ids: Vec<String>,
    trace_summary: String,
    error: Option<String>,
    rounds_executed: Option<u32>,
    forced_ranking: Option<bool>,
}

#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct MxaInput {
    query: String,
}

struct MxaSearcher {
    task: Predict<HarnessSection>,
    workflow: Predict<HarnessSection>,
    retrieval: Predict<HarnessSection>,
    context_management: Predict<HarnessSection>,
    examples: Predict<HarnessSection>,
    final_rules: Predict<HarnessSection>,
    tool_search_corpus: Predict<HarnessToolDesc>,
    tool_filter_chunks: Predict<HarnessToolDesc>,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
}

dspy_rs::predictors!(MxaSearcher {
    task,
    workflow,
    retrieval,
    context_management,
    examples,
    final_rules,
    tool_search_corpus,
    tool_filter_chunks
});

impl Module for MxaSearcher {
    type Input = MxaInput;
    type Output = MxaOutput;

    async fn forward(&self, input: MxaInput) -> Result<Predicted<MxaOutput>, PredictError> {
        // Resolve each slot: ambient candidate wins, seed otherwise. Only
        // texts that differ from seed are sent — untouched slots keep the
        // harness's production bytes.
        let mut changed: BTreeMap<String, String> = BTreeMap::new();
        for (name, sidecar_key) in SLOTS {
            if let Some(text) = fx::current_instruction(name) {
                let seed = self.seeds.get(name).map(String::as_str).unwrap_or("");
                if text != seed {
                    changed.insert(sidecar_key.to_string(), text);
                }
            }
        }
        let response = self
            .sidecar
            .rollout(&input.query, &changed)
            .await
            .unwrap_or_else(|e| RolloutResponse {
                error: Some(format!("sidecar transport error: {e}")),
                ..RolloutResponse::default()
            });
        Ok(Predicted::new(
            MxaOutput {
                ranked_ids: response.ranked_ids,
                trace_summary: if response.trace_summary.is_empty() {
                    response.feedback
                } else {
                    response.trace_summary
                },
                error: response.error,
                rounds_executed: response.rounds_executed,
                forced_ranking: response.forced_ranking,
            },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Metric: binary nDCG@k / recall@k (exact scoring.py port)
// ---------------------------------------------------------------------------

fn dedup_at_k(ranked: &[String], k: usize) -> Vec<String> {
    let mut seen = Vec::new();
    for id in ranked {
        if !seen.contains(id) {
            seen.push(id.clone());
        }
        if seen.len() == k {
            break;
        }
    }
    seen
}

fn ndcg_at_k(ranked: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let deduped = dedup_at_k(ranked, k);
    let dcg: f64 = deduped
        .iter()
        .enumerate()
        .map(|(i, id)| {
            if relevant.contains(id) { 1.0 / ((i + 2) as f64).log2() } else { 0.0 }
        })
        .sum();
    let ideal: f64 = (0..k.min(relevant.len())).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    if ideal > 0.0 { dcg / ideal } else { 0.0 }
}

fn recall_at_k(ranked: &[String], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let deduped = dedup_at_k(ranked, k);
    let hits = relevant.iter().filter(|r| deduped.contains(r)).count();
    hits as f64 / relevant.len() as f64
}

struct MxaMetric {
    metric_k: usize,
    /// "ndcg" (the RL reward) or "recall" (LongSEAL's headline metric).
    score_kind: String,
}

impl MxaMetric {
    fn grade(&self, example: &MxaRow, output: &MxaOutput) -> (f64, String) {
        if let Some(error) = &output.error {
            return (0.0, format!("ROLLOUT FAILED (scored 0):\n{error}"));
        }
        let k = self.metric_k;
        let ndcg = ndcg_at_k(&output.ranked_ids, &example.relevant_ids, k);
        let recall = recall_at_k(&output.ranked_ids, &example.relevant_ids, k);
        let score = if self.score_kind == "recall" { recall } else { ndcg };

        let rel = &example.relevant_ids;
        let marks: Vec<String> = dedup_at_k(&output.ranked_ids, k)
            .into_iter()
            .map(|d| {
                let hit = if rel.contains(&d) { "✓" } else { "✗" };
                format!("{d}{hit}")
            })
            .collect();
        let missed: Vec<&String> =
            rel.iter().filter(|r| !output.ranked_ids.contains(r)).collect();
        let mut feedback = format!(
            "nDCG@{k}={ndcg:.3} recall@{k}={recall:.3}\nGOLD DOC IDS ({}): {}\nFINAL RANKING: {}\n",
            rel.len(),
            rel.join(", "),
            if marks.is_empty() { "(empty)".to_string() } else { marks.join(", ") },
        );
        if !missed.is_empty() {
            feedback.push_str(&format!(
                "MISSED GOLD (the search must surface these): {}\n",
                missed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
        feedback.push_str(&output.trace_summary);
        (score, feedback.trim_end().to_string())
    }
}

impl TypedMetric<MxaRow, MxaSearcher> for MxaMetric {
    async fn evaluate(
        &self,
        example: &MxaRow,
        prediction: &Predicted<MxaOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let (score, feedback) = self.grade(example, prediction);
        Ok(Eval::with_feedback(score, feedback))
    }
}

// ---------------------------------------------------------------------------
// Factory / winner assembly
// ---------------------------------------------------------------------------

fn build_module(seeds: &Arc<BTreeMap<String, String>>, sidecar: &Arc<SidecarClient>) -> MxaSearcher {
    let seed = |name: &str| seeds.get(name).cloned().unwrap_or_default();
    MxaSearcher {
        task: Predict::<HarnessSection>::builder()
            .named("task")
            .instruction(seed("task"))
            .build(),
        workflow: Predict::<HarnessSection>::builder()
            .named("workflow")
            .instruction(seed("workflow"))
            .build(),
        retrieval: Predict::<HarnessSection>::builder()
            .named("retrieval")
            .instruction(seed("retrieval"))
            .build(),
        context_management: Predict::<HarnessSection>::builder()
            .named("context_management")
            .instruction(seed("context_management"))
            .build(),
        examples: Predict::<HarnessSection>::builder()
            .named("examples")
            .instruction(seed("examples"))
            .build(),
        final_rules: Predict::<HarnessSection>::builder()
            .named("final_rules")
            .instruction(seed("final_rules"))
            .build(),
        tool_search_corpus: Predict::<HarnessToolDesc>::builder()
            .named("tool_search_corpus")
            .instruction(seed("tool_search_corpus"))
            .build(),
        tool_filter_chunks: Predict::<HarnessToolDesc>::builder()
            .named("tool_filter_chunks")
            .instruction(seed("tool_filter_chunks"))
            .build(),
        seeds: seeds.clone(),
        sidecar: sidecar.clone(),
    }
}

/// Winner's full slot assignment: seeds, overlaid with post-install module
/// state, overlaid with the candidate's slots, touched slot last (same
/// robustness contract as the HoVer rig's winner_map).
fn winner_map(
    seeds: &BTreeMap<String, String>,
    module: Option<&MxaSearcher>,
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

// ---------------------------------------------------------------------------
// Optimize
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn optimize(
    dir: &str,
    bench: &str,
    seeds: Arc<BTreeMap<String, String>>,
    sidecar: Arc<SidecarClient>,
    metric: &MxaMetric,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("MXA_OPT", "ratchet");
    let train_file = env_str("MXA_TRAIN_FILE", &format!("{bench}_train.jsonl"));
    let val_file = env_str("MXA_VAL_FILE", &format!("{bench}_val.jsonl"));
    let trainset = load_rows(&format!("{dir}/data/{train_file}"))?;
    let val = load_rows(&format!("{dir}/data/{val_file}"))?;
    println!(
        "{bench}: {} train / {} val, arm {opt_name}, {} slots, scoring {}@{}",
        trainset.len(),
        val.len(),
        SLOTS.len(),
        metric.score_kind,
        metric.metric_k
    );

    let seed = env_usize("MXA_SEED", 0) as u64;
    let budget = env_usize("MXA_BUDGET", 300);
    let concurrency = env_usize("MXA_CONCURRENCY", 4);
    let mut module = build_module(&seeds, &sidecar);

    let prefix = env_str("MXA_OUT_PREFIX", &format!("{bench}_{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.json");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.json");

    if opt_name == "gepa" {
        let iters = env_usize("MXA_ITERS", 30);
        let minibatch = env_usize("MXA_MINIBATCH", 3);
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
    } else if opt_name == "ratchet" {
        let iters = env_usize("MXA_ITERS", 40);
        let minibatch = env_usize("MXA_MINIBATCH", 3);
        let champion = env_usize("MXA_CHAMPION", 20);
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
            optimizer
        };
        let result = optimizer
            .compile_module_with_valset(&mut module, &trainset, Some(&val), metric)
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
        bail!("unknown MXA_OPT {opt_name:?} (ratchet|gepa)");
    }
    println!("\nwinner written to {out}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let dir = env_str("MXA_DIR", "benchmarks/mxagent");
    let bench = env_str("MXA_BENCH", "longseal");
    let sidecar_base = env_str("MXA_SIDECAR", "http://127.0.0.1:8130");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(1500))
        .build()?;
    let health = client
        .get(format!("{sidecar_base}/health"))
        .send()
        .await
        .map_err(|e| {
            anyhow!(
                "sidecar not reachable at {sidecar_base}: {e} — start \
                 benchmarks/mxagent/sidecar.py (see its header)"
            )
        })?;
    if !health.status().is_success() {
        bail!("sidecar unhealthy at {sidecar_base}");
    }

    let sidecar = Arc::new(SidecarClient {
        base: sidecar_base,
        bench: bench.clone(),
        gen_seed: env_usize("MXA_GEN_SEED", 1005) as u64,
        client,
    });

    // Seeds = the harness's production prompt fragments, extracted by the
    // sidecar from the live render. Snapshot them for provenance.
    let slots_response = sidecar.fetch_slots().await.context("GET /slots failed")?;
    let seeds = Arc::new(slots_response.slots.clone());
    for (name, _) in SLOTS {
        let key = SLOTS.iter().find(|(n, _)| *n == name).map(|(_, k)| *k).unwrap();
        if !seeds.contains_key(key) && !seeds.contains_key(name) {
            bail!("sidecar /slots is missing slot {key:?} — sidecar/example out of sync");
        }
    }
    // Sidecar keys use "tool:<name>"; normalize to predictor names.
    let seeds: Arc<BTreeMap<String, String>> = Arc::new(
        SLOTS
            .iter()
            .map(|(name, key)| {
                let text = slots_response
                    .slots
                    .get(*key)
                    .or_else(|| slots_response.slots.get(*name))
                    .cloned()
                    .unwrap_or_default();
                (name.to_string(), text)
            })
            .collect(),
    );
    std::fs::write(
        format!("{dir}/seeds_{bench}_snapshot.json"),
        serde_json::to_string_pretty(&*seeds)?,
    )?;
    let metric_k = slots_response.config["metric_k"].as_u64().unwrap_or(10) as usize;
    let metric = MxaMetric {
        metric_k,
        score_kind: env_str("MXA_SCORE", "ndcg"),
    };
    println!(
        "bench {bench}: {} seed slots, metric_k {metric_k}, config {}",
        seeds.len(),
        slots_response.config
    );

    let reflection_lm = LM::builder()
        .model(env_str("MXA_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;
    configure(reflection_lm.clone());

    optimize(&dir, &bench, seeds, sidecar, &metric, reflection_lm).await
}
