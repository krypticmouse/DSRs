//! HoVer multi-hop retrieval rig (the GEPA paper's 4-module claim-verification
//! pipeline; mirrors the IFBench rig, docs/handoff-showdown-campaign.md) —
//! the first MULTI-MODULE rig: four optimizable Predicts, jointly searched.
//!
//! The program (GEPA paper appendix / DSPy hover tutorial, faithfully):
//! three BM25 retrieval hops over the wiki17 abstracts corpus. Hop 1 searches
//! the raw claim (no query Predict, as in the paper's program); the two query
//! writers generate the queries for hops 2 and 3:
//!
//!   docs1 = search(claim)
//!   summary_1 = summarize1(claim, passages=docs1)
//!   q2 = hop1_query(claim, summary_1)            # first generated hop query
//!   docs2 = search(q2)
//!   summary_2 = summarize2(claim, context=summary_1, passages=docs2)
//!   q3 = hop2_query(claim, summary_1, summary_2) # second generated hop query
//!   docs3 = search(q3)
//!   score = |gold ∩ (docs1 ∪ docs2 ∪ docs3).titles| / |gold|
//!
//! Four leaves registered via `predictors!`: hop1_query, summarize1,
//! hop2_query, summarize2 — these names are the candidate slot keys, the
//! serve-protocol override keys, and the winner-JSON keys everywhere in the
//! rig. The metric is pure title-set comparison against the row's
//! `gold_titles` (deterministic, judge-free, zero API cost); feedback lists
//! the missed gold documents and which hop retrieved junk.
//!
//! One binary, two modes:
//!
//! - `HOVER_MODE=optimize` (default): optimize the four instructions with
//!   `HOVER_OPT=temper|ratchet|hybrid|gepa` and write the winner as the rig's
//!   shared 4-key JSON: {"hop1_query": ..., "summarize1": ...,
//!   "hop2_query": ..., "summarize2": ...}.
//!   NOTE on current engine semantics: candidates carry a full 4-slot
//!   assignment (`GEPACandidate::slots`) but the final `install` writes the
//!   winner's single touched slot; the winner JSON is assembled from seeds +
//!   post-install module state + the candidate's full slot assignment, so it
//!   stays correct as the engine's joint multi-leaf install lands.
//! - `HOVER_MODE=serve`: JSONL rollout server on stdin/stdout — the Python
//!   reference arm (`benchmarks/hover/run_python_gepa.py`) and every eval
//!   script drive rollouts through this, so all arms share byte-identical
//!   pipeline calls and retrieval scoring. The protocol takes per-request
//!   instruction overrides FOR ALL FOUR modules (null / missing = seed):
//!   Request:  {"id":1,"instructions":{"hop1_query":"..."|null,...}|null,
//!              "split":"val","idx":3}
//!   Response: {"id":1,"score":0.667,"correct":false,"n_gold":3,"n_found":2,
//!              "found_titles":[...],"missed_titles":[...],"feedback":"...",
//!              "trace":{"hop1_query":"<q2>","summarize1":"<s1>",
//!                       "hop2_query":"<q3>","summarize2":"<s2>",
//!                       "hops":[{"query":"...","titles":[...],"gold_hits":1}]}}
//!
//! The pipeline is tool-free: stages call the local BM25 search server
//! (`benchmarks/hover/search_server.py`, :8125) over plain HTTP between
//! Predicts — no agent loop, no stop tool.
//!
//! Env: HOVER_DIR (benchmarks/hover), HOVER_MODE, HOVER_OPT, HOVER_SEED,
//! HOVER_STUDENT, HOVER_TEACHER (openrouter:openai/gpt-5.6-terra),
//! HOVER_ITERS, HOVER_MINIBATCH, HOVER_BUDGET (600), HOVER_CHAMPION,
//! HOVER_VAL_FILE, HOVER_OUT_PREFIX, HOVER_CONCURRENCY (8), HOVER_SEARCH
//! (http://127.0.0.1:8125), HOVER_TOPK (5), HOVER_MAX_TOKENS,
//! HOVER_MXBAI_URL.
//!
//! Run from the repo root (search server must be up):
//!     set -a; source .env; set +a
//!     cd benchmarks/hover && .venv/bin/python search_server.py &
//!     HOVER_OPT=ratchet cargo run --release --example 34-hover

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use dspy_rs::{
    Eval, GEPA, LM, Module, Predict, PredictError, Predicted, Predictors, Ratchet, Signature,
    SlotCredit, TypedMetric, compound, configure, init_tracing, temper,
};
use serde::{Deserialize, Serialize};

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// The four module names, in pipeline order — candidate slot keys, serve
/// override keys, and winner-JSON keys. Must match `predictors!` below and
/// MODULES in benchmarks/hover/common.py.
const MODULE_NAMES: [&str; 4] = ["hop1_query", "summarize1", "hop2_query", "summarize2"];

/// Seed instructions (DSPy auto-generated style, the GEPA paper's HoVer
/// seeds). MUST stay byte-identical to SEED_INSTRUCTIONS in
/// benchmarks/hover/common.py.
const SEED_HOP1_QUERY: &str = "Given the fields claim, summary_1, produce the fields query.";
const SEED_SUMMARIZE1: &str = "Given the fields claim, passages, produce the fields summary.";
const SEED_HOP2_QUERY: &str =
    "Given the fields claim, summary_1, summary_2, produce the fields query.";
const SEED_SUMMARIZE2: &str =
    "Given the fields claim, context, passages, produce the fields summary.";

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct HoverRow {
    id: String,
    claim: String,
    gold_titles: Vec<String>,
    num_hops: u32,
}

fn load_rows(path: &str) -> Result<Vec<HoverRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run fetch_data.py first, from the repo root"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<HoverRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

// ---------------------------------------------------------------------------
// The four optimizable signatures (2 query writers + 2 summarizers)
// ---------------------------------------------------------------------------

/// Given the fields claim, summary_1, produce the fields query.
#[derive(Signature, Clone, Debug)]
struct Hop1Query {
    /// The claim being verified.
    #[input]
    claim: String,

    /// Summary of the evidence retrieved so far.
    #[input]
    summary_1: String,

    /// A search query over Wikipedia abstracts.
    #[output]
    query: String,
}

/// Given the fields claim, passages, produce the fields summary.
#[derive(Signature, Clone, Debug)]
struct Summarize1 {
    /// The claim being verified.
    #[input]
    claim: String,

    /// Retrieved Wikipedia passages.
    #[input]
    passages: String,

    /// A summary of what the passages establish about the claim.
    #[output]
    summary: String,
}

/// Given the fields claim, summary_1, summary_2, produce the fields query.
#[derive(Signature, Clone, Debug)]
struct Hop2Query {
    /// The claim being verified.
    #[input]
    claim: String,

    /// Summary of the first hop's evidence.
    #[input]
    summary_1: String,

    /// Summary of the second hop's evidence.
    #[input]
    summary_2: String,

    /// A search query over Wikipedia abstracts.
    #[output]
    query: String,
}

/// Given the fields claim, context, passages, produce the fields summary.
#[derive(Signature, Clone, Debug)]
struct Summarize2 {
    /// The claim being verified.
    #[input]
    claim: String,

    /// Summary of the evidence retrieved in earlier hops.
    #[input]
    context: String,

    /// Retrieved Wikipedia passages.
    #[input]
    passages: String,

    /// A summary of what the passages establish about the claim.
    #[output]
    summary: String,
}

// ---------------------------------------------------------------------------
// BM25 search client (the local search server; deterministic, no API cost)
// ---------------------------------------------------------------------------

#[derive(Deserialize, Clone)]
struct SearchHit {
    title: String,
    snippet: String,
}

#[derive(Deserialize)]
struct SearchResults {
    results: Vec<SearchHit>,
}

struct SearchClient {
    base: String,
    client: reqwest::Client,
    topk: usize,
}

/// Minimal percent-encoding for a query-string value (reqwest 0.13 gates
/// `.query()` behind a feature this workspace doesn't enable).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

impl SearchClient {
    /// Top-k BM25 search; retries transient failures, returns empty on
    /// persistent failure (the metric then reports the hop as empty).
    async fn search(&self, query: &str) -> Vec<SearchHit> {
        let query = if query.trim().is_empty() { "(empty)" } else { query };
        let url = format!("{}/search?q={}&k={}", self.base, urlencode(query), self.topk);
        let mut last = String::new();
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(1 << (attempt - 1))).await;
            }
            let sent = self.client.get(&url).send().await;
            match sent {
                Ok(resp) => match resp.json::<SearchResults>().await {
                    Ok(r) => return r.results,
                    Err(e) => last = e.to_string(),
                },
                Err(e) => last = e.to_string(),
            }
        }
        eprintln!("search server unreachable, empty hop: {last}");
        Vec::new()
    }
}

fn format_passages(hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return "(no results)".to_string();
    }
    hits.iter()
        .enumerate()
        .map(|(i, h)| format!("[{}] {} | {}", i + 1, h.title, h.snippet))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// The pipeline under optimization
// ---------------------------------------------------------------------------

/// What one rollout produced — everything the metric and the serve trace need.
#[dspy_rs::Schema]
#[derive(Clone, Debug, Default)]
struct HoverOutput {
    /// Union of retrieved titles across the three hops (deduped, in order).
    retrieved_titles: Vec<String>,
    /// The three executed search queries: [claim, q2, q3].
    queries: Vec<String>,
    /// Per-hop retrieved titles, JSON-encoded [[..],[..],[..]].
    hop_titles: String,
    summary_1: String,
    summary_2: String,
}

/// Pipeline input (rows project into it by field name).
#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct HoverInput {
    claim: String,
}

struct HoverPipeline {
    hop1_query: Predict<Hop1Query>,
    summarize1: Predict<Summarize1>,
    hop2_query: Predict<Hop2Query>,
    summarize2: Predict<Summarize2>,
    search: Arc<SearchClient>,
}

// All four leaves are optimizable surfaces — the optimizer sees 4 slots.
dspy_rs::predictors!(HoverPipeline { hop1_query, summarize1, hop2_query, summarize2 });

/// open-instruct's remove_thinking_section: reasoning students wrap the real
/// output in think/answer tags; downstream stages must see what follows them.
fn strip_thinking(response: &str) -> String {
    let response = response.replace("<|assistant|>", "");
    let response = response.rsplit("</think>").next().unwrap_or(&response);
    response.replace("<answer>", "").replace("</answer>", "").trim().to_string()
}

/// Runs one Predict stage with retries. A parse failure still carries the
/// model's final text (missing field markers etc.) — downstream stages
/// consume free text, so use it instead of failing the rollout. Returns
/// `None` on persistent failure; callers apply a deterministic fallback so
/// the retrieval pipeline always completes and gets scored.
async fn run_stage<Fut>(name: &str, mut attempt_fn: impl FnMut() -> Fut) -> Option<String>
where
    Fut: std::future::Future<Output = Result<String, PredictError>>,
{
    let mut last: Option<PredictError> = None;
    for attempt in 0..3u32 {
        if attempt > 0 {
            tokio::time::sleep(std::time::Duration::from_secs(3u64 << (attempt - 1))).await;
        }
        match attempt_fn().await {
            Ok(text) => {
                let text = strip_thinking(&text);
                if !text.is_empty() {
                    return Some(text);
                }
                last = None;
            }
            Err(PredictError::Parse { raw_response, .. }) if !raw_response.trim().is_empty() => {
                let text = strip_thinking(&raw_response);
                if !text.is_empty() {
                    return Some(text);
                }
                last = None;
            }
            Err(e) => last = Some(e),
        }
    }
    eprintln!("stage {name} failed after retries, using fallback: {last:?}");
    None
}

fn truncate_words(text: &str, words: usize) -> String {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() <= words {
        return text.to_string();
    }
    format!("{} ...", parts[..words].join(" "))
}

impl Module for HoverPipeline {
    type Input = HoverInput;
    type Output = HoverOutput;

    async fn forward(&self, input: HoverInput) -> Result<Predicted<HoverOutput>, PredictError> {
        let claim = input.claim;

        // Hop 1 — search the raw claim (the paper's program: no query Predict).
        let hits1 = self.search.search(&claim).await;
        let passages1 = format_passages(&hits1);

        let summary_1 = run_stage("summarize1", || {
            let (claim, passages) = (claim.clone(), passages1.clone());
            async move {
                self.summarize1
                    .call(Summarize1Input { claim, passages })
                    .await
                    .map(|out| out.summary.trim().to_string())
            }
        })
        .await
        // Fallback: pass the raw passages through, truncated — keeps the
        // downstream query writers informed when the summarizer is down.
        .unwrap_or_else(|| truncate_words(&passages1, 60));

        // Hop 2 — first generated query.
        let q2 = run_stage("hop1_query", || {
            let (claim, summary_1) = (claim.clone(), summary_1.clone());
            async move {
                self.hop1_query
                    .call(Hop1QueryInput { claim, summary_1 })
                    .await
                    .map(|out| out.query.trim().to_string())
            }
        })
        .await
        .unwrap_or_else(|| claim.clone()); // fallback: re-search the claim

        let hits2 = self.search.search(&q2).await;
        let passages2 = format_passages(&hits2);

        let summary_2 = run_stage("summarize2", || {
            let (claim, context, passages) =
                (claim.clone(), summary_1.clone(), passages2.clone());
            async move {
                self.summarize2
                    .call(Summarize2Input { claim, context, passages })
                    .await
                    .map(|out| out.summary.trim().to_string())
            }
        })
        .await
        .unwrap_or_else(|| truncate_words(&passages2, 60));

        // Hop 3 — second generated query.
        let q3 = run_stage("hop2_query", || {
            let (claim, summary_1, summary_2) =
                (claim.clone(), summary_1.clone(), summary_2.clone());
            async move {
                self.hop2_query
                    .call(Hop2QueryInput { claim, summary_1, summary_2 })
                    .await
                    .map(|out| out.query.trim().to_string())
            }
        })
        .await
        .unwrap_or_else(|| claim.clone());

        let hits3 = self.search.search(&q3).await;

        let per_hop: Vec<Vec<String>> = [&hits1, &hits2, &hits3]
            .iter()
            .map(|hits| hits.iter().map(|h| h.title.clone()).collect())
            .collect();
        let mut retrieved_titles = Vec::new();
        for titles in &per_hop {
            for t in titles {
                if !retrieved_titles.contains(t) {
                    retrieved_titles.push(t.clone());
                }
            }
        }
        Ok(Predicted::new(
            HoverOutput {
                retrieved_titles,
                queries: vec![claim, q2, q3],
                hop_titles: serde_json::to_string(&per_hop).unwrap_or_else(|_| "[]".into()),
                summary_1,
                summary_2,
            },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

/// Instruction set for one pipeline build (seed = the four SEED_* consts).
#[derive(Clone)]
struct Instructions {
    hop1_query: String,
    summarize1: String,
    hop2_query: String,
    summarize2: String,
}

impl Default for Instructions {
    fn default() -> Self {
        Self {
            hop1_query: SEED_HOP1_QUERY.to_string(),
            summarize1: SEED_SUMMARIZE1.to_string(),
            hop2_query: SEED_HOP2_QUERY.to_string(),
            summarize2: SEED_SUMMARIZE2.to_string(),
        }
    }
}

impl Instructions {
    /// Seed overridden per module by a serve request's map (null = seed).
    fn with_overrides(overrides: Option<&HashMap<String, Option<String>>>) -> Self {
        let mut instructions = Self::default();
        let Some(map) = overrides else { return instructions };
        for (name, value) in map {
            let Some(text) = value else { continue };
            match name.as_str() {
                "hop1_query" => instructions.hop1_query = text.clone(),
                "summarize1" => instructions.summarize1 = text.clone(),
                "hop2_query" => instructions.hop2_query = text.clone(),
                "summarize2" => instructions.summarize2 = text.clone(),
                other => eprintln!("unknown module in instructions map: {other:?}"),
            }
        }
        instructions
    }
}

struct PipelineFactory {
    student_lm: LM,
    search: Arc<SearchClient>,
}

impl PipelineFactory {
    fn build(&self, instructions: &Instructions) -> HoverPipeline {
        HoverPipeline {
            hop1_query: Predict::<Hop1Query>::builder()
                .named("hop1_query")
                .lm(self.student_lm.clone())
                .instruction(instructions.hop1_query.clone())
                .build(),
            summarize1: Predict::<Summarize1>::builder()
                .named("summarize1")
                .lm(self.student_lm.clone())
                .instruction(instructions.summarize1.clone())
                .build(),
            hop2_query: Predict::<Hop2Query>::builder()
                .named("hop2_query")
                .lm(self.student_lm.clone())
                .instruction(instructions.hop2_query.clone())
                .build(),
            summarize2: Predict::<Summarize2>::builder()
                .named("summarize2")
                .lm(self.student_lm.clone())
                .instruction(instructions.summarize2.clone())
                .build(),
            search: self.search.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Metric: gold-document retrieval (deterministic, judge-free)
// ---------------------------------------------------------------------------

/// Everything one graded rollout produced, for serve-mode responses.
struct Graded {
    score: f64,
    correct: bool,
    n_gold: usize,
    n_found: usize,
    found_titles: Vec<String>,
    missed_titles: Vec<String>,
    feedback: String,
    per_hop_gold_hits: Vec<usize>,
}

fn grade(example: &HoverRow, output: &HoverOutput) -> Graded {
    let n_gold = example.gold_titles.len();
    let found_titles: Vec<String> = example
        .gold_titles
        .iter()
        .filter(|t| output.retrieved_titles.contains(t))
        .cloned()
        .collect();
    let missed_titles: Vec<String> = example
        .gold_titles
        .iter()
        .filter(|t| !output.retrieved_titles.contains(*t))
        .cloned()
        .collect();
    let n_found = found_titles.len();
    let score = if n_gold == 0 { 0.0 } else { n_found as f64 / n_gold as f64 };
    let correct = n_gold > 0 && n_found == n_gold;

    let per_hop: Vec<Vec<String>> =
        serde_json::from_str(&output.hop_titles).unwrap_or_default();
    let per_hop_gold_hits: Vec<usize> = per_hop
        .iter()
        .map(|titles| titles.iter().filter(|t| example.gold_titles.contains(t)).count())
        .collect();

    let mut feedback = if correct {
        format!("Correct. All {n_gold} gold supporting documents were retrieved.\n")
    } else {
        format!(
            "Retrieved {n_found}/{n_gold} gold supporting documents.\nMISSED gold documents \
             (the queries must surface these): {}\n",
            missed_titles.join("; ")
        )
    };
    for (i, titles) in per_hop.iter().enumerate() {
        let hits = per_hop_gold_hits.get(i).copied().unwrap_or(0);
        let query = output
            .queries
            .get(i)
            .map(|q| truncate_words(q, 30))
            .unwrap_or_default();
        let verdict = if titles.is_empty() {
            "EMPTY (search failed or no results)".to_string()
        } else if hits == 0 {
            "0 gold hits — this hop retrieved junk".to_string()
        } else {
            format!("{hits} gold hit(s)")
        };
        let label = if i == 0 { "hop 1 (query = the claim)".to_string() } else { format!("hop {}", i + 1) };
        feedback.push_str(&format!(
            "{label}: query \"{query}\" -> [{}] — {verdict}\n",
            titles.join("; ")
        ));
    }
    Graded {
        score,
        correct,
        n_gold,
        n_found,
        found_titles,
        missed_titles,
        feedback: feedback.trim_end().to_string(),
        per_hop_gold_hits,
    }
}

struct HoverMetric;

impl TypedMetric<HoverRow, HoverPipeline> for HoverMetric {
    async fn evaluate(
        &self,
        example: &HoverRow,
        prediction: &Predicted<HoverOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let graded = grade(example, prediction);
        Ok(Eval::with_feedback(graded.score, graded.feedback))
    }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    /// Per-module instruction overrides; null / missing key = that module's
    /// seed instruction. `null` for the whole map = all-seed program.
    instructions: Option<HashMap<String, Option<String>>>,
    split: String,
    idx: usize,
}

#[derive(Serialize)]
struct HopTrace {
    query: String,
    titles: Vec<String>,
    gold_hits: usize,
}

#[derive(Serialize)]
struct ServeTrace {
    hop1_query: String,
    summarize1: String,
    hop2_query: String,
    summarize2: String,
    hops: Vec<HopTrace>,
}

#[derive(Serialize)]
struct ServeResponse {
    id: u64,
    score: f64,
    correct: bool,
    n_gold: usize,
    n_found: usize,
    found_titles: Vec<String>,
    missed_titles: Vec<String>,
    feedback: String,
    trace: ServeTrace,
}

async fn serve(dir: &str, factory: Arc<PipelineFactory>) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    // Every data/*.jsonl is addressable as a split, so eval scripts can
    // target paraphrase-variant files without code changes.
    let mut splits: HashMap<String, Vec<HoverRow>> = HashMap::new();
    for entry in std::fs::read_dir(format!("{dir}/data"))? {
        let path = entry?.path();
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        if name.starts_with("corpus") || name.starts_with("hover_") {
            continue; // the 5M-doc corpus / raw pools are not eval splits
        }
        splits.insert(name.to_string(), load_rows(&path.to_string_lossy())?);
    }
    eprintln!(
        "serve: splits loaded: {:?}",
        splits.iter().map(|(k, v)| (k.clone(), v.len())).collect::<Vec<_>>()
    );

    let concurrency = env_usize("HOVER_CONCURRENCY", 8);
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
        let (factory, splits, stdout, semaphore) =
            (factory.clone(), splits.clone(), stdout.clone(), semaphore.clone());
        tasks.push(tokio::spawn(async move {
            let _permit = semaphore.acquire().await.expect("semaphore open");
            let Some(row) = splits.get(&request.split).and_then(|s| s.get(request.idx)) else {
                eprintln!("unknown split/idx: {}/{}", request.split, request.idx);
                return;
            };
            let instructions = Instructions::with_overrides(request.instructions.as_ref());
            let pipeline = factory.build(&instructions);
            let prediction = pipeline
                .forward(HoverInput { claim: row.claim.clone() })
                .await
                .expect("forward degrades to fallbacks, never Err");
            let graded = grade(row, &prediction);
            let per_hop: Vec<Vec<String>> =
                serde_json::from_str(&prediction.hop_titles).unwrap_or_default();
            let hops = per_hop
                .into_iter()
                .enumerate()
                .map(|(i, titles)| HopTrace {
                    query: prediction.queries.get(i).cloned().unwrap_or_default(),
                    titles,
                    gold_hits: graded.per_hop_gold_hits.get(i).copied().unwrap_or(0),
                })
                .collect();
            let response = ServeResponse {
                id: request.id,
                score: graded.score,
                correct: graded.correct,
                n_gold: graded.n_gold,
                n_found: graded.n_found,
                found_titles: graded.found_titles,
                missed_titles: graded.missed_titles,
                feedback: graded.feedback,
                trace: ServeTrace {
                    // Trace keys carry the *generated* stage outputs: the
                    // module named hop1_query produced hop 2's query, etc.
                    hop1_query: prediction.queries.get(1).cloned().unwrap_or_default(),
                    summarize1: prediction.summary_1.clone(),
                    hop2_query: prediction.queries.get(2).cloned().unwrap_or_default(),
                    summarize2: prediction.summary_2.clone(),
                    hops,
                },
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

/// Assembles the winner's full 4-module assignment: seeds, overlaid with the
/// post-install module state (when provided), overlaid with the candidate's
/// full slot assignment, with the candidate's touched slot last. Robust to
/// both the current single-slot install and the joint multi-leaf install.
fn winner_map(
    module: Option<&HoverPipeline>,
    slots: &BTreeMap<String, String>,
    module_name: &str,
    instruction: &str,
) -> BTreeMap<String, String> {
    let seeds = Instructions::default();
    let mut map = BTreeMap::from([
        ("hop1_query".to_string(), seeds.hop1_query),
        ("summarize1".to_string(), seeds.summarize1),
        ("hop2_query".to_string(), seeds.hop2_query),
        ("summarize2".to_string(), seeds.summarize2),
    ]);
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
    factory: &PipelineFactory,
    metric: &HoverMetric,
    reflection_lm: LM,
) -> Result<()> {
    let opt_name = env_str("HOVER_OPT", "ratchet");
    let train = load_rows(&format!("{dir}/data/train.jsonl"))?;
    let val_file = env_str("HOVER_VAL_FILE", "val.jsonl");
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
        "{dir}: {} train (+variants) / {} val, arm {opt_name}, 4 modules",
        trainset.len(),
        val.len()
    );

    let seed = env_usize("HOVER_SEED", 0) as u64;
    let budget = env_usize("HOVER_BUDGET", 600);
    let concurrency = env_usize("HOVER_CONCURRENCY", 8);
    let mut pipeline = factory.build(&Instructions::default());

    let prefix = env_str("HOVER_OUT_PREFIX", &format!("{opt_name}_winner"));
    let out = format!("{dir}/{prefix}_s{seed}.json");
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.json");

    if opt_name == "gepa" {
        let iters = env_usize("HOVER_ITERS", 40);
        let minibatch = env_usize("HOVER_MINIBATCH", 4);
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
            .compile_module_with_valset(&mut pipeline, &trainset, Some(&val), metric)
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
            Some(&pipeline),
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
            .map(|c| winner_map(None, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    } else {
        // COMPOUND keeps its preset cycle shape (iters 150 / minibatch 3 /
        // P×N 2×2) unless overridden; the v2-era arms keep theirs.
        let is_compound = opt_name == "compound";
        let iters = env_usize("HOVER_ITERS", if is_compound { 150 } else { 60 });
        let minibatch = env_usize("HOVER_MINIBATCH", if is_compound { 3 } else { 5 });
        let champion = env_usize("HOVER_CHAMPION", 30);
        println!(
            "{opt_name} seed {seed}, iters {iters}, minibatch {minibatch}, budget {budget}, champion {champion}"
        );
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
            // RATCHET v4 COMPOUND (docs/ratchet-v4-design.md): HoVer is the
            // measured no-headroom cell, so this arm tests the false-ship
            // guard — certification must refuse noise and ship the seed.
            // train150 carve: champion 30 + certify 60 leaves 60 batch cols.
            "compound" => {
                let mut optimizer = compound();
                optimizer.prompt_model = Some(reflection_lm);
                optimizer.certify_reserve = env_usize("HOVER_CERTIFY_RESERVE", 60);
                optimizer.certify_budget = env_usize("HOVER_CERTIFY_BUDGET", 150);
                optimizer
            }
            other => bail!("unknown HOVER_OPT {other:?} (temper|ratchet|hybrid|compound|gepa)"),
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
            // HOVER_REPAIR=<probes> turns on counterfactual stage-repair
            // credit assignment (SlotCredit::Repair) with that many probes
            // per generation. 0 / unset keeps round-robin.
            let repair_probes = env_usize("HOVER_REPAIR", 0);
            if repair_probes > 0 {
                optimizer.slot_credit = SlotCredit::Repair;
                optimizer.repair_probes = repair_probes;
                optimizer.repair_attempts = env_usize("HOVER_REPAIR_ATTEMPTS", 2);
                println!(
                    "stage-repair credit assignment ON ({repair_probes} probes/gen, {} attempts)",
                    optimizer.repair_attempts
                );
            }
            optimizer
        };
        let result = optimizer
            .compile_module_with_valset(&mut pipeline, &trainset, Some(&val), metric)
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
        if result.certify_cols_used > 0 || result.certified.is_some() {
            println!(
                "CERTIFICATION: {:?} after {} cols, mean paired delta {:+.4}, e-value {:.2}",
                result.certified,
                result.certify_cols_used,
                result.certify_mean_delta,
                result.certify_e_value
            );
        }
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
            Some(&pipeline),
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
            .map(|c| winner_map(None, &c.slots, &c.module_name, &c.instruction))
            .unwrap_or_else(|| winner.clone());
        write_winner(&out_argmax, &argmax)?;
    }
    println!("\nwinner written to {out}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let mode = env_str("HOVER_MODE", "optimize");
    if mode != "serve" {
        init_tracing()?;
    }
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let dir = env_str("HOVER_DIR", "benchmarks/hover");
    let search_base = env_str("HOVER_SEARCH", "http://127.0.0.1:8125");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let health = client
        .get(format!("{search_base}/health"))
        .send()
        .await
        .map_err(|e| {
            anyhow!(
                "search server not reachable at {search_base}: {e} — start \
                 benchmarks/hover/search_server.py"
            )
        })?;
    if !health.status().is_success() {
        bail!("search server unhealthy at {search_base}");
    }

    // "mxbai:<model>" routes to the Mixedbread OpenAI-compatible endpoint;
    // anything else goes through OpenRouter. Queries and summaries are short;
    // 1024 tokens is generous.
    let student_spec = env_str("HOVER_STUDENT", "openai/gpt-oss-120b");
    let max_tokens = env_usize("HOVER_MAX_TOKENS", 1024) as u32;
    let student_lm = if let Some(model) = student_spec.strip_prefix("mxbai:") {
        LM::builder()
            .base_url(env_str("HOVER_MXBAI_URL", "https://api.mixedbread.com/v1"))
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
        .model(env_str("HOVER_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;
    // Global LM only renders reflection prompts; scoring never touches an LM
    // (the metric is pure title-set comparison).
    configure(reflection_lm.clone());

    let search = Arc::new(SearchClient {
        base: search_base,
        client,
        topk: env_usize("HOVER_TOPK", 5),
    });
    let factory = PipelineFactory { student_lm, search };
    let metric = HoverMetric;

    match mode.as_str() {
        "serve" => serve(&dir, Arc::new(factory)).await,
        "optimize" => optimize(&dir, &factory, &metric, reflection_lm).await,
        other => bail!("unknown HOVER_MODE {other:?} (optimize|serve)"),
    }
}
