//! FORAGE (v5) end-to-end simulation — the content-bandwidth mechanism
//! measured in isolation, deterministically, no LLM in the loop.
//!
//! Same world as `test_ratchet_compound.rs` (50 pinnable facts at the
//! measured DocsQA density, ±0.25 judge noise, scripted student that
//! answers with whatever fact tokens its instruction carries). Both arms
//! run the SAME `compound()` engine at the same seed and budget; the only
//! difference is what the Rewrite teacher can see:
//!
//! - baseline: `TeacherContext::Minibatch` — the tiniest-win teacher pins
//!   exactly one fact per proposal from its 3-example reflection chunk
//!   (the measured v2–v4 content bandwidth).
//! - forage: `TeacherContext::Digest` — the same teacher, given the
//!   engine-computed contrastive digest of every paid rollout, pins up to
//!   10 open-check facts per proposal.
//!
//! Every pinned fact is a true improvement by construction, so shipped
//! fact count is the exact true quality gap — the simulation measures
//! whether whole-ledger evidence closes the harvest gap that minibatch
//! reflection leaves (the gepa2000-vs-compound gap measured live in
//! `docs/compound-validation-prereg.md` amendments 4–5).
//!
//! A third arm exercises `TeacherContext::Tools` end-to-end: the scripted
//! teacher answers the Rewrite call with a `forage_open_checks` tool call,
//! reads the tool result, and pins from it — proving the depth-1 agentic
//! reflection loop works over scripted clients. Its plain-predictor
//! operator calls deliberately no-op, so every shipped fact is attributable
//! to the tool loop.

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::Result;
use dspy_rs::optimizer::forage::{CheckExtractor, TeacherContext};
use dspy_rs::optimizer::ratchet::{Ratchet, RatchetResult, compound};
use dspy_rs::{
    Eval, LM, LMClient, Module, Predict, PredictError, Predicted, ScriptedCompletionModel,
    Signature, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::{Text, ToolCall, ToolFunction};

const FACTS: usize = 50;
const BASE: f64 = 0.35;
const FACT_LIFT: f64 = 0.5;
const NOISE: f64 = 0.25;
const TRAIN: usize = 240;
const VAL: usize = 60;
const PIN_CAP: usize = 10;

fn fact_of(qid: usize) -> usize {
    qid % FACTS
}

fn fact_token(k: usize) -> String {
    format!("F{k:02}")
}

fn facts_in(text: &str) -> BTreeSet<usize> {
    (0..FACTS)
        .filter(|&k| text.contains(&fact_token(k)))
        .collect()
}

fn judge_noise(qid: usize, answer: &str) -> f64 {
    let mut hash = 1469598103934665603u64;
    for byte in answer.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash ^= qid as u64;
    hash = hash.wrapping_mul(1099511628211);
    ((hash % 10_001) as f64 / 10_000.0 - 0.5) * 2.0 * NOISE
}

#[derive(Signature, Clone, Debug)]
/// Answer the question from the docs.
struct QaSig {
    #[input]
    question: String,

    #[output]
    answer: String,
}

struct QaModule {
    predictor: Predict<QaSig>,
}

dspy_rs::predictors!(QaModule { predictor });

impl Module for QaModule {
    type Input = QaSigInput;
    type Output = QaSigOutput;

    async fn forward(&self, input: QaSigInput) -> Result<Predicted<QaSigOutput>, PredictError> {
        self.predictor.call(input).await
    }
}

struct FactMetric;

impl TypedMetric<(QaSigInput, QaSigOutput), QaModule> for FactMetric {
    async fn evaluate(
        &self,
        example: &(QaSigInput, QaSigOutput),
        prediction: &Predicted<QaSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let qid: usize = example.0.question.trim_start_matches('q').parse()?;
        let needed = fact_of(qid);
        let covered = prediction.answer.contains(&fact_token(needed));
        let lift = if covered { FACT_LIFT } else { 0.0 };
        let score = (BASE + lift + judge_noise(qid, &prediction.answer)).clamp(0.0, 1.0);
        let feedback = if covered {
            "ok".to_string()
        } else {
            format!("MISSING::{}", fact_token(needed))
        };
        Ok(Eval::with_feedback(score, feedback))
    }
}

fn field_response(fields: &[(&str, &str)]) -> AssistantContent {
    let mut text = String::new();
    for (name, value) in fields {
        text.push_str(&format!("[[ ## {name} ## ]]\n{value}\n\n"));
    }
    text.push_str("[[ ## completed ## ]]\n");
    AssistantContent::Text(Text { text })
}

async fn make_lm(client: ScriptedCompletionModel) -> LM {
    temp_env::async_with_vars(
        [("OPENAI_API_KEY", Some("test"))],
        LM::builder().model("openai:gpt-4o-mini".to_string()).build(),
    )
    .await
    .unwrap()
    .with_client(LMClient::Scripted(client))
    .await
    .unwrap()
}

fn student_client() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let mut answer = String::from("facts:");
        for k in 0..FACTS {
            let token = fact_token(k);
            if haystack.contains(&token) {
                answer.push(' ');
                answer.push_str(&token);
            }
        }
        Ok(field_response(&[("answer", &answer)]))
    })
}

const INSTRUCTION_FIELDS: [&str; 3] = ["current_instruction", "instruction_a", "instruction_b"];

fn field_section(haystack: &str, name: &str) -> String {
    let marker = format!("{name} ## ]]");
    let mut out = String::new();
    let mut rest = haystack;
    while let Some(pos) = rest.find(&marker) {
        let after = &rest[pos + marker.len()..];
        let end = after.find("[[ ##").unwrap_or(after.len());
        out.push_str(&after[..end]);
        out.push('\n');
        rest = &after[end..];
    }
    out
}

fn current_facts(haystack: &str) -> BTreeSet<usize> {
    let mut current: BTreeSet<usize> = BTreeSet::new();
    for field in INSTRUCTION_FIELDS {
        current.extend(facts_in(&field_section(haystack, field)));
    }
    current
}

fn pinned_response(facts: &BTreeSet<usize>, summary: &str) -> AssistantContent {
    let list = facts
        .iter()
        .map(|&k| fact_token(k))
        .collect::<Vec<_>>()
        .join(" ");
    let instruction = format!("Answer the question. Facts: {list}")
        .trim_end()
        .to_string();
    field_response(&[
        ("improved_instruction", &instruction),
        ("edit_summary", summary),
    ])
}

/// The tiniest-win teacher (baseline content bandwidth): pins exactly ONE
/// fact named `MISSING::` in its minibatch feedback.
fn minibatch_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let current = current_facts(&haystack);
        let missing: BTreeSet<usize> = (0..FACTS)
            .filter(|&k| haystack.contains(&format!("MISSING::{}", fact_token(k))))
            .filter(|k| !current.contains(k))
            .collect();
        let (facts, summary) = match missing.iter().next() {
            Some(&new_fact) => {
                let mut facts = current.clone();
                facts.insert(new_fact);
                (facts, format!("pin {}", fact_token(new_fact)))
            }
            None => (current, "no new fact to pin".to_string()),
        };
        Ok(pinned_response(&facts, &summary))
    })
}

/// The same teacher given whole-ledger evidence: pins up to PIN_CAP facts
/// named in the FORAGE DIGEST (falling back to the minibatch `MISSING::`
/// lines when the digest names nothing new).
fn digest_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let current = current_facts(&haystack);
        let digest = haystack
            .find("FORAGE DIGEST")
            .map(|pos| &haystack[pos..])
            .unwrap_or("");
        let mut fresh: Vec<usize> = facts_in(digest)
            .into_iter()
            .filter(|k| !current.contains(k))
            .collect();
        if fresh.is_empty() {
            fresh = (0..FACTS)
                .filter(|&k| haystack.contains(&format!("MISSING::{}", fact_token(k))))
                .filter(|k| !current.contains(k))
                .collect();
        }
        fresh.truncate(PIN_CAP);
        if fresh.is_empty() {
            return Ok(pinned_response(&current, "no new fact to pin"));
        }
        let summary = format!(
            "pin {}",
            fresh
                .iter()
                .map(|&k| fact_token(k))
                .collect::<Vec<_>>()
                .join(" ")
        );
        let mut facts = current;
        facts.extend(fresh);
        Ok(pinned_response(&facts, &summary))
    })
}

/// Tools-mode teacher: answers a tooled Rewrite call with a
/// `forage_open_checks` tool call, then pins from the tool result. Plain
/// (untooled) operator calls deliberately no-op so every shipped fact is
/// attributable to the tool loop.
fn tools_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let history = format!("{:?}", request.chat_history);
        let has_forage_tools = format!("{:?}", request.tools).contains("forage_open_checks");
        let preamble = format!("{:?}", request.preamble);
        let haystack = format!("{preamble} {history}");
        let current = current_facts(&haystack);
        if !has_forage_tools {
            // Plain predictor (Patch/Simplify/Merge or repair): no-op.
            return Ok(pinned_response(&current, "no-op"));
        }
        // Tool result marker: the open_checks listing header.
        if let Some(pos) = history.find("open checks 0..") {
            let mut fresh: Vec<usize> = facts_in(&history[pos..])
                .into_iter()
                .filter(|k| !current.contains(k))
                .collect();
            fresh.truncate(PIN_CAP);
            let summary = format!("pin {} facts from the forage tool", fresh.len());
            let mut facts = current;
            facts.extend(fresh);
            return Ok(pinned_response(&facts, &summary));
        }
        Ok(AssistantContent::ToolCall(ToolCall::new(
            "tc-forage".to_string(),
            ToolFunction {
                name: "forage_open_checks".to_string(),
                arguments: serde_json::json!({"offset": 0}),
            },
        )))
    })
}

fn check_extractor() -> CheckExtractor {
    Arc::new(|feedback: &str| {
        feedback
            .lines()
            .filter_map(|line| line.strip_prefix("MISSING::"))
            .map(str::to_string)
            .collect()
    })
}

fn dataset(range: std::ops::Range<usize>) -> Vec<(QaSigInput, QaSigOutput)> {
    range
        .map(|qid| {
            (
                QaSigInput {
                    question: format!("q{qid}"),
                },
                QaSigOutput { answer: "-".into() },
            )
        })
        .collect()
}

async fn run_arm(
    mut optimizer: Ratchet,
    teacher: ScriptedCompletionModel,
) -> Result<(RatchetResult, BTreeSet<usize>)> {
    optimizer.prompt_model = Some(make_lm(teacher).await);
    optimizer.max_rollouts = Some(3200);
    optimizer.seed = Some(0);
    optimizer.eval_concurrency = 16;
    let mut module = QaModule {
        predictor: Predict::<QaSig>::builder()
            .instruction("Answer the question. Facts:")
            .lm(make_lm(student_client()).await)
            .build(),
    };
    let train = dataset(0..TRAIN);
    let val = dataset(TRAIN..TRAIN + VAL);
    let result = optimizer
        .compile_module_with_valset(&mut module, &train, Some(&val), &FactMetric)
        .await?;
    let shipped = facts_in(&result.best_candidate.instruction);
    Ok((result, shipped))
}

fn report(name: &str, result: &RatchetResult, facts: &BTreeSet<usize>) {
    println!(
        "{name}: shipped {} facts, certified {:?} on {} cols (mean {:+.4}, e {:.1}), {} rollouts",
        facts.len(),
        result.certified,
        result.certify_cols_used,
        result.certify_mean_delta,
        result.certify_e_value,
        result.total_rollouts,
    );
}

#[tokio::test]
async fn forage_digest_closes_the_content_bandwidth_gap() {
    // Baseline: compound engine, minibatch teacher context.
    let (base_result, base_facts) = run_arm(compound(), minibatch_teacher())
        .await
        .expect("baseline arm runs");

    // FORAGE: identical engine + digest context + the check extractor.
    let mut forage = compound();
    forage.teacher_context = TeacherContext::Digest;
    forage.check_extractor = Some(check_extractor());
    let (forage_result, forage_facts) = run_arm(forage, digest_teacher())
        .await
        .expect("forage arm runs");

    report("minibatch", &base_result, &base_facts);
    report("forage", &forage_result, &forage_facts);

    assert!(base_result.total_rollouts <= 3200);
    assert!(forage_result.total_rollouts <= 3200);

    // The mechanism under test: whole-ledger evidence must out-harvest
    // minibatch reflection by a clear margin at equal budget.
    assert!(
        forage_facts.len() >= base_facts.len() + 10,
        "forage should out-harvest minibatch context: forage {} vs minibatch {}",
        forage_facts.len(),
        base_facts.len()
    );

    // The honesty invariant is untouched: a non-seed ship is certified.
    match forage_result.certified {
        Some(true) => {
            assert!(forage_result.certify_e_value >= 10.0);
            assert!(forage_result.certify_mean_delta > 0.0);
        }
        other => panic!(
            "forage should certify its harvested pile, got {other:?} after {} cols (mean {:+.4})",
            forage_result.certify_cols_used, forage_result.certify_mean_delta
        ),
    }

    // True (noise-free) quality: each fact is FACT_LIFT on its column share.
    let true_gain = |facts: &BTreeSet<usize>| facts.len() as f64 * FACT_LIFT / FACTS as f64;
    assert!(
        true_gain(&forage_facts) - true_gain(&base_facts) >= 0.05,
        "true shipped gain: forage {:+.4} vs minibatch {:+.4}",
        true_gain(&forage_facts),
        true_gain(&base_facts)
    );
}

/// Anchored-accretion teacher: first attempt deliberately drops the base
/// (style-rewrite behavior); the ANCHOR VIOLATION retry addendum makes it
/// comply. Exercises the guard's retry path on every proposal.
fn anchored_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let current = current_facts(&haystack);
        let digest = haystack
            .find("FORAGE DIGEST")
            .map(|pos| &haystack[pos..])
            .unwrap_or("");
        let mut facts = current.clone();
        facts.extend(facts_in(digest).into_iter().take(PIN_CAP));
        let list = facts
            .iter()
            .map(|&k| fact_token(k))
            .collect::<Vec<_>>()
            .join(" ");
        if haystack.contains("ANCHOR VIOLATION") {
            let instruction = format!("Answer the question. Facts: {list}")
                .trim_end()
                .to_string();
            return Ok(field_response(&[
                ("improved_instruction", &instruction),
                ("edit_summary", "compliant retry"),
            ]));
        }
        // Violating first attempt: base restyled away.
        let instruction = format!("Be terse. PINNED: {list}").trim_end().to_string();
        Ok(field_response(&[
            ("improved_instruction", &instruction),
            ("edit_summary", "styled rewrite"),
        ]))
    })
}

const SEED_BASE: &str = "Answer the question. Facts:";
const VERBOSE_PER_FACT: usize = 130;

/// Density teacher: accretion proposals write VERBOSE one-per-line entries
/// (~60 chars/fact); Condense requests compress every present fact into the
/// telegraphic list, dropping none. Measures the representation-vs-content
/// split the Condense operator claims.
fn density_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let current = current_facts(&haystack);
        if haystack.contains("merged or generalized") {
            // Condense: compact representation, zero content loss.
            let list = current
                .iter()
                .map(|&k| fact_token(k))
                .collect::<Vec<_>>()
                .join(" ");
            let instruction = format!("{SEED_BASE} {list}").trim_end().to_string();
            return Ok(field_response(&[
                ("improved_instruction", &instruction),
                ("edit_summary", "condensed entries"),
            ]));
        }
        let digest = haystack
            .find("FORAGE DIGEST")
            .map(|pos| &haystack[pos..])
            .unwrap_or("");
        let fresh: Vec<usize> = facts_in(digest)
            .into_iter()
            .filter(|k| !current.contains(k))
            .take(PIN_CAP)
            .collect();
        let mut facts = current.clone();
        facts.extend(fresh.iter().copied());
        // Verbose accretion, dedupe-compliant (the density directive):
        // already-present facts stay as they are; only NEWLY pinned facts
        // get a padded NOTE line. Uncondensed lineages still bloat at
        // ~VERBOSE_PER_FACT chars per fact — but a Condense promotion
        // resets the accumulated bloat, and later proposals only re-add
        // notes for genuinely new pins.
        let list = facts
            .iter()
            .map(|&k| fact_token(k))
            .collect::<Vec<_>>()
            .join(" ");
        let notes = fresh
            .iter()
            .map(|&k| {
                format!(
                    "- NOTE: token {} was missed in evaluation and matters a lot; the graders repeatedly flagged its absence across the sampled batch, so keep it stated.",
                    fact_token(k)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let instruction = format!("{SEED_BASE} {list}\n{notes}").trim_end().to_string();
        Ok(field_response(&[
            ("improved_instruction", &instruction),
            ("edit_summary", "verbose pins"),
        ]))
    })
}

#[tokio::test]
async fn condense_shrinks_representation_without_losing_teachings() {
    let mut arm = compound();
    arm.teacher_context = TeacherContext::Digest;
    arm.check_extractor = Some(check_extractor());
    arm.anchored_accretion = true;
    let (result, facts) = run_arm(arm, density_teacher())
        .await
        .expect("density arm runs");
    report("condense", &result, &facts);
    println!("ops: {:?}", result.operator_stats);
    for trial in &result.trials {
        if trial.operator.name() == "condense" {
            println!("condense trial gen {}: {:?}", trial.generation, trial.outcome);
        }
    }

    let condense_promoted = result.trials.iter().any(|trial| {
        matches!(trial.outcome, dspy_rs::optimizer::ratchet::TrialOutcome::Promoted { .. })
            && trial.operator.name() == "condense"
    });
    assert!(condense_promoted, "a Condense child should have been promoted");
    assert!(result.best_candidate.instruction.starts_with(SEED_BASE));
    assert!(facts.len() >= 15, "harvest stays healthy, got {}", facts.len());
    // Representation bound: an uncondensed lineage carries ~60 chars/fact
    // of NOTE bloat; a condensed winner must sit well under that.
    assert!(
        result.best_candidate.instruction.len()
            < SEED_BASE.len() + VERBOSE_PER_FACT * facts.len(),
        "winner should be denser than the verbose bound: {} chars for {} facts",
        result.best_candidate.instruction.len(),
        facts.len()
    );
}

/// A teacher that never complies with the anchor.
fn stubborn_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|_request| {
        Ok(field_response(&[
            ("improved_instruction", "Be terse. PINNED: F00"),
            ("edit_summary", "styled rewrite"),
        ]))
    })
}

#[tokio::test]
async fn anchored_accretion_forces_the_base_through_the_retry() {
    let mut arm = compound();
    arm.teacher_context = TeacherContext::Digest;
    arm.check_extractor = Some(check_extractor());
    arm.anchored_accretion = true;
    let (result, facts) = run_arm(arm, anchored_teacher())
        .await
        .expect("anchored arm runs");
    report("anchored", &result, &facts);

    // Every proposal violated once and complied on retry: the harvest
    // still lands, every winner opens with the frozen base, and Simplify
    // never got a slot.
    assert!(result.anchor_retries > 0, "the guard should have retried");
    assert_eq!(result.anchor_skipped, 0);
    assert!(
        result
            .best_candidate
            .instruction
            .starts_with("Answer the question. Facts:"),
        "winner must open with the frozen base: {:?}",
        &result.best_candidate.instruction[..60.min(result.best_candidate.instruction.len())]
    );
    assert!(facts.len() >= 20, "anchored harvest still lands, got {}", facts.len());
    assert!(
        !result
            .operator_stats
            .iter()
            .any(|(op, _, _)| op.contains("simplify")),
        "Simplify must be ineligible under anchoring"
    );
}

#[tokio::test]
async fn stubborn_violations_are_skipped_and_the_seed_ships() {
    let mut arm = compound();
    arm.teacher_context = TeacherContext::Digest;
    arm.check_extractor = Some(check_extractor());
    arm.anchored_accretion = true;
    arm.num_iterations = 4;
    let (result, facts) = run_arm(arm, stubborn_teacher())
        .await
        .expect("stubborn arm runs");
    report("stubborn", &result, &facts);
    assert!(result.anchor_skipped > 0, "persistent violations must be skipped");
    assert!(facts.is_empty(), "no unanchored content may ship");
}

// ---------------------------------------------------------------------------
// Junk-pin mix world (ASSAY Stage 1) — the honesty upgrade the live arms
// demanded. The original world gives every pinnable fact a true +.01, which
// is exactly the assumption five preregistered arms falsified: harvested
// sentences are plausible but NOT all worth keeping. Here the 50 facts are
// a 15/20/15 mix by `k % 10`:
//
//   - k%10 ∈ {0,1,2} → TRUE:  +0.5 on the fact's columns (+.01 pooled)
//   - k%10 ∈ {3..=6} → ZERO:   0.0 (plausible, closes its check, worthless)
//   - k%10 ∈ {7,8,9} → JUNK:  −0.25 (wrong/stale; −.005 pooled)
//
// Every class still reports `MISSING::Fxx` when uncovered — a junk fact
// LOOKS exactly like an open check, so an indiscriminate teacher pins it.
// Junk harm is DIFFUSE, matching the live regression channel: a wrong or
// stale claim carried in the prompt taxes every rollout a little (a judge
// dinging unsupported claims), rather than tanking one question — which is
// exactly why a 3-column gate is structurally blind to it while ~150
// columns of cumulative evidence are not. Success = the re-audit arm ships
// a certified bundle whose harmful-fact count is near zero while the
// plain-forage arm's argmax is the indiscriminate pile, at the same
// budget, with certified evidence reflecting the difference. (Zero-value
// facts are score-invisible by construction: no candidate-level mechanism
// can price a +0.0; stripping those is Stage 2's per-pin job.)
// ---------------------------------------------------------------------------

const JUNK_VAL: usize = 120;
const JUNK_TRAIN: usize = 1_000;
/// Telegraphic accretion granularity: 1–2 sentences per proposal (forage3's
/// measured granularity), which is what per-sentence evidence can price.
const JUNK_PIN_CAP: usize = 3;
/// Per-column tax for each junk fact the answer carries: −.005 pooled per
/// junk fact, the measured wrong/stale-sentence class.
const JUNK_TAX: f64 = 0.005;
/// Judge noise for this world: the live DocsQA identical-arm floor
/// measured ±.011 on 100-column means (per-column paired sd ≈ .11), so
/// ±.15 uniform is the honest calibration — the original world's ±.25 is
/// a coverage-judge stress level that buries −.005-class pins entirely.
const JUNK_NOISE: f64 = 0.15;
const JUNK_BUDGET: usize = 12_000;

fn junk_judge_noise(qid: usize, answer: &str) -> f64 {
    judge_noise(qid, answer) / NOISE * JUNK_NOISE
}

fn is_true_fact(k: usize) -> bool {
    k % 10 <= 2
}

fn is_junk_fact(k: usize) -> bool {
    k % 10 >= 7
}

fn true_facts(facts: &BTreeSet<usize>) -> usize {
    facts.iter().filter(|&&k| is_true_fact(k)).count()
}

fn junk_facts(facts: &BTreeSet<usize>) -> usize {
    facts.iter().filter(|&&k| is_junk_fact(k)).count()
}

fn zero_facts(facts: &BTreeSet<usize>) -> usize {
    facts.len() - true_facts(facts) - junk_facts(facts)
}

struct JunkMetric;

impl TypedMetric<(QaSigInput, QaSigOutput), QaModule> for JunkMetric {
    async fn evaluate(
        &self,
        example: &(QaSigInput, QaSigOutput),
        prediction: &Predicted<QaSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let qid: usize = example.0.question.trim_start_matches('q').parse()?;
        let needed = fact_of(qid);
        let covered = prediction.answer.contains(&fact_token(needed));
        // Own-column reward only for TRUE facts; zero and junk facts close
        // their check without paying anything.
        let lift = if covered && is_true_fact(needed) { FACT_LIFT } else { 0.0 };
        // Diffuse junk tax: every junk fact the answer carries costs a
        // little on EVERY column (−.005 pooled per junk fact) — invisible
        // to a 3-column gate, visible to cumulative evidence.
        let tax = JUNK_TAX
            * facts_in(&prediction.answer)
                .iter()
                .filter(|&&k| is_junk_fact(k))
                .count() as f64;
        let score =
            (BASE + lift - tax + junk_judge_noise(qid, &prediction.answer)).clamp(0.0, 1.0);
        // The deception under test: an uncovered check reads as MISSING no
        // matter what covering it is actually worth.
        let feedback = if covered {
            "ok".to_string()
        } else {
            format!("MISSING::{}", fact_token(needed))
        };
        Ok(Eval::with_feedback(score, feedback))
    }
}

/// One-teaching-per-line accretion format: the engine's Stage 2 pin unit
/// is the appended line, so the junk world writes its pile the way forage3
/// does — base first, one telegraphic entry per line.
fn junk_pinned_response(facts: &BTreeSet<usize>, summary: &str) -> AssistantContent {
    let mut instruction = String::from(SEED_BASE);
    for &k in facts {
        instruction.push('\n');
        instruction.push_str(&fact_token(k));
    }
    field_response(&[
        ("improved_instruction", &instruction),
        ("edit_summary", summary),
    ])
}

/// Indiscriminate digest teacher: pins up to JUNK_PIN_CAP facts drawn
/// pseudo-randomly (request-hash keyed) from the digest's fresh facts —
/// it cannot tell true from junk, exactly like a live teacher reading
/// plausible open checks. The hash draw gives sibling proposals different
/// bags, which is the variation selection needs to work with.
fn indiscriminate_teacher() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let current = current_facts(&haystack);
        let digest = haystack
            .find("FORAGE DIGEST")
            .map(|pos| &haystack[pos..])
            .unwrap_or("");
        let mut fresh: Vec<usize> = facts_in(digest)
            .into_iter()
            .filter(|k| !current.contains(k))
            .collect();
        if fresh.is_empty() {
            fresh = (0..FACTS)
                .filter(|&k| haystack.contains(&format!("MISSING::{}", fact_token(k))))
                .filter(|k| !current.contains(k))
                .collect();
        }
        if fresh.is_empty() {
            return Ok(junk_pinned_response(&current, "no new fact to pin"));
        }
        // Deterministic per-request shuffle: FNV over the haystack seeds a
        // small LCG that picks the bag.
        let mut state = 1469598103934665603u64;
        for byte in haystack.as_bytes() {
            state ^= u64::from(*byte);
            state = state.wrapping_mul(1099511628211);
        }
        let mut bag: BTreeSet<usize> = BTreeSet::new();
        while bag.len() < JUNK_PIN_CAP.min(fresh.len()) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bag.insert(fresh[(state >> 33) as usize % fresh.len()]);
        }
        let summary = format!(
            "pin {}",
            bag.iter().map(|&k| fact_token(k)).collect::<Vec<_>>().join(" ")
        );
        let mut facts = current;
        facts.extend(bag);
        Ok(junk_pinned_response(&facts, &summary))
    })
}

async fn run_junk_arm(mut optimizer: Ratchet) -> Result<(RatchetResult, BTreeSet<usize>)> {
    optimizer.prompt_model = Some(make_lm(indiscriminate_teacher()).await);
    optimizer.max_rollouts = Some(JUNK_BUDGET);
    optimizer.seed = Some(0);
    optimizer.eval_concurrency = 16;
    // The live-arm accretion economics: gate accept + scout slice; the
    // certification (frozen) takes the ship decision in both arms.
    optimizer.accept_mode = dspy_rs::optimizer::ratchet::AcceptMode::Gate;
    optimizer.scout_size = 16;
    // Concentrate accretion on the compounding chain: telegraphic bags of
    // ~3 need chain depth ~10+ to pin the corpus, and the sim measures
    // composition, not exploration breadth. Same value in both arms.
    optimizer.parent_greed = 0.8;
    optimizer.teacher_context = TeacherContext::Digest;
    optimizer.check_extractor = Some(check_extractor());
    let mut module = QaModule {
        predictor: Predict::<QaSig>::builder()
            .instruction("Answer the question. Facts:")
            .lm(make_lm(student_client()).await)
            .build(),
    };
    let train = dataset(0..JUNK_TRAIN);
    let val = dataset(JUNK_TRAIN..JUNK_TRAIN + JUNK_VAL);
    let result = optimizer
        .compile_module_with_valset(&mut module, &train, Some(&val), &JunkMetric)
        .await?;
    let shipped = facts_in(&result.best_candidate.instruction);
    Ok((result, shipped))
}

fn report_junk(name: &str, result: &RatchetResult, facts: &BTreeSet<usize>) {
    println!(
        "{name}: shipped {} facts (true {}, zero {}, junk {}), certified {:?} \
         (mean {:+.4}, e {:.1}, {} cols), audit {} evicted / {} rollouts, \
         pins {}/{} kept ({} probe rollouts), {} total rollouts, \
         {} promoted / {} gate-killed",
        facts.len(),
        true_facts(facts),
        zero_facts(facts),
        junk_facts(facts),
        result.certified,
        result.certify_mean_delta,
        result.certify_e_value,
        result.certify_cols_used,
        result.audit_evicted,
        result.audit_rollouts,
        result.pins_kept,
        result.pins_tested,
        result.pin_probe_rollouts,
        result.total_rollouts,
        result.all_candidates.len(),
        result.gate_rejected,
    );
}

#[tokio::test]
async fn assay_composes_certified_pins_while_the_plain_arm_ships_junk() {
    // Plain FORAGE on the junk world: same engine, no assay.
    let (plain_result, plain_facts) = run_junk_arm(compound())
        .await
        .expect("plain junk arm runs");

    // ASSAY: identical config + Stage 1 (re-audit evidence scheduling) +
    // Stage 2 (per-pin ablation probes, stopped e-BH composition).
    let mut assay = compound();
    assay.reaudit_budget_frac = 0.4;
    assay.reaudit_block = 8;
    assay.pin_assay = true;
    assay.pin_probe_cols = 20;
    assay.pin_probe_budget = 2_200;
    let (assay_result, assay_facts) = run_junk_arm(assay)
        .await
        .expect("assay junk arm runs");

    // The plain arm's search product is its argmax — certification refusing
    // a junk-diluted pile is the known-good frozen behavior, so the
    // indiscriminate-pile claim is measured on what the SEARCH built.
    let plain_argmax = facts_in(
        &plain_result
            .best_by_mean
            .as_ref()
            .map(|candidate| candidate.instruction.clone())
            .unwrap_or_else(|| plain_result.best_candidate.instruction.clone()),
    );
    let assay_argmax = facts_in(
        &assay_result
            .best_by_mean
            .as_ref()
            .map(|candidate| candidate.instruction.clone())
            .unwrap_or_else(|| assay_result.best_candidate.instruction.clone()),
    );

    report_junk("plain", &plain_result, &plain_facts);
    report_junk("assay", &assay_result, &assay_facts);
    println!(
        "plain argmax: true {} zero {} junk {} | assay argmax: true {} zero {} junk {}",
        true_facts(&plain_argmax),
        zero_facts(&plain_argmax),
        junk_facts(&plain_argmax),
        true_facts(&assay_argmax),
        zero_facts(&assay_argmax),
        junk_facts(&assay_argmax),
    );

    // Budget parity: the audit is paid out of the same cap.
    assert!(plain_result.total_rollouts <= JUNK_BUDGET);
    assert!(assay_result.total_rollouts <= JUNK_BUDGET);

    // The plain arm pins indiscriminately: its search product carries a
    // substantial worthless fraction and ships real junk (this is what
    // tonight's live arms did).
    assert!(
        junk_facts(&plain_argmax) + zero_facts(&plain_argmax) >= 12,
        "the plain arm's argmax should be the indiscriminate pile, carried {} worthless facts",
        junk_facts(&plain_argmax) + zero_facts(&plain_argmax)
    );
    assert!(
        junk_facts(&plain_facts) >= 4,
        "the plain arm should ship harmful pins, shipped {}",
        junk_facts(&plain_facts)
    );

    // The mechanism under test: per-pin evidence prices junk AND inert
    // pins out of the shipped bundle — near-zero junk fraction.
    assert!(
        junk_facts(&assay_facts) <= 1,
        "the assay should reject harmful pins: shipped {} junk facts (plain argmax carried {})",
        junk_facts(&assay_facts),
        junk_facts(&plain_argmax)
    );
    assert!(
        zero_facts(&assay_facts) <= 2,
        "inert pins cannot earn evidence and must be dropped: shipped {} zero-value facts",
        zero_facts(&assay_facts)
    );
    // ...without buying purity by refusing to harvest.
    assert!(
        true_facts(&assay_facts) >= 10,
        "the harvest must stay healthy: {} of 15 true facts",
        true_facts(&assay_facts)
    );
    // Both stages engaged: evidence was actually re-scheduled and pins were
    // actually probed and filtered.
    assert!(assay_result.audit_rollouts > 0, "stage 1 must have spent its allowance");
    assert!(
        assay_result.pins_tested > 0
            && assay_result.pins_kept < assay_result.pins_tested,
        "stage 2 must probe and filter: kept {} of {}",
        assay_result.pins_kept,
        assay_result.pins_tested
    );

    // Certified evidence must reflect the difference: the audited bundle
    // certifies affirmatively, and a certified plain pile would carry a
    // visibly junk-diluted mean.
    assert!(
        matches!(assay_result.certified, Some(true)),
        "a clean pile must certify, got {:?} (mean {:+.4}, e {:.1})",
        assay_result.certified,
        assay_result.certify_mean_delta,
        assay_result.certify_e_value
    );
    if matches!(plain_result.certified, Some(true)) {
        // The composed bundle certifies at least as strongly as the pile
        // (its true mean is ~.04 higher) while carrying a fraction of the
        // content — evidence-dense, not just smaller.
        assert!(
            assay_result.certify_mean_delta >= plain_result.certify_mean_delta - 0.005,
            "certified means must reflect the junk gap: assay {:+.4} vs plain {:+.4}",
            assay_result.certify_mean_delta,
            plain_result.certify_mean_delta
        );
        assert!(
            assay_facts.len() * 2 <= plain_facts.len(),
            "the composed bundle should be at most half the pile: {} vs {}",
            assay_facts.len(),
            plain_facts.len()
        );
    }
}

/// Defaults byte-preserve: `reaudit_budget_frac` 0.0 spends nothing, evicts
/// nobody, and leaves every report field at its v2 value.
#[test]
fn assay_knobs_default_off() {
    let optimizer = compound();
    assert_eq!(optimizer.reaudit_budget_frac, 0.0);
    assert_eq!(optimizer.reaudit_block, 8);
    assert!(!optimizer.pin_assay);
    assert_eq!(optimizer.pin_fdr_alpha, 0.1);
    assert_eq!(optimizer.pin_probe_repeats, 1);
}

#[tokio::test]
async fn forage_tools_loop_pins_through_the_tool_call() {
    let mut arm = compound();
    arm.teacher_context = TeacherContext::Tools;
    arm.check_extractor = Some(check_extractor());
    let (result, facts) = run_arm(arm, tools_teacher()).await.expect("tools arm runs");
    report("tools", &result, &facts);

    // Plain-operator calls no-op by construction, so any shipped fact
    // proves the tool loop ran end-to-end: tool call → executor →
    // result → pinned instruction → promoted → certified ship.
    assert!(
        facts.len() >= 3,
        "the tooled Rewrite path should ship tool-sourced facts, got {}",
        facts.len()
    );
    assert_eq!(result.certified, Some(true));
}

/// E3 SUBSET SCREEN on the junk world: same engine and budget as the
/// stage-2 assay arm, but per-pin credit comes from one shared pool of
/// subset-ablation rollouts (lasso on inclusion vectors) instead of one
/// starved e-process per pin. Success = the same three-class separation
/// (junk ≈ 0, inert stripped, harvest healthy) with a certified composed
/// bundle — the screen only points the frozen certification at a better
/// bundle, it never ships anything itself.
#[tokio::test]
async fn subset_screen_composes_true_pins_from_one_shared_rollout_pool() {
    let mut screen = compound();
    screen.reaudit_budget_frac = 0.4;
    screen.reaudit_block = 8;
    screen.pin_assay = true;
    screen.pin_probe_budget = 2_200;
    // Conditioning over coverage-slack: the slice needs one column per
    // PILE pin (stratified round-robin fills those first), and every
    // column saved buys more draws — the regression's sample size.
    // ~32 cols / ~68 draws puts draws/pins ≈ 2.2, where the noise
    // floors hold.
    screen.pin_screen_draws = 70;
    screen.pin_screen_cols = 32;
    let (result, facts) = run_junk_arm(screen).await.expect("screen junk arm runs");
    report_junk("screen", &result, &facts);

    assert!(result.total_rollouts <= JUNK_BUDGET);
    assert!(
        result.pin_probe_rollouts <= 2_200,
        "the screen must respect the probe budget: {}",
        result.pin_probe_rollouts
    );
    assert!(
        junk_facts(&facts) <= 1,
        "the screen should reject harmful pins: shipped {} junk facts",
        junk_facts(&facts)
    );
    assert!(
        zero_facts(&facts) <= 2,
        "inert pins price at ~0 and must be dropped: shipped {}",
        zero_facts(&facts)
    );
    assert!(
        true_facts(&facts) >= 10,
        "the harvest must stay healthy: {} of 15 true facts",
        true_facts(&facts)
    );
    assert!(
        result.pins_tested > 0 && result.pins_kept < result.pins_tested,
        "the screen must price and filter: kept {} of {}",
        result.pins_kept,
        result.pins_tested
    );
    assert!(
        matches!(result.certified, Some(true)),
        "the composed bundle must certify, got {:?} (mean {:+.4}, e {:.1})",
        result.certified,
        result.certify_mean_delta,
        result.certify_e_value
    );
}

/// The rank fallback under a deliberately STARVED screen (few draws, the
/// live-noise regime where honest floors keep almost nothing): composition
/// must still be true-ENRICHED — the certifier owns the ship decision, so
/// the screen only has to beat the indiscriminate pile, not certify pins.
#[tokio::test]
async fn screen_rank_fallback_stays_true_enriched_when_floors_starve() {
    let mut screen = compound();
    screen.reaudit_budget_frac = 0.4;
    screen.reaudit_block = 8;
    screen.pin_assay = true;
    // ~23 draws for ~31 pins: floors correctly refuse near-everything.
    screen.pin_probe_budget = 1_000;
    screen.pin_screen_draws = 40;
    screen.pin_screen_cols = 32;
    screen.pin_screen_min_keep = 10;
    let (result, facts) = run_junk_arm(screen).await.expect("fallback junk arm runs");
    report_junk("fallback", &result, &facts);

    assert!(result.total_rollouts <= JUNK_BUDGET);
    assert!(
        result.pins_kept >= 6,
        "the fallback must compose a bundle even when floors starve: kept {}",
        result.pins_kept
    );
    assert!(
        junk_facts(&facts) <= 1,
        "ranking must still exclude harmful pins: shipped {} junk",
        junk_facts(&facts)
    );
    assert!(
        true_facts(&facts) >= 6,
        "the ranked bundle must be true-enriched: {} true",
        true_facts(&facts)
    );
}
