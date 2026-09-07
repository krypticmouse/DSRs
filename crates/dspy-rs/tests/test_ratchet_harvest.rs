//! RATCHET v6 HARVEST simulation — the DocsQA economics as the 2026-09-01
//! forensics actually measured them (scratchpad reports F1/F2/F3/F4; memory
//! `docsqa-cite-term-2026-09-01`), no LLM in the loop:
//!
//! - ONE global answer policy (a citation footer, `RULE_CITE`) worth +.10
//!   on ~94% of columns — 100% of pip-gepa's holdout gain on the live cell;
//! - corpus facts worth +.45 on their own column (~+.01 pooled each);
//! - JUNK pins (−.005 diffuse tax per token) and WRONG pins (−.15 on the
//!   two columns they mislead) — the teacher's measured miss rate;
//! - judge noise ±.20, deterministic per (question, answer) = CRN.
//!
//! The scripted teacher proposes the policy at a LOW base rate — and only
//! when it is told to look for answer policies (the `policy_channel`
//! directive is in its task description): our live arms' content-steered
//! operators never emitted the global template in ~150 proposals across
//! three sessions, while pip-gepa's plain rewrite found it once in 50.
//! Otherwise it pins the first fact the feedback names — TRUE 60% /
//! JUNK 25% / WRONG 15% by hash.
//!
//! Under test is the ENGINE + the channel, not the LLM: does HARVEST
//! (on-target screen, full-slice ranking, no kills, Rewrite-only, policy
//! channel, pawl) ship a certified winner that carries the policy and a
//! cleaner fact pile than the measured live configuration (held-out gate
//! B, scout-16 ranking, mixed operator portfolio) at the same budget?
//!
//! The v6.4 `Ledger` world (bottom of the file) exercises the lineage-diff
//! clause credit and the post-certification Occam pruning: line-per-clause
//! instructions, a policy the teacher finds only after it has pinned a few
//! facts, pins that help only the row they were harvested from unless they
//! transfer (1 in 5), and judge noise common to every candidate on a column
//! (CRN limit) so the attribution is exact.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use dspy_rs::optimizer::ratchet::{AcceptMode, Ratchet, RatchetResult, TrialOutcome, compound, harvest};
use dspy_rs::{
    Eval, LM, LMClient, Module, Predict, PredictError, Predicted, ScriptedCompletionModel,
    Signature, TypedMetric,
};
use rig::completion::AssistantContent;
use rig::message::Text;

const FACTS: usize = 40;
const BASE: f64 = 0.30;
const FACT_LIFT: f64 = 0.45;
const CITE_LIFT: f64 = 0.10;
const WRONG_PEN: f64 = 0.15;
const JUNK_TAX: f64 = 0.005;
const NOISE: f64 = 0.20;
const TRAIN: usize = 120;
const CERT: usize = 200;
const VAL: usize = 45;
/// Per-proposal chance the teacher proposes the policy when it is looking
/// for one and the feedback names the missing citation.
const P_RULE: u64 = 8;
const RULE: &str = "RULE_CITE";
const SEED_TEXT: &str = "Answer the question. Facts:";

/// Whether a pinned fact helps questions OUTSIDE the training rows it was
/// harvested from. The DocsQA forensics measured ~zero net transfer of
/// gepa2000's 73 pinned facts to holdout coverage (F2: topic-pinned rows
/// +.011, other rows −.012); the "additive" world where every fact
/// transfers is the assumption the earlier simulators made — kept here as
/// the other regime so the engine is not tuned to one task.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum World {
    /// Only one fact in five transfers beyond its training rows.
    DocsQa,
    /// Every fact transfers everywhere (the old simulators' assumption).
    Additive,
    /// v6.4 lineage-credit world: DocsQa transfer (1 in 5), but a pin helps
    /// a training row only when harvested from that very row (its source
    /// tag rides along), and judge noise is per column, common to every
    /// candidate — the CRN limit where lineage credit is exact.
    Ledger,
}

fn transfers(world: World, k: usize) -> bool {
    match world {
        World::Additive => true,
        World::DocsQa | World::Ledger => k.is_multiple_of(5),
    }
}

/// The row a Ledger-world pin was harvested from, as a token the student
/// echoes (`Q017`); the question input itself is lowercase (`q17`).
fn source_token(qid: usize) -> String {
    format!("Q{qid:03}")
}

fn fact_of(qid: usize) -> usize {
    qid % FACTS
}

/// 15 of 16 questions require a citation (holdout100: 94 of 100).
fn cite_needed(qid: usize) -> bool {
    qid % 16 != 0
}

fn fact_token(k: usize) -> String {
    format!("F{k:02}")
}

fn wrong_token(k: usize) -> String {
    format!("W{k:02}")
}

fn facts_in(text: &str) -> BTreeSet<usize> {
    (0..FACTS).filter(|&k| text.contains(&fact_token(k))).collect()
}

fn wrongs_in(text: &str) -> BTreeSet<usize> {
    (0..FACTS).filter(|&k| text.contains(&wrong_token(k))).collect()
}

fn junk_in(text: &str) -> usize {
    text.split_whitespace()
        .filter(|tok| tok.starts_with('J') && tok[1..].chars().all(|c| c.is_ascii_digit()))
        .count()
}

fn fnv(text: &str) -> u64 {
    let mut hash = 1469598103934665603u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// Deterministic judge noise per (question, answer content): the same
/// content on the same column always scores identically (common random
/// numbers), different content redraws.
fn judge_noise(qid: usize, answer: &str) -> f64 {
    let mut hash = fnv(answer);
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

struct DocsMetric {
    world: World,
}

impl TypedMetric<(QaSigInput, QaSigOutput), QaModule> for DocsMetric {
    async fn evaluate(
        &self,
        example: &(QaSigInput, QaSigOutput),
        prediction: &Predicted<QaSigOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let qid: usize = example.0.question.trim_start_matches('q').parse()?;
        let needed = fact_of(qid);
        let answer = &prediction.answer;
        // A pinned fact always helps the training rows it was harvested
        // from; beyond them only if it transfers in this world.
        let in_train = qid < TRAIN;
        // Ledger world: a pin helps a training row only if it was harvested
        // from that very row (or transfers); elsewhere any training row.
        let harvested_here = match self.world {
            World::Ledger => in_train && answer.contains(&source_token(qid)),
            _ => in_train,
        };
        let covered = answer.contains(&fact_token(needed))
            && (harvested_here || transfers(self.world, needed));
        let cited = !cite_needed(qid) || answer.contains("SOURCES");
        let wrongs = wrongs_in(answer);
        let misled = wrongs.contains(&needed) || wrongs.contains(&((needed + 1) % FACTS));
        let junk = junk_in(answer) as f64;
        let score = (BASE
            + if covered { FACT_LIFT } else { 0.0 }
            + if cited { CITE_LIFT } else { 0.0 }
            - if misled { WRONG_PEN } else { 0.0 }
            - JUNK_TAX * junk
            + match self.world {
                // Column-only noise: identical for every candidate.
                World::Ledger => judge_noise(qid, "column"),
                _ => judge_noise(qid, answer),
            })
        .clamp(0.0, 1.0);
        let mut feedback = format!("Score {score:.2}.");
        if !covered {
            feedback.push_str(&format!("\nMissing or wrong:\n  - MISSING::{}", fact_token(needed)));
            if self.world == World::Ledger {
                feedback.push_str(&format!(" (source {})", source_token(qid)));
            }
        }
        if misled {
            feedback.push_str("\nContradicts the docs:\n  - a pinned claim about this topic is wrong");
        }
        if !cited {
            feedback.push_str(
                "\nAdvice: cite the relevant docs page; end with a Sources line (Missing citation).",
            );
        }
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

/// The scripted student: answers with every fact/junk/wrong token its
/// rendered prompt carries, and appends the Sources footer iff the
/// instruction carries the policy.
fn student_client() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let mut answer = String::from("facts:");
        for tok in haystack.split(|c: char| !c.is_ascii_alphanumeric()) {
            let is_fact = tok.len() == 3 && tok.starts_with('F') && tok[1..].chars().all(|c| c.is_ascii_digit());
            let is_wrong = tok.len() == 3 && tok.starts_with('W') && tok[1..].chars().all(|c| c.is_ascii_digit());
            let is_junk = tok.len() >= 2 && tok.starts_with('J') && tok[1..].chars().all(|c| c.is_ascii_digit());
            let is_source = tok.len() == 4 && tok.starts_with('Q') && tok[1..].chars().all(|c| c.is_ascii_digit());
            if (is_fact || is_wrong || is_junk || is_source) && !answer.contains(tok) {
                answer.push(' ');
                answer.push_str(tok);
            }
        }
        if haystack.contains(RULE) {
            answer.push_str(" SOURCES");
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

/// The measured teacher: keeps everything the parent instruction carries
/// and adds ONE thing. If it has been told to look for answer policies
/// (the `policy_channel` directive is present), the feedback names a
/// missing citation, and the policy is not pinned yet, it proposes the
/// policy with probability `P_RULE`%. Otherwise it pins the first fact the
/// feedback names as missing: TRUE 60% / JUNK 25% / WRONG 15% by hash.
/// The DocsQa teacher's move on one reflection input: the token pile it
/// will emit (facts, wrongs, junk pins, the policy last) and its one-line
/// summary. Shared by the plain teacher and the bloating (v6.5) one.
fn docsqa_teacher_move(haystack: &str) -> (Vec<String>, String) {
    let mut current_text = String::new();
    for field in INSTRUCTION_FIELDS {
        current_text.push_str(&field_section(haystack, field));
    }
    let mut facts = facts_in(&current_text);
    let mut wrongs = wrongs_in(&current_text);
    let mut junks: Vec<String> = current_text
        .split_whitespace()
        .filter(|tok| tok.starts_with('J') && tok[1..].chars().all(|c| c.is_ascii_digit()))
        .map(|tok| tok.to_string())
        .collect();
    junks.sort();
    junks.dedup();
    let mut has_rule = current_text.contains(RULE);
    let hash = fnv(haystack);
    let policy_visible = haystack.contains("PROPOSAL SCOPE");
    let missing_cite = haystack.contains("Missing citation");
    // The dedicated Policy operator (its signature text carries "ANSWER
    // POLICY") looks for the recurring pattern on purpose: it proposes
    // the policy half the time the feedback names the missing citation,
    // and never pins facts.
    let policy_op = haystack.contains("ANSWER POLICY");
    let summary = if policy_op {
        if missing_cite && !has_rule && hash % 100 < 50 {
            has_rule = true;
            "policy: end every answer with a Sources footer".to_string()
        } else {
            "policy: nothing recurring beyond what is pinned".to_string()
        }
    } else if policy_visible && missing_cite && !has_rule && hash % 100 < P_RULE {
        has_rule = true;
        "policy: end every answer with a Sources footer".to_string()
    } else {
        let missing: Vec<usize> = (0..FACTS)
            .filter(|&k| haystack.contains(&format!("MISSING::{}", fact_token(k))))
            .filter(|k| !facts.contains(k))
            .collect();
        match missing.first() {
            Some(&k) => {
                let roll = (hash / 100) % 100;
                if roll < 60 {
                    facts.insert(k);
                    format!("pin {}", fact_token(k))
                } else if roll < 85 {
                    junks.push(format!("J{}", hash % 1000));
                    format!("junk pin for {}", fact_token(k))
                } else {
                    wrongs.insert(k);
                    format!("wrong pin {}", wrong_token(k))
                }
            }
            None => "nothing new".to_string(),
        }
    };
    let mut tokens: Vec<String> = facts.iter().map(|&k| fact_token(k)).collect();
    tokens.extend(wrongs.iter().map(|&k| wrong_token(k)));
    tokens.extend(junks);
    if has_rule {
        tokens.push(RULE.to_string());
    }
    (tokens, summary)
}

fn teacher_client() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let (tokens, summary) = docsqa_teacher_move(&haystack);
        let instruction = format!("{SEED_TEXT} {}", tokens.join(" ")).trim_end().to_string();
        Ok(field_response(&[
            ("improved_instruction", &instruction),
            ("edit_summary", &summary),
        ]))
    })
}

/// A scoreless per-topic reminder — the kind of line the live HARVEST
/// winners fill up with (8–13k chars on DocsQA while a 3.8k prompt carried
/// the whole gain). The student ignores it; only the size grows.
const REMINDER: &str =
    "Reminder: before answering, re-read the retrieved passage on this topic and \
     double-check every name, number, and identifier it contains.";

/// The v6.5 bloating teacher: the DocsQa teacher's move plus ONE new
/// reminder line per proposal (every inherited reminder kept) — unless the
/// reflection input carries a LENGTH BUDGET block, in which case it folds:
/// no reminders at all, and if the token pile itself would still exceed
/// the budget it was given, junk pins go first and then the newest facts.
/// `saw_budget` records whether the block ever reached it.
fn budget_teacher_client(saw_budget: Arc<AtomicBool>) -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(move |request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let (tokens, summary) = docsqa_teacher_move(&haystack);
        let reminders = haystack.matches("Reminder: before answering").count();
        let budget: Option<usize> = haystack.find("LENGTH BUDGET:").and_then(|at| {
            let rest = &haystack[at..];
            let start = rest.find("the budget is ")? + "the budget is ".len();
            rest[start..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .ok()
        });
        if budget.is_some() {
            saw_budget.store(true, Ordering::SeqCst);
        }
        let render = |tokens: &[String], reminders: usize| -> String {
            let mut text = format!("{SEED_TEXT} {}", tokens.join(" ")).trim_end().to_string();
            for _ in 0..reminders {
                text.push('\n');
                text.push_str(REMINDER);
            }
            text
        };
        let instruction = match budget {
            None => render(&tokens, reminders + 1),
            Some(budget) => {
                let mut kept = tokens.clone();
                while render(&kept, 0).len() > budget {
                    if let Some(pos) = kept.iter().position(|tok| tok.starts_with('J')) {
                        kept.remove(pos);
                    } else if let Some(pos) = kept.iter().rposition(|tok| tok.starts_with('F')) {
                        kept.remove(pos);
                    } else {
                        break;
                    }
                }
                render(&kept, 0)
            }
        };
        Ok(field_response(&[
            ("improved_instruction", &instruction),
            ("edit_summary", &summary),
        ]))
    })
}

/// Pins the Ledger-world teacher can read back from an instruction:
/// `(source row, fact)` in order of appearance.
fn ledger_pins(text: &str) -> Vec<(usize, usize)> {
    let mut pins = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find("Fact from Q") {
        let after = &rest[pos + "Fact from Q".len()..];
        let src: String = after.chars().take(3).collect();
        let fact: Option<usize> = after
            .find(": F")
            .and_then(|at| after[at + 3..].chars().take(2).collect::<String>().parse().ok());
        if let (Ok(src), Some(fact)) = (src.parse::<usize>(), fact) {
            pins.push((src, fact));
        }
        rest = after;
    }
    pins
}

fn pin_line(src: usize, fact: usize) -> String {
    format!("Fact from {}: {}.", source_token(src), fact_token(fact))
}

const RULE_LINE: &str = "RULE_CITE: end every answer with a Sources line.";
/// Pinned lines the Ledger teacher needs before it notices the policy.
const POLICY_AFTER: usize = 4;

/// The v6.4 Ledger teacher: one clause per line, ONE new line per child.
/// The Policy operator proposes the citation rule (50%) only once the
/// parent carries `POLICY_AFTER` pins — the pattern is noticed late, after
/// the pile has grown. The Rewrite operator pins the first fact the
/// feedback names as missing, tagged with the row it came from; a pin
/// re-harvested for another row is a new (zero-transfer) line.
fn ledger_teacher_client() -> ScriptedCompletionModel {
    ScriptedCompletionModel::new(|request| {
        let haystack = format!("{:?} {:?}", request.preamble, request.chat_history);
        let mut current_text = String::new();
        for field in INSTRUCTION_FIELDS {
            current_text.push_str(&field_section(&haystack, field));
        }
        let mut pins = ledger_pins(&current_text);
        let mut has_rule = current_text.contains(RULE);
        let hash = fnv(&haystack);
        let missing_cite = haystack.contains("Missing citation");
        let policy_op = haystack.contains("ANSWER POLICY");
        let summary = if policy_op {
            if missing_cite && !has_rule && pins.len() >= POLICY_AFTER && hash % 100 < 50 {
                has_rule = true;
                "policy: end every answer with a Sources footer".to_string()
            } else {
                "policy: nothing recurring beyond what is pinned".to_string()
            }
        } else {
            // "MISSING::F17 (source Q017)" — first one not yet pinned for
            // that row.
            let mut found: Option<(usize, usize)> = None;
            let mut rest = haystack.as_str();
            while let Some(pos) = rest.find("MISSING::F") {
                let after = &rest[pos + "MISSING::F".len()..];
                let fact: Option<usize> = after.chars().take(2).collect::<String>().parse().ok();
                let src: Option<usize> = after
                    .strip_prefix(|c: char| c.is_ascii_digit())
                    .and_then(|s| s.strip_prefix(|c: char| c.is_ascii_digit()))
                    .and_then(|s| s.strip_prefix(" (source Q"))
                    .and_then(|s| s.chars().take(3).collect::<String>().parse().ok());
                match (fact, src) {
                    (Some(fact), Some(src)) if !pins.iter().any(|(s, _)| *s == src) => {
                        found = Some((src, fact));
                        break;
                    }
                    _ => {}
                }
                rest = after;
            }
            match found {
                Some((src, fact)) => {
                    pins.push((src, fact));
                    format!("pin {} from {}", fact_token(fact), source_token(src))
                }
                None => "nothing new".to_string(),
            }
        };
        let mut lines: Vec<String> = vec![SEED_TEXT.to_string()];
        lines.extend(pins.iter().map(|&(src, fact)| pin_line(src, fact)));
        if has_rule {
            lines.push(RULE_LINE.to_string());
        }
        Ok(field_response(&[
            ("improved_instruction", &lines.join("\n")),
            ("edit_summary", &summary),
        ]))
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

struct ArmReport {
    name: &'static str,
    result: RatchetResult,
}

impl ArmReport {
    fn winner(&self) -> &str {
        &self.result.best_candidate.instruction
    }
    fn argmax(&self) -> &str {
        self.result
            .best_by_mean
            .as_ref()
            .map(|c| c.instruction.as_str())
            .unwrap_or(self.winner())
    }
    fn has_rule(&self) -> bool {
        self.winner().contains(RULE)
    }
    fn promoted(&self) -> usize {
        self.result
            .trials
            .iter()
            .filter(|t| matches!(t.outcome, TrialOutcome::Promoted { .. }))
            .count()
    }
    fn print(&self) {
        let w = self.winner();
        let a = self.argmax();
        println!(
            "{:>10}: rollouts {} promoted {} gate_rejected {} race_killed {} faded {} | \
             certified {:?} (cols {}, mean {:+.4}, e {:.2}) | shipped: rule {} true {} wrong {} junk {} | \
             argmax: rule {} true {} wrong {} junk {}",
            self.name,
            self.result.total_rollouts,
            self.promoted(),
            self.result.gate_rejected,
            self.result.race_killed,
            self.result.faded,
            self.result.certified,
            self.result.certify_cols_used,
            self.result.certify_mean_delta,
            self.result.certify_e_value,
            w.contains(RULE),
            facts_in(w).len(),
            wrongs_in(w).len(),
            junk_in(w),
            a.contains(RULE),
            facts_in(a).len(),
            wrongs_in(a).len(),
            junk_in(a),
        );
        let ops: Vec<String> = self
            .result
            .operator_stats
            .iter()
            .map(|(op, attempts, wins)| format!("{op} {attempts}/{wins}"))
            .collect();
        println!("{:>10}  operators: {}", "", ops.join("  "));
    }
}

async fn run_arm(
    name: &'static str,
    optimizer: Ratchet,
    seed: u64,
    budget: usize,
    world: World,
) -> Result<ArmReport> {
    let teacher = if world == World::Ledger { ledger_teacher_client() } else { teacher_client() };
    run_arm_with(name, optimizer, seed, budget, world, teacher).await
}

async fn run_arm_with(
    name: &'static str,
    mut optimizer: Ratchet,
    seed: u64,
    budget: usize,
    world: World,
    teacher: ScriptedCompletionModel,
) -> Result<ArmReport> {
    optimizer.prompt_model = Some(make_lm(teacher).await);
    optimizer.max_rollouts = Some(budget);
    optimizer.seed = Some(seed);
    optimizer.eval_concurrency = 16;
    optimizer.allow_memorization = true;
    // Harness-supplied certification split (the DocsQA rig's pool_certify):
    // rows appended to the trainset, named as explicit certify columns.
    optimizer.certify_reserve = 0;
    optimizer.certify_cols = (TRAIN..TRAIN + CERT).collect();
    optimizer.champion_fresh = 30;
    let mut module = QaModule {
        predictor: Predict::<QaSig>::builder()
            .instruction(SEED_TEXT)
            .lm(make_lm(student_client()).await)
            .build(),
    };
    let train = dataset(0..TRAIN + CERT);
    let val = dataset(TRAIN + CERT..TRAIN + CERT + VAL);
    let result = optimizer
        .compile_module_with_valset(&mut module, &train, Some(&val), &DocsMetric { world })
        .await?;
    Ok(ArmReport { name, result })
}

/// The measured live configuration (escrow-class): compound preset, gate
/// on held-out B, scout-16 ranking, mixed operator portfolio, no policy
/// channel, certification on the harness split.
fn live_control() -> Ratchet {
    let mut optimizer = compound();
    optimizer.accept_mode = AcceptMode::Gate;
    optimizer.scout_size = 16;
    optimizer.champion_k = 4;
    optimizer.certify_budget = 400;
    optimizer
}

const BUDGET: usize = 2600;

/// The headline claim on the DocsQA-calibrated world: at equal budget,
/// HARVEST ships a CERTIFIED winner that carries the answer policy pip-gepa
/// found by luck, and its certified gain clears the measured live
/// configuration's by the policy's worth.
#[tokio::test]
async fn harvest_ships_the_certified_policy_the_live_config_never_proposes() -> Result<()> {
    let harvest_arm = run_arm("harvest", harvest(), 0, BUDGET, World::DocsQa).await?;
    let control = run_arm("control", live_control(), 0, BUDGET, World::DocsQa).await?;
    harvest_arm.print();
    control.print();

    assert!(
        harvest_arm.has_rule(),
        "harvest must ship the policy; shipped: {}",
        harvest_arm.winner()
    );
    assert_eq!(
        harvest_arm.result.certified,
        Some(true),
        "harvest must certify its ship (mean {:+.4}, e {:.2} over {} cols)",
        harvest_arm.result.certify_mean_delta,
        harvest_arm.result.certify_e_value,
        harvest_arm.result.certify_cols_used
    );
    assert!(harvest_arm.result.total_rollouts <= BUDGET);
    assert!(control.result.total_rollouts <= BUDGET);
    // The control's teacher is never told to look for policies — the
    // measured live behavior — so its ship carries none, and whatever it
    // certifies is worth less than the policy.
    assert!(!control.has_rule(), "control shipped the policy: {}", control.winner());
    let control_gain = if control.result.certified == Some(true) {
        control.result.certify_mean_delta
    } else {
        0.0
    };
    assert!(
        harvest_arm.result.certify_mean_delta >= control_gain + 0.03,
        "harvest certified {:+.4} vs control {:+.4}",
        harvest_arm.result.certify_mean_delta,
        control_gain
    );
    // Rewrite-only portfolio actually held.
    for (op, attempts, _) in &harvest_arm.result.operator_stats {
        if op != "rewrite" && op != "policy" {
            assert_eq!(*attempts, 0, "harvest ran a non-Rewrite operator: {op}");
        }
    }
    Ok(())
}

/// The on-target screen admits no child that scored below its parent on the
/// examples it was written from (pip-gepa's 0/33). And the engine, not
/// luck: across seeds the policy is found, kept, and certified.
#[tokio::test]
async fn on_target_screen_admits_no_negative_children_and_finds_the_policy_across_seeds() -> Result<()> {
    let mut found = 0usize;
    let mut certified = 0usize;
    for seed in 0..4u64 {
        let arm = run_arm("harvest", harvest(), seed, BUDGET, World::DocsQa).await?;
        arm.print();
        for trial in &arm.result.trials {
            if let TrialOutcome::GateKilled { parent_gate, child_gate } = trial.outcome {
                assert!(
                    child_gate <= parent_gate + 1e-9,
                    "a screen kill must mean the child did not beat its parent on target"
                );
            }
        }
        found += arm.has_rule() as usize;
        certified += (arm.result.certified == Some(true)) as usize;
    }
    println!("harvest across 4 seeds: policy shipped {found}/4, certified {certified}/4");
    assert!(found >= 3, "policy shipped in only {found}/4 seeds");
    assert!(certified >= 3, "certified in only {certified}/4 seeds");
    Ok(())
}

/// The other regime — every fact transfers — reported for the design
/// discussion (evaluation depth per child vs number of children): HARVEST
/// pays the full ranking slice per accepted child, the live control pays a
/// 16-column scout and promotes ~2× more children. Asserted: HARVEST still
/// ships the certified policy; the fact counts are printed, not judged.
#[tokio::test]
async fn additive_world_reports_the_depth_vs_breadth_tradeoff() -> Result<()> {
    let harvest_arm = run_arm("harvest", harvest(), 1, BUDGET, World::Additive).await?;
    let control = run_arm("control", live_control(), 1, BUDGET, World::Additive).await?;
    harvest_arm.print();
    control.print();
    assert!(harvest_arm.has_rule());
    assert_eq!(harvest_arm.result.certified, Some(true));
    println!(
        "additive world: true facts shipped — harvest {} vs control {} (depth-vs-breadth; see design §)",
        facts_in(harvest_arm.winner()).len(),
        facts_in(control.winner()).len()
    );
    Ok(())
}

/// v6.4 (handoff-harvest-v6 §5 items 2 and 4): in the Ledger world the
/// teacher's children carry ONE true policy clause plus junk (pins that
/// help nothing beyond the row they came from). (i) The lineage-diff
/// credit — zero extra rollouts — assigns the policy clause positive
/// evidence and the junk ~zero; (ii) with `prune_after_certify` the
/// zero-evidence clauses come out as a bundle, the pruned prompt
/// re-certifies against the seed, and the shipped winner is shorter than
/// the un-pruned run's and still carries the policy.
#[tokio::test]
async fn lineage_credit_separates_policy_from_junk_and_occam_pruning_ships_shorter() -> Result<()> {
    let fact_of_pin = |text: &str| -> Option<usize> {
        text.find(": F")
            .and_then(|at| text[at + 3..].chars().take(2).collect::<String>().parse().ok())
    };
    let mut shorter = 0usize;
    let mut removed_total = 0usize;
    for seed in 0..2u64 {
        let plain = run_arm("ledger", harvest(), seed, BUDGET, World::Ledger).await?;
        let mut pruning = harvest();
        pruning.prune_after_certify = true;
        pruning.prune_budget = 90;
        let pruned = run_arm("pruned", pruning, seed, BUDGET, World::Ledger).await?;
        plain.print();
        println!("{}", plain.result.clause_credit_report());
        pruned.print();
        println!("{}", pruned.result.clause_credit_report());

        // (i) attribution: policy positive, junk ~zero, true pins small +.
        assert!(plain.has_rule(), "seed {seed}: plain arm shipped no policy: {}", plain.winner());
        assert_eq!(plain.result.certified, Some(true));
        let credits = &plain.result.clause_credits;
        let policy = credits
            .iter()
            .find(|credit| credit.text.contains(RULE))
            .expect("the policy clause is in the credit table");
        assert!(policy.in_champion);
        assert!(policy.ema > 0.05, "policy ema {:+.4}", policy.ema);
        assert!(policy.cols_won >= 30, "policy won {} cols", policy.cols_won);
        let mut junk_in_winner = 0usize;
        for credit in credits.iter().filter(|credit| credit.in_champion) {
            let Some(fact) = fact_of_pin(&credit.text) else { continue };
            if transfers(World::Ledger, fact) && credit.times_credited > 0 && credit.ema > 0.0 {
                assert!(credit.ema < 0.03, "true pin ema {:+.4}: {}", credit.ema, credit.text);
            } else {
                junk_in_winner += 1;
                assert!(
                    credit.ema.abs() < 0.005,
                    "junk pin ema {:+.4}: {}",
                    credit.ema,
                    credit.text
                );
            }
        }
        let seed_clause = credits.iter().find(|credit| credit.text == SEED_TEXT).unwrap();
        assert_eq!((seed_clause.added_at_gen, seed_clause.times_credited), (0, 0));
        assert!(seed_clause.in_champion);
        println!("seed {seed}: junk pins in the un-pruned winner: {junk_in_winner}");

        // (ii) pruning: still certified, re-certified, shorter, policy kept,
        // no zero-transfer pin left, within budget.
        assert_eq!(pruned.result.certified, Some(true));
        assert!(pruned.has_rule(), "pruned winner lost the policy: {}", pruned.winner());
        assert!(pruned.result.total_rollouts <= BUDGET);
        assert!(pruned.result.prune_chars_before > 0, "the pruning phase never ran");
        assert!(pruned.result.prune_chars_after <= pruned.result.prune_chars_before);
        assert!(pruned.winner().len() <= plain.winner().len());
        if pruned.result.prune_removed > 0 {
            assert_eq!(pruned.result.prune_certified, Some(true));
            assert!(pruned.result.prune_chars_after < pruned.result.prune_chars_before);
            assert!(pruned.winner().len() < plain.winner().len());
            shorter += 1;
            removed_total += pruned.result.prune_removed;
            for line in pruned.winner().lines() {
                if let Some(fact) = fact_of_pin(line) {
                    assert!(transfers(World::Ledger, fact), "zero-transfer pin survived: {line}");
                }
            }
        }
        println!(
            "seed {seed}: chars {} -> {} ({} clauses removed, {} rollouts, re-certified {:?})",
            pruned.result.prune_chars_before,
            pruned.result.prune_chars_after,
            pruned.result.prune_removed,
            pruned.result.prune_rollouts,
            pruned.result.prune_certified
        );
    }
    assert!(shorter >= 1, "pruning removed nothing on either seed");
    assert!(removed_total >= 1);
    Ok(())
}

/// The v6.5 length prior's budget for the bloating-teacher world: the seed
/// (27 chars) plus the whole 40-fact pile and the policy fit (~200 chars);
/// two reminder lines alone do not.
const LENGTH_BUDGET: usize = 240;

/// v6.5 length prior (soft, teacher-facing). With the bloating teacher and
/// the prior OFF the shipped HARVEST winner outgrows the budget and the
/// teacher never sees a budget (byte-identical inputs); with the prior ON
/// the LENGTH BUDGET block reaches the teacher through both operators, it
/// folds, and the shipped winner fits the budget, still certifies vs the
/// seed, and still carries the policy. Both configurations replay
/// identically on the same seed.
#[tokio::test]
async fn length_prior_keeps_the_certified_winner_within_budget() -> Result<()> {
    async fn run(seed: u64, prior: bool) -> Result<(ArmReport, bool)> {
        let saw_budget = Arc::new(AtomicBool::new(false));
        let mut optimizer = harvest();
        if prior {
            optimizer.length_prior = Some(LENGTH_BUDGET);
        }
        let arm = run_arm_with(
            if prior { "prior" } else { "bloat" },
            optimizer,
            seed,
            BUDGET,
            World::DocsQa,
            budget_teacher_client(saw_budget.clone()),
        )
        .await?;
        Ok((arm, saw_budget.load(Ordering::SeqCst)))
    }
    for seed in 0..2u64 {
        let (bloat, bloat_saw) = run(seed, false).await?;
        let (prior, prior_saw) = run(seed, true).await?;
        bloat.print();
        prior.print();
        println!(
            "seed {seed}: winner chars bloat {} vs prior {} (budget {LENGTH_BUDGET}, hard cap \
             {:?}); reminder lines {} vs {}; length retries/skips bloat {}/{} prior {}/{}",
            bloat.winner().len(),
            prior.winner().len(),
            prior.result.length_cap_in_force,
            bloat.winner().matches("Reminder:").count(),
            prior.winner().matches("Reminder:").count(),
            bloat.result.length_retries,
            bloat.result.length_skipped,
            prior.result.length_retries,
            prior.result.length_skipped,
        );

        // (a) prior off: the teacher never sees a budget, nothing is
        // capped, and the shipped winner outgrows the budget.
        assert!(!bloat_saw, "seed {seed}: a LENGTH BUDGET reached the teacher with the prior off");
        assert_eq!(bloat.result.length_prior, None);
        assert_eq!(bloat.result.length_cap_in_force, None);
        assert_eq!(bloat.result.length_retries + bloat.result.length_skipped, 0);
        assert!(
            bloat.winner().len() > LENGTH_BUDGET,
            "seed {seed}: the un-steered winner stayed within budget ({} chars): {}",
            bloat.winner().len(),
            bloat.winner()
        );

        // (b) prior on: briefed, within budget, certified, policy kept.
        assert!(prior_saw, "seed {seed}: the LENGTH BUDGET block never reached the teacher");
        assert_eq!(prior.result.length_prior, Some(LENGTH_BUDGET));
        assert_eq!(prior.result.length_cap_in_force, Some(300));
        assert!(
            prior.winner().len() <= LENGTH_BUDGET,
            "seed {seed}: the steered winner is over budget ({} chars): {}",
            prior.winner().len(),
            prior.winner()
        );
        assert!(!prior.winner().contains("Reminder:"), "seed {seed}: a reminder survived the fold");
        assert_eq!(
            prior.result.certified,
            Some(true),
            "seed {seed}: the steered winner did not certify (mean {:+.4}, e {:.2} over {} cols)",
            prior.result.certify_mean_delta,
            prior.result.certify_e_value,
            prior.result.certify_cols_used
        );
        assert!(prior.has_rule(), "seed {seed}: the steered winner lost the policy: {}", prior.winner());
        assert!(prior.result.total_rollouts <= BUDGET);

        // (c) both configurations are deterministic per seed.
        let (bloat_again, _) = run(seed, false).await?;
        let (prior_again, _) = run(seed, true).await?;
        assert_eq!(bloat.winner(), bloat_again.winner(), "seed {seed}: prior-off replay differs");
        assert_eq!(bloat.result.total_rollouts, bloat_again.result.total_rollouts);
        assert_eq!(prior.winner(), prior_again.winner(), "seed {seed}: prior-on replay differs");
        assert_eq!(prior.result.total_rollouts, prior_again.result.total_rollouts);
    }
    Ok(())
}


#[tokio::test]
async fn search_mode_uses_small_budgets_without_partial_validation_or_certification() -> Result<()> {
    for budget in [105, 205, 600] {
        let mut optimizer = dspy_rs::harvest_search();
        optimizer.prompt_model = Some(make_lm(teacher_client()).await);
        optimizer.max_rollouts = Some(budget);
        optimizer.seed = Some(2);
        optimizer.eval_concurrency = 8;
        let mut module = QaModule {
            predictor: Predict::<QaSig>::builder()
                .instruction(SEED_TEXT)
                .lm(make_lm(student_client()).await)
                .build(),
        };
        let train = dataset(0..TRAIN);
        let val = dataset(TRAIN..TRAIN + VAL);
        let result = optimizer.compile_module_with_valset(
            &mut module, &train, Some(&val), &DocsMetric { world: World::DocsQa },
        ).await?;
        assert!(result.total_rollouts <= budget, "{} > {budget}", result.total_rollouts);
        assert!(result.total_rollouts > VAL, "tail budget never reached search");
        assert_eq!(result.best_candidate.example_scores.len(), VAL);
        assert_eq!(result.certified, None);
        assert_eq!(result.certify_cols_used, 0);
        assert_eq!(result.fresh_selection_cols, 0);
        assert!(result.best_by_mean.is_none());
    }
    Ok(())
}
