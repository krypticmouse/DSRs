//! RATCHET / HARVEST arms of the optimizer benchmark: optimize a HotpotQA
//! instruction under the exact conditions of the GEPA arms
//! (benchmarks/ratchet-hotpot/run_python_gepa.py and example 29 with
//! HOTPOT_DIR): same student, same bare rendering, same splits, same budget,
//! same teacher.
//!
//! `HOTPOT_OPT=ratchet` (default) is the original RATCHET arm, unchanged.
//! `HOTPOT_OPT=harvest` is RATCHET v6 HARVEST (docs/ratchet-v6-harvest-design.md):
//! on-target screen, full-val ranking, Rewrite + Policy operators, sequential
//! paired certification vs the seed on reserved train columns (train150 carve:
//! champion 40 + certify 45 leaves 65 batch columns; the reserve is carved
//! inside the optimizer, no driver-side split).
//!
//! Env: HOTPOT_DIR (default benchmarks/ratchet-hotpot), HOTPOT_OPT, HOTPOT_SEED,
//! HOTPOT_SEED_PROMPT, HOTPOT_STUDENT (OpenRouter model id, e.g.
//! openai/gpt-oss-20b; default meta-llama/llama-3.2-3b-instruct),
//! HOTPOT_TEACHER (full LM spec, default openrouter:openai/gpt-5.6-sol — the
//! grid uses openrouter:openai/gpt-5.6-terra; run_python_gepa.py reads the
//! same variable and strips the `openrouter:` prefix), HOTPOT_MAX_TOKENS
//! (student max_tokens, default 256; benchmarks/ratchet-hotpot/common.py reads
//! the same variable so every arm's student is configured identically),
//! HOTPOT_VAL_FILE, HOTPOT_ITERS, HOTPOT_MINIBATCH, HOTPOT_BUDGET,
//! HOTPOT_CHAMPION, HOTPOT_OUT_PREFIX (default `<opt>_winner`, i.e.
//! ratchet_winner / harvest_winner), HOTPOT_CERTIFY_RESERVE (45),
//! HOTPOT_CERTIFY_BUDGET (150).
//! HARVEST ablation knobs (unset = preset; mirror of 33-pupa.rs): HOTPOT_SCREEN
//! (a|b), HOTPOT_REWRITE_ONLY, HOTPOT_POLICY, HOTPOT_POLICY_SHARE,
//! HOTPOT_LEDGER_GATE_KILLS, HOTPOT_PARENT_GREED, HOTPOT_GATE_SIZE,
//! HOTPOT_SCOUT, HOTPOT_PARENTS, HOTPOT_PROPOSALS, HOTPOT_CHAMPION_K,
//! HOTPOT_LENGTH_PRIOR (chars; soft teacher budget, hard guard ×
//! HOTPOT_LENGTH_SLACK 1.25).
//!
//! Run from the repo root:
//!     set -a; source .env; set +a
//!     HOTPOT_SEED_PROMPT="Answer the question." cargo run --release --example 30-ratchet-hotpot
//!     HOTPOT_OPT=harvest HOTPOT_SEED_PROMPT="Answer the question." \
//!         cargo run --release --example 30-ratchet-hotpot

use anyhow::{Context, Result, bail};
use dspy_rs::ir::RenderMode;
use dspy_rs::{
    Eval, LM, Module, Predict, PredictError, Predicted, Ratchet, Signature, TypedMetric,
    configure, harvest, init_tracing,
};
use serde::{Deserialize, Serialize};

fn student_model() -> String {
    match std::env::var("HOTPOT_STUDENT") {
        Ok(m) => format!("openrouter:{m}"),
        Err(_) => "openrouter:meta-llama/llama-3.2-3b-instruct".to_string(),
    }
}
/// Teacher / reflection LM. `HOTPOT_TEACHER` overrides (full spec, e.g.
/// `openrouter:openai/gpt-5.6-terra`); unset keeps the original hard-coded
/// gpt-5.6-sol so existing runs are reproducible.
const DEFAULT_TEACHER_MODEL: &str = "openrouter:openai/gpt-5.6-sol";

fn teacher_model() -> String {
    std::env::var("HOTPOT_TEACHER").unwrap_or_else(|_| DEFAULT_TEACHER_MODEL.to_string())
}

/// Answer the question using the provided context paragraphs. Give only the
/// exact answer — a few words, no explanation.
#[derive(Signature, Clone, Debug)]
struct HotpotQA {
    #[input]
    context: String,

    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, dspy_rs::Example)]
struct HotpotRow {
    context: String,
    question: String,
    answer: String,
}

/// Retrying student, mirroring the other arms' 3-attempt `call_student`.
struct Student {
    answerer: Predict<HotpotQA>,
}

dspy_rs::predictors!(Student { answerer });

impl Module for Student {
    type Input = HotpotQAInput;
    type Output = HotpotQAOutput;

    async fn forward(&self, input: HotpotQAInput) -> Result<Predicted<HotpotQAOutput>, PredictError> {
        let mut last = None;
        for attempt in 0..3u32 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(3u64 << (attempt - 1))).await;
            }
            match self.answerer.call(input.clone()).await {
                Ok(out) => return Ok(out),
                Err(e) => last = Some(e),
            }
        }
        eprintln!("rollout failed after retries, scoring 0: {:?}", last);
        Ok(Predicted::new(
            HotpotQAOutput {
                answer: String::new(),
            },
            dspy_rs::CallMetadata::default(),
        ))
    }
}

fn normalize(text: &str) -> String {
    let lowered: String = text.to_lowercase().chars().filter(|c| !c.is_ascii_punctuation()).collect();
    let articles = regex::Regex::new(r"\b(a|an|the)\b").expect("valid article regex");
    articles.replace_all(&lowered, " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

struct AnswerEmMetric;

impl TypedMetric<HotpotRow, Student> for AnswerEmMetric {
    async fn evaluate(
        &self,
        example: &HotpotRow,
        prediction: &dspy_rs::Predicted<HotpotQAOutput>,
        _trace: Option<&dspy_rs::Trace>,
    ) -> Result<Eval> {
        let answer = prediction.answer.trim();
        let f1 = if normalize(answer) == normalize(&example.answer) { 1.0 } else { 0.0 };
        Ok(Eval::with_feedback(
            f1,
            format!(
                "Predicted: {answer:?}. Gold answer: {:?}. Answer EM = {f1:.0}. \
                 The full answer must match after lowercasing and \
                 dropping punctuation/articles — extra words beyond the answer \
                 span reduce it.",
                example.answer
            ),
        ))
    }
}

fn load_rows(path: &str) -> Result<Vec<HotpotRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {path} — run from the repo root"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<HotpotRow>(l).with_context(|| format!("bad row: {l:.80}")))
        .collect()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;
    let teacher_model = teacher_model();
    configure(LM::builder().model(teacher_model.clone()).build().await?);

    let dir =
        std::env::var("HOTPOT_DIR").unwrap_or_else(|_| "benchmarks/ratchet-hotpot".to_string());
    let train = load_rows(&format!("{dir}/train.jsonl"))?;
    let val_file =
        std::env::var("HOTPOT_VAL_FILE").unwrap_or_else(|_| "val.jsonl".to_string());
    let val = load_rows(&format!("{dir}/{val_file}"))?;
    println!("{dir}: {} train / {} val", train.len(), val.len());

    // HOTPOT_MAX_TOKENS: student completion cap, default 256 (the llama-3b
    // cells); a reasoning student (gpt-oss-20b) needs ~1024. common.py reads
    // the same variable so the pip-gepa arm and the evaluator match.
    let max_tokens = env_usize("HOTPOT_MAX_TOKENS", 256) as u32;
    let student_lm = if let Ok(base_url) = std::env::var("HOTPOT_STUDENT_BASE_URL") {
        LM::builder()
        .base_url(base_url)
        .api_key("local-benchmark-gateway".to_string())
        .model(env_str("HOTPOT_STUDENT", "openai/gpt-oss-20b"))
        .temperature(0.2)
        .max_tokens(max_tokens)
        .max_retries(0)
        .request_timeout_secs(130)
        .build().await?
    } else { LM::builder().model(student_model())
        .temperature(0.2)
        .max_tokens(max_tokens)
        .build()
        .await? };
    println!(
        "student: {}  max_tokens: {max_tokens}  teacher: {teacher_model}",
        student_model()
    );
    let answerer = Predict::<HotpotQA>::builder()
        .named("hotpot_answerer")
        .render_mode(RenderMode::Bare)
        .lm(student_lm);
    let answerer = match std::env::var("HOTPOT_SEED_PROMPT") {
        Ok(prompt) => answerer.instruction(prompt).build(),
        Err(_) => answerer.build(),
    };
    let mut student = Student { answerer };

    let reflection_lm = LM::builder()
        .model(teacher_model.clone())
        .temperature(1.0)
        .max_tokens(6000)
        .build()
        .await?;

    let opt_name = env_str("HOTPOT_OPT", "ratchet");
    let is_harvest = matches!(opt_name.as_str(), "harvest" | "harvest_search");
    let seed: u64 = env_usize("HOTPOT_SEED", 0) as u64;
    // HARVEST keeps its preset cycle shape (iters 150 / minibatch 5 /
    // P×N 2×2) unless overridden; the RATCHET arm keeps its original 60.
    let iters = env_usize("HOTPOT_ITERS", if is_harvest { 150 } else { 60 });
    let minibatch = env_usize("HOTPOT_MINIBATCH", 5);
    let budget = env_usize("HOTPOT_BUDGET", 600);
    let champion = env_usize("HOTPOT_CHAMPION", if opt_name == "harvest_search" { 0 } else { 40 });
    println!(
        "{opt_name} seed: {seed}, iterations: {iters}, minibatch: {minibatch}, budget: {budget}, champion: {champion}"
    );

    let optimizer = match opt_name.as_str() {
        // The original RATCHET arm, byte-for-byte the same configuration.
        "ratchet" => Ratchet::builder()
            .num_iterations(iters)
            .minibatch_size(minibatch)
            .race_block(8)
            .champion_fresh(champion)
            .max_rollouts(budget)
            .prompt_model(reflection_lm)
            .seed(seed)
            .eval_concurrency(8)
            .track_stats(true)
            .build(),
        // HARVEST (RATCHET v6, docs/ratchet-v6-harvest-design.md): on-target
        // screen, full-val ranking, Rewrite + Policy operators, sequential
        // paired certification vs the seed on reserved train columns.
        "harvest" | "harvest_search" => {
            let mut optimizer = if opt_name == "harvest_search" {
                    dspy_rs::harvest_search()
                } else { harvest() };
            optimizer.prompt_model = Some(reflection_lm);
            optimizer.certify_reserve = env_usize("HOTPOT_CERTIFY_RESERVE", 45);
            optimizer.certify_budget = env_usize("HOTPOT_CERTIFY_BUDGET", 150);
            // One-factor ablation knobs (mirror of 33-pupa.rs): each
            // overrides the `harvest()` preset only when set; unset keeps
            // the preset value.
            match env_str("HOTPOT_SCREEN", "").as_str() {
                "a" => optimizer.screen_rule = dspy_rs::ScreenRule::OnTargetA,
                "b" => optimizer.screen_rule = dspy_rs::ScreenRule::HeldOutB,
                _ => {}
            }
            if let Ok(v) = std::env::var("HOTPOT_REWRITE_ONLY") {
                optimizer.rewrite_only = v == "1";
            }
            if let Ok(v) = std::env::var("HOTPOT_POLICY") {
                optimizer.policy_channel = v == "1";
            }
            if let Ok(v) = std::env::var("HOTPOT_POLICY_SHARE") {
                optimizer.policy_share = v.parse().unwrap_or(optimizer.policy_share);
            }
            if let Ok(v) = std::env::var("HOTPOT_LEDGER_GATE_KILLS") {
                optimizer.ledger_gate_kills = v == "1";
            }
            if let Ok(v) = std::env::var("HOTPOT_PARENT_GREED") {
                optimizer.parent_greed = v.parse().unwrap_or(optimizer.parent_greed);
            }
            if let Ok(v) = std::env::var("HOTPOT_GATE_SIZE") {
                optimizer.gate_size = v.parse().ok();
            }
            // v6.5 length prior (soft, teacher-facing; unset = off):
            // HOTPOT_LENGTH_PRIOR=N chars, hard guard N × HOTPOT_LENGTH_SLACK
            // (default 1.25) through the length_cap path.
            if let Ok(v) = std::env::var("HOTPOT_LENGTH_PRIOR") {
                optimizer.length_prior = v.parse().ok().filter(|n: &usize| *n > 0);
            }
            if let Ok(v) = std::env::var("HOTPOT_LENGTH_SLACK") {
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
            optimizer.scout_size = env_usize("HOTPOT_SCOUT", optimizer.scout_size);
            optimizer.parents_per_step = env_usize("HOTPOT_PARENTS", optimizer.parents_per_step);
            optimizer.proposals_per_parent =
                env_usize("HOTPOT_PROPOSALS", optimizer.proposals_per_parent);
            optimizer.champion_k = env_usize("HOTPOT_CHAMPION_K", optimizer.champion_k);
            // Shared cycle knobs (same values the RATCHET arm's builder sets).
            optimizer.num_iterations = iters;
            optimizer.minibatch_size = minibatch;
            optimizer.champion_fresh = champion;
            optimizer.max_rollouts = Some(budget);
            optimizer.seed = Some(seed);
            optimizer.eval_concurrency = env_usize("HOTPOT_CONCURRENCY", 4);
            optimizer.track_stats = true;
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
            optimizer
        }
        other => bail!("unknown HOTPOT_OPT {other:?} (ratchet|harvest)"),
    };

    let result = optimizer
        .compile_module_with_valset(&mut student, &train, Some(&val), &AnswerEmMetric)
        .await?;

    println!("\n== {} result ==", opt_name.to_uppercase());
    println!(
        "rollouts: {}  lm_calls: {}  promoted: {}",
        result.total_rollouts,
        result.total_lm_calls,
        result.all_candidates.len()
    );
    println!(
        "gate_rejected: {}  race_killed: {}  skipped: {}  leak_retries: {}  leak_skipped: {}",
        result.gate_rejected,
        result.race_killed,
        result.skipped,
        result.leak_retries,
        result.leak_skipped
    );
    println!("operator stats (name, attempts, wins): {:?}", result.operator_stats);
    println!("fresh_selection_cols: {}", result.fresh_selection_cols);
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
    println!("evolution: {:?}", result.evolution_history);
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
        "\nchosen candidate (gen {}, val avg {:.3}):\n---\n{}\n---",
        result.best_candidate.generation,
        result.best_candidate.average_score(),
        result.best_candidate.instruction
    );
    if let Some(argmax) = &result.best_by_mean {
        println!(
            "argmax-by-val-mean differs (gen {}, val avg {:.3}):\n---\n{}\n---",
            argmax.generation,
            argmax.average_score(),
            argmax.instruction
        );
    }

    let prefix = env_str("HOTPOT_OUT_PREFIX", &format!("{opt_name}_winner"));
    if let Ok(path) = std::env::var("HOTPOT_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    }
    let out = format!("{dir}/{prefix}_s{seed}.txt");
    std::fs::write(&out, &result.best_candidate.instruction)?;
    let argmax_instruction = result
        .best_by_mean
        .as_ref()
        .map(|c| c.instruction.as_str())
        .unwrap_or(&result.best_candidate.instruction);
    let out_argmax = format!("{dir}/{prefix}_argmax_s{seed}.txt");
    std::fs::write(&out_argmax, argmax_instruction)?;
    println!("\nchosen instruction written to {out}\nargmax instruction written to {out_argmax}");
    Ok(())
}


#[cfg(test)]
mod native_metric_tests {
    use super::*;
    #[test]
    fn official_answer_normalization_and_no_partial_overlap() {
        assert_eq!(normalize("The U.S.A."),normalize("USA"));
        assert_eq!(normalize("an  Apple"),normalize("apple"));
        assert_eq!(normalize("(The) Answer"),normalize("Answer"));
        assert_ne!(normalize("yes, it does"),normalize("yes"));
        assert_ne!(normalize("New York City"),normalize("New York"));
    }
}
