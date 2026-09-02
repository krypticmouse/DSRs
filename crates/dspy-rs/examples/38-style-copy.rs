//! Style copy with GRPO-style prompt optimization.
//!
//! Learns to write replies in one person's voice from (incoming message →
//! their real reply) pairs. The optimizer is [`GRPO`]: for each training
//! message it samples a group of drafts at temperature, scores every draft
//! against the real reply, and a teacher turns the within-group contrast
//! into style rules. The learned rules are the extracted style guide; the
//! writer's instruction is the seed plus those rules.
//!
//! Data: `STYLE_DATA=path.jsonl`, one `{"context": ..., "reply": ...}` per
//! line — an email or Slack export where `context` is the message being
//! replied to and `reply` is what the person actually wrote. Defaults to
//! the bundled synthetic persona in `examples/data/style-persona.jsonl`.
//!
//! Cold start: a generic writer's drafts are all equally far from the
//! persona, so early groups are flat. `GRPO` then contrasts the group
//! against the real reply instead (`reference_fallback`); once the rules
//! create variance between samples, the within-group contrast takes over.
//!
//! Reward = 0.7 × LLM judge (voice match vs the reference, 0-10) + 0.3 ×
//! deterministic stylometry (casing, greetings, exclamation, contractions,
//! sentence length, length, multi-line). Tune with `STYLE_STYLO_W`.
//!
//! Arms (`STYLE_ARM`): `grpo` (default) | `fewshot` | `grpo+fewshot` | `none`.
//! Models: `STYLE_MODEL` (writer), `STYLE_JUDGE`, `STYLE_TEACHER` — OpenRouter ids.
//! Knobs: `STYLE_STEPS`, `STYLE_BATCH`, `STYLE_GROUP`, `STYLE_TEMP`,
//! `STYLE_TEST_SAMPLES`, `STYLE_SEED`, `STYLE_FEWSHOT_K`.
//!
//! Run:
//! ```text
//! OPENROUTER_API_KEY=... cargo run --release --example 38-style-copy
//! ```

use anyhow::{Context, Result};
use dspy_rs::{
    Demo, Eval, Example, GRPO, LM, Module, ModuleState, Predict, PredictError, Predicted,
    Signature, Trace, TypedMetric, average_score, configure, evaluate_trainset, init_tracing,
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

/// Write the reply this person would send to the incoming message, in their
/// own voice.
#[derive(Signature, Clone, Debug)]
struct WriteReply {
    /// The message being replied to (an email, a Slack message, a thread).
    #[input]
    context: String,

    /// The reply, exactly as this person would write it.
    #[output]
    reply: String,
}

/// One row of the export. `context` projects into `WriteReplyInput` by
/// name; `reply` is the reference the metric compares against.
#[derive(Example, Clone, Debug, Serialize, Deserialize)]
struct StyleRow {
    context: String,
    reply: String,
}

struct StyleWriter {
    writer: Predict<WriteReply>,
}

dspy_rs::predictors!(StyleWriter { writer });

impl Module for StyleWriter {
    type Input = WriteReplyInput;
    type Output = WriteReplyOutput;

    async fn forward(
        &self,
        input: WriteReplyInput,
    ) -> Result<Predicted<WriteReplyOutput>, PredictError> {
        self.writer.call(input).await
    }
}

/// Score how closely a candidate reply matches the reference author's VOICE.
///
/// Compare the candidate reply with the reference reply the real author sent
/// to the same message. Judge voice only: tone, register, formality, length,
/// sentence rhythm, openers and sign-offs, punctuation and capitalisation
/// habits, vocabulary and idioms, structure (bullets, paragraphs). Ignore
/// whether the facts match; the candidate does not know the author's private
/// information. Score 0 to 10: 10 is indistinguishable from the author, 5 is
/// the right register but clearly a different person, 0 is a generic
/// assistant. List the 2-3 most important differences first, most damaging
/// first.
#[derive(Signature, Clone, Debug)]
struct JudgeStyle {
    #[input]
    incoming_message: String,

    #[input]
    reference_reply: String,

    #[input]
    candidate_reply: String,

    /// The 2-3 most important voice differences, most damaging first.
    #[output]
    differences: String,

    /// 0 to 10.
    #[output]
    score: f64,
}

struct StyleMetric {
    judge: Predict<JudgeStyle>,
    stylometry_weight: f64,
}

impl TypedMetric<StyleRow, StyleWriter> for StyleMetric {
    async fn evaluate(
        &self,
        example: &StyleRow,
        prediction: &Predicted<WriteReplyOutput>,
        _trace: Option<&Trace>,
    ) -> Result<Eval> {
        let candidate = prediction.reply.trim();
        let stylometry = stylometry_similarity(&example.reply, candidate);
        let judged = self
            .judge
            .call(JudgeStyleInput {
                incoming_message: example.context.clone(),
                reference_reply: example.reply.clone(),
                candidate_reply: candidate.to_string(),
            })
            .await;
        let (judge, differences) = match judged {
            Ok(out) => (
                out.score.clamp(0.0, 10.0) / 10.0,
                out.differences.trim().to_string(),
            ),
            Err(err) => (stylometry, format!("judge call failed: {err}")),
        };
        let score = (1.0 - self.stylometry_weight) * judge + self.stylometry_weight * stylometry;
        Ok(Eval::with_feedback(
            score,
            format!(
                "judge={:.1}/10 stylometry={stylometry:.2}\ndifferences: {differences}",
                judge * 10.0
            ),
        ))
    }
}

// ---------------------------------------------------------------------------
// Deterministic stylometry: cheap, ungameable-by-flattery half of the reward.
// ---------------------------------------------------------------------------

fn sentences(text: &str) -> Vec<&str> {
    text.split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Seven voice features, each in [0, 1].
fn style_features(text: &str) -> [f64; 7] {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n_words = words.len().max(1) as f64;
    let sents = sentences(text);
    let n_sents = sents.len().max(1) as f64;

    let lower_starts = sents
        .iter()
        .filter(|s| {
            s.chars()
                .find(|c| c.is_alphabetic())
                .is_some_and(char::is_lowercase)
        })
        .count() as f64
        / n_sents;
    let exclaim = (text.matches('!').count() as f64 / n_sents).min(1.0);
    let head = text.trim_start().to_lowercase();
    let greeting = [
        "hi",
        "hey",
        "hello",
        "dear",
        "good morning",
        "good afternoon",
        "thanks for",
    ]
    .iter()
    .any(|g| head.starts_with(g)) as u8 as f64;
    let contractions = (words
        .iter()
        .filter(|w| w.contains('\'') || w.contains('\u{2019}'))
        .count() as f64
        / n_words
        * 5.0)
        .min(1.0);
    let words_per_sentence = (n_words / n_sents / 25.0).min(1.0);
    let multiline = (text.trim().lines().count() > 1) as u8 as f64;
    let length = (n_words.ln_1p() / 6.0).min(1.0);
    [
        lower_starts,
        exclaim,
        greeting,
        contractions,
        words_per_sentence,
        multiline,
        length,
    ]
}

/// 1 − mean absolute feature difference.
fn stylometry_similarity(reference: &str, candidate: &str) -> f64 {
    let a = style_features(reference);
    let b = style_features(candidate);
    1.0 - a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f64>() / a.len() as f64
}

// ---------------------------------------------------------------------------
// Data + env helpers
// ---------------------------------------------------------------------------

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn load_rows(path: &str) -> Result<Vec<StyleRow>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("{path}:{} is not a StyleRow", i + 1))
        })
        .collect()
}

/// Mean reward of `module` on `rows`, averaged over `samples` independent
/// passes (the writer samples at temperature, so one pass is noisy).
async fn score_split(
    module: &StyleWriter,
    rows: &[StyleRow],
    metric: &StyleMetric,
    samples: usize,
) -> Result<f64> {
    let mut evals = Vec::new();
    for _ in 0..samples.max(1) {
        evals.extend(evaluate_trainset(module, rows, metric).await?);
    }
    Ok(average_score(&evals))
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;
    std::env::var("OPENROUTER_API_KEY").context("OPENROUTER_API_KEY not set")?;

    let arm = env_str("STYLE_ARM", "grpo");
    let seed: u64 = env_parse("STYLE_SEED", 7);
    let default_data = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/data/style-persona.jsonl"
    );
    let data_path = env_str("STYLE_DATA", default_data);

    let mut rows = load_rows(&data_path)?;
    let mut rng = StdRng::seed_from_u64(seed);
    rows.shuffle(&mut rng);
    let n_test = (rows.len() / 5).max(1);
    let n_val = (rows.len() / 5).max(1);
    let test: Vec<StyleRow> = rows.drain(..n_test).collect();
    let val: Vec<StyleRow> = rows.drain(..n_val).collect();
    let train = rows;
    println!(
        "data: {} rows from {data_path} -> train {} / val {} / test {}; arm={arm}",
        train.len() + val.len() + test.len(),
        train.len(),
        val.len(),
        test.len()
    );

    let writer_lm = LM::builder()
        .model(env_str("STYLE_MODEL", "openrouter:openai/gpt-5.6-sol"))
        .temperature(env_parse("STYLE_TEMP", 1.0))
        .max_tokens(600)
        .build()
        .await?;
    let judge_lm = LM::builder()
        .model(env_str("STYLE_JUDGE", "openrouter:openai/gpt-5.6-sol"))
        .temperature(0.0)
        .max_tokens(600)
        .build()
        .await?;
    let teacher_lm = LM::builder()
        .model(env_str("STYLE_TEACHER", "openrouter:openai/gpt-5.6-terra"))
        .temperature(1.0)
        .max_tokens(2000)
        .build()
        .await?;
    configure(writer_lm);

    let metric = StyleMetric {
        judge: Predict::<JudgeStyle>::builder()
            .named("judge")
            .lm(judge_lm)
            .build(),
        stylometry_weight: env_parse("STYLE_STYLO_W", 0.3),
    };

    // Few-shot arm: k training replies as demos (the baseline to beat for voice).
    let demos: Vec<Demo<WriteReply>> = if arm.contains("fewshot") {
        let k: usize = env_parse("STYLE_FEWSHOT_K", 4);
        train
            .iter()
            .take(k)
            .map(|row| {
                Demo::new(
                    WriteReplyInput {
                        context: row.context.clone(),
                    },
                    WriteReplyOutput {
                        reply: row.reply.clone(),
                    },
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut module = StyleWriter {
        writer: Predict::<WriteReply>::builder()
            .named("writer")
            .with_demos(demos)
            .build(),
    };

    let test_samples: usize = env_parse("STYLE_TEST_SAMPLES", 2);
    let before = score_split(&module, &test, &metric, test_samples).await?;
    println!("test before: {before:.3} ({test_samples} passes)");

    if arm.contains("grpo") {
        let grpo = GRPO::builder()
            .steps(env_parse("STYLE_STEPS", 6))
            .batch_size(env_parse("STYLE_BATCH", 4))
            .group_size(env_parse("STYLE_GROUP", 6))
            .min_spread(0.05)
            .max_rules(10)
            .prompt_model(teacher_lm)
            .eval_concurrency(8)
            .seed(seed)
            .build();
        let report = grpo
            .compile_module_with_valset(&mut module, &train, Some(&val), &metric)
            .await?;

        for step in &report.steps {
            let spreads: Vec<String> = step.spreads.iter().map(|s| format!("{s:.2}")).collect();
            println!(
                "step {}: spreads [{}], {} informative / {} flat ({} vs reference), insights {}, val {:.3} -> {}, {}, rules {}",
                step.step,
                spreads.join(" "),
                step.informative_groups,
                step.flat_groups,
                step.reference_contrasts,
                step.insights,
                step.parent_score,
                step.child_score
                    .map_or("-".to_string(), |s| format!("{s:.3}")),
                if step.accepted {
                    "accepted"
                } else {
                    "rejected"
                },
                step.rules_after
            );
        }
        println!(
            "val: {:.3} -> {:.3}; accepted {} / rejected {}; spend {} rollouts, {} lm calls",
            report.baseline_score,
            report.final_score,
            report.accepted,
            report.rejected,
            report.spend.metric_calls,
            report.spend.lm_calls
        );
        println!("\nlearned style rules:");
        for (leaf, rules) in &report.rules {
            for rule in rules {
                println!("  [{leaf}] - {rule}");
            }
        }
        println!();
    }

    let after = score_split(&module, &test, &metric, test_samples).await?;
    println!("test after: {after:.3} (delta {:+.3})", after - before);

    for row in test.iter().take(2) {
        let draft = module
            .call(WriteReplyInput {
                context: row.context.clone(),
            })
            .await?;
        println!(
            "\n--- incoming:\n{}\n--- reference:\n{}\n--- draft:\n{}",
            row.context,
            row.reply,
            draft.reply.trim()
        );
    }

    let out = env_str("STYLE_OUT", "style-writer.json");
    ModuleState::from_module(&module)?.save(&out)?;
    println!("\nsaved optimized writer state to {out}");
    Ok(())
}
