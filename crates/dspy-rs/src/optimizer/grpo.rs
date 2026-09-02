//! GRPO-style prompt optimization with no weight updates: sample a *group*
//! of rollouts per input, score them, and let a teacher turn the
//! within-group contrast into rules the module follows.
//!
//! The one ingredient borrowed from GRPO is the **group baseline**. Every
//! sample's advantage is its reward minus the mean of the samples drawn for
//! the *same input* under the *same prompt*. That separates "the prompt is
//! weak" from "this input is hard": on a hard input every sample scores low,
//! the spread is ~0, and the group is skipped for free. Only groups whose
//! samples actually disagree spend a teacher call. No critic, no clipping,
//! and no std normalization (mean subtraction only, as in Dr. GRPO; dividing
//! by std amplifies noise in near-tied groups).
//!
//! Prior art: Training-Free GRPO (Tencent Youtu, 2025) and ACE (Stanford,
//! 2025) both keep the LM frozen and write the "semantic advantage" into an
//! experience library carried in the context. This is that loop on the
//! shared [`Engine`].

use std::collections::BTreeMap;

use anyhow::{Result, anyhow};
use bon::Builder;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::core::ToInput;
use crate::evaluate::TypedMetric;
use crate::optimizer::engine::{Candidate, Engine, EngineConfig, EvalOutcome, Spend};
use crate::optimizer::target::LeafInfo;
use crate::optimizer::{OptimizeTarget, Optimizer, OptimizerCommon, Report};
use crate::trace::{Eval, Trace};
use crate::utils::truncate;
use crate::{Module, Predict, Predictors, Signature};

/// Distill one rule from a within-input sample group.
///
/// You are improving one module inside an LLM pipeline by editing the rules
/// it follows. You see several independent samples the module produced for
/// the SAME input under the SAME rules, sorted from highest to lowest reward,
/// each with its advantage (reward minus the group mean) and the metric's
/// feedback. The example row is shown too and may include a reference
/// output. Only behaviours that VARY between samples explain the reward gap,
/// so do not restate what every sample already does. Write ONE concise,
/// general, actionable rule that would move the low-advantage samples toward
/// the high-advantage ones without harming the best one. The rule is about
/// behaviour, not this particular input: never quote the reference output
/// verbatim, and do not repeat a rule already in the current rules. If no
/// rule can be drawn, answer exactly NONE.
#[derive(Signature, Clone, Debug)]
struct ContrastGroup {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The rules the module currently follows (may be empty).
    #[input]
    current_rules: String,

    /// The example row: the input, plus any reference output the metric compared against.
    #[input]
    example: String,

    /// The group's samples, best first, each with reward, advantage, feedback, and output.
    #[input]
    samples: String,

    /// One rule, or NONE.
    #[output]
    rule: String,
}

/// Merge new insights into a compact rule list.
///
/// You maintain the list of rules a module follows. Merge the new insights
/// into the current rules: keep rules that still apply, generalize rules
/// that overlap into one, drop rules that a new insight contradicts or that
/// only fit a single input, and never exceed the maximum count. Preserve the
/// wording of rules that are already good. Output the full updated list, one
/// rule per line, each line starting with "- ", and nothing else.
#[derive(Signature, Clone, Debug)]
struct ConsolidateRules {
    /// The current rule list, one "- " line per rule (may be empty).
    #[input]
    current_rules: String,

    /// New rules distilled this step, one "- " line per rule.
    #[input]
    new_insights: String,

    /// Maximum number of rules in the output.
    #[input]
    max_rules: String,

    /// The updated rule list, one "- " line per rule.
    #[output]
    rules: String,
}

/// Distill one rule from a flat group by contrasting it with the reference.
///
/// You are improving one module inside an LLM pipeline by editing the rules
/// it follows. You see several independent samples the module produced for
/// the SAME input under the SAME rules. They all scored about the same, so
/// the differences between them carry no signal; the signal is the gap
/// between all of them and the reference output in the example row, which
/// is what a perfect answer looks like. Identify the most important thing
/// the reference does that none of the samples do (or that every sample does
/// and the reference does not) and write ONE concise, general, actionable
/// rule that closes that gap. The rule is about behaviour, not this
/// particular input: never quote the reference verbatim, and do not repeat a
/// rule already in the current rules. If the samples already match the
/// reference in every way that matters, answer exactly NONE.
#[derive(Signature, Clone, Debug)]
struct ContrastReference {
    /// The module's input and output fields with their descriptions.
    #[input]
    task_description: String,

    /// The rules the module currently follows (may be empty).
    #[input]
    current_rules: String,

    /// The example row: the input, plus the reference output the metric compared against.
    #[input]
    example: String,

    /// The group's samples, each with reward, feedback, and output.
    #[input]
    samples: String,

    /// One rule, or NONE.
    #[output]
    rule: String,
}

/// What a contrast call produced.
enum Proposal {
    Rule(String),
    /// The teacher quoted the example verbatim; dropped by the leak guard.
    Leaked,
}

/// What one GRPO step did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrpoStep {
    /// Step index (0-based).
    pub step: usize,
    /// Training columns sampled this step (one group each).
    pub batch: Vec<usize>,
    /// Groups whose reward spread reached `min_spread` (teacher called).
    pub informative_groups: usize,
    /// Groups whose samples all scored about the same (spread below
    /// `min_spread`).
    pub flat_groups: usize,
    /// Flat groups that were contrasted against the reference instead of
    /// skipped (`reference_fallback`).
    pub reference_contrasts: usize,
    /// Reward spread (max − min) of each group, in `batch` order.
    pub spreads: Vec<f64>,
    /// Rules the teacher distilled this step (before consolidation).
    pub insights: usize,
    /// Proposed rules dropped by the leak guard (quoted the example verbatim).
    pub leaked: usize,
    /// The current program's mean on the scoring columns (the gate threshold).
    pub parent_score: f64,
    /// The proposed child's mean on the scoring columns; `None` when no
    /// child was proposed (no insights) or the budget ran out.
    pub child_score: Option<f64>,
    /// Whether the child became the new current program.
    pub accepted: bool,
    /// Total rules across all leaves after this step.
    pub rules_after: usize,
}

/// What a [`GRPO`] run did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GrpoReport {
    /// Mean score of the unmodified module on the scoring columns.
    pub baseline_score: f64,
    /// Mean score of the installed program on the scoring columns.
    pub final_score: f64,
    /// The learned rules, per leaf. These are the whole artifact: the
    /// installed instruction is the seed plus this list.
    pub rules: BTreeMap<String, Vec<String>>,
    /// Per-step outcomes, in order.
    pub steps: Vec<GrpoStep>,
    /// Steps whose child was accepted.
    pub accepted: usize,
    /// Steps whose child was rejected by the gate.
    pub rejected: usize,
    /// Engine spend for the whole run (teacher calls included via charge).
    pub spend: Spend,
}

/// GRPO-style rule learning over the shared [`Engine`]. The LM stays frozen;
/// the "policy update" is an edit to a bounded rule list appended to the
/// leaf's instruction.
///
/// Each step:
///
/// 1. **Sample** `batch_size` training examples and draw `group_size`
///    independent rollouts of the current program for each one
///    ([`Engine::sample_rollouts`], cache bypassed). The module's LM must
///    sample at a temperature above zero or every group is flat.
/// 2. **Filter** — a group whose reward spread (max − min) is below
///    `min_spread` carries no within-group signal. It is skipped, unless
///    `reference_fallback` is on and its mean is below `reference_ceiling`:
///    then it is the cold-start case (every sample equally far from the
///    reference) and the teacher contrasts the group against the example's
///    reference output instead.
/// 3. **Contrast** — for each informative group, one teacher call
///    (`prompt_model`) reads the samples best-first with their advantages,
///    feedback, and outputs, plus the example row, and distills one rule (or
///    NONE). Without a `prompt_model` the worst sample's metric feedback is
///    the rule.
/// 4. **Consolidate** — the step's new rules are merged into the leaf's
///    rule list by a second teacher call (add / generalize / drop, capped at
///    `max_rules`); without a `prompt_model`, append + dedupe + drop-oldest.
/// 5. **Gate** — the child (seed instruction + rules) is scored on the
///    scoring columns (the valset when one was supplied, else the trainset)
///    and accepted only if it does not fall below the incumbent by more
///    than `min_gain` allows. The incumbent is re-drawn on the same columns
///    each step (`incumbent_redraw`) and compared through its running mean,
///    so a lucky cached draw does not become a permanent bar. Set
///    `gate(false)` to always accept, the ungated accretion of the original
///    recipe.
///
/// The winner is installed once, through [`OptimizeTarget::install`], when
/// `compile` returns. The learned rules are also returned in the report:
/// for a task like voice or style copying they *are* the extracted style
/// guide.
///
/// # Hyperparameters
///
/// - **`steps`** (default: 6) — optimization steps.
/// - **`batch_size`** (default: 4) — training examples (groups) per step.
/// - **`group_size`** (default: 4) — samples per example. Must be ≥ 2.
/// - **`min_spread`** (default: 0.05) — minimum max−min reward inside a
///   group for it to be contrasted within itself. Raise it on noisy metrics.
/// - **`reference_fallback`** (default: true) / **`reference_ceiling`**
///   (default: 0.8) — contrast flat groups whose mean is below the ceiling
///   against the reference output (cold start). Off = pure GRPO signal.
/// - **`max_rules`** (default: 12) — rule-list cap per leaf.
/// - **`gate`** (default: true) / **`min_gain`** (default: 0.0) — acceptance
///   rule: `child ≥ incumbent + min_gain` on the scoring columns.
/// - **`incumbent_redraw`** (default: true) — fresh incumbent draw per gated
///   step, folded into a running mean.
/// - **`prompt_model`** — the teacher. Strongly recommended.
/// - **`max_metric_calls`** / **`max_lm_calls`** — engine budget caps; the
///   run stops cleanly when a step no longer fits.
/// - **`eval_concurrency`** (default: 16) — rollouts in flight.
/// - **`seed`** — fixes batch sampling.
///
/// # Cost
///
/// Scoring columns once for the baseline, then per step:
/// `batch_size × group_size` rollouts, at most `batch_size` contrast calls,
/// at most one consolidation call per touched leaf, and one scoring pass for
/// the child (cached if the rules did not change). Flat groups are free
/// beyond their rollouts.
///
/// ```ignore
/// let grpo = GRPO::builder()
///     .steps(6)
///     .batch_size(4)
///     .group_size(4)
///     .prompt_model(teacher_lm)
///     .build();
/// let report = grpo.compile_module_with_valset(&mut module, &train, Some(&val), &metric).await?;
/// for rule in &report.rules["writer"] { println!("- {rule}"); }
/// ```
#[derive(Builder)]
pub struct GRPO {
    /// Optimization steps to attempt.
    #[builder(default = 6)]
    pub steps: usize,

    /// Training examples (one group each) sampled per step.
    #[builder(default = 4)]
    pub batch_size: usize,

    /// Independent samples drawn per example. Must be at least 2.
    #[builder(default = 4)]
    pub group_size: usize,

    /// Minimum reward spread (max − min) for a group to be contrasted.
    #[builder(default = 0.05)]
    pub min_spread: f64,

    /// Maximum rules kept per leaf.
    #[builder(default = 12)]
    pub max_rules: usize,

    /// Cold-start signal: contrast a flat group (spread below `min_spread`)
    /// against the example's reference output instead of skipping it. Off,
    /// the run is pure within-group contrast.
    #[builder(default = true)]
    pub reference_fallback: bool,

    /// Flat groups whose mean reward is at or above this are treated as
    /// solved and never reference-contrasted.
    #[builder(default = 0.8)]
    pub reference_ceiling: f64,

    /// Whether to gate the child on the scoring columns. `false` accepts
    /// every step (ungated accretion).
    #[builder(default = true)]
    pub gate: bool,

    /// Minimum improvement over the current program the gate requires.
    /// `0.0` accepts non-decreasing children.
    #[builder(default = 0.0)]
    pub min_gain: f64,

    /// Re-draw the incumbent on the scoring columns every gated step and
    /// gate against its running mean. At temperature, a single cached draw
    /// of the incumbent is a biased bar: a lucky one rejects every child.
    /// Costs one scoring pass per step.
    #[builder(default = true)]
    pub incumbent_redraw: bool,

    /// The teacher LM that distills and consolidates rules. Without it, rules
    /// degrade to the worst sample's metric feedback.
    pub prompt_model: Option<crate::LM>,

    /// Hard cap on metric calls (rollouts). `None` = unlimited.
    pub max_metric_calls: Option<usize>,
    /// Hard cap on LM call units (rollouts + teacher calls). `None` = unlimited.
    pub max_lm_calls: Option<usize>,

    /// Concurrent rollouts in flight during sampling and evaluation.
    #[builder(default = crate::evaluate::DEFAULT_EVAL_CONCURRENCY)]
    pub eval_concurrency: usize,

    /// Seed for batch sampling. `None` uses a nondeterministic seed.
    pub seed: Option<u64>,
}

/// Header under which the learned rules are appended to the seed instruction.
const RULES_HEADER: &str = "Rules:";

/// Reward spread (max − min) of a group; `0.0` for an empty group.
fn spread(scores: &[f64]) -> f64 {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
    if scores.is_empty() { 0.0 } else { max - min }
}

/// Group-relative advantages: each score minus the group mean (no std
/// normalization).
fn advantages(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    scores.iter().map(|score| score - mean).collect()
}

/// The seed instruction with the rule list appended (the seed alone when
/// there are no rules).
fn render_instruction(base: &str, rules: &[String]) -> String {
    if rules.is_empty() {
        return base.to_string();
    }
    let mut text = String::from(base.trim_end());
    text.push_str("\n\n");
    text.push_str(RULES_HEADER);
    for rule in rules {
        text.push_str("\n- ");
        text.push_str(rule);
    }
    text
}

/// A rule list as the teacher sees it.
fn render_rules(rules: &[String]) -> String {
    if rules.is_empty() {
        return "(none)".to_string();
    }
    rules
        .iter()
        .map(|rule| format!("- {rule}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Dedupe key: case-insensitive, trimmed, trailing period dropped.
fn rule_key(rule: &str) -> String {
    rule.trim().trim_end_matches('.').to_lowercase()
}

/// Parses a teacher-written rule list: one rule per line, list markers and
/// numbering stripped, blanks and duplicates dropped.
fn parse_rules(text: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut rules = Vec::new();
    for line in text.lines() {
        let mut item = line.trim();
        for marker in ["- ", "* ", "• "] {
            if let Some(rest) = item.strip_prefix(marker) {
                item = rest.trim();
            }
        }
        // "1. rule" / "1) rule"
        let digits = item.chars().take_while(char::is_ascii_digit).count();
        if digits > 0
            && let Some(rest) = item[digits..]
                .strip_prefix('.')
                .or_else(|| item[digits..].strip_prefix(')'))
        {
            item = rest.trim();
        }
        let item = item.trim_matches('"').trim();
        if item.is_empty()
            || item.eq_ignore_ascii_case("none")
            || item.eq_ignore_ascii_case("(none)")
        {
            continue;
        }
        if seen.insert(rule_key(item)) {
            rules.push(item.to_string());
        }
    }
    rules
}

/// Fallback consolidation: append new rules, dedupe, keep the newest `cap`.
fn merge_rules(existing: &[String], new: &[String], cap: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();
    for rule in existing.iter().chain(new) {
        let rule = rule.trim();
        if rule.is_empty() {
            continue;
        }
        if seen.insert(rule_key(rule)) {
            merged.push(rule.to_string());
        }
    }
    let cap = cap.max(1);
    if merged.len() > cap {
        merged.drain(0..merged.len() - cap);
    }
    merged
}

/// The leaf a group's rule targets: the one with the most spans across the
/// group's traces (first leaf on ties or when no trace has spans).
fn rule_target<'a>(leaves: &'a [LeafInfo], traces: &[&Trace]) -> &'a LeafInfo {
    let mut target = &leaves[0];
    let mut most = 0usize;
    for leaf in leaves {
        let count: usize = traces
            .iter()
            .map(|trace| trace.for_component(&leaf.name).count())
            .sum();
        if count > most {
            most = count;
            target = leaf;
        }
    }
    target
}

/// The leaf's final output on a rollout, rendered for the teacher.
fn leaf_output_text(trace: &Trace, leaf: &str) -> String {
    let Some(span) = trace.for_component(leaf).last() else {
        return "-".to_string();
    };
    match (&span.output, &span.error) {
        (Some(map), _) => {
            let mut text = String::new();
            for (key, value) in map {
                let rendered = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                text.push_str(key);
                text.push_str(": ");
                text.push_str(truncate(&rendered, 1500));
                text.push('\n');
            }
            text.trim_end().to_string()
        }
        (None, Some(error)) => format!(
            "<{}: {}>",
            error.kind.as_str(),
            truncate(span.raw_output.as_deref().unwrap_or(&error.message), 500)
        ),
        (None, None) => "-".to_string(),
    }
}

/// Lowercased word tokens (letters, digits, apostrophes).
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '\'' || c == '\u{2019}'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase().replace('\u{2019}', "'"))
        .collect()
}

/// Leak guard: whether `rule` contains a run of `n` or more consecutive
/// words that also appears verbatim in `example` (the row, reference output
/// included). A rule that quotes the example is a memorized answer, not a
/// behaviour.
fn leaks(rule: &str, example: &str, n: usize) -> bool {
    let rule = words(rule);
    let example = words(example);
    if rule.len() < n || example.len() < n {
        return false;
    }
    rule.windows(n)
        .any(|run| example.windows(n).any(|window| window == run))
}

/// Minimum verbatim run (in words) that counts as a leak.
const LEAK_RUN: usize = 4;

/// Leak guard for labels: whether `rule` contains, verbatim, the value the
/// example row holds under any of the leaf's output-field names (the gold
/// label when the row carries one). Catches short answers the run-length
/// guard cannot ("the answer is Paris"). Values under 3 characters are
/// ignored.
fn leaks_label(rule: &str, example: &serde_json::Value, leaf: &LeafInfo) -> bool {
    let Some(map) = example.as_object() else {
        return false;
    };
    let rule = rule.to_lowercase();
    leaf.output_fields.iter().any(|(name, _)| {
        map.get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| value.chars().count() >= 3)
            .is_some_and(|value| rule.contains(&value.to_lowercase()))
    })
}

impl GRPO {
    fn common(&self) -> OptimizerCommon {
        OptimizerCommon {
            eval_concurrency: self.eval_concurrency,
            max_metric_calls: self.max_metric_calls,
            max_lm_calls: self.max_lm_calls,
            seed: self.seed,
            ..OptimizerCommon::default()
        }
    }

    /// Renders one group for the contrast prompt: samples best-first with
    /// reward, advantage, feedback, and the target leaf's output.
    fn render_samples(samples: &[(Eval, Trace)], leaf: &str) -> String {
        use std::fmt::Write as _;

        let scores: Vec<f64> = samples.iter().map(|(eval, _)| eval.score).collect();
        let advantages = advantages(&scores);
        let mut order: Vec<usize> = (0..samples.len()).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut text = String::new();
        for (rank, &i) in order.iter().enumerate() {
            let (eval, trace) = &samples[i];
            let _ = writeln!(
                text,
                "### sample {} (reward={:.3}, advantage={:+.3})",
                rank + 1,
                eval.score,
                advantages[i]
            );
            let _ = writeln!(
                text,
                "feedback: {}",
                eval.feedback
                    .as_deref()
                    .map(|feedback| truncate(feedback, 800))
                    .unwrap_or("-")
            );
            let _ = writeln!(text, "output:\n{}\n", leaf_output_text(trace, leaf));
        }
        text.trim_end().to_string()
    }

    /// One contrast call for an informative group. Returns the rule (if the
    /// teacher found one) and the number of teacher calls consumed.
    #[allow(clippy::too_many_arguments)]
    async fn propose_insight(
        &self,
        target: &OptimizeTarget<'_>,
        idx: usize,
        leaf: &LeafInfo,
        current_rules: &[String],
        samples: &[(Eval, Trace)],
        teachers: Option<&(Predict<ContrastGroup>, Predict<ContrastReference>)>,
        against_reference: bool,
    ) -> (Option<Proposal>, usize) {
        let Some((contraster, referencer)) = teachers else {
            // Fallback: the worst sample's metric feedback is the rule.
            let worst = samples.iter().min_by(|a, b| {
                a.0.score
                    .partial_cmp(&b.0.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let rule = worst
                .and_then(|(eval, _)| eval.feedback.as_deref())
                .map(|feedback| truncate(feedback.trim(), 400).to_string())
                .filter(|rule| !rule.is_empty())
                .map(Proposal::Rule);
            return (rule, 0);
        };

        let example_json = target.example_json(idx);
        let example = example_json
            .as_ref()
            .and_then(|value| serde_json::to_string_pretty(value).ok())
            .unwrap_or_else(|| "-".to_string());
        let task_description = leaf.schema_for_reflection();
        let current_rules = render_rules(current_rules);
        let example = truncate(&example, 4000).to_string();
        let samples = Self::render_samples(samples, &leaf.name);
        let called = if against_reference {
            referencer
                .call(ContrastReferenceInput {
                    task_description,
                    current_rules,
                    example: example.clone(),
                    samples,
                })
                .await
                .map(|predicted| predicted.rule.clone())
        } else {
            contraster
                .call(ContrastGroupInput {
                    task_description,
                    current_rules,
                    example: example.clone(),
                    samples,
                })
                .await
                .map(|predicted| predicted.rule.clone())
        };

        match called {
            Ok(rule) => {
                let rule = rule
                    .trim()
                    .trim_start_matches("- ")
                    .trim_matches('"')
                    .trim()
                    .to_string();
                let none = rule.is_empty() || rule.to_ascii_uppercase().starts_with("NONE");
                if none {
                    return (None, 1);
                }
                let label_leak = example_json
                    .as_ref()
                    .is_some_and(|value| leaks_label(&rule, value, leaf));
                if label_leak || leaks(&rule, &example, LEAK_RUN) {
                    tracing::warn!(
                        target = leaf.name,
                        rule,
                        "GRPO rule quotes the example verbatim; dropped"
                    );
                    return (Some(Proposal::Leaked), 1);
                }
                (Some(Proposal::Rule(rule)), 1)
            }
            Err(err) => {
                tracing::warn!(target = leaf.name, error = %err, "GRPO contrast call failed; group skipped");
                (None, 1)
            }
        }
    }

    /// Merges new rules into a leaf's list. Returns the list and the number
    /// of teacher calls consumed (0 when the fallback ran or when there was
    /// nothing to reconcile).
    async fn consolidate(
        &self,
        existing: &[String],
        new: &[String],
        consolidator: Option<&Predict<ConsolidateRules>>,
    ) -> (Vec<String>, usize) {
        let fallback = || merge_rules(existing, new, self.max_rules);
        let Some(consolidator) = consolidator else {
            return (fallback(), 0);
        };
        // Nothing to reconcile: a single first rule simply becomes the list.
        // Several first rules still go through the teacher, since one batch
        // often distills the same insight in different words.
        if existing.is_empty() && new.len() <= 1 {
            return (fallback(), 0);
        }

        let input = ConsolidateRulesInput {
            current_rules: render_rules(existing),
            new_insights: render_rules(new),
            max_rules: self.max_rules.to_string(),
        };
        match consolidator.call(input).await {
            Ok(predicted) => {
                let mut rules = parse_rules(&predicted.rules);
                if rules.is_empty() {
                    tracing::warn!("GRPO consolidation returned no rules; using append fallback");
                    return (fallback(), 1);
                }
                rules.truncate(self.max_rules.max(1));
                (rules, 1)
            }
            Err(err) => {
                tracing::warn!(error = %err, "GRPO consolidation call failed; using append fallback");
                (fallback(), 1)
            }
        }
    }

    /// Convenience: optimizes a typed module over a trainset with this
    /// optimizer's default engine. The trainset serves as both the sampling
    /// pool and the scoring columns.
    pub async fn compile_module<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        metric: &MT,
    ) -> Result<GrpoReport>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        self.compile_module_with_valset(module, trainset, None, metric)
            .await
    }

    /// [`compile_module`](Self::compile_module) with a separate validation
    /// set: groups are sampled from `trainset`, the gate scores on `valset`.
    pub async fn compile_module_with_valset<E, M, MT>(
        &self,
        module: &mut M,
        trainset: &[E],
        valset: Option<&[E]>,
        metric: &MT,
    ) -> Result<GrpoReport>
    where
        E: ToInput<M::Input> + serde::Serialize + Send + Sync,
        M: Module + Predictors,
        MT: TypedMetric<E, M>,
    {
        let mut target = OptimizeTarget::module_with_valset(module, trainset, valset, metric);
        let mut engine = Engine::new(Optimizer::engine_config(self));
        let report = Optimizer::compile(self, &mut target, &mut engine).await?;
        report
            .into_grpo()
            .ok_or_else(|| anyhow!("GRPO must return a GRPO report"))
    }
}

#[async_trait::async_trait(?Send)]
impl Optimizer for GRPO {
    fn engine_config(&self) -> EngineConfig {
        self.common().engine_config()
    }

    async fn compile(
        &self,
        target: &mut OptimizeTarget<'_>,
        engine: &mut Engine,
    ) -> Result<Report> {
        if self.group_size < 2 {
            return Err(anyhow!("group_size must be at least 2"));
        }
        let leaves = target.leaves().to_vec();
        if leaves.is_empty() {
            return Err(anyhow!("no optimizable predictors found"));
        }
        let val_columns = target.val_columns();
        let train_columns = target.train_columns();
        if train_columns.is_empty() || val_columns.is_empty() {
            return Err(anyhow!(
                "GRPO needs at least one training and one scoring example"
            ));
        }

        let teachers = self.prompt_model.as_ref().map(|lm| {
            (
                Predict::<ContrastGroup>::builder().lm(lm.clone()).build(),
                Predict::<ContrastReference>::builder()
                    .lm(lm.clone())
                    .build(),
            )
        });
        let consolidator = self.prompt_model.as_ref().map(|lm| {
            Predict::<ConsolidateRules>::builder()
                .lm(lm.clone())
                .build()
        });
        let mut rng = self.common().rng();

        // Baseline: the empty candidate on the scoring columns.
        let mut current = Candidate::new();
        let mut current_idx = engine.register(current.clone());
        let baseline_score = match engine
            .evaluate(target, current_idx, Some(&val_columns))
            .await?
        {
            EvalOutcome::Complete(eval) => eval.mean(),
            EvalOutcome::BudgetExhausted { needed } => {
                return Err(anyhow!(
                    "budget too small for the baseline pass ({needed} rollouts needed)"
                ));
            }
        };
        let mut current_score = baseline_score;
        // Scoring-column draws behind `current_score` (its running mean).
        let mut current_draws = 1usize;
        let mut rules: BTreeMap<String, Vec<String>> = BTreeMap::new();

        let mut steps = Vec::new();
        let mut accepted = 0usize;
        let mut rejected = 0usize;

        for step in 0..self.steps {
            // 1. Sample a batch of training columns, one group each.
            let batch_size = train_columns.len().min(self.batch_size.max(1));
            let mut batch: Vec<usize> = train_columns
                .choose_multiple(&mut rng, batch_size)
                .copied()
                .collect();
            batch.sort_unstable();

            let Some(groups) = engine
                .sample_rollouts(target, current_idx, &batch, self.group_size)
                .await?
            else {
                break;
            };

            // 2 + 3. Filter flat groups; contrast the rest. Teacher calls
            // for one batch are independent, so they run concurrently.
            let mut insights: BTreeMap<String, Vec<String>> = BTreeMap::new();
            let mut informative = 0usize;
            let mut flat = 0usize;
            let mut reference_contrasts = 0usize;
            let mut spreads = Vec::with_capacity(batch.len());
            let mut pending = Vec::new();
            for (&idx, samples) in batch.iter().zip(&groups) {
                let scores: Vec<f64> = samples.iter().map(|(eval, _)| eval.score).collect();
                let group_spread = spread(&scores);
                spreads.push(group_spread);
                let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
                let against_reference = if group_spread < self.min_spread {
                    flat += 1;
                    if !self.reference_fallback || mean >= self.reference_ceiling {
                        continue;
                    }
                    reference_contrasts += 1;
                    true
                } else {
                    informative += 1;
                    false
                };
                let traces: Vec<&Trace> = samples.iter().map(|(_, trace)| trace).collect();
                let leaf = rule_target(&leaves, &traces);
                let current_rules = rules.get(&leaf.name).cloned().unwrap_or_default();
                pending.push((leaf, idx, samples, current_rules, against_reference));
            }
            let proposals = futures::future::join_all(pending.iter().map(
                |(leaf, idx, samples, current_rules, against_reference)| {
                    self.propose_insight(
                        target,
                        *idx,
                        leaf,
                        current_rules,
                        samples,
                        teachers.as_ref(),
                        *against_reference,
                    )
                },
            ))
            .await;
            let mut leaked = 0usize;
            for ((leaf, ..), (proposal, calls)) in pending.iter().zip(proposals) {
                engine.charge(0, calls);
                match proposal {
                    Some(Proposal::Rule(insight)) => {
                        insights.entry(leaf.name.clone()).or_default().push(insight);
                    }
                    Some(Proposal::Leaked) => leaked += 1,
                    None => {}
                }
            }

            let num_insights: usize = insights.values().map(Vec::len).sum();
            let rules_total = |rules: &BTreeMap<String, Vec<String>>| -> usize {
                rules.values().map(Vec::len).sum()
            };
            if num_insights == 0 {
                steps.push(GrpoStep {
                    step,
                    batch,
                    informative_groups: informative,
                    flat_groups: flat,
                    reference_contrasts,
                    spreads,
                    insights: 0,
                    leaked,
                    parent_score: current_score,
                    child_score: None,
                    accepted: false,
                    rules_after: rules_total(&rules),
                });
                continue;
            }

            // 4. Consolidate per touched leaf.
            let mut next_rules = rules.clone();
            for (name, new) in &insights {
                let existing = next_rules.get(name).cloned().unwrap_or_default();
                let (merged, calls) = self
                    .consolidate(&existing, new, consolidator.as_ref())
                    .await;
                engine.charge(0, calls);
                next_rules.insert(name.clone(), merged);
            }

            // 5. Gate the child on the scoring columns.
            let mut child = current.clone();
            for (name, list) in &next_rules {
                let leaf = leaves
                    .iter()
                    .find(|leaf| &leaf.name == name)
                    .expect("rules are keyed by leaf name");
                child.set_instruction(name, render_instruction(&leaf.instruction, list));
            }
            // Fresh incumbent draw, folded into its running mean, so the bar
            // is not one lucky cached sample.
            if self.gate && self.incumbent_redraw {
                match engine
                    .sample_rollouts(target, current_idx, &val_columns, 1)
                    .await?
                {
                    Some(draws) => {
                        let fresh = draws
                            .iter()
                            .filter_map(|group| group.first())
                            .map(|(eval, _)| eval.score)
                            .sum::<f64>()
                            / draws.len().max(1) as f64;
                        current_draws += 1;
                        current_score += (fresh - current_score) / current_draws as f64;
                    }
                    None => {
                        steps.push(GrpoStep {
                            step,
                            batch,
                            informative_groups: informative,
                            flat_groups: flat,
                            reference_contrasts,
                            spreads,
                            insights: num_insights,
                            leaked,
                            parent_score: current_score,
                            child_score: None,
                            accepted: false,
                            rules_after: rules_total(&rules),
                        });
                        break;
                    }
                }
            }
            let parent_score = current_score;

            let child_idx = engine.register(child.clone());
            let child_score = match engine
                .evaluate(target, child_idx, Some(&val_columns))
                .await?
            {
                EvalOutcome::Complete(eval) => eval.mean(),
                EvalOutcome::BudgetExhausted { .. } => {
                    steps.push(GrpoStep {
                        step,
                        batch,
                        informative_groups: informative,
                        flat_groups: flat,
                        reference_contrasts,
                        spreads,
                        insights: num_insights,
                        leaked,
                        parent_score: current_score,
                        child_score: None,
                        accepted: false,
                        rules_after: rules_total(&rules),
                    });
                    break;
                }
            };

            let accept = !self.gate || child_score + 1e-9 >= parent_score + self.min_gain;
            if accept {
                accepted += 1;
                current = child;
                current_idx = child_idx;
                current_score = child_score;
                current_draws = 1;
                rules = next_rules;
            } else {
                rejected += 1;
            }
            steps.push(GrpoStep {
                step,
                batch,
                informative_groups: informative,
                flat_groups: flat,
                reference_contrasts,
                spreads,
                insights: num_insights,
                leaked,
                parent_score,
                child_score: Some(child_score),
                accepted: accept,
                rules_after: rules_total(&rules),
            });
        }

        // The one mutation of the run: install the winner.
        if !current.is_empty() {
            target.install(&current)?;
        }

        Ok(Report::Grpo(GrpoReport {
            baseline_score,
            final_score: current_score,
            rules,
            steps,
            accepted,
            rejected,
            spend: *engine.spend(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn advantages_are_mean_subtracted_not_normalized() {
        let adv = advantages(&[1.0, 0.0, 0.5, 0.5]);
        assert_eq!(adv, vec![0.5, -0.5, 0.0, 0.0]);
        assert!(advantages(&[]).is_empty());
        assert_eq!(spread(&[0.2, 0.9, 0.4]), 0.7);
        assert_eq!(spread(&[]), 0.0);
    }

    #[test]
    fn leak_guard_catches_verbatim_runs() {
        let example = r#"{"context":"Just got word from legal","reply":"ok. i'll tell the team, but 90 min is 90 min"}"#;
        assert!(leaks(
            "Frame it personally (“i’ll tell the team”) and add an aside",
            example,
            4
        ));
        assert!(!leaks(
            "Acknowledge briefly, then add a dry aside",
            example,
            4
        ));
        assert!(!leaks("tell the team", example, 4));
    }

    #[test]
    fn label_leak_guard_catches_short_gold_answers() {
        let leaf = LeafInfo {
            name: "agent".into(),
            instruction: String::new(),
            default_instruction: String::new(),
            demos: vec![],
            input_fields: vec![("question".into(), String::new())],
            output_fields: vec![("answer".into(), String::new())],
        };
        let row = serde_json::json!({"question": "Which river?", "answer": "Danube", "gold_docids": ["1"]});
        assert!(leaks_label(
            "When asked about rivers, answer Danube.",
            &row,
            &leaf
        ));
        assert!(!leaks_label(
            "Verify the river name in an opened document.",
            &row,
            &leaf
        ));
        let short = serde_json::json!({"question": "q", "answer": "no"});
        assert!(!leaks_label("Do not answer no by default.", &short, &leaf));
    }

    #[test]
    fn render_instruction_appends_rules_under_header() {
        assert_eq!(render_instruction("seed", &[]), "seed");
        let text = render_instruction("seed\n", &rules(&["a", "b"]));
        assert_eq!(text, "seed\n\nRules:\n- a\n- b");
    }

    #[test]
    fn parse_rules_strips_markers_and_dedupes() {
        let parsed = parse_rules(
            "- Keep it short.\n* keep it short\n1. Lowercase only\n2) \"No greetings\"\n\nNONE\n",
        );
        assert_eq!(
            parsed,
            rules(&["Keep it short.", "Lowercase only", "No greetings"])
        );
    }

    #[test]
    fn merge_rules_dedupes_and_keeps_newest() {
        let merged = merge_rules(&rules(&["a", "b"]), &rules(&["B", "c", "d"]), 3);
        assert_eq!(merged, rules(&["b", "c", "d"]));
        let merged = merge_rules(&[], &rules(&["x", " ", "x."]), 5);
        assert_eq!(merged, rules(&["x"]));
    }
}
