//! FORAGE — the contrastive rollout ledger behind RATCHET v5's tool-using
//! teacher (`TeacherContext`).
//!
//! Motivation (measured): on content-compounding tasks the Rewrite teacher
//! is bandwidth-starved — it sees one `minibatch_size` chunk of feedback per
//! proposal, so content arrives at ~5 examples/step and pip-gepa's blind
//! 150-iteration accumulation out-harvests it 16.4k chars to ~3k (DocsQA
//! compound-mem3/4 arms, docs/compound-validation-prereg.md amendments 4–5).
//! The 2026 literature is convergent on the fix: computed error aggregates
//! over the WHOLE eval ledger beat narrated minibatch reflection (SPEAR,
//! arXiv:2605.26275 — python-tool ablation is load-bearing), matched
//! fail/pass pairs on the same input beat pooled reflection (ContraPrompt,
//! arXiv:2604.17937; Contrastive Reflection, arXiv:2606.30840), and
//! deterministic engine-managed aggregation beats model-managed exploration
//! at sub-window ledger sizes (LCM, arXiv:2605.04050, vs RLM,
//! arXiv:2512.24601). FORAGE is all three behind one knob:
//!
//! - Every rollout the search pays for is folded into a per-column ledger
//!   (best/worst score + feedback per train/val column) — no extra rollout
//!   cost, pure reuse of already-paid evidence.
//! - `TeacherContext::Digest` renders a bounded contrastive digest into the
//!   Rewrite teacher's `execution_feedback`: OPEN checks (what the best
//!   rollout on a column STILL misses — the verified harvest frontier) and
//!   FLIP contrasts (checks one candidate covers and another misses on the
//!   same column — proof the behavior is writable).
//! - `TeacherContext::Tools` additionally hands the teacher three read-only
//!   drill-down tools over a step snapshot (RLM-style depth-1 agentic
//!   reflection; the digest stays as the map).
//!
//! The "verifiable rewards" update loop the design serves: checks are
//! (re)derived every step from the metric's own feedback via the
//! harness-supplied `CheckExtractor`, retire automatically once the best
//! rollout covers them (ARBOR, arXiv:2606.03239; DR Tulu variance
//! filtering, arXiv:2511.19399), and never touch the ship decision — the
//! frozen judge + e-process certification remains the only authority
//! (every working evolving-verifier system keeps an immutable anchor:
//! Cooper, EAPO, OnlineRubrics). Defaults preserve measured v2–v4 behavior
//! byte-for-byte.

use std::collections::BTreeMap;
use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::{ToolDyn, ToolError};
use rig::wasm_compat::WasmBoxedFuture;
use serde::{Deserialize, Serialize};

/// Harness-supplied verifiable-check extractor: parses one rollout's metric
/// feedback into the discrete checks it failed (e.g. the DocsQA judge's
/// "Missing or wrong:" bullets). Task feedback formats are harness
/// knowledge, so the harness supplies the parser — same seam philosophy as
/// `certify_cols`. `None` degrades the digest to score-contrast + feedback
/// excerpts (still useful, less surgical).
pub type CheckExtractor = Arc<dyn Fn(&str) -> Vec<String> + Send + Sync>;

/// How the Rewrite teacher sees the run's accumulated evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeacherContext {
    /// v2–v4 default: the reflection minibatch's feedback only.
    Minibatch,
    /// Minibatch feedback + the bounded FORAGE digest (deterministic
    /// engine-computed contrastive aggregate over every paid rollout).
    Digest,
    /// Digest + three read-only forage tools on the Rewrite call (the
    /// teacher becomes a depth-1 agent over the step's ledger snapshot).
    Tools,
}

/// Per-column contrastive record: the best and worst rollout seen on this
/// column across every candidate the search has paid to evaluate.
#[derive(Clone, Debug, Default)]
pub(crate) struct ForageEntry {
    pub attempts: u32,
    pub best_score: f64,
    pub best_row: usize,
    pub best_feedback: String,
    pub worst_score: f64,
    pub worst_row: usize,
    pub worst_feedback: String,
}

/// The run-level contrastive ledger. Observation is free (rollouts are
/// already paid for); rendering is bounded by the forage caps.
#[derive(Clone, Debug, Default)]
pub(crate) struct ForageLedger {
    pub entries: BTreeMap<usize, ForageEntry>,
    pub observations: usize,
}

/// One rendered digest plus the columns it exposed (the leak guard extends
/// its forbidden set with these columns' leak-class strings when the guard
/// is on — digest evidence must not become an unguarded leak channel).
pub(crate) struct ForageDigest {
    pub text: String,
    pub columns: Vec<usize>,
}

const FLIP_SPREAD: f64 = 0.15;
const EXCERPT: usize = 300;

fn truncate_str(text: &str, cap: usize) -> &str {
    if text.len() <= cap {
        return text;
    }
    let mut end = cap;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

impl ForageLedger {
    /// Folds one paid rollout into the ledger. Feedback-less evals still
    /// move the score contrast.
    pub fn observe(&mut self, column: usize, row: usize, score: f64, feedback: Option<&str>) {
        self.observations += 1;
        let entry = self.entries.entry(column).or_insert_with(|| ForageEntry {
            attempts: 0,
            best_score: f64::NEG_INFINITY,
            best_row: row,
            best_feedback: String::new(),
            worst_score: f64::INFINITY,
            worst_row: row,
            worst_feedback: String::new(),
        });
        entry.attempts += 1;
        if score > entry.best_score {
            entry.best_score = score;
            entry.best_row = row;
            entry.best_feedback = feedback.unwrap_or_default().to_string();
        }
        if score < entry.worst_score {
            entry.worst_score = score;
            entry.worst_row = row;
            entry.worst_feedback = feedback.unwrap_or_default().to_string();
        }
    }

    /// Folds a whole [`CandidateEval`]'s rollouts in (absolute example
    /// indices; the engine row is the candidate id shown in flip lines).
    pub fn observe_eval(&mut self, eval: &crate::optimizer::engine::CandidateEval) {
        for rollout in &eval.rollouts {
            self.observe(
                rollout.example,
                eval.candidate,
                rollout.eval.score,
                rollout.eval.feedback.as_deref(),
            );
        }
    }

    /// Open checks per column: what the column's BEST rollout still misses.
    fn open_checks(&self, extractor: &CheckExtractor) -> Vec<(usize, &ForageEntry, Vec<String>)> {
        let mut rows: Vec<(usize, &ForageEntry, Vec<String>)> = self
            .entries
            .iter()
            .filter_map(|(&col, entry)| {
                let checks = extractor(&entry.best_feedback);
                (!checks.is_empty()).then_some((col, entry, checks))
            })
            .collect();
        rows.sort_by(|a, b| {
            a.1.best_score
                .partial_cmp(&b.1.best_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        rows
    }

    /// Flip contrasts per column: checks the worst rollout missed that the
    /// best rollout covers, where the score spread makes the pair a real
    /// contrast rather than noise.
    fn flips(&self, extractor: &CheckExtractor) -> Vec<(usize, &ForageEntry, Vec<String>)> {
        let mut rows: Vec<(usize, &ForageEntry, Vec<String>)> = self
            .entries
            .iter()
            .filter_map(|(&col, entry)| {
                if entry.attempts < 2 || entry.best_score - entry.worst_score < FLIP_SPREAD {
                    return None;
                }
                let best: Vec<String> = extractor(&entry.best_feedback);
                let flipped: Vec<String> = extractor(&entry.worst_feedback)
                    .into_iter()
                    .filter(|check| !best.contains(check))
                    .collect();
                (!flipped.is_empty()).then_some((col, entry, flipped))
            })
            .collect();
        rows.sort_by(|a, b| {
            (b.1.best_score - b.1.worst_score)
                .partial_cmp(&(a.1.best_score - a.1.worst_score))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        rows
    }

    /// Renders the bounded digest. Deterministic in the ledger state and
    /// caps; consumes no rng, no budget.
    pub fn render_digest(
        &self,
        extractor: Option<&CheckExtractor>,
        generation: usize,
        open_cap: usize,
        contrast_cap: usize,
        char_cap: usize,
    ) -> ForageDigest {
        if self.entries.is_empty() {
            return ForageDigest {
                text: String::new(),
                columns: Vec::new(),
            };
        }
        let mut columns: Vec<usize> = Vec::new();
        let mut text = format!(
            "=== FORAGE DIGEST (generation {generation}; contrastive ledger over every \
             rollout this search has paid for — {} columns, {} rollouts) ===\n\
             OPEN CHECKS are failures the best-known rollout on that column STILL makes: \
             each is a verified, still-unharvested win. FLIPS are failures one candidate \
             makes and another avoids on the same column: proof the behavior is writable. \
             Prefer itemized, append-style edits — keep existing validated content intact \
             and add targeted rules or facts; do not restructure what already works.\n",
            self.entries.len(),
            self.observations,
        );

        if let Some(extractor) = extractor {
            let open = self.open_checks(extractor);
            let mut open_lines = 0usize;
            if !open.is_empty() {
                text.push_str("\nOPEN CHECKS (pin or fix every one you can):\n");
                'open: for (col, entry, checks) in &open {
                    for check in checks {
                        if open_lines >= open_cap || text.len() >= char_cap {
                            break 'open;
                        }
                        text.push_str(&format!(
                            "  [q{col} best {:.2} tries {}] {}\n",
                            entry.best_score,
                            entry.attempts,
                            truncate_str(check, EXCERPT)
                        ));
                        open_lines += 1;
                    }
                    columns.push(*col);
                }
            }
            let flips = self.flips(extractor);
            if !flips.is_empty() && text.len() < char_cap {
                text.push_str(
                    "\nFLIP CONTRASTS (covered by one candidate, missed by another — keep \
                     the covering behavior):\n",
                );
                let mut flip_lines = 0usize;
                'flip: for (col, entry, checks) in &flips {
                    for check in checks {
                        if flip_lines >= contrast_cap || text.len() >= char_cap {
                            break 'flip;
                        }
                        text.push_str(&format!(
                            "  [q{col}] cand#{} ({:.2}) covers what cand#{} ({:.2}) missed: {}\n",
                            entry.best_row,
                            entry.best_score,
                            entry.worst_row,
                            entry.worst_score,
                            truncate_str(check, EXCERPT)
                        ));
                        flip_lines += 1;
                    }
                    if !columns.contains(col) {
                        columns.push(*col);
                    }
                }
            }
        } else {
            // No extractor: fall back to raw feedback excerpts on the
            // weakest columns — generic, less surgical, still contrastive.
            let mut weakest: Vec<(&usize, &ForageEntry)> = self.entries.iter().collect();
            weakest.sort_by(|a, b| {
                a.1.best_score
                    .partial_cmp(&b.1.best_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(b.0))
            });
            text.push_str("\nWEAKEST COLUMNS (best rollout feedback):\n");
            for (col, entry) in weakest.into_iter().take(open_cap.max(1)) {
                if text.len() >= char_cap {
                    break;
                }
                text.push_str(&format!(
                    "  [q{col} best {:.2} worst {:.2} tries {}] {}\n",
                    entry.best_score,
                    entry.worst_score,
                    entry.attempts,
                    truncate_str(entry.best_feedback.trim(), EXCERPT)
                ));
                columns.push(*col);
            }
        }
        text.truncate({
            let mut end = char_cap.max(1).min(text.len());
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            end
        });
        ForageDigest { text, columns }
    }

    /// Immutable step snapshot backing the read-only teacher tools.
    pub fn snapshot(&self, extractor: Option<&CheckExtractor>) -> ForageSnapshot {
        let mut columns: BTreeMap<usize, ForageColumn> = BTreeMap::new();
        for (&col, entry) in &self.entries {
            let open = extractor
                .map(|extract| extract(&entry.best_feedback))
                .unwrap_or_default();
            columns.insert(
                col,
                ForageColumn {
                    attempts: entry.attempts,
                    best_score: entry.best_score,
                    best_row: entry.best_row,
                    best_feedback: entry.best_feedback.clone(),
                    worst_score: entry.worst_score,
                    worst_row: entry.worst_row,
                    worst_feedback: entry.worst_feedback.clone(),
                    open_checks: open,
                },
            );
        }
        ForageSnapshot {
            observations: self.observations,
            columns,
        }
    }
}

/// One column in a [`ForageSnapshot`].
#[derive(Clone, Debug, Serialize)]
pub struct ForageColumn {
    pub attempts: u32,
    pub best_score: f64,
    pub best_row: usize,
    pub best_feedback: String,
    pub worst_score: f64,
    pub worst_row: usize,
    pub worst_feedback: String,
    pub open_checks: Vec<String>,
}

/// Immutable ledger snapshot handed to the teacher's forage tools for one
/// step. Read-only by construction: tools cannot spend budget or mutate
/// search state.
#[derive(Clone, Debug, Default)]
pub struct ForageSnapshot {
    pub observations: usize,
    pub columns: BTreeMap<usize, ForageColumn>,
}

#[derive(Clone, Copy, Debug)]
enum ForageToolKind {
    Overview,
    Column,
    OpenChecks,
}

/// A read-only forage tool over one step's snapshot.
struct ForageTool {
    snapshot: Arc<ForageSnapshot>,
    kind: ForageToolKind,
}

#[derive(Deserialize, Default)]
struct ColumnArgs {
    column: usize,
}

#[derive(Deserialize, Default)]
struct PageArgs {
    #[serde(default)]
    offset: usize,
}

const OPEN_CHECKS_PAGE: usize = 50;
const OVERVIEW_ROWS: usize = 30;

impl ForageTool {
    fn overview(&self) -> String {
        let mut rows: Vec<(&usize, &ForageColumn)> = self.snapshot.columns.iter().collect();
        rows.sort_by(|a, b| {
            a.1.best_score
                .partial_cmp(&b.1.best_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(b.0))
        });
        let open_total: usize = rows.iter().map(|(_, c)| c.open_checks.len()).sum();
        let mut out = format!(
            "{} columns observed over {} rollouts; {} open checks total.\n\
             Weakest {} columns (col, best, worst, attempts, open checks):\n",
            rows.len(),
            self.snapshot.observations,
            open_total,
            OVERVIEW_ROWS.min(rows.len()),
        );
        for (col, column) in rows.into_iter().take(OVERVIEW_ROWS) {
            out.push_str(&format!(
                "  q{col}: best {:.2} worst {:.2} tries {} open {}\n",
                column.best_score,
                column.worst_score,
                column.attempts,
                column.open_checks.len(),
            ));
        }
        out.push_str("Use forage_column {\"column\": N} for full detail on one column.");
        out
    }

    fn column(&self, args: ColumnArgs) -> String {
        match self.snapshot.columns.get(&args.column) {
            Some(column) => format!(
                "q{}: {} attempts.\nBEST rollout (cand#{}, {:.2}) feedback:\n{}\n\n\
                 WORST rollout (cand#{}, {:.2}) feedback:\n{}\n\nOpen checks:\n{}",
                args.column,
                column.attempts,
                column.best_row,
                column.best_score,
                truncate_str(&column.best_feedback, 2_000),
                column.worst_row,
                column.worst_score,
                truncate_str(&column.worst_feedback, 2_000),
                if column.open_checks.is_empty() {
                    "  (none — the best rollout passes every extracted check)".to_string()
                } else {
                    column
                        .open_checks
                        .iter()
                        .map(|check| format!("  - {}", truncate_str(check, EXCERPT)))
                        .collect::<Vec<_>>()
                        .join("\n")
                },
            ),
            None => format!("column q{} has no recorded rollouts yet", args.column),
        }
    }

    fn open_checks(&self, args: PageArgs) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (col, column) in &self.snapshot.columns {
            for check in &column.open_checks {
                lines.push(format!(
                    "[q{col} best {:.2}] {}",
                    column.best_score,
                    truncate_str(check, EXCERPT)
                ));
            }
        }
        let total = lines.len();
        let page: Vec<String> = lines
            .into_iter()
            .skip(args.offset)
            .take(OPEN_CHECKS_PAGE)
            .collect();
        if page.is_empty() {
            return format!("{total} open checks; offset {} is past the end", args.offset);
        }
        format!(
            "open checks {}..{} of {total}:\n{}",
            args.offset,
            args.offset + page.len(),
            page.join("\n")
        )
    }
}

impl ToolDyn for ForageTool {
    fn name(&self) -> String {
        match self.kind {
            ForageToolKind::Overview => "forage_overview".to_string(),
            ForageToolKind::Column => "forage_column".to_string(),
            ForageToolKind::OpenChecks => "forage_open_checks".to_string(),
        }
    }

    fn definition<'a>(&'a self, _prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(async move {
            match self.kind {
                ForageToolKind::Overview => ToolDefinition {
                    name: "forage_overview".to_string(),
                    description: "Summary of the search's whole rollout ledger: weakest \
                                  columns with best/worst scores and open-check counts."
                        .to_string(),
                    parameters: serde_json::json!({"type": "object", "properties": {}}),
                },
                ForageToolKind::Column => ToolDefinition {
                    name: "forage_column".to_string(),
                    description: "Full contrastive detail for one column: best and worst \
                                  rollout feedback plus the checks the best rollout still \
                                  fails."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"column": {"type": "integer"}},
                        "required": ["column"]
                    }),
                },
                ForageToolKind::OpenChecks => ToolDefinition {
                    name: "forage_open_checks".to_string(),
                    description: "Paginated list of every open check across the ledger — \
                                  failures the best-known rollout on each column still \
                                  makes. Each is a verified, unharvested win."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"offset": {"type": "integer", "default": 0}}
                    }),
                },
            }
        })
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let parse = |args: &str| -> Result<serde_json::Value, ToolError> {
                if args.trim().is_empty() {
                    Ok(serde_json::Value::Object(serde_json::Map::new()))
                } else {
                    serde_json::from_str(args).map_err(ToolError::JsonError)
                }
            };
            let value = parse(&args)?;
            Ok(match self.kind {
                ForageToolKind::Overview => self.overview(),
                ForageToolKind::Column => {
                    let args: ColumnArgs =
                        serde_json::from_value(value).map_err(ToolError::JsonError)?;
                    self.column(args)
                }
                ForageToolKind::OpenChecks => {
                    let args: PageArgs =
                        serde_json::from_value(value).map_err(ToolError::JsonError)?;
                    self.open_checks(args)
                }
            })
        })
    }
}

/// The three read-only forage tools over one step snapshot.
pub(crate) fn forage_tools(snapshot: Arc<ForageSnapshot>) -> Vec<Arc<dyn ToolDyn>> {
    vec![
        Arc::new(ForageTool {
            snapshot: Arc::clone(&snapshot),
            kind: ForageToolKind::Overview,
        }),
        Arc::new(ForageTool {
            snapshot: Arc::clone(&snapshot),
            kind: ForageToolKind::Column,
        }),
        Arc::new(ForageTool {
            snapshot,
            kind: ForageToolKind::OpenChecks,
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extractor() -> CheckExtractor {
        Arc::new(|feedback: &str| {
            feedback
                .lines()
                .filter_map(|line| line.strip_prefix("MISS:"))
                .map(str::to_string)
                .collect()
        })
    }

    #[test]
    fn digest_lists_open_checks_and_flips() {
        let mut ledger = ForageLedger::default();
        // Column 3: best rollout still misses fact_a → OPEN.
        ledger.observe(3, 1, 0.4, Some("MISS:fact_a\nMISS:fact_b"));
        ledger.observe(3, 2, 0.5, Some("MISS:fact_a"));
        // Column 7: worst missed fact_c, best covers it → FLIP.
        ledger.observe(7, 1, 0.2, Some("MISS:fact_c"));
        ledger.observe(7, 4, 0.9, Some(""));
        let extractor = extractor();
        let digest = ledger.render_digest(Some(&extractor), 5, 50, 10, 10_000);
        assert!(digest.text.contains("[q3 best 0.50 tries 2] fact_a"));
        assert!(!digest.text.contains("fact_b")); // recovered: absent from best
        assert!(digest.text.contains("[q7] cand#4 (0.90) covers what cand#1 (0.20) missed: fact_c"));
        assert!(digest.columns.contains(&3) && digest.columns.contains(&7));
    }

    #[test]
    fn digest_without_extractor_excerpts_weakest_columns() {
        let mut ledger = ForageLedger::default();
        ledger.observe(0, 1, 0.1, Some("bad answer"));
        ledger.observe(1, 1, 0.9, Some("good answer"));
        let digest = ledger.render_digest(None, 0, 5, 5, 10_000);
        assert!(digest.text.contains("WEAKEST COLUMNS"));
        assert!(digest.text.contains("bad answer"));
        assert_eq!(digest.columns.first(), Some(&0));
    }

    #[test]
    fn digest_respects_char_cap() {
        let mut ledger = ForageLedger::default();
        for col in 0..100 {
            ledger.observe(col, 1, 0.1, Some(&format!("MISS:fact_{col:03}")));
        }
        let extractor = extractor();
        let digest = ledger.render_digest(Some(&extractor), 0, 1_000, 10, 900);
        assert!(digest.text.len() <= 900);
    }

    #[tokio::test]
    async fn tools_serve_the_snapshot_read_only() {
        let mut ledger = ForageLedger::default();
        ledger.observe(12, 3, 0.3, Some("MISS:fact_x"));
        ledger.observe(12, 5, 0.8, Some(""));
        let extractor = extractor();
        let tools = forage_tools(Arc::new(ledger.snapshot(Some(&extractor))));
        let overview = tools[0].call(String::new()).await.unwrap();
        assert!(overview.contains("q12"));
        let column = tools[1].call(r#"{"column": 12}"#.to_string()).await.unwrap();
        assert!(column.contains("cand#5") || column.contains("0.80"));
        let checks = tools[2].call("{}".to_string()).await.unwrap();
        // fact_x is on the BEST rollout? No — best (0.8) has no misses, so
        // the check retired; the listing reports zero open checks.
        assert!(checks.contains("0 open checks"));
    }
}
