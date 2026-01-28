/*
Example demonstrating trajectory analysis with Rlm and #[rlm_type] data shapes.

Run with:
```
cargo run --example rlm_trajectory --features rlm
```

Requires OPENAI_API_KEY in the environment.
*/

use anyhow::Result;
use dspy_rs::rlm::{Rlm, RlmResult};
use dspy_rs::{configure, rlm_type, ChatAdapter, RlmType, Signature, LM};

/// A trajectory step (simplified; real datasets may include more fields).
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(repr = "Step({self.source}: {self.content:60}...)")]
pub struct Step {
    pub source: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[rlm_type]
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: String,
}

/// A trajectory - the main input type.
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(
    repr = "Trajectory({len(self.steps)} steps, session={self.session_id:10}...)",
    iter = "steps",
    index = "steps",
)]
pub struct Trajectory {
    pub session_id: String,

    #[rlm(
        desc = "All conversation steps; use .user_steps for user-only entries",
        filter_property = "user_steps",
        filter_value = "user",
        filter_field = "source"
    )]
    pub steps: Vec<Step>,
}

/// A documentation pattern that reduces tool calls.
#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(repr = "Pattern({self.name}, {len(self.examples)} examples)")]
pub struct Pattern {
    pub id: String,
    pub name: String,
    pub description: String,
    pub documentation_section: String,
    pub example_trajectory_ids: Vec<String>,
    pub examples: Vec<PatternExample>,
    pub estimated_calls_saved: i32,
}

#[rlm_type]
#[derive(Clone, Debug)]
pub struct PatternExample {
    pub trajectory_id: String,
    pub step_range: String,
    pub description: String,
    pub suggested_doc: String,
}

/// Analyze trajectories to identify documentation patterns that reduce tool calls.
#[derive(Signature, Clone, Debug)]
struct AnalyzeTrajectories {
    #[input(desc = "Trajectories to analyze - use .user_steps, .steps[i], len()")]
    trajectories: Vec<Trajectory>,

    #[input(desc = "Existing patterns to update or extend")]
    existing_patterns: Vec<Pattern>,

    #[output]
    #[check("len(this) >= 1", label = "has_patterns")]
    updated_patterns: Vec<Pattern>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let trajectories = sample_trajectories();
    let existing_patterns = seed_patterns();

    let input = AnalyzeTrajectoriesInput {
        trajectories,
        existing_patterns,
    };

    // Configure DSRs with global LM settings
    let lm = LM::builder()
        .model("openai:gpt-4o-mini".to_string())
        .temperature(0.2)
        .build()
        .await?;

    configure(lm, ChatAdapter);

    let rlm = Rlm::<AnalyzeTrajectories>::new();
    let result = rlm.call(input).await?;

    print_patterns(&result);
    print_trajectory_history(&result.input.trajectories);
    print_stats(&result);

    Ok(())
}

fn print_patterns(result: &RlmResult<AnalyzeTrajectories>) {
    println!("Found {} patterns:", result.output.updated_patterns.len());
    for pattern in &result.output.updated_patterns {
        println!(
            "  - {} ({} examples, ~{} calls saved)",
            pattern.name,
            pattern.examples.len(),
            pattern.estimated_calls_saved
        );
    }
}

fn print_trajectory_history(trajectories: &[Trajectory]) {
    println!("\nTrajectory history:");
    for trajectory in trajectories {
        println!(
            "Session {} ({} steps):",
            trajectory.session_id,
            trajectory.steps.len()
        );
        for (idx, step) in trajectory.steps.iter().enumerate() {
            println!("  {idx:02}: {} -> {}", step.source, step.content);
        }
    }
}

fn print_stats(result: &RlmResult<AnalyzeTrajectories>) {
    let summary = &result.constraint_summary;
    println!(
        "\nStats: {} iterations, {} LLM calls, fallback={}",
        result.iterations,
        result.llm_calls,
        result.is_fallback()
    );
    println!(
        "Constraint summary: {} checks passed, {} checks failed, {} assertions passed",
        summary.checks_passed, summary.checks_failed, summary.assertions_passed
    );

    if result.has_constraint_warnings() {
        println!("\nWarnings:");
        for check in result.failed_checks() {
            println!("  - {}: {}", check.label, check.expression);
        }
    }
}

fn sample_trajectories() -> Vec<Trajectory> {
    vec![
        Trajectory {
            session_id: "sess-001".to_string(),
            steps: vec![
                Step {
                    source: "user".to_string(),
                    content: "Summarize the incident response guide for on-call.".to_string(),
                    tool_calls: None,
                },
                Step {
                    source: "agent".to_string(),
                    content: "I will look up the incident response guide and summarize it.".to_string(),
                    tool_calls: Some(vec![ToolCall {
                        tool_name: "search_docs".to_string(),
                        arguments: "{\"query\":\"incident response guide\"}".to_string(),
                    }]),
                },
                Step {
                    source: "agent".to_string(),
                    content: "Summary: triage, contain, communicate, resolve, and postmortem."
                        .to_string(),
                    tool_calls: None,
                },
            ],
        },
        Trajectory {
            session_id: "sess-002".to_string(),
            steps: vec![
                Step {
                    source: "user".to_string(),
                    content: "How do I rotate database credentials safely?".to_string(),
                    tool_calls: None,
                },
                Step {
                    source: "agent".to_string(),
                    content: "Checking the credential rotation runbook.".to_string(),
                    tool_calls: Some(vec![ToolCall {
                        tool_name: "search_docs".to_string(),
                        arguments: "{\"query\":\"credential rotation runbook\"}".to_string(),
                    }]),
                },
                Step {
                    source: "agent".to_string(),
                    content: "Steps: create new secret, deploy, verify, revoke old secret."
                        .to_string(),
                    tool_calls: None,
                },
            ],
        },
    ]
}

fn seed_patterns() -> Vec<Pattern> {
    vec![Pattern {
        id: "pattern-doc-shortcuts".to_string(),
        name: "Doc shortcuts for common runbooks".to_string(),
        description: "Inline the top runbook steps to avoid repeated searches.".to_string(),
        documentation_section: "Runbooks: On-call essentials".to_string(),
        example_trajectory_ids: vec!["sess-001".to_string()],
        examples: vec![PatternExample {
            trajectory_id: "sess-001".to_string(),
            step_range: "1-3".to_string(),
            description: "Agent repeatedly searched for the same guide.".to_string(),
            suggested_doc: "Include the incident response summary directly.".to_string(),
        }],
        estimated_calls_saved: 2,
    }]
}
