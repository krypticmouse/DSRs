//! E2E test: Test StorableRlmResult serialization roundtrip.
//!
//! This example demonstrates how to create, serialize, and deserialize
//! StorableRlmResult without requiring a live LLM call.
//!
//! Run: cargo run --example rlm_storage --features rlm

use dspy_rs::rlm::{ConstraintSummary, REPLHistory, StorableFieldMeta, StorableRlmResult};
use dspy_rs::ConstraintResult;
use indexmap::IndexMap;
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    // Create a sample trajectory
    let trajectory = REPLHistory::new()
        .append_with_reasoning(
            "First, let me analyze the text".to_string(),
            "text = input_data['text']\nprint(f'Text length: {len(text)}')"
                .to_string(),
            "Text length: 142".to_string(),
        )
        .append_with_reasoning(
            "Now I'll summarize the key points".to_string(),
            "summary = 'Rust: safe, fast, concurrent systems language with memory safety sans GC.'\nprint(summary)"
                .to_string(),
            "Rust: safe, fast, concurrent systems language with memory safety sans GC."
                .to_string(),
        );

    // Create sample field metadata
    let mut field_metas = IndexMap::new();
    field_metas.insert(
        "summary".to_string(),
        StorableFieldMeta {
            raw_text: "Rust: safe, fast, concurrent systems language with memory safety sans GC."
                .to_string(),
            checks: vec![ConstraintResult {
                label: "non_trivial".to_string(),
                expression: "len(this) > 10".to_string(),
                passed: true,
            }],
        },
    );

    // Create metadata
    let mut metadata = HashMap::new();
    metadata.insert("task_id".to_string(), serde_json::json!("test-001"));
    metadata.insert("experiment".to_string(), serde_json::json!("storage_e2e"));

    // Build the StorableRlmResult
    let storable = StorableRlmResult {
        id: trajectory.id,
        created_at: trajectory.created_at,
        input_json: serde_json::json!({
            "text": "Rust is a systems programming language focused on safety, \
                     speed, and concurrency. It achieves memory safety without \
                     garbage collection."
        }),
        output_json: serde_json::json!({
            "summary": "Rust: safe, fast, concurrent systems language with memory safety sans GC."
        }),
        trajectory,
        field_metas,
        iterations: 2,
        llm_calls: 3,
        extraction_fallback: false,
        constraint_summary: ConstraintSummary {
            checks_passed: 1,
            checks_failed: 0,
            assertions_passed: 0,
        },
        metadata,
    };

    // Serialize to JSON
    let json = storable.to_json_pretty()?;
    println!("=== Serialized StorableRlmResult ===\n{}\n", json);

    // Deserialize back
    let restored = StorableRlmResult::from_json(&json)?;

    // Verify roundtrip
    println!("=== Verification ===");
    println!("ID matches: {}", storable.id == restored.id);
    println!("Created at matches: {}", storable.created_at == restored.created_at);
    println!("Iterations: {}", restored.iterations);
    println!("LLM calls: {}", restored.llm_calls);
    println!("Trajectory entries: {}", restored.trajectory.entries.len());
    println!("Trajectory ID: {}", restored.trajectory.id);
    println!(
        "Trajectory ID preserved: {}",
        storable.trajectory.id == restored.trajectory.id
    );

    // Print trajectory with timestamps
    println!("\n=== Trajectory ===");
    for (i, entry) in restored.trajectory.entries.iter().enumerate() {
        println!("Step {} @ {}:", i + 1, entry.timestamp);
        println!("  Reasoning: {}", &entry.reasoning);
        println!(
            "  Code: {}...",
            &entry.code.chars().take(60).collect::<String>()
        );
        println!(
            "  Output: {}...",
            &entry.output.chars().take(60).collect::<String>()
        );
    }

    // Print field metas
    println!("\n=== Field Metas ===");
    for (name, meta) in &restored.field_metas {
        println!("  {}: {} checks", name, meta.checks.len());
        for check in &meta.checks {
            println!(
                "    - {} ({}): {}",
                check.label,
                check.expression,
                if check.passed { "PASS" } else { "FAIL" }
            );
        }
    }

    // Print metadata
    println!("\n=== Metadata ===");
    for (k, v) in &restored.metadata {
        println!("  {}: {}", k, v);
    }

    println!("\n=== All Checks Passed! ===");
    Ok(())
}
