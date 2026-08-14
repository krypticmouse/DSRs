use dspy_rs::{Eval, MIPROv2, PromptCandidate, PromptingTips, Signature, Trace, TraceOutcome};
use rstest::*;

#[derive(Signature, Clone, Debug)]
struct TestSignature {
    #[input]
    question: String,

    #[output]
    answer: String,
}

/// A rollout trace carrying only a whole-program score, the projection MIPRO's
/// candidate generation reads.
fn scored_trace(score: Option<f64>) -> Trace {
    Trace {
        outcome: score.map(|score| TraceOutcome {
            output: None,
            error: None,
            eval: Some(Eval::score(score)),
            duration_us: 0,
        }),
        ..Trace::default()
    }
}

fn trace_score(trace: &Trace) -> Option<f64> {
    trace
        .outcome
        .as_ref()
        .and_then(|outcome| outcome.eval.as_ref())
        .map(|eval| eval.score)
}

#[rstest]
fn test_prompting_tips_default() {
    let tips = PromptingTips::default_tips();

    assert!(!tips.tips.is_empty());
    assert!(tips.tips.len() >= 15);
}

#[rstest]
fn test_prompting_tips_formatting() {
    let tips = PromptingTips::default_tips();
    let formatted = tips.format_for_prompt();

    assert!(formatted.contains("1."));
    assert!(formatted.contains("\n"));
}

#[rstest]
fn test_prompt_candidate_creation() {
    let candidate = PromptCandidate::new("Test instruction".to_string());

    assert_eq!(candidate.instruction, "Test instruction");
    assert_eq!(candidate.score, 0.0);
}

#[rstest]
fn test_prompt_candidate_with_score() {
    let candidate = PromptCandidate::new("test".to_string()).with_score(0.85);
    assert_eq!(candidate.score, 0.85);
}

#[rstest]
fn test_miprov2_default_configuration() {
    let optimizer = MIPROv2::builder().build();

    assert_eq!(optimizer.num_candidates, 10);
    assert_eq!(optimizer.num_trials, 20);
    assert_eq!(optimizer.minibatch_size, 25);
}

#[rstest]
fn test_select_best_traces_descending_order() {
    let optimizer = MIPROv2::builder().build();

    let traces = vec![
        scored_trace(Some(0.1)),
        scored_trace(Some(0.5)),
        scored_trace(Some(0.3)),
    ];

    let best = optimizer.select_best_traces(&traces, 2);
    assert_eq!(best.len(), 2);
    assert_eq!(trace_score(best[0]), Some(0.5));
    assert_eq!(trace_score(best[1]), Some(0.3));
}

#[rstest]
fn test_select_best_traces_ignores_unscored() {
    let optimizer = MIPROv2::builder().build();

    let traces = vec![scored_trace(None), scored_trace(Some(0.8))];

    let best = optimizer.select_best_traces(&traces, 2);
    assert_eq!(best.len(), 1);
    assert_eq!(trace_score(best[0]), Some(0.8));
}

#[rstest]
fn test_create_prompt_candidates_uses_all_instructions() {
    let optimizer = MIPROv2::builder().build();
    let candidates = optimizer.create_prompt_candidates(vec![
        "instruction-1".to_string(),
        "instruction-2".to_string(),
    ]);

    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].instruction, "instruction-1");
    assert_eq!(candidates[1].instruction, "instruction-2");
}

#[rstest]
fn test_format_schema_fields_reads_typed_schema() {
    let optimizer = MIPROv2::builder().build();
    let rendered = optimizer.format_schema_fields(TestSignature::schema());

    assert!(rendered.contains("Input Fields:"));
    assert!(rendered.contains("question"));
    assert!(rendered.contains("Output Fields:"));
    assert!(rendered.contains("answer"));
}
