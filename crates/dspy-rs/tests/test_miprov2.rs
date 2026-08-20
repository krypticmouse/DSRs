use dspy_rs::{MIPROv2, PromptingTips};
use rstest::*;

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
fn test_miprov2_default_configuration() {
    let optimizer = MIPROv2::builder().build();

    assert_eq!(optimizer.num_candidates, 10);
    assert_eq!(optimizer.num_trials, 20);
    assert_eq!(optimizer.minibatch_size, 25);
}
