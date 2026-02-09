use dspy_rs::{Signature, WithReasoning};

#[derive(Signature, Clone, Debug, PartialEq)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[test]
fn with_reasoning_deref_exposes_inner_fields() {
    let output = WithReasoning {
        reasoning: "thinking".to_string(),
        inner: QAOutput {
            answer: "Paris".to_string(),
        },
    };

    assert_eq!(output.reasoning, "thinking");
    assert_eq!(output.answer, "Paris");

    let WithReasoning { reasoning, inner } = output;
    assert_eq!(reasoning, "thinking");
    assert_eq!(inner.answer, "Paris");
}
