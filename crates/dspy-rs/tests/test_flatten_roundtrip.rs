use dspy_rs::{Augmented, ChatAdapter, Example, Message, Reasoning, Signature, WithReasoning};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[test]
fn augmented_demo_roundtrips_through_adapter() {
    let adapter = ChatAdapter;
    let demo = Example::<Augmented<QA, Reasoning>>::new(
        QAInput {
            question: "What is 2+2?".to_string(),
        },
        WithReasoning {
            reasoning: "Add the numbers".to_string(),
            inner: QAOutput {
                answer: "4".to_string(),
            },
        },
    );

    let (user_msg, assistant_msg) = adapter.format_demo_typed::<Augmented<QA, Reasoning>>(&demo);
    let schema = <Augmented<QA, Reasoning> as Signature>::schema();
    let output_names: Vec<&str> = schema.output_fields().iter().map(|f| f.lm_name).collect();

    assert!(user_msg.contains("question"));
    assert!(assistant_msg.contains("reasoning"));
    assert!(assistant_msg.contains("answer"));

    let response = Message::assistant(assistant_msg);
    let (parsed, _meta) = adapter
        .parse_response_typed::<Augmented<QA, Reasoning>>(&response)
        .expect("typed parse should succeed");

    assert_eq!(parsed.reasoning, "Add the numbers");
    assert_eq!(parsed.answer, "4");

    assert_eq!(output_names, vec!["reasoning", "answer"]);
}
