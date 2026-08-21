use dspy_rs::ir::SignatureDef;
use dspy_rs::{Augmented, ChatAdapter, Demo, Message, Reasoning, Signature, WithReasoning};
use serde_json::Value;

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

fn json_map<T: serde::Serialize>(value: &T) -> serde_json::Map<String, Value> {
    match serde_json::to_value(value).expect("serializable") {
        Value::Object(map) => map,
        other => panic!("expected object, got {other:?}"),
    }
}

#[test]
fn augmented_demo_roundtrips_through_adapter() {
    let adapter = ChatAdapter;
    let demo = Demo::<Augmented<QA, Reasoning>>::new(
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

    let def = SignatureDef::of::<Augmented<QA, Reasoning>>();
    let types = SignatureDef::types_of::<Augmented<QA, Reasoning>>();
    // `WithReasoning` flattens: the serialized demo output keys flat by leaf
    // name (`reasoning`, `answer`) — exactly the def's canonical field names.
    let user_msg = adapter.format_input_def(def, &json_map(&demo.input));
    let assistant_msg = adapter.format_output_def(def, &json_map(&demo.output));
    let schema = <Augmented<QA, Reasoning> as Signature>::schema();
    let output_names: Vec<&str> = schema.output_fields().iter().map(|f| f.lm_name).collect();

    assert!(user_msg.contains("question"));
    assert!(assistant_msg.contains("reasoning"));
    assert!(assistant_msg.contains("answer"));

    let response = Message::assistant(assistant_msg);
    let (output_map, _meta) = adapter
        .parse_output_def(def, types, &response)
        .expect("def parse should succeed");
    let parsed: WithReasoning<QAOutput> =
        serde_json::from_value(Value::Object(output_map)).expect("typed assembly");

    assert_eq!(parsed.reasoning, "Add the numbers");
    assert_eq!(parsed.answer, "4");

    assert_eq!(output_names, vec!["reasoning", "answer"]);
}
