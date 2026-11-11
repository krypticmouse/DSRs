use dspy_rs::ir::{FieldDef, RenderSpec, SignatureDef};
use dspy_rs::{ChatAdapter, FieldType, JsonMap};
use serde_json::json;

fn fixture() -> (SignatureDef, JsonMap) {
    let mut question = FieldDef::new("question", FieldType::String).aliased("query");
    question.render = RenderSpec::Jinja(
        "{{ this }}|{{ input.query }}|{{ input.context }}|{{ field.name }}|{{ vars|length }}"
            .into(),
    );
    let mut context = FieldDef::new("context", FieldType::String);
    context.render = RenderSpec::Jinja(
        "{{ this }}|{{ input.question }}|{{ input.query }}|{{ field.rust_name }}".into(),
    );
    let def = SignatureDef {
        name: "SharedContext".into(),
        instruction: "Answer using the context.".into(),
        inputs: vec![question, context].into_boxed_slice(),
        outputs: vec![FieldDef::new("answer", FieldType::String)].into_boxed_slice(),
    };
    let input = serde_json::from_value(json!({"question":"one","context":"two"})).unwrap();
    (def, input)
}

#[test]
fn jinja_fields_share_aliases_without_mutating_or_retaining_inputs() {
    let (def, mut input) = fixture();
    let original = input.clone();
    let rendered = ChatAdapter.try_format_input_def(&def, &input).unwrap();
    assert!(rendered.contains("[[ ## query ## ]]\none|one|two|query|0"));
    assert!(rendered.contains("[[ ## context ## ]]\ntwo|one|one|context"));
    assert_eq!(input, original);

    input.insert("question".into(), json!("changed"));
    let rendered = ChatAdapter.try_format_input_def(&def, &input).unwrap();
    assert!(rendered.contains("changed|changed|two|query|0"));
    assert!(rendered.contains("two|changed|changed|context"));
}

#[test]
fn explicit_alias_keys_win_and_absent_fields_are_skipped() {
    let (mut def, mut input) = fixture();
    input.insert("query".into(), json!("explicit"));
    let rendered = ChatAdapter.try_format_input_def(&def, &input).unwrap();
    assert!(rendered.contains("one|explicit|two|query|0"));
    assert!(rendered.contains("two|one|explicit|context"));

    input.remove("question");
    def.inputs[1].render = RenderSpec::Default;
    let rendered = ChatAdapter.try_format_input_def(&def, &input).unwrap();
    assert!(!rendered.contains("[[ ## query ## ]]"));
    assert!(rendered.contains("[[ ## context ## ]]\ntwo"));
}

#[test]
fn failed_render_does_not_poison_later_calls_or_hide_the_error() {
    let (def, mut input) = fixture();
    input.remove("context");
    assert!(ChatAdapter.try_format_input_def(&def, &input).is_err());
    assert_eq!(
        ChatAdapter.format_input_def(&def, &input),
        "[[ ## input_render_error ## ]]"
    );
    input.insert("context".into(), json!("recovered"));
    let rendered = ChatAdapter.try_format_input_def(&def, &input).unwrap();
    assert!(rendered.contains("one|one|recovered|query|0"));
    assert!(rendered.contains("recovered|one|one|context"));
}
