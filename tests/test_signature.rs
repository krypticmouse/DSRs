use dspy_rs::field::{Field, In, Out};
use dspy_rs::signature::Signature;
use indexmap::IndexMap;
use rstest::*;

#[rstest]
fn test_field_initizalization() {
    let input_field = In::default();

    let output_field = Out::default();
    assert_eq!(input_field.desc(), "desc");
    assert_eq!(output_field.desc(), "desc");
}

#[rstest]
fn test_signature_from_string() {
    let signature = Signature::from("inp1, inp2 -> out1, out2");

    assert_eq!(
        signature.instruction,
        "Given a inputs inp1, inp2, return outputs out1, out2"
    );
    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);
}

#[rstest]
fn test_signature_insert() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2");
    signature.insert_input("inp3".to_string(), In::default(), 0);

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
}

#[rstest]
fn test_signature_append() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2");
    signature.append_input("inp3".to_string(), In::default());

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
}

#[rstest]
fn test_signature_prepend() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2");
    signature.prepend_input("inp3".to_string(), In::default());

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
}

#[rstest]
fn test_signature_builder() {
    let signature = Signature::builder()
        .name("test".to_string())
        .instruction("given a input, return a output".to_string())
        .input_fields(IndexMap::from_iter(vec![
            ("inp1".to_string(), In::default()),
            ("inp2".to_string(), In::default()),
        ]))
        .output_fields(IndexMap::from_iter(vec![
            ("out1".to_string(), Out::default()),
            ("out2".to_string(), Out::default()),
        ]))
        .build()
        .unwrap();

    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);

    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "desc 1");
    assert_eq!(signature.input_fields.get("inp2").unwrap().desc(), "desc 2");

    assert_eq!(
        signature.output_fields.get("out1").unwrap().desc(),
        "desc 1"
    );

    assert_eq!(
        signature.output_fields.get("out2").unwrap().desc(),
        "desc 2"
    );
}

#[rstest]
fn test_signature_no_builder() {
    let signature = Signature {
        name: "QASignature".to_string(),
        instruction: "You'll be given a question and a context, and you'll need to answer the question based on the context".to_string(),
        input_fields: IndexMap::from_iter(vec![
            ("question".to_string(), In::default()),
        ]),
        output_fields: IndexMap::from_iter(vec![
            ("answer".to_string(), Out::default()),
        ]),
    };

    assert_eq!(signature.input_fields.len(), 1);
    assert_eq!(signature.output_fields.len(), 1);
    assert_eq!(
        signature.input_fields.get("question").unwrap().desc(),
        "The question to answer"
    );
    assert_eq!(
        signature.output_fields.get("answer").unwrap().desc(),
        "The answer to the question"
    );
}
