use dspy_rs::signature::field::Field;
use dspy_rs::signature::signature::Signature;
use indexmap::IndexMap;
use rstest::*;

#[rstest]
fn test_field_initizalization() {
    let input_field = Field::In("desc");

    let output_field = Field::Out("desc");
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
    signature.insert("inp3".to_string(), Field::In("desc"), 0);

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
}

#[rstest]
fn test_signature_append() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2");
    signature.append("inp3".to_string(), Field::In("desc"));

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc(), "");
}

#[rstest]
fn test_signature_prepend() {
    let mut signature = Signature::from("inp1, inp2 -> out1, out2");
    signature.prepend("inp3".to_string(), Field::In("desc"));

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc(), "desc");
}

#[rstest]
fn test_signature_builder() {
    let signature = Signature::builder()
        .name("test")
        .instruction("given a input, return a output".to_string())
        .input_fields(IndexMap::from_iter(vec![
            ("inp1".to_string(), Field::In("desc 1")),
            ("inp2".to_string(), Field::In("desc 2")),
        ]))
        .output_fields(IndexMap::from_iter(vec![
            ("out1".to_string(), Field::Out("desc 1")),
            ("out2".to_string(), Field::Out("desc 2")),
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
