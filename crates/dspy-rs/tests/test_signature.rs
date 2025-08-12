use dspy_rs::field::{In, Out};
use dspy_rs::internal::{MetaField, MetaSignature};
use dspy_rs::sign;
use indexmap::IndexMap;
use rstest::*;

#[rstest]
fn test_field_initizalization() {
    let input_field = In::<String>::default();

    let output_field = Out::<String>::default();
    assert_eq!(input_field.desc, "");
    assert_eq!(output_field.desc, "");
}

#[rstest]
fn test_signature_from_string() {
    let signature = sign! {
        (inp1: String, inp2: String) -> out1: String, out2: String
    };

    assert_eq!(signature.instruction, "");
    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);
}

#[rstest]
fn test_signature_insert() {
    let mut signature = sign! {
        (inp1: String, inp2: String) -> out1: String, out2: String
    };
    signature.insert("inp3".to_string(), In::<String>::default(), 0);

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc, "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc, "");
}

#[rstest]
fn test_signature_append() {
    let mut signature = sign! {
        (inp1: String, inp2: String) -> out1: String, out2: String
    };
    signature.append("inp3".to_string(), In::<String>::default());

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc, "");
    assert_eq!(signature.input_fields.get("inp1").unwrap().desc, "");
}

#[rstest]
fn test_signature_prepend() {
    let mut signature = sign! {
        (inp1: String, inp2: String) -> out1: String, out2: String
    };
    signature.prepend("inp3".to_string(), In::<String>::default());

    assert_eq!(signature.input_fields.len(), 3);
    assert_eq!(signature.input_fields.get("inp3").unwrap().desc, "");
}

#[rstest]
fn test_signature_builder() {
    let signature = MetaSignature {
        name: "test".to_string(),
        instruction: "given a input, return a output".to_string(),
        input_fields: IndexMap::from_iter(vec![
            (
                "inp1".to_string(),
                MetaField {
                    desc: "".to_string(),
                    schema: "".to_string(),
                    data_type: "".to_string(),
                    __dsrs_field_type: "Input".to_string(),
                },
            ),
            (
                "inp2".to_string(),
                MetaField {
                    desc: "".to_string(),
                    schema: "".to_string(),
                    data_type: "".to_string(),
                    __dsrs_field_type: "Input".to_string(),
                },
            ),
        ]),
        output_fields: IndexMap::from_iter(vec![
            (
                "out1".to_string(),
                MetaField {
                    desc: "".to_string(),
                    schema: "".to_string(),
                    data_type: "".to_string(),
                    __dsrs_field_type: "Output".to_string(),
                },
            ),
            (
                "out2".to_string(),
                MetaField {
                    desc: "".to_string(),
                    schema: "".to_string(),
                    data_type: "".to_string(),
                    __dsrs_field_type: "Output".to_string(),
                },
            ),
        ]),
    };

    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);

    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);

    assert_eq!(signature.input_fields.get("inp1").unwrap().desc, "");
    assert_eq!(signature.input_fields.get("inp2").unwrap().desc, "");

    assert_eq!(signature.output_fields.get("out1").unwrap().desc, "");

    assert_eq!(signature.output_fields.get("out2").unwrap().desc, "");
}

#[rstest]
fn test_signature_no_builder() {
    let signature = MetaSignature {
        name: "QASignature".to_string(),
        instruction: "You'll be given a question and a context, and you'll need to answer the question based on the context".to_string(),
        input_fields: IndexMap::from_iter(vec![
            ("question".to_string(), MetaField {
                desc: "".to_string(),
                schema: "".to_string(),
                data_type: "".to_string(),
                __dsrs_field_type: "Input".to_string(),
            }),
        ]),
        output_fields: IndexMap::from_iter(vec![
            ("answer".to_string(), MetaField {
                desc: "".to_string(),
                schema: "".to_string(),
                data_type: "".to_string(),
                __dsrs_field_type: "Output".to_string(),
            }),
        ]),
    };

    assert_eq!(signature.input_fields.len(), 1);
    assert_eq!(signature.output_fields.len(), 1);
    assert_eq!(signature.input_fields.get("question").unwrap().desc, "");
    assert_eq!(signature.output_fields.get("answer").unwrap().desc, "");
}

#[rstest]
fn test_signature_macro() {
    let signature = sign! {
        (inp1: String, inp2: bool) -> out1: String, out2: i8
    };

    assert_eq!(signature.name, "InlineSignature");
    assert_eq!(signature.instruction, "");
    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 2);

    assert_eq!(
        signature
            .input_fields
            .get("inp1")
            .unwrap()
            .__dsrs_field_type,
        "Input"
    );
    assert_eq!(
        signature
            .input_fields
            .get("inp2")
            .unwrap()
            .__dsrs_field_type,
        "Input"
    );
    assert_eq!(
        signature
            .output_fields
            .get("out1")
            .unwrap()
            .__dsrs_field_type,
        "Output"
    );
    assert_eq!(
        signature
            .output_fields
            .get("out2")
            .unwrap()
            .__dsrs_field_type,
        "Output"
    );
}
