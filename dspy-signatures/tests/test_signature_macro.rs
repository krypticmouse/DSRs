use dspy_rs::field::{FieldType, In, Out};
use dspy_signatures::Signature;
use schemars::JsonSchema;

#[derive(Signature)]
struct TestSignature {
    input: In<String>,
    output: Out<i8>,
}

#[derive(JsonSchema)]
struct TestOutput {
    output1: i8,
    output2: String,
    output3: bool,
}

#[derive(Signature)]
struct TestSignature2 {
    /// This is a test input
    ///
    /// What is the meaning of life?
    input1: In<String>,
    input2: In<i8>,
    output1: Out<TestOutput>,
}

#[test]
fn test_signature_macro() {
    let signature = TestSignature::new();
    assert_eq!(signature.name, "TestSignature");
    assert_eq!(signature.instruction.is_empty(), true);
    assert_eq!(signature.input_fields.len(), 1);
    assert_eq!(signature.output_fields.len(), 1);

    assert_eq!(
        signature
            .input_fields
            .get("input")
            .unwrap()
            .__dsrs_field_type,
        FieldType::Input
    );
    assert_eq!(
        signature
            .output_fields
            .get("output")
            .unwrap()
            .__dsrs_field_type,
        FieldType::Output
    );
}

#[test]
fn test_signature_macro_2() {
    let signature = TestSignature2::new();
    assert_eq!(signature.name, "TestSignature2");
    assert_eq!(
        signature.instruction,
        "This is a test input\n\nWhat is the meaning of life?"
    );
    assert_eq!(signature.input_fields.len(), 2);
    assert_eq!(signature.output_fields.len(), 1);

    assert_eq!(
        signature
            .input_fields
            .get("input1")
            .unwrap()
            .__dsrs_field_type,
        FieldType::Input
    );
    assert_eq!(
        signature
            .input_fields
            .get("input2")
            .unwrap()
            .__dsrs_field_type,
        FieldType::Input
    );
    assert_eq!(
        signature
            .output_fields
            .get("output1")
            .unwrap()
            .__dsrs_field_type,
        FieldType::Output
    );

    assert_eq!(
        signature.output_fields.get("output1").unwrap().schema,
        "{\"output1\":{\"format\":\"int8\",\"maximum\":127,\"minimum\":-128,\"type\":\"integer\"},\"output2\":{\"type\":\"string\"},\"output3\":{\"type\":\"boolean\"}}"
    );
    assert_eq!(
        signature.input_fields.get("input1").unwrap().schema,
        "".to_string()
    );
    assert_eq!(
        signature.input_fields.get("input2").unwrap().schema,
        "".to_string()
    );

    assert_eq!(
        signature.input_fields.get("input1").unwrap().data_type,
        "String"
    );
    assert_eq!(
        signature.input_fields.get("input2").unwrap().data_type,
        "i8"
    );
    assert_eq!(
        signature.output_fields.get("output1").unwrap().data_type,
        "TestOutput"
    );
}
