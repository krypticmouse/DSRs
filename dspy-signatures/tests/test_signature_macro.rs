use dspy_signatures::Signature;
use dspy_rs::field::{In, Out};
use indexmap::IndexMap;

#[derive(Signature)]
struct TestSignature {
    input: In,
    output: Out,
}

#[derive(Signature)]
struct TestSignature2 {
    /// This is a test input
    ///
    /// What is the meaning of life?

    input: In,
    output: Out,
}

#[test]
fn test_signature_macro() {
    let signature = TestSignature::new();
    assert_eq!(signature.name, "TestSignature");
    assert_eq!(signature.instruction.is_empty(), true);
    assert_eq!(signature.input_fields.len(), 1);
    assert_eq!(signature.output_fields.len(), 1);
}

#[test]
fn test_signature_macro_2() {
    let signature = TestSignature2::new();
    assert_eq!(signature.name, "TestSignature2");
    assert_eq!(signature.instruction, "This is a test input\n\nWhat is the meaning of life?");
    assert_eq!(signature.input_fields.len(), 1);
    assert_eq!(signature.output_fields.len(), 1);
} 