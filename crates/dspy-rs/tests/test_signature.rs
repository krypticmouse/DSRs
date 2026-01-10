use dspy_rs::{LegacySignature, MetaSignature, field};
use rstest::*;

#[LegacySignature]
struct InlineSignature {
    #[input]
    inp1: String,
    #[input]
    inp2: String,
    #[output]
    out1: String,
    #[output]
    out2: String,
}

#[rstest]
fn test_signature_from_string() {
    let signature = InlineSignature::new();

    assert_eq!(signature.instruction, "");
    assert_eq!(signature.input_fields_len(), 2);
    assert_eq!(signature.output_fields_len(), 2);
}

#[rstest]
fn test_signature_append() {
    let mut signature = InlineSignature::new();
    let field_obj = field! {
        input => inp3 : String
    };
    let _ = signature.append("inp3", field_obj["inp3"].clone());

    assert_eq!(signature.input_fields_len(), 3);
    assert_eq!(
        signature.input_fields.get("inp3").unwrap()["__dsrs_field_type"],
        "input"
    );
    assert_eq!(signature.input_fields.get("inp3").unwrap()["desc"], "");
    assert_eq!(
        signature.input_fields.get("inp1").unwrap()["__dsrs_field_type"],
        "input"
    );
    assert_eq!(signature.input_fields.get("inp1").unwrap()["desc"], "");
}
