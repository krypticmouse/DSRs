use dspy_rs::Signature as SignatureTrait;

/// Test instruction
#[derive(dsrs_macros::Signature)]
struct TestSig {
    #[input]
    #[alias("question_text")]
    question: String,

    #[output]
    #[check("this != ''", label = "non_empty")]
    answer: String,
}

#[test]
fn test_generates_input_struct() {
    let input = TestSigInput {
        question: "test".to_string(),
    };
    assert_eq!(input.question, "test");
}

#[test]
fn test_generates_signature_impl() {
    assert_eq!(<TestSig as SignatureTrait>::instruction(), "Test instruction");

    let input_fields = <TestSig as SignatureTrait>::input_fields();
    assert_eq!(input_fields.len(), 1);
    assert_eq!(input_fields[0].name, "question_text");

    let output_fields = <TestSig as SignatureTrait>::output_fields();
    assert_eq!(output_fields.len(), 1);
    assert_eq!(output_fields[0].constraints.len(), 1);
    assert_eq!(output_fields[0].constraints[0].label, "non_empty");
}

#[test]
fn test_from_parts_into_parts() {
    let input = TestSigInput {
        question: "q".to_string(),
    };
    let output = __TestSigOutput {
        answer: "a".to_string(),
    };

    let full = TestSig::from_parts(input.clone(), output.clone());
    assert_eq!(full.question, "q");
    assert_eq!(full.answer, "a");

    let (input2, output2) = full.into_parts();
    assert_eq!(input2.question, "q");
    assert_eq!(output2.answer, "a");
}

#[test]
fn test_baml_type_impl() {
    let _ = <TestSig as dspy_rs::baml_bridge::BamlType>::baml_output_format();
}
