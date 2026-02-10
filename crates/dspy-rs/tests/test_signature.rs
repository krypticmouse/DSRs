use dspy_rs::Signature;

#[derive(Signature, Clone, Debug)]
struct BasicSignature {
    /// Provide a concise answer.

    #[input(desc = "Question to answer")]
    question: String,

    #[output(desc = "Final answer")]
    answer: String,
}

#[test]
fn signature_instruction_and_schema_fields_are_exposed() {
    let schema = BasicSignature::schema();

    let instruction = BasicSignature::instruction();
    assert!(
        instruction.is_empty() || instruction.contains("Provide a concise answer"),
        "unexpected instruction rendering: {instruction:?}"
    );
    assert_eq!(schema.input_fields().len(), 1);
    assert_eq!(schema.output_fields().len(), 1);

    let input = &schema.input_fields()[0];
    assert_eq!(input.rust_name, "question");
    assert_eq!(input.lm_name, "question");
    assert_eq!(input.docs, "Question to answer");

    let output = &schema.output_fields()[0];
    assert_eq!(output.rust_name, "answer");
    assert_eq!(output.lm_name, "answer");
    assert_eq!(output.docs, "Final answer");
}

#[test]
fn signature_metadata_tables_match_schema_fields() {
    let input_meta = BasicSignature::input_field_metadata();
    let output_meta = BasicSignature::output_field_metadata();

    assert_eq!(input_meta.len(), 1);
    assert_eq!(output_meta.len(), 1);

    assert_eq!(input_meta[0].rust_name, "question");
    assert_eq!(output_meta[0].rust_name, "answer");
    assert_eq!(input_meta[0].alias, None);
    assert_eq!(output_meta[0].alias, None);
}
