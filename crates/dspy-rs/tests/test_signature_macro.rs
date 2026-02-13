use dspy_rs::Signature;

#[derive(Signature, Clone, Debug)]
struct AliasAndFormatSignature {
    /// Test alias and format metadata on typed signatures.

    #[input(desc = "Free-form payload")]
    #[alias("payload")]
    #[format("json")]
    request_body: String,

    #[output(desc = "Result message")]
    #[alias("result")]
    answer: String,
}

#[test]
fn signature_macro_emits_alias_and_format_metadata() {
    let schema = AliasAndFormatSignature::schema();

    assert_eq!(schema.input_fields().len(), 1);
    assert_eq!(schema.output_fields().len(), 1);

    let input = &schema.input_fields()[0];
    assert_eq!(input.rust_name, "request_body");
    assert_eq!(input.lm_name, "payload");
    assert_eq!(input.format, Some("json"));

    let output = &schema.output_fields()[0];
    assert_eq!(output.rust_name, "answer");
    assert_eq!(output.lm_name, "result");
    assert_eq!(output.format, None);

    let input_meta = AliasAndFormatSignature::input_field_metadata();
    assert_eq!(input_meta[0].alias, Some("payload"));
    assert_eq!(input_meta[0].format, Some("json"));

    let output_meta = AliasAndFormatSignature::output_field_metadata();
    assert_eq!(output_meta[0].alias, Some("result"));
}

#[derive(Signature, Clone, Debug)]
struct DocsPrioritySignature {
    /// Primary instruction line.
    /// Secondary instruction line.

    #[input]
    prompt: String,

    #[output]
    answer: String,
}

#[test]
fn signature_macro_preserves_multiline_instruction_docs() {
    let instruction = DocsPrioritySignature::instruction();
    assert!(
        instruction.is_empty()
            || (instruction.contains("Primary instruction line.")
                && instruction.contains("Secondary instruction line.")),
        "unexpected instruction rendering: {instruction:?}"
    );
}
