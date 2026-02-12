use dspy_rs::{BamlType, Facet, Signature as SignatureTrait, SignatureSchema};

/// Test instruction
#[derive(dsrs_macros::Signature, Clone, Debug)]
struct TestSig {
    #[input]
    #[alias("question_text")]
    question: String,

    #[output]
    #[check("this != ''", label = "non_empty")]
    answer: String,
}

/// Test logical operators are normalized to Jinja syntax.
#[derive(dsrs_macros::Signature, Clone, Debug)]
struct NormalizedConstraintSig {
    #[input]
    question: String,

    #[output]
    #[check("this >= 0.0 && this <= 1.0", label = "valid_range")]
    score: f64,
}

#[derive(dsrs_macros::Signature, Clone, Debug)]
struct LiteralConstraintSig {
    #[input]
    question: String,

    #[output]
    #[check(
        "this == \"value||value\" && this != \"foo&&bar\"",
        label = "literal_ops"
    )]
    answer: String,
}

#[derive(Clone, Debug)]
#[BamlType]
struct GenericCtx {
    question: String,
}

#[derive(dsrs_macros::Signature, Clone, Debug)]
struct GenericFlattenSig<T: BamlType + for<'a> Facet<'a> + Clone + Send + Sync> {
    #[input]
    #[flatten]
    context: T,

    #[output]
    answer: String,
}

#[test]
fn generates_typed_input_and_output_helpers() {
    let input = TestSigInput::new("test".to_string());
    assert_eq!(input.question, "test");

    let _output = TestSigOutput::new("ok".to_string());
}

#[test]
fn generates_signature_impl_and_metadata() {
    assert_eq!(
        <TestSig as SignatureTrait>::instruction(),
        "Test instruction"
    );

    let input_metadata = <TestSig as SignatureTrait>::input_field_metadata();
    assert_eq!(input_metadata.len(), 1);
    assert_eq!(input_metadata[0].rust_name, "question");
    assert_eq!(input_metadata[0].alias, Some("question_text"));

    let output_metadata = <TestSig as SignatureTrait>::output_field_metadata();
    assert_eq!(output_metadata.len(), 1);
    assert_eq!(output_metadata[0].rust_name, "answer");
    assert_eq!(output_metadata[0].constraints.len(), 1);
    assert_eq!(output_metadata[0].constraints[0].label, "non_empty");
}

#[test]
fn constraint_operator_normalization_is_preserved() {
    let output_metadata = <NormalizedConstraintSig as SignatureTrait>::output_field_metadata();
    assert_eq!(output_metadata.len(), 1);
    assert_eq!(output_metadata[0].constraints.len(), 1);
    assert_eq!(
        output_metadata[0].constraints[0].expression,
        "this >= 0.0 and this <= 1.0"
    );
}

#[test]
fn literal_constraint_operators_are_preserved() {
    let output_metadata = <LiteralConstraintSig as SignatureTrait>::output_field_metadata();
    assert_eq!(output_metadata.len(), 1);
    let expr = &output_metadata[0].constraints[0].expression;
    assert_eq!(
        expr,
        &"this == \"value||value\" and this != \"foo&&bar\"".to_string()
    );
}

#[test]
fn derives_generic_helpers_and_flatten_paths() {
    let _typed_input = GenericFlattenSigInput::<GenericCtx> {
        context: GenericCtx {
            question: "Where?".to_string(),
        },
    };
    let _typed_output = GenericFlattenSigOutput::<GenericCtx>::new("Here".to_string());

    let schema = SignatureSchema::of::<GenericFlattenSig<GenericCtx>>();
    let input_paths: Vec<Vec<&str>> = schema
        .input_fields()
        .iter()
        .map(|field| field.path().iter().collect())
        .collect();
    assert_eq!(input_paths, vec![vec!["context", "question"]]);

    let output_names: Vec<&str> = schema.output_fields().iter().map(|f| f.lm_name).collect();
    assert_eq!(output_names, vec!["answer"]);
}
