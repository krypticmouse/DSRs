use dspy_rs::{BamlType, MetaSignature, Predict, Signature, SignatureSchema};

#[derive(Clone, Debug)]
#[BamlType]
struct DetailInput {
    note: String,
}

#[derive(Clone, Debug)]
#[BamlType]
struct DetailOutput {
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Nested schema test signature.
struct NestedSig {
    #[input]
    question: String,

    #[input]
    #[flatten]
    detail: DetailInput,

    #[output]
    #[flatten]
    result: DetailOutput,

    #[output]
    #[alias("score")]
    confidence: f32,
}

#[derive(Signature, Clone, Debug)]
/// Signature intentionally colliding output aliases.
struct CollisionSig {
    #[input]
    question: String,

    #[output]
    answer: String,

    #[output]
    #[flatten]
    result: DetailOutput,
}

#[test]
fn schema_contains_flattened_paths_and_aliases() {
    let schema = SignatureSchema::of::<NestedSig>();

    let input_paths: Vec<Vec<&str>> = schema
        .input_fields()
        .iter()
        .map(|field| field.path().iter().collect())
        .collect();
    assert_eq!(input_paths, vec![vec!["question"], vec!["detail", "note"]]);

    let output_paths: Vec<Vec<&str>> = schema
        .output_fields()
        .iter()
        .map(|field| field.path().iter().collect())
        .collect();
    assert_eq!(
        output_paths,
        vec![vec!["result", "answer"], vec!["confidence"]]
    );

    let output_names: Vec<&str> = schema
        .output_fields()
        .iter()
        .map(|field| field.lm_name)
        .collect();
    assert_eq!(output_names, vec!["answer", "score"]);

    let expected = <<NestedSig as Signature>::Output as BamlType>::baml_output_format();
    assert_eq!(
        schema.output_format().target.diagnostic_repr().to_string(),
        expected.target.diagnostic_repr().to_string()
    );
}

#[test]
fn schema_panics_on_flattened_lm_name_collision() {
    let result = std::panic::catch_unwind(|| {
        let _ = SignatureSchema::of::<CollisionSig>();
    });
    assert!(result.is_err(), "expected schema collision panic");
}

#[test]
fn legacy_meta_signature_uses_lm_names_for_flattened_fields() {
    let predict = Predict::<NestedSig>::new();
    let output_fields = predict.output_fields();
    let obj = output_fields
        .as_object()
        .expect("output_fields should be an object");

    assert!(obj.contains_key("answer"));
    assert!(obj.contains_key("score"));
    assert!(!obj.contains_key("result.answer"));
}
