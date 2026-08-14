//! IR-1 (RFC 0002 §1): value-level signatures.
//!
//! Golden rule under test: for representative signatures, the owned
//! `SignatureDef` built by hand equals the one bridged from the derive via
//! `SignatureDef::of`, and the value lane round-trips through serde.

use dspy_rs::ir::{ConstraintDef, FieldDef, FieldType, RenderSpec, SigError, SignatureDef};
use dspy_rs::modules::Reasoning;
use dspy_rs::{Augmented, BamlType, Schema, Signature};

#[derive(Signature, Clone, Debug)]
/// Answer questions accurately and concisely.
struct PlainQA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Grade an answer against a question.
struct Graded {
    /// The question being graded.
    #[input]
    question: String,

    #[input]
    #[format("json")]
    evidence: Vec<String>,

    #[output]
    #[alias("score")]
    #[check("this >= 0.0 && this <= 1.0", label = "range")]
    confidence: f64,

    #[output]
    #[assert("this|length > 0")]
    verdict: String,
}

#[derive(Clone, Debug)]
#[BamlType]
struct Citation {
    /// Source URL.
    url: String,
    title: String,
}

#[derive(Clone, Debug)]
#[BamlType]
enum Stance {
    Support,
    Refute,
    Neutral,
}

#[derive(Signature, Clone, Debug)]
/// Extract citations and a stance.
struct Structured {
    #[input]
    document: String,
    #[output]
    citations: Vec<Citation>,
    #[output]
    stance: Stance,
}

#[derive(Clone, Debug)]
#[BamlType]
struct DetailOutput {
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Flattened signature.
struct Flattened {
    #[input]
    question: String,
    #[output]
    #[flatten]
    result: DetailOutput,
    #[output]
    confidence: f32,
}

#[test]
fn derive_and_builder_defs_are_equal_plain() {
    let derived = SignatureDef::of::<PlainQA>();
    let built = SignatureDef::build("PlainQA")
        .instruction("Answer questions accurately and concisely.")
        .input("question", FieldType::String)
        .output("answer", FieldType::String)
        .finish()
        .unwrap();
    assert_eq!(*derived, built);
}

#[test]
fn derive_and_builder_defs_are_equal_with_alias_constraints_format_docs() {
    let derived = SignatureDef::of::<Graded>();
    let built = SignatureDef::build("Graded")
        .instruction("Grade an answer against a question.")
        .input_full(
            FieldDef::new("question", FieldType::String).with_docs("The question being graded."),
        )
        .input_full(
            FieldDef::new("evidence", FieldType::List(Box::new(FieldType::String)))
                .with_render(RenderSpec::Format("json".into())),
        )
        .output_full(
            FieldDef::new("confidence", FieldType::Float)
                .aliased("score")
                // The derive normalizes `&&` to jinja's `and`.
                .with_constraint(ConstraintDef::check("range", "this >= 0.0 and this <= 1.0")),
        )
        .output_full(
            FieldDef::new("verdict", FieldType::String)
                .with_constraint(ConstraintDef::assert("this|length > 0")),
        )
        .finish()
        .unwrap();
    assert_eq!(*derived, built);
}

#[test]
fn derive_bridge_carries_class_and_enum_tables() {
    let derived = SignatureDef::of::<Structured>();
    let types = SignatureDef::types_of::<Structured>();

    let citation_name = <Citation as Schema>::internal_name();
    let stance_name = <Stance as Schema>::internal_name();

    assert_eq!(
        derived.outputs[0].ty,
        FieldType::List(Box::new(FieldType::Class(citation_name.clone())))
    );
    assert_eq!(derived.outputs[1].ty, FieldType::Enum(stance_name.clone()));

    let citation = types.classes.get(&citation_name).expect("Citation class");
    assert_eq!(citation.fields.len(), 2);
    assert_eq!(citation.fields[0].name, "url");
    assert_eq!(citation.fields[0].docs.as_deref(), Some("Source URL."));

    let stance = types.enums.get(&stance_name).expect("Stance enum");
    let values: Vec<&str> = stance.values.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(values, ["Support", "Refute", "Neutral"]);
}

#[test]
fn flattened_fields_use_leaf_names() {
    let derived = SignatureDef::of::<Flattened>();
    let output_names: Vec<&str> = derived.outputs.iter().map(|f| &*f.name).collect();
    assert_eq!(output_names, ["answer", "confidence"]);
}

#[test]
fn generic_augmented_signature_bridges() {
    let derived = SignatureDef::of::<Augmented<PlainQA, Reasoning>>();
    assert_eq!(derived.name.as_ref(), "Augmented<PlainQA, Reasoning>");
    let output_names: Vec<&str> = derived.outputs.iter().map(|f| &*f.name).collect();
    assert_eq!(output_names, ["reasoning", "answer"]);
    // Augmentation keeps the base signature's instruction.
    assert_eq!(
        derived.instruction,
        SignatureDef::of::<PlainQA>().instruction
    );
}

#[test]
fn augmented_with_mirrors_the_type_level_augmentation() {
    let augmented = SignatureDef::of::<PlainQA>()
        .augmented_with(&[FieldDef::new("reasoning", FieldType::String)]);
    augmented
        .matches::<Augmented<PlainQA, Reasoning>>()
        .expect("value-lane augmentation should match Augmented<PlainQA, Reasoning>");
}

#[test]
fn matches_accepts_structural_equals_and_ignores_instruction() {
    let def = SignatureDef::build("anything")
        .instruction("totally different instruction")
        .input("question", FieldType::String)
        .output("answer", FieldType::String)
        .finish()
        .unwrap();
    def.matches::<PlainQA>().unwrap();
}

#[test]
fn matches_rejects_shape_differences() {
    let wrong_type = SignatureDef::build("x")
        .input("question", FieldType::String)
        .output("answer", FieldType::Int)
        .finish()
        .unwrap();
    assert!(wrong_type.matches::<PlainQA>().is_err());

    let wrong_name = SignatureDef::build("x")
        .input("question", FieldType::String)
        .output("response", FieldType::String)
        .finish()
        .unwrap();
    assert!(wrong_name.matches::<PlainQA>().is_err());

    let wrong_arity = SignatureDef::build("x")
        .input("question", FieldType::String)
        .output("answer", FieldType::String)
        .output("extra", FieldType::String)
        .finish()
        .unwrap();
    assert!(wrong_arity.matches::<PlainQA>().is_err());
}

#[test]
fn serde_round_trips() {
    let def = SignatureDef::of::<Graded>().clone();
    let json = serde_json::to_string(&def).unwrap();
    let back: SignatureDef = serde_json::from_str(&json).unwrap();
    assert_eq!(def, back);

    // Spot-check the wire shape: render specs are snake_case, plain fields omit
    // empty constraint lists.
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["inputs"][1]["render"]["format"], "json");
    assert!(value["inputs"][0].get("constraints").is_none());
}

#[test]
fn type_table_serde_round_trips() {
    let types = SignatureDef::types_of::<Structured>().clone();
    let json = serde_json::to_string(&types).unwrap();
    let back: dspy_rs::ir::TypeTable = serde_json::from_str(&json).unwrap();
    assert_eq!(types, back);
}

#[test]
fn builder_rejects_what_the_derive_rejects() {
    // Empty sides.
    assert!(matches!(
        SignatureDef::build("s")
            .output("answer", FieldType::String)
            .finish(),
        Err(SigError::EmptyInputs { .. })
    ));
    assert!(matches!(
        SignatureDef::build("s")
            .input("question", FieldType::String)
            .finish(),
        Err(SigError::EmptyOutputs { .. })
    ));

    // Duplicate lm_name after aliasing.
    assert!(matches!(
        SignatureDef::build("s")
            .input("question", FieldType::String)
            .output_full(FieldDef::new("a", FieldType::String).aliased("answer"))
            .output_full(FieldDef::new("b", FieldType::String).aliased("answer"))
            .finish(),
        Err(SigError::DuplicateLmName { .. })
    ));

    // Duplicate canonical name.
    assert!(matches!(
        SignatureDef::build("s")
            .input("question", FieldType::String)
            .input("question", FieldType::Int)
            .output("answer", FieldType::String)
            .finish(),
        Err(SigError::DuplicateName { .. })
    ));

    // Bad #[format] value.
    assert!(matches!(
        SignatureDef::build("s")
            .input_full(
                FieldDef::new("q", FieldType::String).with_render(RenderSpec::Format("xml".into()))
            )
            .output("answer", FieldType::String)
            .finish(),
        Err(SigError::InvalidFormat { .. })
    ));

    // Invalid Jinja template.
    assert!(matches!(
        SignatureDef::build("s")
            .input_full(
                FieldDef::new("q", FieldType::String)
                    .with_render(RenderSpec::Jinja("{% if".into()))
            )
            .output("answer", FieldType::String)
            .finish(),
        Err(SigError::InvalidJinja { .. })
    ));

    // Check without a label.
    assert!(matches!(
        SignatureDef::build("s")
            .input("question", FieldType::String)
            .output_full(
                FieldDef::new("answer", FieldType::String)
                    .with_constraint(ConstraintDef::check("", "this|length > 0"))
            )
            .finish(),
        Err(SigError::CheckMissingLabel { .. })
    ));

    // Malformed constraint expression.
    assert!(matches!(
        SignatureDef::build("s")
            .input("question", FieldType::String)
            .output_full(
                FieldDef::new("answer", FieldType::String)
                    .with_constraint(ConstraintDef::assert("this >>>"))
            )
            .finish(),
        Err(SigError::InvalidConstraintExpr { .. })
    ));

    // Non-string map keys, however deeply nested.
    assert!(matches!(
        SignatureDef::build("s")
            .input("question", FieldType::String)
            .output(
                "scores",
                FieldType::List(Box::new(FieldType::Map(
                    Box::new(FieldType::Int),
                    Box::new(FieldType::Float),
                )))
            )
            .finish(),
        Err(SigError::NonStringMapKey { .. })
    ));
}

#[test]
fn static_cache_is_pointer_stable() {
    let a = SignatureDef::of::<PlainQA>();
    let b = SignatureDef::of::<PlainQA>();
    assert!(std::ptr::eq(a, b));

    // The legacy schema façade resolves through the same single cache.
    let s1 = PlainQA::schema();
    let s2 = PlainQA::schema();
    assert!(std::ptr::eq(s1, s2));
    assert_eq!(s1.instruction(), derived_instruction());
}

fn derived_instruction() -> &'static str {
    "Answer questions accurately and concisely."
}
