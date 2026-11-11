use dspy_rs::ir::{FieldDef, SignatureDef};
use dspy_rs::{Augmented, ChainOfThought, Demo, FieldType, Predict, Reasoning, Schema, Signature};
use serde::{Serialize, de::DeserializeOwned};

#[Schema]
#[derive(Clone, Debug, PartialEq)]
struct Document {
    title: String,
    tags: Vec<String>,
}

#[derive(Signature, Clone, Debug)]
/// Answer with supporting documents.
struct QA {
    #[input]
    question: String,
    #[input]
    context: Vec<Document>,
    #[output]
    answer: String,
    #[output]
    citations: Vec<Document>,
}

fn roundtrip<T: Serialize + DeserializeOwned>(value: &T) -> T {
    serde_json::from_value(serde_json::to_value(value).unwrap()).unwrap()
}

#[test]
fn schema_and_generated_input_output_serialize_nested_values_without_manual_derives() {
    let document = Document {
        title: "source".into(),
        tags: vec!["reference".into()],
    };
    assert_eq!(roundtrip(&document), document);
    let input = QAInput {
        question: "q".into(),
        context: vec![document.clone()],
    };
    let output = QAOutput {
        answer: "a".into(),
        citations: vec![document],
    };
    let recovered: QAInput = roundtrip(&input);
    assert_eq!(recovered.question, input.question);
    assert_eq!(recovered.context, input.context);
    let recovered: QAOutput = roundtrip(&output);
    assert_eq!(recovered.answer, output.answer);
    assert_eq!(recovered.citations, output.citations);
    let demo = Demo::<QA>::new(input, output);
    let _predict = Predict::<QA>::builder().demo(demo).build();
}

#[test]
fn augmentations_add_typed_fields_without_changing_the_base_signature() {
    let base = SignatureDef::of::<QA>();
    let augmented = SignatureDef::of::<Augmented<QA, Reasoning>>();
    assert_eq!(base.outputs.len(), 2);
    assert_eq!(augmented.outputs.len(), 3);
    assert_eq!(&*augmented.outputs[0].name, "reasoning");
    assert_eq!(&*augmented.outputs[1].name, "answer");
    let _: ChainOfThought<QA> = Predict::<Augmented<QA, Reasoning>>::new();
}

#[test]
fn owned_signature_builder_replaces_json_field_macro_and_roundtrips() {
    let def = SignatureDef::build("InlineQA")
        .input_full(FieldDef::new("question", FieldType::String))
        .output("answer", FieldType::String)
        .finish()
        .unwrap();
    let recovered: SignatureDef = roundtrip(&def);
    assert_eq!(recovered, def);
    let values = std::collections::HashMap::from([("answer", "4")]);
    assert_eq!(values["answer"], "4");
}
