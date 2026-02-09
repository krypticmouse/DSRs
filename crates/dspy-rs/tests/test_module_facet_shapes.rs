use dspy_rs::__macro_support::bamltype::facet::{self, Type, UserType};
use dspy_rs::{ChainOfThought, Facet, ModuleExt, ReAct, Signature};

#[derive(Signature, Clone, Debug, facet::Facet)]
#[facet(crate = facet)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

fn shape_of<T: for<'a> Facet<'a>>(_: &T) -> &'static facet::Shape {
    <T as Facet<'static>>::SHAPE
}

fn struct_fields(shape: &'static facet::Shape) -> &'static [facet::Field] {
    match shape.ty {
        Type::User(UserType::Struct(struct_ty)) => struct_ty.fields,
        _ => panic!(
            "expected struct shape for {}, got {:?}",
            shape.type_identifier, shape.ty
        ),
    }
}

fn find_field(shape: &'static facet::Shape, name: &str) -> &'static facet::Field {
    struct_fields(shape)
        .iter()
        .find(|field| field.name == name)
        .unwrap_or_else(|| {
            let available = struct_fields(shape)
                .iter()
                .map(|field| field.name)
                .collect::<Vec<_>>();
            panic!(
                "field `{name}` not found on shape `{}` (available: {:?})",
                shape.type_identifier, available
            )
        })
}

fn drop_reasoning(output: dspy_rs::WithReasoning<QAOutput>) -> QAOutput {
    output.inner
}

#[test]
fn chain_of_thought_shape_exposes_predictor_field() {
    let module = ChainOfThought::<QA>::new();
    let shape = shape_of(&module);
    let predictor = find_field(shape, "predictor");

    assert!(!predictor.should_skip_deserializing());
    assert_eq!(predictor.shape().type_identifier, "Predict");
}

#[test]
fn react_shape_exposes_action_and_extract_and_skips_non_parameters() {
    let module = ReAct::<QA>::new();
    let shape = shape_of(&module);

    let action = find_field(shape, "action");
    let extract = find_field(shape, "extract");
    assert!(!action.should_skip_deserializing());
    assert!(!extract.should_skip_deserializing());
    assert_eq!(action.shape().type_identifier, "Predict");
    assert_eq!(extract.shape().type_identifier, "Predict");

    let tools = find_field(shape, "tools");
    let max_steps = find_field(shape, "max_steps");
    assert!(tools.should_skip_deserializing());
    assert!(max_steps.should_skip_deserializing());
}

#[test]
fn map_shape_exposes_inner_chain_of_thought_shape() {
    let mapped = ChainOfThought::<QA>::new()
        .map(drop_reasoning as fn(dspy_rs::WithReasoning<QAOutput>) -> QAOutput);
    let map_shape = shape_of(&mapped);
    let inner = find_field(map_shape, "inner");

    assert!(!inner.should_skip_deserializing());
    assert_eq!(inner.shape().type_identifier, "ChainOfThought");

    let nested_predictor = find_field(inner.shape(), "predictor");
    assert_eq!(nested_predictor.shape().type_identifier, "Predict");
}
