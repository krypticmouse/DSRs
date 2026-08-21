use dspy_rs::{ChainOfThought, Facet, Signature};

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

#[test]
fn chain_of_thought_is_a_predict_leaf() {
    // `ChainOfThought<S>` is an alias for `Predict<Augmented<S, Reasoning>>`,
    // so the module itself is the optimizable leaf — no wrapper field.
    let module = ChainOfThought::<QA>::new();
    let shape = shape_of(&module);

    assert_eq!(shape.type_identifier, "Predict");
}
