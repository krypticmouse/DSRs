use dspy_rs::{Optimizable, Predict, Signature};

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Optimizable)]
struct Pipeline {
    #[parameter]
    qa: Predict<QA>,
}

fn main() {
    let mut pipeline = Pipeline {
        qa: Predict::<QA>::new(),
    };
    let params = dspy_rs::core::module::Optimizable::parameters(&mut pipeline);
    let _qa = params.get("qa").expect("qa parameter should be present");
}
