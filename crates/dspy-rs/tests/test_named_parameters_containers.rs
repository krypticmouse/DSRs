use dspy_rs::{NamedParametersError, Predict, Signature, named_parameters};
use dspy_rs::__macro_support::bamltype::facet;

#[derive(Signature, Clone, Debug, PartialEq, facet::Facet)]
#[facet(crate = facet)]
struct QA {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct ContainerModule {
    predictors: Vec<Predict<QA>>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct PointerContainerModule {
    predictor: Box<Predict<QA>>,
}

#[test]
fn named_parameters_container_error_for_vec_predict() {
    let mut module = ContainerModule {
        predictors: vec![Predict::<QA>::new()],
    };

    let err = match named_parameters(&mut module) {
        Ok(_) => panic!("containers should error for Slice 5"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        NamedParametersError::Container {
            path: "predictors".to_string(),
            ty: "Vec",
        }
    );
}

#[test]
fn named_parameters_container_error_for_box_predict() {
    let mut module = PointerContainerModule {
        predictor: Box::new(Predict::<QA>::new()),
    };

    let err = match named_parameters(&mut module) {
        Ok(_) => panic!("containers should error for Slice 5"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        NamedParametersError::Container {
            path: "predictor".to_string(),
            ty: "Box",
        }
    );
}
