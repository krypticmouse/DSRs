use std::collections::HashMap;
use std::rc::Rc;

use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{NamedParametersError, Predict, Signature, named_parameters, named_parameters_ref};

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
    maybe: Option<Predict<QA>>,
    predictors: Vec<Predict<QA>>,
    by_name: HashMap<String, Predict<QA>>,
    boxed: Box<Predict<QA>>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct OptionalModule {
    maybe: Option<Predict<QA>>,
    fallback: Predict<QA>,
}

#[test]
fn named_parameters_traverses_supported_containers_with_canonical_paths() {
    let mut module = ContainerModule {
        maybe: Some(Predict::<QA>::new()),
        predictors: vec![Predict::<QA>::new()],
        by_name: HashMap::from([
            ("z".to_string(), Predict::<QA>::new()),
            ("a'b\\c\n".to_string(), Predict::<QA>::new()),
            ("alpha".to_string(), Predict::<QA>::new()),
        ]),
        boxed: Box::new(Predict::<QA>::new()),
    };

    module.predictors.push(Predict::<QA>::new());

    let paths = named_parameters(&mut module)
        .expect("containers should be traversed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        vec![
            "maybe".to_string(),
            "predictors[0]".to_string(),
            "predictors[1]".to_string(),
            "by_name['a\\'b\\\\c\\u{A}']".to_string(),
            "by_name['alpha']".to_string(),
            "by_name['z']".to_string(),
            "boxed".to_string(),
        ]
    );
}

#[test]
fn named_parameters_skips_none_option() {
    let mut module = OptionalModule {
        maybe: None,
        fallback: Predict::<QA>::new(),
    };

    let paths = named_parameters(&mut module)
        .expect("none option should not fail")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    assert_eq!(paths, vec!["fallback".to_string()]);
}

#[test]
fn named_parameters_ref_matches_mutable_with_containers() {
    let mut module = ContainerModule {
        maybe: Some(Predict::<QA>::new()),
        predictors: vec![Predict::<QA>::new(), Predict::<QA>::new()],
        by_name: HashMap::from([
            ("z".to_string(), Predict::<QA>::new()),
            ("a".to_string(), Predict::<QA>::new()),
        ]),
        boxed: Box::new(Predict::<QA>::new()),
    };

    let mutable_paths = named_parameters(&mut module)
        .expect("mutable traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let ref_paths = named_parameters_ref(&module)
        .expect("shared traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();

    assert_eq!(ref_paths, mutable_paths);
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct RcContainerModule {
    predictor: Rc<Predict<QA>>,
}

#[test]
fn named_parameters_container_error_for_rc_predict() {
    let mut module = RcContainerModule {
        predictor: Rc::new(Predict::<QA>::new()),
    };

    let err = match named_parameters(&mut module) {
        Ok(_) => panic!("Rc is not supported for mutable traversal"),
        Err(err) => err,
    };
    assert_eq!(
        err,
        NamedParametersError::Container {
            path: "predictor".to_string(),
            ty: "Rc",
        }
    )
}
