use std::collections::HashMap;
use std::rc::Rc;

use dspy_rs::__macro_support::bamltype::facet;
use dspy_rs::{NamedParametersError, Predict as DsPredict, Signature, named_parameters, named_parameters_ref};

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
    maybe: Option<DsPredict<QA>>,
    predictors: Vec<DsPredict<QA>>,
    by_name: HashMap<String, DsPredict<QA>>,
    boxed: Box<DsPredict<QA>>,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct OptionalModule {
    maybe: Option<DsPredict<QA>>,
    fallback: DsPredict<QA>,
}

#[test]
fn named_parameters_traverses_supported_containers_with_canonical_paths() {
    let mut module = ContainerModule {
        maybe: Some(DsPredict::<QA>::new()),
        predictors: vec![DsPredict::<QA>::new()],
        by_name: HashMap::from([
            ("z".to_string(), DsPredict::<QA>::new()),
            ("a'b\\c\n".to_string(), DsPredict::<QA>::new()),
            ("alpha".to_string(), DsPredict::<QA>::new()),
        ]),
        boxed: Box::new(DsPredict::<QA>::new()),
    };

    module.predictors.push(DsPredict::<QA>::new());

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
        fallback: DsPredict::<QA>::new(),
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
        maybe: Some(DsPredict::<QA>::new()),
        predictors: vec![DsPredict::<QA>::new(), DsPredict::<QA>::new()],
        by_name: HashMap::from([
            ("z".to_string(), DsPredict::<QA>::new()),
            ("a".to_string(), DsPredict::<QA>::new()),
        ]),
        boxed: Box::new(DsPredict::<QA>::new()),
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

#[test]
fn named_parameters_container_path_order_is_stable_across_mut_and_ref_runs() {
    let mut module = ContainerModule {
        maybe: Some(DsPredict::<QA>::new()),
        predictors: vec![DsPredict::<QA>::new(), DsPredict::<QA>::new()],
        by_name: HashMap::from([
            ("z".to_string(), DsPredict::<QA>::new()),
            ("a'b\\c\n".to_string(), DsPredict::<QA>::new()),
            ("alpha".to_string(), DsPredict::<QA>::new()),
        ]),
        boxed: Box::new(DsPredict::<QA>::new()),
    };

    let expected_mut_paths = named_parameters(&mut module)
        .expect("initial mutable traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();
    let expected_ref_paths = named_parameters_ref(&module)
        .expect("initial shared traversal should succeed")
        .into_iter()
        .map(|(path, _)| path)
        .collect::<Vec<_>>();

    for _ in 0..32 {
        let mut_paths = named_parameters(&mut module)
            .expect("mutable traversal should remain stable")
            .into_iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>();
        let ref_paths = named_parameters_ref(&module)
            .expect("shared traversal should remain stable")
            .into_iter()
            .map(|(path, _)| path)
            .collect::<Vec<_>>();
        assert_eq!(mut_paths, expected_mut_paths);
        assert_eq!(ref_paths, expected_ref_paths);
        assert_eq!(ref_paths, mut_paths);
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct RcContainerModule {
    predictor: Rc<DsPredict<QA>>,
}

#[test]
fn named_parameters_container_error_for_rc_predict() {
    let mut module = RcContainerModule {
        predictor: Rc::new(DsPredict::<QA>::new()),
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

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct RcFakePredictModule {
    predictor: Rc<Predict>,
}

#[test]
fn named_parameters_container_error_for_rc_predict_like_leaf_without_accessor() {
    let mut module = RcFakePredictModule {
        predictor: Rc::new(Predict { marker: 11 }),
    };

    let err = match named_parameters(&mut module) {
        Ok(_) => panic!("Rc should error when pointee is a parameter-like leaf"),
        Err(err) => err,
    };

    assert_eq!(
        err,
        NamedParametersError::Container {
            path: "predictor".to_string(),
            ty: "Rc",
        }
    );
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct Predict {
    marker: i32,
}

#[derive(facet::Facet)]
#[facet(crate = facet)]
struct FakePredictModule {
    predictor: Predict,
}

#[test]
fn named_parameters_missing_accessor_reports_predict_like_leaf_path() {
    let mut module = FakePredictModule {
        predictor: Predict { marker: 7 },
    };

    let err = match named_parameters(&mut module) {
        Ok(_) => panic!("predict-like shapes should fail without accessor registration"),
        Err(err) => err,
    };
    let message = err.to_string();
    match err {
        NamedParametersError::MissingAttr { path } => {
            assert_eq!(path, "predictor");
            assert!(
                message.contains("S2 fallback"),
                "diagnostic should mention fallback status"
            );
        }
        other => panic!("expected MissingAttr, got {other:?}"),
    }
}

#[test]
fn named_parameters_ref_missing_accessor_reports_predict_like_leaf_path() {
    let module = FakePredictModule {
        predictor: Predict { marker: 7 },
    };

    let err = match named_parameters_ref(&module) {
        Ok(_) => panic!("predict-like shapes should fail without accessor registration"),
        Err(err) => err,
    };
    let message = err.to_string();
    match err {
        NamedParametersError::MissingAttr { path } => {
            assert_eq!(path, "predictor");
            assert!(
                message.contains("S2 fallback"),
                "diagnostic should mention fallback status"
            );
        }
        other => panic!("expected MissingAttr, got {other:?}"),
    }
}
