use dspy_rs::{Optimizable, Predict, sign};
use rstest::*;

#[derive(Optimizable)]
struct Leaf {
    #[parameter]
    predictor: Predict,
}

#[derive(Optimizable)]
struct Parent {
    #[parameter]
    a: Predict,
    #[parameter]
    b: Leaf,
}

#[derive(Optimizable)]
struct GrandParent {
    #[parameter]
    p: Parent,
    #[parameter]
    c: Predict,
}

fn new_predict() -> Predict {
    Predict::new(sign! { (question: String) -> answer: String })
}

#[rstest]
fn flattens_two_levels_and_updates() {
    let mut parent = Parent {
        a: new_predict(),
        b: Leaf {
            predictor: new_predict(),
        },
    };

    // Check flattened names
    let mut names: Vec<String> = parent.parameters().keys().cloned().collect();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b.predictor".to_string()]);

    // Update all signatures via returned params
    for (name, param) in parent.parameters() {
        param
            .update_signature_instruction(format!("X {name}"))
            .unwrap();
    }

    assert_eq!(parent.a.signature.instruction(), "X a");
    assert_eq!(parent.b.predictor.signature.instruction(), "X b.predictor");
}

#[rstest]
fn flattens_three_levels_and_updates() {
    let mut grand = GrandParent {
        p: Parent {
            a: new_predict(),
            b: Leaf {
                predictor: new_predict(),
            },
        },
        c: new_predict(),
    };

    // Check flattened names
    let mut names: Vec<String> = grand.parameters().keys().cloned().collect();
    names.sort();
    assert_eq!(
        names,
        vec![
            "c".to_string(),
            "p.a".to_string(),
            "p.b.predictor".to_string(),
        ]
    );

    // Update all signatures via returned params
    for (name, param) in grand.parameters() {
        param
            .update_signature_instruction(format!("Y {name}"))
            .unwrap();
    }

    assert_eq!(grand.c.signature.instruction(), "Y c");
    assert_eq!(grand.p.a.signature.instruction(), "Y p.a");
    assert_eq!(
        grand.p.b.predictor.signature.instruction(),
        "Y p.b.predictor"
    );
}
