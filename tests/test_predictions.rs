use rstest::*;

use dsrs::data::prediction::{LmUsage, Prediction};
use std::collections::HashMap;

#[rstest]
fn test_prediction_initialization() {
    let data = HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ]);
    let prediction = Prediction::new(data);
    assert_eq!(
        prediction.data,
        HashMap::from([
            ("a".to_string(), "1".to_string()),
            ("b".to_string(), "2".to_string())
        ])
    );
    assert_eq!(prediction.lm_usage, LmUsage::default());
}

#[rstest]
fn test_prediction_get() {
    let data = HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ]);
    let prediction = Prediction::new(data);
    assert_eq!(prediction.get("a", None), "1");
    assert_eq!(prediction.get("b", None), "2");
    assert_eq!(prediction.get("c", None), "");
    assert_eq!(prediction.get("c", Some("3")), "3");
}

#[rstest]
fn test_prediction_keys() {
    let data = HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ]);
    let prediction = Prediction::new(data);

    let mut keys = prediction.keys();
    keys.sort();
    assert_eq!(keys, vec!["a", "b"]);
}

#[rstest]
fn test_prediction_values() {
    let data = HashMap::from([
        ("a".to_string(), "1".to_string()),
        ("b".to_string(), "2".to_string()),
    ]);
    let prediction = Prediction::new(data);

    let mut values = prediction.values();
    values.sort();
    assert_eq!(values, vec!["1", "2"]);
}

#[rstest]
fn test_prediction_set_lm_usage() {
    let mut prediction = Prediction::new(HashMap::new());
    let lm_usage = LmUsage {
        prompt_tokens: 10,
        completion_tokens: 20,
    };
    prediction.set_lm_usage(lm_usage);
    assert_eq!(
        prediction.lm_usage,
        LmUsage {
            prompt_tokens: 10,
            completion_tokens: 20
        }
    );
}
