use dsrs::data::prediction::{Prediction, LmUsage};
use std::collections::HashMap;


#[test]
fn test_prediction_initialization() {
    let data = HashMap::from([("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())]);
    let prediction = Prediction::new(data);
    assert_eq!(prediction.data, HashMap::from([("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())]));
    assert_eq!(prediction.lm_usage, LmUsage::default());
}

#[test]
fn test_prediction_get() {
    let data = HashMap::from([("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())]);
    let prediction = Prediction::new(data);
    assert_eq!(prediction.get("a", None), "1");
    assert_eq!(prediction.get("b", None), "2");
    assert_eq!(prediction.get("c", None), "");
    assert_eq!(prediction.get("c", Some("3")), "3");
}

#[test]
fn test_prediction_keys() {
    let data = HashMap::from([("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())]);
    let prediction = Prediction::new(data);
    assert_eq!(prediction.keys(), vec!["a", "b"]);
}

#[test]
fn test_prediction_values() {
    let data = HashMap::from([("a".to_string(), "1".to_string()), ("b".to_string(), "2".to_string())]);
    let prediction = Prediction::new(data);

    assert_eq!(prediction.values(), vec!["1", "2"]);
}

#[test]
fn test_prediction_set_lm_usage() {
    let mut prediction = Prediction::new(HashMap::new());
    let lm_usage = LmUsage { prompt_tokens: 10, completion_tokens: 20 };
    prediction.set_lm_usage(lm_usage);
    assert_eq!(prediction.lm_usage, LmUsage { prompt_tokens: 10, completion_tokens: 20 });
}
