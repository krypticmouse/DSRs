use dsrs_predict::{DynPredictor, Example, Predict, PredictState, Signature};

#[derive(Signature, Clone, Debug)]
struct StateRoundTripSig {
    /// Answer the question.
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[test]
fn predict_state_serializes_and_restores_instruction_and_demos() {
    let demo = Example::new(
        StateRoundTripSigInput {
            question: "What is 2 + 2?".to_string(),
        },
        StateRoundTripSigOutput {
            answer: "4".to_string(),
        },
    );
    let predict = Predict::<StateRoundTripSig>::builder()
        .instruction("Answer as an integer.")
        .demo(demo)
        .build();

    let state = predict.dump_state();
    let encoded = serde_json::to_string(&state).expect("PredictState should serialize");
    let decoded: PredictState =
        serde_json::from_str(&encoded).expect("PredictState should deserialize");

    let mut restored = Predict::<StateRoundTripSig>::new();
    restored
        .load_state(decoded)
        .expect("serialized state should restore into the same predictor type");
    let restored_state = restored.dump_state();

    assert_eq!(
        restored_state.instruction_override.as_deref(),
        Some("Answer as an integer.")
    );
    assert_eq!(restored_state.demos.len(), 1);
    assert_eq!(
        serde_json::to_value(&restored_state).expect("restored state should serialize"),
        serde_json::to_value(&state).expect("original state should serialize")
    );
}
