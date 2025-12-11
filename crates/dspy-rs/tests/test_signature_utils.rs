use dspy_rs::trace::signature_utils::{fuse_signatures, modify_signature, split_signature};
use dspy_rs::{MetaSignature, Signature};
use rstest::*;
use serde_json::json;

#[Signature]
struct SimpleSignature {
    /// Answer the question
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[Signature]
struct ComplexSignature {
    /// Process the data
    #[input]
    data: String,

    #[input]
    context: String,

    #[output]
    result: String,

    #[output]
    confidence: f32,
}

#[Signature]
struct AnotherSignature {
    /// Analyze the text
    #[input]
    text: String,

    #[output]
    sentiment: String,
}

#[rstest]
fn test_split_signature_basic() {
    let original = SimpleSignature::new();

    let split_metadata = vec![
        json!({
            "name": "QuestionProcessor",
            "inputs": ["question"],
            "outputs": ["intermediate"]
        }),
        json!({
            "name": "AnswerGenerator",
            "inputs": ["intermediate"],
            "outputs": ["answer"]
        }),
    ];

    let result = split_signature(&original, split_metadata).unwrap();

    assert_eq!(result.len(), 2);

    // Check first signature
    let sig1 = result[0].as_ref();
    assert!(sig1.instruction().contains("QuestionProcessor"));
    assert!(
        sig1.input_fields()
            .as_object()
            .unwrap()
            .contains_key("question")
    );
    assert!(
        sig1.output_fields()
            .as_object()
            .unwrap()
            .contains_key("intermediate")
    );

    // Check second signature
    let sig2 = result[1].as_ref();
    assert!(sig2.instruction().contains("AnswerGenerator"));
    assert!(
        sig2.input_fields()
            .as_object()
            .unwrap()
            .contains_key("intermediate")
    );
    assert!(
        sig2.output_fields()
            .as_object()
            .unwrap()
            .contains_key("answer")
    );
}

#[rstest]
fn test_split_signature_complex() {
    let original = ComplexSignature::new();

    let split_metadata = vec![
        json!({
            "name": "DataProcessor",
            "inputs": ["data", "context"],
            "outputs": ["processed_data"]
        }),
        json!({
            "name": "ResultGenerator",
            "inputs": ["processed_data"],
            "outputs": ["result", "confidence"]
        }),
    ];

    let result = split_signature(&original, split_metadata).unwrap();

    assert_eq!(result.len(), 2);

    // First signature should have both original inputs
    let sig1 = result[0].as_ref();
    assert!(
        sig1.input_fields()
            .as_object()
            .unwrap()
            .contains_key("data")
    );
    assert!(
        sig1.input_fields()
            .as_object()
            .unwrap()
            .contains_key("context")
    );
    assert!(
        sig1.output_fields()
            .as_object()
            .unwrap()
            .contains_key("processed_data")
    );

    // Second signature should use processed_data from first
    let sig2 = result[1].as_ref();
    assert!(
        sig2.input_fields()
            .as_object()
            .unwrap()
            .contains_key("processed_data")
    );
    assert!(
        sig2.output_fields()
            .as_object()
            .unwrap()
            .contains_key("result")
    );
    assert!(
        sig2.output_fields()
            .as_object()
            .unwrap()
            .contains_key("confidence")
    );
}

#[rstest]
fn test_split_signature_without_names() {
    let original = SimpleSignature::new();

    let split_metadata = vec![json!({
        "inputs": ["question"],
        "outputs": ["answer"]
    })];

    let result = split_signature(&original, split_metadata).unwrap();

    assert_eq!(result.len(), 1);
    let sig = result[0].as_ref();
    assert!(sig.instruction().contains("Part 1"));
}

#[rstest]
fn test_fuse_signatures_basic() {
    let sig1 = SimpleSignature::new();
    let sig2 = AnotherSignature::new();

    let fused = fuse_signatures(&[&sig1 as &dyn MetaSignature, &sig2 as &dyn MetaSignature]);

    // Check that fused signature has union of inputs
    let inputs = fused.input_fields();
    assert!(inputs.as_object().unwrap().contains_key("question"));
    assert!(inputs.as_object().unwrap().contains_key("text"));

    // Check that fused signature has union of outputs
    let outputs = fused.output_fields();
    assert!(outputs.as_object().unwrap().contains_key("answer"));
    assert!(outputs.as_object().unwrap().contains_key("sentiment"));

    // Check instruction contains both tasks
    let instruction = fused.instruction();
    assert!(instruction.contains("Task 1"));
    assert!(instruction.contains("Task 2"));
    assert!(instruction.contains("Combined Task"));
}

#[rstest]
fn test_fuse_signatures_empty() {
    let fused = fuse_signatures(&[]);

    assert_eq!(fused.instruction(), "Empty signature");
    assert!(fused.input_fields().as_object().unwrap().is_empty());
    assert!(fused.output_fields().as_object().unwrap().is_empty());
}

#[rstest]
fn test_fuse_signatures_single() {
    let sig1 = SimpleSignature::new();

    let fused = fuse_signatures(&[&sig1 as &dyn MetaSignature]);

    assert_eq!(fused.input_fields(), sig1.input_fields());
    assert_eq!(fused.output_fields(), sig1.output_fields());
    assert!(fused.instruction().contains("Task 1"));
}

#[rstest]
fn test_fuse_signatures_three() {
    let sig1 = SimpleSignature::new();
    let sig2 = AnotherSignature::new();
    let sig3 = ComplexSignature::new();

    let fused = fuse_signatures(&[
        &sig1 as &dyn MetaSignature,
        &sig2 as &dyn MetaSignature,
        &sig3 as &dyn MetaSignature,
    ]);

    let inputs = fused.input_fields();
    assert!(inputs.as_object().unwrap().contains_key("question"));
    assert!(inputs.as_object().unwrap().contains_key("text"));
    assert!(inputs.as_object().unwrap().contains_key("data"));
    assert!(inputs.as_object().unwrap().contains_key("context"));

    let outputs = fused.output_fields();
    assert!(outputs.as_object().unwrap().contains_key("answer"));
    assert!(outputs.as_object().unwrap().contains_key("sentiment"));
    assert!(outputs.as_object().unwrap().contains_key("result"));
    assert!(outputs.as_object().unwrap().contains_key("confidence"));
}

#[rstest]
fn test_modify_signature_change_instruction() {
    let original = SimpleSignature::new();
    let original_instruction = original.instruction();

    let modified = modify_signature(
        &original,
        Some("New instruction".to_string()),
        None,
        None,
        None,
    );

    assert_eq!(modified.instruction(), "New instruction");
    assert_ne!(modified.instruction(), original_instruction);
    assert_eq!(modified.input_fields(), original.input_fields());
    assert_eq!(modified.output_fields(), original.output_fields());
}

#[rstest]
fn test_modify_signature_add_inputs() {
    let original = SimpleSignature::new();
    let original_input_count = original.input_fields().as_object().unwrap().len();

    let new_field = json!({
        "type": "String",
        "desc": "Additional context",
        "__dsrs_field_type": "input"
    });

    let modified = modify_signature(
        &original,
        None,
        Some(&[("context".to_string(), new_field)]),
        None,
        None,
    );

    let inputs = modified.input_fields();
    assert_eq!(inputs.as_object().unwrap().len(), original_input_count + 1);
    assert!(inputs.as_object().unwrap().contains_key("context"));
    assert!(inputs.as_object().unwrap().contains_key("question"));
}

#[rstest]
fn test_modify_signature_add_outputs() {
    let original = SimpleSignature::new();
    let original_output_count = original.output_fields().as_object().unwrap().len();

    let new_field = json!({
        "type": "f32",
        "desc": "Confidence score",
        "__dsrs_field_type": "output"
    });

    let modified = modify_signature(
        &original,
        None,
        None,
        Some(&[("confidence".to_string(), new_field)]),
        None,
    );

    let outputs = modified.output_fields();
    assert_eq!(
        outputs.as_object().unwrap().len(),
        original_output_count + 1
    );
    assert!(outputs.as_object().unwrap().contains_key("confidence"));
    assert!(outputs.as_object().unwrap().contains_key("answer"));
}

#[rstest]
fn test_modify_signature_remove_fields() {
    let original = ComplexSignature::new();
    let original_input_count = original.input_fields().as_object().unwrap().len();
    let original_output_count = original.output_fields().as_object().unwrap().len();

    let modified = modify_signature(
        &original,
        None,
        None,
        None,
        Some(&["context".to_string(), "confidence".to_string()]),
    );

    let inputs = modified.input_fields();
    let outputs = modified.output_fields();

    assert_eq!(inputs.as_object().unwrap().len(), original_input_count - 1);
    assert_eq!(
        outputs.as_object().unwrap().len(),
        original_output_count - 1
    );
    assert!(!inputs.as_object().unwrap().contains_key("context"));
    assert!(!outputs.as_object().unwrap().contains_key("confidence"));
    assert!(inputs.as_object().unwrap().contains_key("data"));
    assert!(outputs.as_object().unwrap().contains_key("result"));
}

#[rstest]
fn test_modify_signature_multiple_operations() {
    let original = SimpleSignature::new();

    let new_input = json!({
        "type": "String",
        "desc": "Context",
        "__dsrs_field_type": "input"
    });

    let new_output = json!({
        "type": "f32",
        "desc": "Score",
        "__dsrs_field_type": "output"
    });

    let modified = modify_signature(
        &original,
        Some("Updated instruction".to_string()),
        Some(&[("context".to_string(), new_input)]),
        Some(&[("score".to_string(), new_output)]),
        Some(&["question".to_string()]),
    );

    assert_eq!(modified.instruction(), "Updated instruction");
    assert!(
        !modified
            .input_fields()
            .as_object()
            .unwrap()
            .contains_key("question")
    );
    assert!(
        modified
            .input_fields()
            .as_object()
            .unwrap()
            .contains_key("context")
    );
    assert!(
        modified
            .output_fields()
            .as_object()
            .unwrap()
            .contains_key("answer")
    );
    assert!(
        modified
            .output_fields()
            .as_object()
            .unwrap()
            .contains_key("score")
    );
}

#[rstest]
fn test_modify_signature_no_changes() {
    let original = SimpleSignature::new();

    let modified = modify_signature(&original, None, None, None, None);

    assert_eq!(modified.instruction(), original.instruction());
    assert_eq!(modified.input_fields(), original.input_fields());
    assert_eq!(modified.output_fields(), original.output_fields());
}

#[rstest]
fn test_split_then_fuse() {
    let original = ComplexSignature::new();

    // Split into two parts
    let split_metadata = vec![
        json!({
            "name": "Part1",
            "inputs": ["data"],
            "outputs": ["intermediate"]
        }),
        json!({
            "name": "Part2",
            "inputs": ["intermediate", "context"],
            "outputs": ["result", "confidence"]
        }),
    ];

    let split_sigs = split_signature(&original, split_metadata).unwrap();
    assert_eq!(split_sigs.len(), 2);

    // Fuse them back
    let sig_refs: Vec<&dyn MetaSignature> = split_sigs.iter().map(|s| s.as_ref()).collect();
    let fused = fuse_signatures(&sig_refs);

    // Fused should have all original inputs and outputs
    let inputs = fused.input_fields();
    assert!(inputs.as_object().unwrap().contains_key("data"));
    assert!(inputs.as_object().unwrap().contains_key("context"));
    assert!(inputs.as_object().unwrap().contains_key("intermediate"));

    let outputs = fused.output_fields();
    assert!(outputs.as_object().unwrap().contains_key("intermediate"));
    assert!(outputs.as_object().unwrap().contains_key("result"));
    assert!(outputs.as_object().unwrap().contains_key("confidence"));
}

#[rstest]
fn test_modify_then_split() {
    let original = SimpleSignature::new();

    // First modify to add a field
    let new_input = json!({
        "type": "String",
        "desc": "Context",
        "__dsrs_field_type": "input"
    });

    let modified = modify_signature(
        &original,
        None,
        Some(&[("context".to_string(), new_input)]),
        None,
        None,
    );

    // Then split
    let split_metadata = vec![json!({
        "name": "Part1",
        "inputs": ["question", "context"],
        "outputs": ["answer"]
    })];

    let split_sigs = split_signature(modified.as_ref(), split_metadata).unwrap();
    assert_eq!(split_sigs.len(), 1);

    let sig = split_sigs[0].as_ref();
    assert!(
        sig.input_fields()
            .as_object()
            .unwrap()
            .contains_key("question")
    );
    assert!(
        sig.input_fields()
            .as_object()
            .unwrap()
            .contains_key("context")
    );
    assert!(
        sig.output_fields()
            .as_object()
            .unwrap()
            .contains_key("answer")
    );
}
