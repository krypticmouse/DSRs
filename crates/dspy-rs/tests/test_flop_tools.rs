use dspy_rs::optimizer::flop_tools::{FieldDef, FuseArgs, ModifyArgs, SplitArgs, SplitPoint};
use dspy_rs::optimizer::flop_tools::{FuseSignatureTool, ModifySignatureTool, SplitSignatureTool};
use dspy_rs::{MetaSignature, Signature};
use rig::tool::Tool;
use serde_json::json;

#[Signature]
struct TestSignature {
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

#[tokio::test]
async fn test_split_signature_tool() {
    let tool = SplitSignatureTool;

    // Create JSON representation of a signature
    let sig = TestSignature::new();
    let original_json = json!({
        "instruction": sig.instruction(),
        "inputs": sig.input_fields(),
        "outputs": sig.output_fields(),
        "demos": []
    });

    let args = SplitArgs {
        original_signature: serde_json::to_string(&original_json).unwrap(),
        split_points: vec![
            SplitPoint {
                name: "QuestionProcessor".to_string(),
                inputs: vec!["question".to_string()],
                outputs: vec!["intermediate".to_string()],
            },
            SplitPoint {
                name: "AnswerGenerator".to_string(),
                inputs: vec!["intermediate".to_string()],
                outputs: vec!["answer".to_string()],
            },
        ],
    };

    let result = tool.call(args).await.unwrap();

    assert_eq!(result.len(), 2);

    // Parse and verify first signature
    let sig1_json: serde_json::Value = serde_json::from_str(&result[0]).unwrap();
    assert!(
        sig1_json["instruction"]
            .as_str()
            .unwrap()
            .contains("QuestionProcessor")
    );
    assert!(
        sig1_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("question")
    );
    assert!(
        sig1_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("intermediate")
    );

    // Parse and verify second signature
    let sig2_json: serde_json::Value = serde_json::from_str(&result[1]).unwrap();
    assert!(
        sig2_json["instruction"]
            .as_str()
            .unwrap()
            .contains("AnswerGenerator")
    );
    assert!(
        sig2_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("intermediate")
    );
    assert!(
        sig2_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("answer")
    );
}

#[tokio::test]
async fn test_fuse_signature_tool() {
    let tool = FuseSignatureTool;

    // Create JSON representations of two signatures
    let sig1 = TestSignature::new();
    let sig1_json = json!({
        "instruction": sig1.instruction(),
        "inputs": sig1.input_fields(),
        "outputs": sig1.output_fields(),
        "demos": []
    });

    let sig2 = ComplexSignature::new();
    let sig2_json = json!({
        "instruction": sig2.instruction(),
        "inputs": sig2.input_fields(),
        "outputs": sig2.output_fields(),
        "demos": []
    });

    let args = FuseArgs {
        signatures: vec![
            serde_json::to_string(&sig1_json).unwrap(),
            serde_json::to_string(&sig2_json).unwrap(),
        ],
        merged_name: "CombinedSignature".to_string(),
    };

    let result = tool.call(args).await.unwrap();

    // Parse and verify merged signature
    let merged_json: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert!(
        merged_json["instruction"]
            .as_str()
            .unwrap()
            .contains("CombinedSignature")
    );

    let inputs = merged_json["inputs"].as_object().unwrap();
    assert!(inputs.contains_key("question"));
    assert!(inputs.contains_key("data"));
    assert!(inputs.contains_key("context"));

    let outputs = merged_json["outputs"].as_object().unwrap();
    assert!(outputs.contains_key("answer"));
    assert!(outputs.contains_key("result"));
    assert!(outputs.contains_key("confidence"));
}

#[tokio::test]
async fn test_fuse_signature_tool_empty() {
    let tool = FuseSignatureTool;

    let args = FuseArgs {
        signatures: vec![],
        merged_name: "Empty".to_string(),
    };

    let result = tool.call(args).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_modify_signature_tool_change_instruction() {
    let tool = ModifySignatureTool;

    let sig = TestSignature::new();
    let sig_json = json!({
        "instruction": sig.instruction(),
        "inputs": sig.input_fields(),
        "outputs": sig.output_fields(),
        "demos": []
    });

    let args = ModifyArgs {
        signature: serde_json::to_string(&sig_json).unwrap(),
        new_instruction: Some("New instruction text".to_string()),
        add_inputs: None,
        add_outputs: None,
        remove_fields: None,
    };

    let result = tool.call(args).await.unwrap();
    let modified_json: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert_eq!(modified_json["instruction"], "New instruction text");
    assert_eq!(modified_json["inputs"], sig.input_fields());
    assert_eq!(modified_json["outputs"], sig.output_fields());
}

#[tokio::test]
async fn test_modify_signature_tool_add_fields() {
    let tool = ModifySignatureTool;

    let sig = TestSignature::new();
    let sig_json = json!({
        "instruction": sig.instruction(),
        "inputs": sig.input_fields(),
        "outputs": sig.output_fields(),
        "demos": []
    });

    let args = ModifyArgs {
        signature: serde_json::to_string(&sig_json).unwrap(),
        new_instruction: None,
        add_inputs: Some(vec![FieldDef {
            name: "context".to_string(),
            desc: "Additional context".to_string(),
            type_name: "String".to_string(),
        }]),
        add_outputs: Some(vec![FieldDef {
            name: "confidence".to_string(),
            desc: "Confidence score".to_string(),
            type_name: "f32".to_string(),
        }]),
        remove_fields: None,
    };

    let result = tool.call(args).await.unwrap();
    let modified_json: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(
        modified_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("context")
    );
    assert!(
        modified_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("question")
    );
    assert!(
        modified_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("confidence")
    );
    assert!(
        modified_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("answer")
    );
}

#[tokio::test]
async fn test_modify_signature_tool_remove_fields() {
    let tool = ModifySignatureTool;

    let sig = ComplexSignature::new();
    let sig_json = json!({
        "instruction": sig.instruction(),
        "inputs": sig.input_fields(),
        "outputs": sig.output_fields(),
        "demos": []
    });

    let args = ModifyArgs {
        signature: serde_json::to_string(&sig_json).unwrap(),
        new_instruction: None,
        add_inputs: None,
        add_outputs: None,
        remove_fields: Some(vec!["context".to_string(), "confidence".to_string()]),
    };

    let result = tool.call(args).await.unwrap();
    let modified_json: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert!(
        !modified_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("context")
    );
    assert!(
        modified_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("data")
    );
    assert!(
        !modified_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("confidence")
    );
    assert!(
        modified_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("result")
    );
}

#[tokio::test]
async fn test_modify_signature_tool_complex() {
    let tool = ModifySignatureTool;

    let sig = TestSignature::new();
    let sig_json = json!({
        "instruction": sig.instruction(),
        "inputs": sig.input_fields(),
        "outputs": sig.output_fields(),
        "demos": []
    });

    let args = ModifyArgs {
        signature: serde_json::to_string(&sig_json).unwrap(),
        new_instruction: Some("Updated instruction".to_string()),
        add_inputs: Some(vec![FieldDef {
            name: "context".to_string(),
            desc: "Context".to_string(),
            type_name: "String".to_string(),
        }]),
        add_outputs: Some(vec![FieldDef {
            name: "score".to_string(),
            desc: "Score".to_string(),
            type_name: "f32".to_string(),
        }]),
        remove_fields: Some(vec!["question".to_string()]),
    };

    let result = tool.call(args).await.unwrap();
    let modified_json: serde_json::Value = serde_json::from_str(&result).unwrap();

    assert_eq!(modified_json["instruction"], "Updated instruction");
    assert!(
        !modified_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("question")
    );
    assert!(
        modified_json["inputs"]
            .as_object()
            .unwrap()
            .contains_key("context")
    );
    assert!(
        modified_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("answer")
    );
    assert!(
        modified_json["outputs"]
            .as_object()
            .unwrap()
            .contains_key("score")
    );
}

#[tokio::test]
async fn test_tool_definitions() {
    let split_tool = SplitSignatureTool;
    let fuse_tool = FuseSignatureTool;
    let modify_tool = ModifySignatureTool;

    let split_def = split_tool.definition("".to_string()).await;
    assert_eq!(split_def.name, "split_signature");
    assert!(split_def.description.contains("Splits"));

    let fuse_def = fuse_tool.definition("".to_string()).await;
    assert_eq!(fuse_def.name, "fuse_signatures");
    assert!(fuse_def.description.contains("Fuses"));

    let modify_def = modify_tool.definition("".to_string()).await;
    assert_eq!(modify_def.name, "modify_signature");
    assert!(modify_def.description.contains("Modifies"));
}
