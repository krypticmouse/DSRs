use dspy_rs::ir::{
    self, Budget, FieldDef, Interpreter, ProgramBuilder, RenderSpec, RuntimeEnv, SignatureDef,
};
use dspy_rs::{
    ChatAdapter, FieldType, JsonMap, LM, LMClient, LMConfig, Message, TestCompletionModel,
};
use rig::completion::AssistantContent;
use serde_json::json;
use std::sync::Arc;

async fn model(
    native: bool,
    replies: impl IntoIterator<Item = AssistantContent>,
) -> (Arc<LM>, TestCompletionModel) {
    let client = TestCompletionModel::new(replies);
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("test".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap()
        .with_custom_output_schema(native);
    (Arc::new(lm), client)
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn one_signature_keeps_native_and_delimited_prompts_separate_across_runs() {
    let (plain, plain_client) = model(
        false,
        (0..2)
            .map(|_| AssistantContent::text("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]")),
    )
    .await;
    let (native, native_client) = model(
        true,
        (0..2).map(|_| AssistantContent::text(r#"{"answer":"Paris"}"#)),
    )
    .await;
    let def = SignatureDef::build("QA")
        .instruction("Answer the question.")
        .input("question", FieldType::String)
        .output("answer", FieldType::String)
        .finish()
        .unwrap();
    let mut builder = ProgramBuilder::new("mixed formats");
    let plain_ref = builder.model("plain", LMConfig::default());
    let native_ref = builder.model("native", LMConfig::default());
    let sig = builder.sig(def.clone());
    let first = ir::predict("first", sig)
        .model(plain_ref)
        .bind("question", ir::input("question"));
    let second = ir::predict("second", sig)
        .model(native_ref)
        .bind("question", ir::out("first", "answer"));
    let program = builder
        .main(
            sig,
            ir::seq([first, second]).out("answer", ir::out("second", "answer")),
        )
        .unwrap();
    let plain_oracle = ChatAdapter.build_system_def(&def, &program.types, Some(&def.instruction));
    let native_oracle =
        ChatAdapter.build_system_json_def(&def, &program.types, Some(&def.instruction), true);
    assert_ne!(plain_oracle, native_oracle);
    let interpreter = Interpreter::load(
        program,
        RuntimeEnv::new()
            .bind_model("plain", plain)
            .bind_model("native", native),
    )
    .await
    .unwrap();
    for question in ["first question", "fresh question"] {
        let input: JsonMap = serde_json::from_value(json!({"question":question})).unwrap();
        assert_eq!(
            interpreter
                .run(input, None, Budget::unlimited())
                .await
                .unwrap()["answer"],
            "Paris"
        );
        let plain = plain_client.last_request().unwrap();
        let native = native_client.last_request().unwrap();
        assert_eq!(plain.preamble.as_deref(), Some(plain_oracle.as_str()));
        assert_eq!(native.preamble.as_deref(), Some(native_oracle.as_str()));
        assert!(plain.output_schema.is_none());
        assert!(native.output_schema.is_some());
    }
    assert_eq!(plain_client.call_count(), 2);
    assert_eq!(native_client.call_count(), 2);
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn native_input_render_errors_are_fallible_and_do_not_reach_the_provider() {
    let mut field = FieldDef::new("question", FieldType::String).aliased("query");
    field.render = RenderSpec::Jinja("{{ missing.name }}".into());
    let def = SignatureDef::build("QA")
        .input_full(field)
        .output("answer", FieldType::String)
        .finish()
        .unwrap();
    let input: JsonMap = serde_json::from_value(json!({"question":"q"})).unwrap();
    assert!(ChatAdapter.try_format_input_json_def(&def, &input).is_err());
    assert_eq!(
        ChatAdapter.format_input_json_def(&def, &input),
        "[[ ## input_render_error ## ]]"
    );
    let (native, client) = model(true, [AssistantContent::text(r#"{"answer":"unused"}"#)]).await;
    let mut builder = ProgramBuilder::new("bad render");
    let model = builder.model("native", LMConfig::default());
    let sig = builder.sig(def);
    let leaf = ir::predict("qa", sig)
        .model(model)
        .bind("question", ir::input("question"));
    let program = builder
        .main(sig, ir::seq([leaf]).out("answer", ir::out("qa", "answer")))
        .unwrap();
    let interpreter = Interpreter::load(program, RuntimeEnv::new().bind_model("native", native))
        .await
        .unwrap();
    assert!(matches!(
        interpreter.run(input, None, Budget::unlimited()).await,
        Err(ir::RunError::Input { .. })
    ));
    assert_eq!(client.call_count(), 0);
}

#[test]
fn native_json_enforces_nested_assertions_and_retains_nested_soft_checks() {
    use dspy_rs::typesys::{
        ClassDef, Constraint, ConstraintKind, FieldDef as TypeField, TypeTable,
    };
    let mut types = TypeTable::default();
    types.classes.insert(
        "Score".into(),
        ClassDef {
            internal_name: "Score".into(),
            rendered_name: "Score".into(),
            docs: None,
            fields: vec![TypeField {
                name: "value".into(),
                rendered_name: "value".into(),
                field_type: FieldType::Int,
                docs: None,
                constraints: vec![
                    Constraint {
                        level: ConstraintKind::Assert,
                        label: Some("positive".into()),
                        expression: "this > 0".into(),
                    },
                    Constraint {
                        level: ConstraintKind::Check,
                        label: Some("small".into()),
                        expression: "this < 10".into(),
                    },
                ],
            }],
            constraints: vec![],
        },
    );
    let def = SignatureDef::build("Scored")
        .input("question", FieldType::String)
        .output("score", FieldType::Class("Score".into()))
        .finish()
        .unwrap();
    assert!(
        ChatAdapter
            .parse_output_json_def(
                &def,
                &types,
                &Message::assistant(r#"{"score":{"value":-1}}"#)
            )
            .is_err()
    );
    let (output, metadata) = ChatAdapter
        .parse_output_json_def(
            &def,
            &types,
            &Message::assistant(r#"{"score":{"value":20}}"#),
        )
        .unwrap();
    assert_eq!(output["score"]["value"], 20);
    assert!(
        metadata["score"]
            .checks
            .iter()
            .any(|check| check.label.contains("small") && !check.passed)
    );
}
