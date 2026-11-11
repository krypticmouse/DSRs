use dspy_rs::ir::{self, Budget, FieldType, Interpreter, ProgramBuilder, RuntimeEnv, SignatureDef};
use dspy_rs::{ChatAdapter, LM, LMClient, LMConfig, Predict, Signature, TestCompletionModel};
use rig::completion::AssistantContent;
use serde_json::json;
use std::sync::Arc;

async fn model(answer: &str) -> Arc<LM> {
    Arc::new(
        LM::builder()
            .base_url("http://localhost:1/v1".into())
            .model(answer.into())
            .build()
            .await
            .unwrap()
            .with_client(LMClient::Test(TestCompletionModel::new([
                AssistantContent::text(format!(
                    "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
                )),
            ])))
            .await
            .unwrap(),
    )
}

#[derive(Signature, Clone, Debug)]
/// Answer the question.
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn cached_system_prompts_follow_each_scoped_instruction_and_keep_live_inputs_fresh() {
    let predictor = Predict::<QA>::builder()
        .named("qa")
        .lm((*model("unused").await).clone())
        .build();
    let def = SignatureDef::of::<QA>();
    let types = Default::default();
    for (index, instruction) in ["first", "second", "first", ""].iter().enumerate() {
        let mut params = dspy_rs::fx::Params::new();
        params.set_instruction("qa", *instruction);
        let input = QAInput {
            question: format!("fresh question {index}"),
        };
        let chat = dspy_rs::fx::with_params(params, predictor.build_chat(&input))
            .await
            .unwrap();
        let oracle = ChatAdapter.build_system_def(&def, &types, Some(instruction));
        assert_eq!(chat.messages[0].content(), oracle);
        assert!(
            chat.messages
                .last()
                .unwrap()
                .content()
                .contains(&input.question)
        );
    }
    let input = QAInput {
        question: "default again".into(),
    };
    let chat = predictor.build_chat(&input).await.unwrap();
    assert_eq!(
        chat.messages[0].content(),
        ChatAdapter.build_system_def(&def, &types, None)
    );
    let mut first = dspy_rs::fx::Params::new();
    first.set_instruction("qa", "concurrent one");
    let mut second = dspy_rs::fx::Params::new();
    second.set_instruction("qa", "concurrent two");
    let (one, two) = tokio::join!(
        dspy_rs::fx::with_params(first, predictor.build_chat(&input)),
        dspy_rs::fx::with_params(second, predictor.build_chat(&input)),
    );
    assert_eq!(
        one.unwrap().messages[0].content(),
        ChatAdapter.build_system_def(&def, &types, Some("concurrent one"))
    );
    assert_eq!(
        two.unwrap().messages[0].content(),
        ChatAdapter.build_system_def(&def, &types, Some("concurrent two"))
    );
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn shared_graph_keeps_independent_bindings_runs_and_owned_lifetime() {
    let mut builder = ProgramBuilder::new("shared");
    let model_id = builder.model("default", LMConfig::default());
    let sig = builder.sig(
        SignatureDef::build("QA")
            .input("question", FieldType::String)
            .output("answer", FieldType::String)
            .finish()
            .unwrap(),
    );
    let node = ir::predict("qa", sig)
        .model(model_id)
        .bind("question", ir::input("question"));
    let program = Arc::new(
        builder
            .main(sig, ir::seq([node]).out("answer", ir::out("qa", "answer")))
            .unwrap(),
    );
    let weak = Arc::downgrade(&program);
    let one = Interpreter::load_shared(
        program.clone(),
        RuntimeEnv::new().bind_model("default", model("one").await),
    )
    .await
    .unwrap();
    let two = Interpreter::load_shared(
        program.clone(),
        RuntimeEnv::new().bind_model("default", model("two").await),
    )
    .await
    .unwrap();
    assert!(Arc::ptr_eq(one.program(), &program));
    assert!(Arc::ptr_eq(two.program(), &program));
    let input: dspy_rs::JsonMap = serde_json::from_value(json!({"question":"q"})).unwrap();
    let (first, second) = tokio::join!(
        one.run(input.clone(), None, Budget::unlimited()),
        two.run(input, None, Budget::unlimited())
    );
    assert_eq!(first.unwrap()["answer"], "one");
    assert_eq!(second.unwrap()["answer"], "two");
    drop(program);
    assert!(weak.upgrade().is_some());
    drop(one);
    assert!(weak.upgrade().is_some());
    drop(two);
    assert!(
        weak.upgrade().is_none(),
        "no global cache may retain a dynamic program"
    );
}
