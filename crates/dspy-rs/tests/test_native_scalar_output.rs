use dspy_rs::ir::{
    self, Budget, ConstraintDef, FieldDef, Interpreter, ProgramBuilder, RuntimeEnv, SignatureDef,
};
use dspy_rs::{
    Demo, FieldType, JsonMap, LM, LMClient, LMConfig, Predict, Signature, TestCompletionModel,
};
use rig::completion::{AssistantContent, ToolDefinition};
use rig::tool::Tool;
use serde_json::json;
use std::sync::Arc;

/// Answer geography questions accurately.
#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,
    #[output]
    #[alias("answer.value")]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct Pair {
    #[input]
    question: String,
    #[output]
    answer: String,
    #[output]
    confidence: f64,
}

#[dspy_rs::Schema]
#[derive(Clone, Debug)]
struct City {
    /// The city name.
    name: String,
}

#[derive(Signature, Clone, Debug)]
struct Profile {
    #[input]
    question: String,
    #[output]
    city: City,
}

/// Square the input integer.
#[derive(Signature, Clone, Debug)]
struct Square {
    #[input]
    x: i64,
    #[output]
    square: i64,
}

/// Is the input strictly greater than zero?
#[derive(Signature, Clone, Debug)]
struct Positive {
    #[input]
    x: i64,
    #[output]
    positive: bool,
}

/// Divide the input number by two.
#[derive(Signature, Clone, Debug)]
struct Half {
    #[input]
    x: f64,
    #[output]
    half: f64,
}

async fn model(cache: bool, replies: &[&str]) -> (LM, TestCompletionModel) {
    let client = TestCompletionModel::new(replies.iter().map(|text| AssistantContent::text(*text)));
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("test".into())
        .cache(cache)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client.clone()))
        .await
        .unwrap()
        .with_custom_output_schema(true)
        .with_scalar_output_schema(true);
    (lm, client)
}

#[tokio::test]
async fn scalar_predictions_preserve_aliases_demos_history_and_conversations() {
    let (lm, client) = model(false, &[r#""Paris""#, r#""Berlin""#]).await;
    let predictor = Predict::<QA>::builder()
        .lm(lm.clone())
        .with_demos([Demo::new(
            QAInput {
                question: "Italy?".into(),
            },
            QAOutput {
                answer: "Rome".into(),
            },
        )])
        .build();
    let chat = predictor
        .build_chat(&QAInput {
            question: "France?".into(),
        })
        .await
        .unwrap();
    assert!(
        chat.messages
            .iter()
            .any(|message| message.content() == r#""Rome""#)
    );
    let replay_chat = chat.clone();
    let (result, trace) = dspy_rs::capture(|| predictor.call_and_parse(chat)).await;
    let (answer, mut chat) = result.unwrap();
    assert_eq!(answer.answer, "Paris");
    assert_eq!(answer.metadata().raw_response, r#""Paris""#);
    let schema =
        serde_json::to_value(client.last_request().unwrap().output_schema.unwrap()).unwrap();
    assert_eq!(schema["type"], "string");
    let history = lm.inspect_history(1).await;
    assert_eq!(history[0].raw_output.as_deref(), Some(r#""Paris""#));
    assert_eq!(history[0].output.as_ref().unwrap()["answer"], "Paris");
    let restored = dspy_rs::Trace::from_jsonl(&trace.to_jsonl().unwrap()).unwrap();
    let (replayed, report) = dspy_rs::replay(&restored, dspy_rs::ReplayMode::Strict, || {
        predictor.call_and_parse(replay_chat)
    })
    .await;
    assert_eq!(replayed.unwrap().0.answer, "Paris");
    assert_eq!(report.served, 1);
    assert_eq!(report.live, 0);
    assert_eq!(client.call_count(), 1);
    chat.push_message(dspy_rs::Message::user("And Germany?"));
    let (answer, _) = predictor.call_and_parse(chat).await.unwrap();
    assert_eq!(answer.answer, "Berlin");
    assert_eq!(client.call_count(), 2);
    assert!(
        lm.copy_with(|config| config.temperature = 0.2)
            .await
            .unwrap()
            .supports_scalar_output_schema()
    );
    assert!(
        !lm.clone()
            .with_custom_output_schema(false)
            .supports_scalar_output_schema()
    );
}

#[tokio::test]
async fn scalar_coercion_keeps_constraints_and_continuation_input_format() {
    let (lm, client) = model(
        false,
        &[r#""Paris""#, r#""""#, r#"["Paris"]"#, r#""Berlin""#],
    )
    .await;
    let def = SignatureDef::build("Count")
        .input("question", FieldType::String)
        .output_full(
            FieldDef::new("count", FieldType::String)
                .aliased("n")
                .with_constraint(ConstraintDef::assert("this != \"\"")),
        )
        .finish()
        .unwrap();
    let mut builder = ProgramBuilder::new("scalar count");
    let model_id = builder.model("local", LMConfig::default());
    let sig = builder.sig(def);
    let program = builder
        .main(
            sig,
            ir::seq([ir::predict("count", sig)
                .model(model_id)
                .bind("question", ir::input("question"))])
            .out("count", ir::out("count", "count")),
        )
        .unwrap();
    let interpreter =
        Interpreter::load(program, RuntimeEnv::new().bind_model("local", Arc::new(lm)))
            .await
            .unwrap();
    let input: JsonMap = serde_json::from_value(json!({"question":"Count?"})).unwrap();
    let (run, chat) = interpreter
        .run_conversation(
            dspy_rs::Chat::new(vec![]),
            Some(input.clone()),
            None,
            Budget::unlimited(),
        )
        .await
        .unwrap();
    assert_eq!(run.output["count"], "Paris");
    assert!(
        interpreter
            .run(input.clone(), None, Budget::unlimited())
            .await
            .is_err()
    );
    assert!(
        interpreter
            .run(input.clone(), None, Budget::unlimited())
            .await
            .is_err()
    );
    let (run, _) = interpreter
        .run_conversation(chat, Some(input), None, Budget::unlimited())
        .await
        .unwrap();
    assert_eq!(run.output["count"], "Berlin");
    let request = client.last_request().unwrap();
    let text = format!("{:?}", request.chat_history);
    assert!(text.contains("Compute `n` from the inputs"));
    assert!(!text.contains("ending with the marker"));
}

#[derive(Clone)]
struct Ready;
impl Tool for Ready {
    const NAME: &'static str = "ready";
    type Args = serde_json::Value;
    type Output = String;
    type Error = std::convert::Infallible;
    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: "ready".into(),
            description: "Ready".into(),
            parameters: json!({}),
        }
    }
    async fn call(&self, _: Self::Args) -> Result<String, Self::Error> {
        Ok("ready".into())
    }
}

#[tokio::test]
async fn multiple_outputs_and_tool_agents_keep_object_schemas() {
    let (lm, client) = model(false, &[r#"{"answer":"Paris","confidence":0.9}"#]).await;
    let predictor = Predict::<Pair>::builder().lm(lm).build();
    assert_eq!(
        predictor
            .call(PairInput {
                question: "France?".into()
            })
            .await
            .unwrap()
            .answer,
        "Paris"
    );
    assert_eq!(
        serde_json::to_value(client.last_request().unwrap().output_schema.unwrap()).unwrap()["type"],
        "object"
    );
    let (lm, client) = model(false, &[r#"{"answer.value":"Paris"}"#]).await;
    let predictor = Predict::<QA>::builder().lm(lm).add_tool(Ready).build();
    assert_eq!(
        predictor
            .call(QAInput {
                question: "France?".into()
            })
            .await
            .unwrap()
            .answer,
        "Paris"
    );
    assert_eq!(
        serde_json::to_value(client.last_request().unwrap().output_schema.unwrap()).unwrap()["type"],
        "object"
    );
    let (lm, client) = model(false, &[r#"{"city":{"name":"Paris"}}"#]).await;
    let predictor = Predict::<Profile>::builder().lm(lm).build();
    assert_eq!(
        predictor
            .call(ProfileInput {
                question: "France?".into()
            })
            .await
            .unwrap()
            .city
            .name,
        "Paris"
    );
    assert_eq!(
        serde_json::to_value(client.last_request().unwrap().output_schema.unwrap()).unwrap()["type"],
        "object"
    );
}

#[tokio::test]
#[ignore = "requires local Ollama with llama3.2 pulled"]
async fn live_ollama_scalar_string_and_numeric_object_fallbacks() {
    let lm = LM::builder()
        .base_url(std::env::var("OLLAMA_BASE_URL").unwrap())
        .model("llama3.2".into())
        .temperature(0.0)
        .max_tokens(64)
        .max_retries(0)
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_custom_output_schema(true)
        .with_scalar_output_schema(true);
    let qa = Predict::<QA>::builder().lm(lm.clone()).build();
    assert_eq!(
        qa.call(QAInput {
            question: "What is the capital of France?".into()
        })
        .await
        .unwrap()
        .answer,
        "Paris"
    );
    let square = Predict::<Square>::builder().lm(lm.clone()).build();
    assert!(
        square
            .build_chat(&SquareInput { x: 10 })
            .await
            .unwrap()
            .to_json()
            .to_string()
            .contains("Square the input integer.")
    );
    assert_eq!(
        square.call(SquareInput { x: 10 }).await.unwrap().square,
        100
    );
    let positive = Predict::<Positive>::builder().lm(lm.clone()).build();
    assert!(
        !positive
            .call(PositiveInput { x: -2 })
            .await
            .unwrap()
            .positive
    );
    let half = Predict::<Half>::builder().lm(lm).build();
    assert_eq!(half.call(HalfInput { x: 1.0 }).await.unwrap().half, 0.5);
}

#[cfg(feature = "cache")]
#[tokio::test]
async fn scalar_and_object_formats_have_independent_cached_responses() {
    let (lm, client) = model(true, &[r#"{"answer.value":"Paris"}"#, r#""Paris""#]).await;
    let object = Predict::<QA>::builder()
        .lm(lm.clone().with_scalar_output_schema(false))
        .build();
    let scalar = Predict::<QA>::builder().lm(lm).build();
    for predictor in [&object, &scalar, &scalar, &object] {
        assert_eq!(
            predictor
                .call(QAInput {
                    question: "France?".into()
                })
                .await
                .unwrap()
                .answer,
            "Paris"
        );
    }
    assert_eq!(client.call_count(), 2);
}
