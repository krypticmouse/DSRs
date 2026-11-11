//! Exercise the signatures and composed modules from the real examples without
//! downloading datasets or running an optimizer search.
#![allow(dead_code)]
use dspy_rs::{LM, Predict, PredictError, Signature};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::{Duration, timeout};

async fn predict<S: Signature>(lm: LM, input: S::Input) -> Result<Value, PredictError>
where
    S::Input: dspy_rs::Schema,
    S::Output: dspy_rs::Schema + serde::Serialize,
{
    let def = dspy_rs::ir::SignatureDef::of::<S>();
    assert!(
        !def.instruction.is_empty(),
        "example task docs must be struct-level"
    );
    let predictor = Predict::<S>::builder().lm(lm).build();
    let chat = predictor.build_chat(&input).await?;
    assert!(chat.to_json().to_string().contains(&*def.instruction));
    let (output, _) = predictor.call_and_parse(chat).await?;
    Ok(serde_json::to_value(output.into_inner()).unwrap())
}

mod simple {
    include!("../examples/01-simple.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        assert!(!dspy_rs::ir::SignatureDef::of::<QA>().instruction.is_empty());
        assert!(
            !dspy_rs::ir::SignatureDef::of::<Rate>()
                .instruction
                .is_empty()
        );
        let module = QARater {
            answerer: Predict::<QA>::builder()
                .lm(lm.clone())
                .instruction(QA_INSTRUCTION)
                .build(),
            rater: Predict::<Rate>::builder()
                .lm(lm)
                .instruction(RATE_INSTRUCTION)
                .build(),
        };
        let output = module
            .call(QAInput {
                question: "What is the capital of France?".into(),
            })
            .await?;
        Ok(serde_json::to_value(output.into_inner()).unwrap())
    }
}
mod optimize {
    include!("../examples/02-module-iteration-and-updation.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        super::predict::<QA>(
            lm,
            QAInput {
                question: "What is the capital of France?".into(),
            },
        )
        .await
    }
}
mod numbers {
    include!("../examples/05-heterogenous-examples.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        let input =
            serde_json::from_value(json!({"number":10,"debug_note":"unused metadata"})).unwrap();
        super::predict::<NumberSignature>(lm, input).await
    }
}
mod batch {
    include!("../examples/06-other-providers-batch.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, dspy_rs::PredictError> {
        assert!(!dspy_rs::ir::SignatureDef::of::<QA>().instruction.is_empty());
        let predictor = Predict::<QA>::builder().lm(lm).build();
        let mut outputs = Vec::new();
        for result in forward_all(&predictor, prompts(), 2).await {
            outputs.push(serde_json::to_value(result?.into_inner()).unwrap());
        }
        Ok(serde_json::Value::Array(outputs))
    }
}
mod history {
    include!("../examples/07-inspect-history.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, dspy_rs::PredictError> {
        super::predict::<QA>(
            lm,
            QAInput {
                question: "What is the capital of France?".into(),
            },
        )
        .await
    }
}
mod sentiment {
    include!("../examples/09-gepa-sentiment.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        super::predict::<SentimentSignature>(
            lm,
            SentimentSignatureInput {
                text: "I love the helpful staff and excellent service.".into(),
            },
        )
        .await
    }
}
#[cfg(feature = "data")]
mod evaluate {
    include!("../examples/03-evaluate-hotpotqa.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        super::predict::<QA>(
            lm,
            QAInput {
                question: "What is the capital of France?".into(),
            },
        )
        .await
    }
}
#[cfg(feature = "data")]
mod copro {
    include!("../examples/04-optimize-hotpotqa.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        super::predict::<QA>(
            lm,
            QAInput {
                question: "What is the capital of France?".into(),
            },
        )
        .await
    }
}
#[cfg(feature = "data")]
mod mipro {
    include!("../examples/08-optimize-mipro.rs");
    pub async fn exercise(lm: LM) -> Result<serde_json::Value, PredictError> {
        super::predict::<QuestionAnswering>(
            lm,
            QuestionAnsweringInput {
                question: "What is the capital of France?".into(),
            },
        )
        .await
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Simple,
    Optimize,
    Numbers,
    Batch,
    History,
    Sentiment,
    #[cfg(feature = "data")]
    Evaluate,
    #[cfg(feature = "data")]
    Copro,
    #[cfg(feature = "data")]
    Mipro,
}

impl Case {
    fn requests(self) -> usize {
        match self {
            Self::Simple => 2,
            Self::Batch => 3,
            _ => 1,
        }
    }
    fn output(self, request: &Value, index: usize) -> Value {
        match self {
            Self::Simple if index == 1 => json!({"rating":8}),
            Self::Simple => json!({"reasoning":"France's capital is Paris.","answer":"Paris"}),
            Self::Numbers => json!({"number_squared":100,"number_cubed":1000}),
            Self::Sentiment => {
                json!({"sentiment":"positive","reasoning":"The text praises the staff and service."})
            }
            Self::Batch => {
                let prompt = request["messages"].to_string();
                let answer = if prompt.contains("Germany") {
                    "Berlin"
                } else if prompt.contains("Italy") {
                    "Rome"
                } else {
                    "Paris"
                };
                json!({"reasoning":"The question names a country.","answer":answer})
            }
            _ => json!({"answer":"Paris"}),
        }
    }
    async fn exercise(self, lm: LM) -> Result<Value, PredictError> {
        match self {
            Self::Simple => simple::exercise(lm).await,
            Self::Optimize => optimize::exercise(lm).await,
            Self::Numbers => numbers::exercise(lm).await,
            Self::Batch => batch::exercise(lm).await,
            Self::History => history::exercise(lm).await,
            Self::Sentiment => sentiment::exercise(lm).await,
            #[cfg(feature = "data")]
            Self::Evaluate => evaluate::exercise(lm).await,
            #[cfg(feature = "data")]
            Self::Copro => copro::exercise(lm).await,
            #[cfg(feature = "data")]
            Self::Mipro => mipro::exercise(lm).await,
        }
    }
    fn check(self, output: &Value) {
        match self {
            Self::Simple => {
                assert!(output["answer"].as_str().unwrap().contains("Paris"));
                assert!(!output["reasoning"].as_str().unwrap().is_empty());
                assert!((1..=10).contains(&output["rating"].as_i64().unwrap()));
            }
            Self::Numbers => {
                assert_eq!(output["number_squared"], 100);
                assert_eq!(output["number_cubed"], 1000);
            }
            Self::Sentiment => assert_eq!(
                output["sentiment"].as_str().unwrap().to_lowercase(),
                "positive"
            ),
            Self::Batch => {
                let outputs = output.as_array().unwrap();
                assert_eq!(outputs.len(), 3);
                for (out, expected) in outputs.iter().zip(["Paris", "Berlin", "Rome"]) {
                    assert!(out["answer"].as_str().unwrap().contains(expected));
                    assert!(!out["reasoning"].as_str().unwrap().is_empty());
                }
            }
            _ => assert!(output["answer"].as_str().unwrap().contains("Paris")),
        }
    }
}

async fn model(base_url: String, structured: bool) -> LM {
    LM::builder()
        .base_url(base_url)
        .model("example-model".into())
        .temperature(0.0)
        .max_retries(0)
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_custom_output_schema(structured)
}

async fn server(listener: TcpListener, case: Case, structured: bool) -> Vec<Value> {
    let mut requests = Vec::new();
    for index in 0..case.requests() {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        let (start, length) = loop {
            let n = stream.read(&mut chunk).await.unwrap();
            assert_ne!(n, 0);
            bytes.extend_from_slice(&chunk[..n]);
            if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                assert!(headers.starts_with("POST /v1/chat/completions "));
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap();
                break (end + 4, length);
            }
        };
        while bytes.len() < start + length {
            let n = stream.read(&mut chunk).await.unwrap();
            assert_ne!(n, 0);
            bytes.extend_from_slice(&chunk[..n]);
        }
        let request: Value = serde_json::from_slice(&bytes[start..start + length]).unwrap();
        let output = case.output(&request, index);
        let content = if structured {
            output.to_string()
        } else {
            let mut text = String::new();
            for (key, value) in output.as_object().unwrap() {
                let value = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                text.push_str(&format!("[[ ## {key} ## ]]\n{value}\n\n"));
            }
            text.push_str("[[ ## completed ## ]]");
            text
        };
        let body=json!({"id":"examples","object":"chat.completion","created":0,"model":"example-model",
            "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":content}}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}).to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        requests.push(request);
    }
    requests
}

#[rstest::rstest]
#[case::simple(Case::Simple)]
#[case::optimize(Case::Optimize)]
#[case::numbers(Case::Numbers)]
#[case::batch(Case::Batch)]
#[case::history(Case::History)]
#[case::sentiment(Case::Sentiment)]
#[cfg_attr(feature = "data", case::evaluate(Case::Evaluate))]
#[cfg_attr(feature = "data", case::copro(Case::Copro))]
#[cfg_attr(feature = "data", case::mipro(Case::Mipro))]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn real_example_signatures_roundtrip_in_both_protocols(#[case] case: Case) {
    for structured in [false, true] {
        timeout(Duration::from_secs(30), async {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
            let server = tokio::spawn(server(listener, case, structured));
            let output = case
                .exercise(model(base_url, structured).await)
                .await
                .unwrap();
            case.check(&output);
            let requests = server.await.unwrap();
            assert_eq!(requests.len(), case.requests());
            for request in requests {
                if structured {
                    assert_eq!(request["response_format"]["type"], "json_schema");
                    assert!(
                        !request["response_format"]["json_schema"]["schema"]["properties"]
                            .as_object()
                            .unwrap()
                            .is_empty()
                    );
                } else {
                    assert!(request.get("response_format").is_none());
                    assert!(
                        request["messages"]
                            .to_string()
                            .contains("[[ ## completed ## ]]")
                    );
                }
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{case:?} timed out"));
    }
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn malformed_example_output_retains_the_response_for_diagnostics() {
    let client = dspy_rs::TestCompletionModel::new([rig::completion::AssistantContent::text(
        "Conversational filler instead of fields",
    )]);
    let lm = model("http://localhost:1/v1".into(), false)
        .await
        .with_client(dspy_rs::LMClient::Test(client))
        .await
        .unwrap();
    let error = numbers::exercise(lm).await.unwrap_err();
    match error {
        PredictError::Parse { raw_response, .. } => {
            assert_eq!(raw_response, "Conversational filler instead of fields")
        }
        other => panic!("expected a typed parse error, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "requires Ollama with llama3.2; runs the real example signatures and batch/module workflows"]
async fn live_ollama_example_matrix() {
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let model_name = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into());
    let lm = LM::builder()
        .base_url(base_url)
        .model(model_name)
        .temperature(0.0)
        .max_tokens(512)
        .max_retries(0)
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_custom_output_schema(true);
    let cases = [
        Case::Simple,
        Case::Optimize,
        Case::Numbers,
        Case::Batch,
        Case::History,
        Case::Sentiment,
        #[cfg(feature = "data")]
        Case::Evaluate,
        #[cfg(feature = "data")]
        Case::Copro,
        #[cfg(feature = "data")]
        Case::Mipro,
    ];
    for case in cases {
        let output = timeout(Duration::from_secs(180), case.exercise(lm.clone()))
            .await
            .unwrap()
            .unwrap();
        case.check(&output);
        println!("verified {case:?}: {output}");
    }
}
