use dspy_rs::{LM, Predict, Signature};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::time::{Duration, timeout};

#[derive(Signature, Clone, Debug)]
/// Answer geography questions accurately.
struct QA {
    #[input]
    /// The geography question to answer.
    question: String,

    #[output]
    /// The city name, without extra commentary.
    answer: String,
}

async fn predictor(base_url: String, model: String, structured: bool) -> Predict<QA> {
    let lm = LM::builder()
        .base_url(base_url)
        .model(model)
        .temperature(0.0)
        .max_retries(0)
        .build()
        .await
        .unwrap()
        .with_custom_output_schema(structured);
    Predict::<QA>::builder().lm(lm).build()
}

#[rstest::rstest]
#[case::delimiters(false)]
#[case::structured(true)]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn local_http_request_preserves_docs_and_output_format(#[case] structured: bool) {
    timeout(Duration::from_secs(15), async {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0; 4096];
            let (body_start, content_length) = loop {
                let n = stream.read(&mut chunk).await.unwrap();
                assert_ne!(n, 0, "connection closed before request headers");
                bytes.extend_from_slice(&chunk[..n]);
                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&bytes[..end]).unwrap();
                    assert!(headers.starts_with("POST /v1/chat/completions "));
                    let length = headers.lines().find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    }).unwrap();
                    break (end + 4, length);
                }
            };
            while bytes.len() < body_start + content_length {
                let n = stream.read(&mut chunk).await.unwrap();
                assert_ne!(n, 0, "connection closed before request body");
                bytes.extend_from_slice(&chunk[..n]);
            }
            let request: Value = serde_json::from_slice(
                &bytes[body_start..body_start + content_length],
            ).unwrap();
            let content = if structured {
                r#"{"answer":"Paris"}"#
            } else {
                "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
            };
            let body = json!({
                "id": "local-test", "object": "chat.completion", "created": 0,
                "model": "llama3.2",
                "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content}
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }).to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(), body,
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        let predictor = predictor(base_url, "llama3.2".into(), structured).await;
        let chat = predictor.build_chat(&QAInput {
            question: "What is the capital of France?".into(),
        }).await.unwrap();
        let prompt = chat.to_json().to_string();
        assert!(prompt.contains("Answer geography questions accurately."));
        assert!(prompt.contains("The geography question to answer."));
        assert!(prompt.contains("The city name, without extra commentary."));
        let (output, _) = predictor.call_and_parse(chat).await.unwrap();
        assert_eq!(output.answer, "Paris");
        let request = server.await.unwrap();
        assert_eq!(request["model"], "llama3.2");
        assert_eq!(request["temperature"], 0.0);
        assert!(request["messages"].to_string().contains("Answer geography questions accurately."));
        if structured {
            assert_eq!(request["response_format"]["type"], "json_schema");
            let schema = &request["response_format"]["json_schema"]["schema"];
            assert_eq!(schema["properties"]["answer"]["type"], "string");
            assert_eq!(schema["required"], json!(["answer"]));
        } else {
            assert!(request.get("response_format").is_none());
            assert!(prompt.contains("[[ ## answer ## ]]"));
        }
    }).await.expect("local HTTP test timed out");
}

#[tokio::test]
#[ignore = "requires local Ollama with llama3.2 pulled"]
async fn live_ollama_structured_prediction() {
    timeout(Duration::from_secs(120), async {
        let predictor = predictor(
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".into()),
            std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "llama3.2".into()),
            true,
        )
        .await;
        let output = predictor
            .call(QAInput {
                question: "What is the capital of France?".into(),
            })
            .await
            .unwrap();
        assert_eq!(output.answer.trim(), "Paris");
    })
    .await
    .expect("Ollama did not complete within 120 seconds");
}
