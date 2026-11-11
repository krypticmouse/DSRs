use dspy_rs::{Chat, CompletionProvider, LM, LMClient, Message, TestCompletionModel};
use rig::completion::{AssistantContent, CompletionError, CompletionRequest};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

async fn server(statuses: Vec<u16>) -> (String, JoinHandle<Vec<(String, Value, String)>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for status in statuses {
            let (stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            assert!(line.starts_with("POST /v1/chat/completions "));
            let mut length = None;
            let mut authorization = String::new();
            loop {
                line.clear();
                assert_ne!(reader.read_line(&mut line).await.unwrap(), 0);
                if line == "\r\n" {
                    break;
                }
                let (name, value) = line.split_once(':').unwrap();
                if name.eq_ignore_ascii_case("content-length") {
                    length = Some(value.trim().parse::<usize>().unwrap());
                }
                if name.eq_ignore_ascii_case("authorization") {
                    authorization = value.trim().to_string();
                }
            }
            let mut bytes = vec![0; length.unwrap()];
            reader.read_exact(&mut bytes).await.unwrap();
            requests.push((
                authorization,
                serde_json::from_slice(&bytes).unwrap(),
                String::from_utf8(bytes).unwrap(),
            ));
            let body = if status == 200 {
                json!({
                    "id": "test", "object": "chat.completion", "created": 0, "model": "test",
                    "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                })
            } else {
                // Deliberately omit the status from the body: classification must
                // use the actual HTTP status rather than error-message keywords.
                json!({"error": {"message": "request failed", "type": "test"}})
            }
            .to_string();
            let response = format!(
                "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            reader
                .get_mut()
                .write_all(response.as_bytes())
                .await
                .unwrap();
        }
        requests
    });
    (url, task)
}

fn chat() -> Chat {
    Chat::new(vec![Message::user("hello")])
}

fn raw_request(additional_params: Value) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: rig::OneOrMany::one("hello".into()),
        documents: vec![],
        tools: vec![],
        temperature: Some(0.0),
        max_tokens: Some(321),
        tool_choice: None,
        additional_params: Some(additional_params),
        output_schema: None,
    }
}

#[rstest::rstest]
#[case("max_tokens")]
#[case("max_completion_tokens")]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn explicit_provider_limits_and_other_parameters_are_preserved(#[case] parameter: &str) {
    timeout(Duration::from_secs(10), async {
        let (url, server) = server(vec![200]).await;
        let client = LMClient::from_local(&url, "custom").unwrap();
        let request = raw_request(json!({parameter: 7, "presence_penalty": 0.2}));
        client.completion(request).await.unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests[0].1[parameter], 7);
        assert_eq!(requests[0].1["presence_penalty"], 0.2);
        let other = if parameter == "max_tokens" {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        assert!(requests[0].1.get(other).is_none());
    })
    .await
    .unwrap();
}

#[tokio::test]
#[cfg_attr(miri, ignore = "requires a native HTTP client")]
async fn invalid_additional_parameters_fail_as_a_typed_request_error() {
    let client = LMClient::from_local("http://127.0.0.1:1/v1", "custom").unwrap();
    let error = client.completion(raw_request(json!([]))).await.unwrap_err();
    let CompletionError::RequestError(cause) = error else {
        panic!("expected a request error")
    };
    assert_eq!(
        cause.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::InvalidInput
    );
}

#[rstest::rstest]
#[case("llama3.2:3b", "max_tokens")]
#[case("gpt-4o-mini", "max_tokens")]
#[case("gpt-oss:20b", "max_tokens")]
#[case("o3-mini", "max_completion_tokens")]
#[case("gpt-5", "max_completion_tokens")]
#[case("gpt-5.1", "max_completion_tokens")]
#[case("gpt-6", "max_completion_tokens")]
#[case("chatgpt-5-latest", "max_completion_tokens")]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn token_limit_and_config_copies_reach_the_http_payload(
    #[case] model: &str,
    #[case] parameter: &str,
) {
    timeout(Duration::from_secs(10), async {
        let (url, server) = server(vec![200, 200]).await;
        let original = LM::builder()
            .base_url(url)
            .model(model.into())
            .max_tokens(321)
            .max_retries(0)
            .cache(false)
            .build()
            .await
            .unwrap();
        original.call(chat(), vec![]).await.unwrap();
        let copy = original
            .copy_with(|config| config.max_tokens = 7)
            .await
            .unwrap();
        copy.call(chat(), vec![]).await.unwrap();
        let requests = server.await.unwrap();
        assert_eq!(requests[0].1[parameter], 321);
        assert_eq!(requests[1].1[parameter], 7);
        let other = if parameter == "max_tokens" {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        for (_, body, raw) in requests {
            assert!(body.get(other).is_none());
            assert_eq!(raw.matches(&format!("\"{parameter}\"")).count(), 1);
        }
        assert_eq!(original.config.max_tokens, 321);
    })
    .await
    .unwrap();
}

#[rstest::rstest]
#[case(429)]
#[case(500)]
#[case(503)]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn transient_status_retries_then_succeeds(#[case] status: u16) {
    timeout(Duration::from_secs(10), async {
        let (url, server) = server(vec![status, 200]).await;
        let lm = LM::builder()
            .base_url(url)
            .model("test".into())
            .max_retries(1)
            .retry_base_delay_ms(0)
            .build()
            .await
            .unwrap();
        assert_eq!(
            lm.call(chat(), vec![]).await.unwrap().output.content(),
            "done"
        );
        assert_eq!(server.await.unwrap().len(), 2);
    })
    .await
    .unwrap();
}

#[rstest::rstest]
#[case(400)]
#[case(401)]
#[case(403)]
#[case(404)]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn permanent_status_fails_without_retry(#[case] status: u16) {
    timeout(Duration::from_secs(10), async {
        let (url, server) = server(vec![status]).await;
        let lm = LM::builder()
            .base_url(url)
            .model("test".into())
            .max_retries(2)
            .retry_base_delay_ms(0)
            .build()
            .await
            .unwrap();
        let error = lm.call(chat(), vec![]).await.unwrap_err();
        assert!(format!("{error:#}").contains(&status.to_string()));
        assert_eq!(server.await.unwrap().len(), 1);
    })
    .await
    .unwrap();
}

#[rstest::rstest]
#[case(0)]
#[case(2)]
#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn retries_are_bounded_and_zero_disables_them(#[case] retries: u32) {
    timeout(Duration::from_secs(10), async {
        let (url, server) = server(vec![503; retries as usize + 1]).await;
        let lm = LM::builder()
            .base_url(url)
            .model("test".into())
            .max_retries(retries)
            .retry_base_delay_ms(0)
            .build()
            .await
            .unwrap();
        let error = lm.call(chat(), vec![]).await.unwrap_err();
        assert!(format!("{error:#}").contains("503"));
        assert_eq!(server.await.unwrap().len(), retries as usize + 1);
    })
    .await
    .unwrap();
}

#[tokio::test]
#[cfg_attr(miri, ignore = "requires localhost TCP")]
async fn provider_identity_changes_rebuild_client_and_allow_clearing_credentials() {
    timeout(Duration::from_secs(10), async {
        let (url, server) = server(vec![200, 200]).await;
        let mock = TestCompletionModel::new([AssistantContent::text("original")]);
        let original = LM::builder()
            .base_url("http://localhost:1/v1".into())
            .model("original".into())
            .temperature(0.3)
            .api_key("original-key".into())
            .build()
            .await
            .unwrap()
            .with_client(LMClient::Test(mock.clone()))
            .await
            .unwrap();
        let changed = original
            .copy_with(|config| {
                config.base_url = Some(url);
                config.model = "changed".into();
                config.api_key = Some("variant-key".into());
            })
            .await
            .unwrap();
        assert_eq!(
            changed.call(chat(), vec![]).await.unwrap().output.content(),
            "done"
        );
        let cleared = changed
            .copy_with(|config| config.api_key = None)
            .await
            .unwrap();
        assert_eq!(
            cleared.call(chat(), vec![]).await.unwrap().output.content(),
            "done"
        );
        let requests = server.await.unwrap();
        assert_eq!(requests[0].0, "Bearer variant-key");
        assert_eq!(requests[1].0, "Bearer dummy-key-for-local-server");
        assert_eq!(requests[0].1["model"], "changed");
        assert_eq!(requests[1].1["model"], "changed");
        assert_eq!(original.config.model, "original");
        assert_eq!(original.config.api_key.as_deref(), Some("original-key"));
        assert!(mock.last_request().is_none());
        assert_eq!(
            original
                .call(chat(), vec![])
                .await
                .unwrap()
                .output
                .content(),
            "original"
        );
    })
    .await
    .unwrap();
}
