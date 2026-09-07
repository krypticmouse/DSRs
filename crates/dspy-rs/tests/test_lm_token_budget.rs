//! Check actual SDK-serialized HTTP bodies, not only the intermediate request.
use dspy_rs::{Chat, LMClient, LMConfig, Message, LM};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn captured_request(kind: &str) -> serde_json::Value {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut data = Vec::new();
        let (start, len) = loop {
            let mut chunk = [0; 4096];
            let n = socket.read(&mut chunk).await.unwrap();
            assert!(n > 0 && data.len() < 100_000);
            data.extend_from_slice(&chunk[..n]);
            if let Some(end) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&data[..end]).to_lowercase();
                let len: usize = headers
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap();
                if data.len() >= end + 4 + len {
                    break (end + 4, len);
                }
            }
        };
        socket.write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").await.unwrap();
        serde_json::from_slice::<serde_json::Value>(&data[start..start + len]).unwrap()
    });
    let config = LMConfig {
        model: "openai:test".to_string(),
        api_key: Some("test-key".to_string()),
        base_url: (kind == "compatible").then(|| base.clone()),
        max_tokens: 37,
        max_retries: 0,
        request_timeout_secs: 5,
        ..Default::default()
    };
    let client = if kind == "openrouter" {
        let client = rig::providers::openrouter::Client::builder()
            .api_key("test-key")
            .base_url(&base)
            .build()
            .unwrap();
        LMClient::from_custom(
            rig::providers::openrouter::completion::CompletionModel::new(
                client,
                "openai/gpt-oss-20b",
            ),
        )
    } else {
        LMClient::from_openai_compatible(&base, "test-key", "test").unwrap()
    };
    let lm = LM::from_config(config)
        .await
        .unwrap()
        .with_client(client)
        .await
        .unwrap();
    let result = lm
        .call(Chat::new(vec![Message::user("hello")]), vec![])
        .await;
    assert!(
        result.is_err(),
        "the local capture server deliberately returns 400"
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test]
async fn token_budget_survives_native_openai_wire_serialization() {
    let body = captured_request("openai").await;
    assert_eq!(body["max_completion_tokens"], 37);
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn token_budget_survives_compatible_wire_serialization() {
    let body = captured_request("compatible").await;
    assert_eq!(body["max_tokens"], 37);
}

#[tokio::test]
async fn token_budget_survives_openrouter_wire_serialization() {
    let body = captured_request("openrouter").await;
    assert_eq!(body["max_tokens"], 37);
}
