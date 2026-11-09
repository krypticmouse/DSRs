use dspy_rs::{Chat, LM, LMClient, LMConfig, Message};
use std::io::{self, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

struct LogWriter(Arc<Mutex<Vec<u8>>>);

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn model_client_trace_omits_url_credentials() {
    let output = Arc::new(Mutex::new(Vec::new()));
    let writer_output = Arc::clone(&output);
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || LogWriter(Arc::clone(&writer_output)))
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        let url = "https://user:password-sentinel@example.com/?token=query-sentinel";
        LMClient::from_openai_compatible(url, "api-key-sentinel", "model").unwrap();
        LMClient::from_local(url, "model").unwrap();
    });

    let events = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    assert!(events.contains("openai-compatible client ready"));
    assert!(events.contains("local openai-compatible client ready"));
    assert!(!events.contains("password-sentinel"));
    assert!(!events.contains("query-sentinel"));
    assert!(!events.contains("api-key-sentinel"));
}

#[tokio::test(flavor = "current_thread")]
async fn model_transport_failure_omits_url_credentials() {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let output = Arc::new(Mutex::new(Vec::new()));
    let writer_output = Arc::clone(&output);
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .with_writer(move || LogWriter(Arc::clone(&writer_output)))
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let lm = LM::from_config(LMConfig {
        base_url: Some(format!(
            "http://127.0.0.1:{port}/v1?token=query-sentinel"
        )),
        api_key: Some("api-key-sentinel".to_string()),
        model: "test-model".to_string(),
        max_retries: 1,
        retry_base_delay_ms: 0,
        ..LMConfig::default()
    })
    .await
    .unwrap();
    let err = lm
        .call(Chat::new(vec![Message::user("hello")]), Vec::new())
        .await
        .unwrap_err();
    let events = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    assert!(events.contains("retrying transient lm completion failure"));
    assert!(err.to_string().contains("HTTP transport error"), "{err}");
    for secret in ["query-sentinel", "api-key-sentinel"] {
        assert!(!events.contains(secret), "trace exposed {secret}");
        assert!(!err.to_string().contains(secret), "error exposed {secret}");
    }
}
