use dspy_rs::{DataLoadError, DataLoader, TypedLoadOptions};
use std::error::Error;

#[derive(Debug, serde::Deserialize, facet::Facet)]
struct Row {
    #[allow(dead_code)]
    value: String,
}

#[test]
fn remote_http_failures_keep_status_in_the_source_chain() {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/rows.json", listener.local_addr().unwrap());
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 2048];
        stream.read(&mut request).unwrap();
        stream
            .write_all(
                b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            )
            .unwrap();
    });
    let error = DataLoader::load_json::<Row>(&url, false, TypedLoadOptions::default()).unwrap_err();
    server.join().unwrap();
    assert!(matches!(
        error.downcast_ref::<DataLoadError>(),
        Some(DataLoadError::Io(_))
    ));
    let http = error
        .chain()
        .find_map(|cause| cause.downcast_ref::<reqwest::Error>())
        .unwrap();
    assert_eq!(http.status(), Some(reqwest::StatusCode::TOO_MANY_REQUESTS));
}

#[test]
fn json_and_mapper_failures_keep_the_typed_original_cause() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.json");
    std::fs::write(&path, "not json").unwrap();
    let error =
        DataLoader::load_json::<Row>(path.to_str().unwrap(), false, TypedLoadOptions::default())
            .unwrap_err();
    let source = error.downcast_ref::<DataLoadError>().unwrap();
    assert!(matches!(source, DataLoadError::Json(_)));
    assert!(
        source
            .source()
            .unwrap()
            .downcast_ref::<serde_json::Error>()
            .is_some()
    );

    std::fs::write(&path, r#"[{"value":"ok"}]"#).unwrap();
    let error = DataLoader::load_json_with::<Row, _>(
        path.to_str().unwrap(),
        false,
        TypedLoadOptions::default(),
        |_| {
            Err(
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "mapper fixture cause")
                    .into(),
            )
        },
    )
    .unwrap_err();
    let source = error.downcast_ref::<DataLoadError>().unwrap();
    assert!(matches!(source, DataLoadError::Mapper { row: 1, .. }));
    assert_eq!(
        source
            .source()
            .unwrap()
            .downcast_ref::<std::io::Error>()
            .unwrap()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
}
