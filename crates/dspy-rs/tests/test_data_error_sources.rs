use dspy_rs::{DataLoadError, DataLoader, TypedLoadOptions};
use std::error::Error;

#[derive(Debug, serde::Deserialize, facet::Facet)]
struct Row {
    #[allow(dead_code)]
    value: String,
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
