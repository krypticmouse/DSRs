use std::fs;
use std::path::Path;

#[test]
fn contract_ui_error_messages_match_legacy() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let legacy_ui = root.join("../baml-bridge/tests/ui");
    let facet_ui = root.join("tests/ui");

    let mut compared = 0usize;
    for entry in fs::read_dir(&legacy_ui).expect("read legacy ui dir") {
        let entry = entry.expect("read dir entry");
        let path = entry.path();

        if path.extension().and_then(|ext| ext.to_str()) != Some("stderr") {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("stderr file name");
        let facet_path = facet_ui.join(file_name);
        assert!(
            facet_path.exists(),
            "missing facet stderr fixture for {file_name}"
        );

        let legacy = fs::read_to_string(&path).expect("read legacy stderr");
        let facet = fs::read_to_string(&facet_path).expect("read facet stderr");
        assert_eq!(legacy, facet, "stderr mismatch for {file_name}");
        compared += 1;
    }

    assert!(compared > 0, "no stderr fixtures compared");
}
