use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use expect_test::expect;
use sha2::{Digest, Sha256};

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[test]
fn contract_ui_error_messages_match_frozen_fixtures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let facet_ui = root.join("tests/ui");

    let mut hashes = BTreeMap::new();
    for entry in fs::read_dir(&facet_ui).expect("read facet ui dir") {
        let entry = entry.expect("read dir entry");
        let path = entry.path();

        if path.extension().and_then(|ext| ext.to_str()) != Some("stderr") {
            continue;
        }

        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("stderr file name")
            .to_string();
        let content = fs::read(&path).expect("read stderr fixture");
        hashes.insert(file_name, sha256_hex(&content));
    }

    assert!(!hashes.is_empty(), "no stderr fixtures discovered");

    let snapshot = serde_json::to_string_pretty(&hashes).expect("serialize fixture digest map");
    expect![[r#"
        {
          "as_enum_data_enum.stderr": "56fbfa048d2c8ee533da741e1521ef5482a2ac8ccd2aaa8bbbe70426cb735d0f",
          "function_type.stderr": "150024c5f2be81e9baa6dae06cb64ae050edd8a0f5f2734c02478e01c559577a",
          "large_int_without_repr.stderr": "55b8916098c952008baaa4e96fadc792727eccfc40d1319b1aed837e291a8a71",
          "map_key_non_string.stderr": "53fdb34520a4b258b21a543103bb9e58c9327d26e6f9ce8026cbb3522cddbbd0",
          "map_key_repr_non_map.stderr": "85ea714df372a6c2746c4eb7398a05d5b1edc31de981746ed2d500759263d72e",
          "non_string_literal_attr.stderr": "285e7b3fb2da106342895f016126d973d30470bad99d4edcc4edac7e6845c18a",
          "serde_default_path.stderr": "7628e1ccec4d97edc09dab392fbd0a39c7fe2b8e55fca8fe35cfa589ac61561d",
          "serde_flatten.stderr": "4ecdf04b1000d85a9b503d0d4a7c1e4567524bfe9c58d6c98e70322bc36bb57a",
          "serde_json_value.stderr": "ef346adbb204631fe7e96dd1a71ab2d47f4edd37d76610ce753ad8509fa6a9b2",
          "serde_skip_variant.stderr": "d89852ff38b3433ab4140d5790dcbaefd9533f88a8eb7c4300f1efd5d2d3b754",
          "serde_untagged.stderr": "dfd185a97bc8b99658e55754dc959c10b84167c6dcc6d1713307be656b1e3bf2",
          "trait_object.stderr": "dca3ad8c84eb406ebba578d7d5df48c0fcdc0f1fdd2c0854221814ee78cd19fe",
          "tuple_enum_variant.stderr": "8dcc1098be3166c6cdbdbe23470f78230fd81f819d6fc58657474d221f3b8e12",
          "tuple_field.stderr": "e0cfa7477dc9ebd083734ba9b4ae6e81da0c7ccc02f609a1a50547aa10adccac",
          "tuple_struct.stderr": "c540e375631f795a8f421ddfbad48e75808e8798ba120e4a7adaa742ceac97a4",
          "unit_struct.stderr": "82b287a2199e20a5febf675be80f9d175047a9f046c0356a90b28c7367e9d269",
          "unsupported_baml_attr.stderr": "6a068b8fe63e74d9d408d85de4345be41bb6517e601024633a115cf0b677d667"
        }"#]].assert_eq(&snapshot);
}
