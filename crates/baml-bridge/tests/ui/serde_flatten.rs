use baml_bridge::BamlType;
use std::collections::HashMap;

#[derive(BamlType)]
struct Bad {
    #[serde(flatten)]
    extras: HashMap<String, String>,
}

fn main() {}
