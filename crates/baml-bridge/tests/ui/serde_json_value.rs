use baml_bridge::BamlType;

#[derive(BamlType)]
struct Bad {
    value: serde_json::Value,
}

fn main() {}
