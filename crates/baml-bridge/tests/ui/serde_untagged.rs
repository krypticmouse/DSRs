use baml_bridge::BamlType;

#[derive(BamlType)]
#[serde(untagged)]
enum Bad {
    A { x: i64 },
    B { y: i64 },
}

fn main() {}
