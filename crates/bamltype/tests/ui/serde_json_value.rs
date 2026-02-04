use bamltype::BamlType;

#[BamlType]
struct Bad {
    value: serde_json::Value,
}

fn main() {}
