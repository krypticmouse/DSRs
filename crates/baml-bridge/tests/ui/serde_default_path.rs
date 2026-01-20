use baml_bridge::BamlType;

#[derive(BamlType)]
struct Bad {
    #[serde(default = "default_age")]
    age: i64,
}

fn default_age() -> i64 {
    0
}

fn main() {}
