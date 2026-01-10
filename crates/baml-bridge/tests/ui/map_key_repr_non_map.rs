use baml_bridge::BamlType;

#[derive(BamlType)]
struct Bad {
    #[baml(map_key_repr = "string")]
    value: Vec<String>,
}

fn main() {}
