use baml_bridge::BamlType;

#[derive(BamlType)]
enum Bad {
    #[serde(skip)]
    A,
    B,
}

fn main() {}
