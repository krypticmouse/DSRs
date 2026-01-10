use baml_bridge::BamlType;

#[derive(BamlType)]
struct Bad {
    value: Box<dyn std::fmt::Debug>,
}

fn main() {}
