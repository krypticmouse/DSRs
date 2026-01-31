use baml_bridge_derive::BamlType;

#[derive(BamlType)]
struct NestedEmpty {
    #[render()]
    content: String,
}

fn main() {}
