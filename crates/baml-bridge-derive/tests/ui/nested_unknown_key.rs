use baml_bridge_derive::BamlType;

#[derive(BamlType)]
struct NestedUnknownKey {
    #[render(nope = 1)]
    content: String,
}

fn main() {}
