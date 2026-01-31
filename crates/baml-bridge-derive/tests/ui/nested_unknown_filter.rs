use baml_bridge_derive::BamlType;

#[derive(BamlType)]
struct NestedUnknownFilter {
    #[render(template = "{{ value | nonexistent_filter }}")]
    content: String,
}

fn main() {}
