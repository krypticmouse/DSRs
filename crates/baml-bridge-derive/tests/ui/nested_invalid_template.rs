use baml_bridge_derive::BamlType;

#[derive(BamlType)]
struct NestedBadTemplate {
    #[render(template = "{{ value")]
    content: String,
}

fn main() {}
