use baml_bridge_derive::BamlType;

#[derive(BamlType)]
struct NestedConflict {
    #[render(style = "json", template = "{{ value.raw }}")]
    content: String,
}

fn main() {}
