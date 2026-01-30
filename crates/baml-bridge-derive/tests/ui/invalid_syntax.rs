use baml_bridge_derive::BamlType;

#[derive(BamlType)]
#[render(template = "{{ value")]
struct BadTemplate {
    value: String,
}

fn main() {}
