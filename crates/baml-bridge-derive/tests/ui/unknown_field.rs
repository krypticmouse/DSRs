use baml_bridge_derive::BamlType;

#[derive(BamlType)]
#[render(template = "{{ value.missing }}")]
struct BadField {
    title: String,
}

fn main() {}
