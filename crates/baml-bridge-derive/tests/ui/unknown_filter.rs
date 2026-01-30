use baml_bridge_derive::BamlType;

#[derive(BamlType)]
#[render(template = "{{ value.title | not_a_filter }}")]
struct BadFilter {
    title: String,
}

fn main() {}
