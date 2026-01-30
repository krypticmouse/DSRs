use baml_bridge_derive::BamlType;

#[derive(BamlType)]
#[render(template = "{{ value.missing }}", allow_dynamic = true)]
struct AllowDynamic {
    title: String,
}

fn main() {}
