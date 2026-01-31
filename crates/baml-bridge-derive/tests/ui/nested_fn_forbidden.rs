use baml_bridge_derive::BamlType;

fn render_it(_: &str) -> String { String::new() }

#[derive(BamlType)]
struct NestedFnForbidden {
    #[render(fn = render_it)]
    content: String,
}

fn main() {}
