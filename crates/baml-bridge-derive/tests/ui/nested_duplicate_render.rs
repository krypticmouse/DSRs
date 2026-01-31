use baml_bridge_derive::BamlType;

#[derive(BamlType)]
struct DuplicateRender {
    #[render(style = "json")]
    #[render(max_string_chars = 10)]
    content: String,
}

fn main() {}
