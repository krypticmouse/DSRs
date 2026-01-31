use baml_bridge_derive::BamlType;

#[derive(BamlType)]
#[baml(tag = "type")]
enum E {
    #[render(style = "json")]
    A { x: String },
}

fn main() {}
