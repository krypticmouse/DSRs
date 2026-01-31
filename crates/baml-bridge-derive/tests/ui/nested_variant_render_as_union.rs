use baml_bridge_derive::BamlType;

#[derive(BamlType)]
#[baml(as_union)]
enum E {
    #[render(style = "json")]
    A,
}

fn main() {}
