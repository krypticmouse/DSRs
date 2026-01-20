use baml_bridge::BamlType;

#[derive(BamlType)]
#[baml(as_enum)]
enum Bad {
    A { value: i64 },
}

fn main() {}
