use baml_bridge::BamlType;

#[derive(BamlType)]
struct Bad {
    callback: fn(i32) -> i32,
}

fn main() {}
