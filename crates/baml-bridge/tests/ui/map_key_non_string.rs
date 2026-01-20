use baml_bridge::BamlType;
use std::collections::HashMap;

#[derive(BamlType)]
struct Bad {
    values: HashMap<u32, String>,
}

fn main() {}
