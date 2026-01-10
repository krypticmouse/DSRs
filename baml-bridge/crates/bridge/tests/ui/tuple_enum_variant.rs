use baml_bridge::BamlType;

#[derive(BamlType)]
enum Bad {
    One(u32),
}

fn main() {}
