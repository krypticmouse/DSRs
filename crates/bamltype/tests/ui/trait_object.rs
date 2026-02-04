use bamltype::BamlType;

#[BamlType]
struct Bad {
    value: Box<dyn std::fmt::Debug>,
}

fn main() {}
