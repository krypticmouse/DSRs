use bamltype::BamlType;

#[BamlType]
#[serde(untagged)]
enum Bad {
    A { x: i64 },
    B { y: i64 },
}

fn main() {}
