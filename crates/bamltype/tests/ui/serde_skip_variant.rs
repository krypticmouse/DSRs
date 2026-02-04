use bamltype::BamlType;

#[BamlType]
enum Bad {
    #[serde(skip)]
    A,
    B,
}

fn main() {}
