use bamltype::BamlType;

#[BamlType]
#[baml(as_enum)]
enum Bad {
    A { value: i64 },
}

fn main() {}
