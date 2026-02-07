use bamltype::BamlType;

#[BamlType]
struct TupleFieldRejected {
    pair: (i32, i32),
}

fn main() {}
