use bamltype::BamlType;

#[BamlType]
struct Bad {
    callback: fn(i32) -> i32,
}

fn main() {}
