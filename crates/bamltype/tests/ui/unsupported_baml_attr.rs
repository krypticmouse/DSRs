use bamltype::BamlType;

#[BamlType]
#[baml(unknown = "x")]
struct UnsupportedAttr {
    value: String,
}

fn main() {}
