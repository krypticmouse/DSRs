use bamltype::BamlType;

#[BamlType]
#[baml(name = 123)]
struct NonStringName {
    value: String,
}

fn main() {}
