use bamltype::BamlType;
type HashMap<K, V> = std::collections::HashMap<K, V>;

#[BamlType]
struct Bad {
    values: HashMap<u32, String>,
}

fn main() {}
