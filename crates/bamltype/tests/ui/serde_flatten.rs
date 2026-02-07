use bamltype::BamlType;
type HashMap<K, V> = std::collections::HashMap<K, V>;

#[BamlType]
struct Bad {
    #[serde(flatten)]
    extras: HashMap<String, String>,
}

fn main() {}
