use dsrs_macros::Signature;
type HashMap<K, V> = std::collections::HashMap<K, V>;

#[derive(Signature)]
struct SignatureNonStringMapKey {
    #[input]
    values: HashMap<u32, String>,

    #[output]
    answer: String,
}

fn main() {}
