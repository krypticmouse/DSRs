use dsrs_macros::Signature;

#[derive(Signature)]
struct SignatureSerdeJsonValue {
    #[input]
    payload: serde_json::Value,

    #[output]
    answer: String,
}

fn main() {}
