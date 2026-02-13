use dsrs_macros::Signature;

#[derive(Signature)]
struct SignatureTupleType {
    #[input]
    pair: (i32, i32),

    #[output]
    answer: String,
}

fn main() {}
