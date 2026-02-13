use dsrs_macros::Signature;

#[derive(Signature)]
struct SignatureLargeInt {
    #[input]
    id: u64,

    #[output]
    answer: String,
}

fn main() {}
