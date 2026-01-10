use dsrs_macros::Signature;

#[derive(Signature)]
struct MissingOutput {
    #[input]
    question: String,
}

fn main() {}
