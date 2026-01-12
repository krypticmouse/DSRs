use dsrs_macros::Signature;

#[derive(Signature)]
struct MissingAttr {
    question: String,
    #[output]
    answer: String,
}

fn main() {}
