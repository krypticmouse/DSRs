use dsrs_macros::Signature;

#[derive(Signature)]
struct MissingInput {
    #[output]
    answer: String,
}

fn main() {}
