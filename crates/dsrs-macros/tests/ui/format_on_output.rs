use dsrs_macros::Signature;

#[derive(Signature)]
struct BadFormatOutput {
    #[input]
    question: String,

    #[output]
    #[format("json")]
    answer: String,
}

fn main() {}
