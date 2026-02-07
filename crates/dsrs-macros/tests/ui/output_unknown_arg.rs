use dsrs_macros::Signature;

#[derive(Signature)]
struct OutputUnknownArg {
    #[input]
    question: String,

    #[output(bad = "x")]
    answer: String,
}

fn main() {}
