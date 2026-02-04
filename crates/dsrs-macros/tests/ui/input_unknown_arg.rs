use dsrs_macros::Signature;

#[derive(Signature)]
struct InputUnknownArg {
    #[input(foo = "bar")]
    question: String,

    #[output]
    answer: String,
}

fn main() {}
