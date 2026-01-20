use dsrs_macros::Signature;

#[derive(Signature)]
struct BadFormatDuplicate {
    #[input]
    #[format("json")]
    #[format("yaml")]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
