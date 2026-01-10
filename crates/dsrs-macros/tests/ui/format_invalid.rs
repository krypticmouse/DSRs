use dsrs_macros::Signature;

#[derive(Signature)]
struct BadFormatValue {
    #[input]
    #[format("xml")]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
