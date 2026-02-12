use dsrs_macros::Signature;

#[derive(Signature)]
struct SignatureTraitObject {
    #[input]
    value: Box<dyn std::fmt::Debug>,

    #[output]
    answer: String,
}

fn main() {}
