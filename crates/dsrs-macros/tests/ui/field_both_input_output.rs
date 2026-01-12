use dsrs_macros::Signature;

#[derive(Signature)]
struct BothAttrs {
    #[input]
    #[output]
    value: String,
}

fn main() {}
