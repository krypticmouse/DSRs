use dsrs_macros::Signature;

#[derive(Signature)]
struct BadSig {
    #[input]
    #[format]
    name: String,

    #[output]
    result: String,
}

fn main() {}
