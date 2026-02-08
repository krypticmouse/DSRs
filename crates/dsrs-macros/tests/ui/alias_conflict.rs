use dsrs_macros::Signature;

#[derive(Signature)]
struct AliasConflict {
    #[input]
    first: String,

    #[input]
    #[alias("first")]
    second: String,

    #[output]
    answer: String,
}

fn main() {}
