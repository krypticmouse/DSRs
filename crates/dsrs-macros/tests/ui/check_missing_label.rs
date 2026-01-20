use dsrs_macros::Signature;

#[derive(Signature)]
struct CheckMissingLabel {
    #[input]
    question: String,

    #[output]
    #[check("this > 0")]
    score: i32,
}

fn main() {}
