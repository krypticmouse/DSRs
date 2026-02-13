use dsrs_macros::Signature;

#[derive(Signature)]
struct SignatureFunctionType {
    #[input]
    callback: fn(i32) -> i32,

    #[output]
    answer: String,
}

fn main() {}
