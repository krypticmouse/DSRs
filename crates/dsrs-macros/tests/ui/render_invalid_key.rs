use dsrs_macros::Signature;

#[derive(Signature)]
struct RenderInvalidKey {
    #[input]
    #[render(template = "{{ this }}")]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
