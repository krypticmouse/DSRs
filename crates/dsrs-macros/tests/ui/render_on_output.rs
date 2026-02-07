use dsrs_macros::Signature;

#[derive(Signature)]
struct RenderOnOutput {
    #[input]
    question: String,

    #[output]
    #[render(jinja = "{{ this }}")]
    answer: String,
}

fn main() {}
