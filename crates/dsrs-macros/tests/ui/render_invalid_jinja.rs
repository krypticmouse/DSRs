use dsrs_macros::Signature;

#[derive(Signature)]
struct RenderInvalidJinja {
    #[input]
    #[render(jinja = "{{ this ")]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
