use dsrs_macros::Signature;

const TEMPLATE: &str = "{{ this }}";

#[derive(Signature)]
struct RenderNonLiteral {
    #[input]
    #[render(jinja = TEMPLATE)]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
