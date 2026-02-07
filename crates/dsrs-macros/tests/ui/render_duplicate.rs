use dsrs_macros::Signature;

#[derive(Signature)]
struct RenderDuplicate {
    #[input]
    #[render(jinja = "{{ this }}")]
    #[render(jinja = "{{ this }}")]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
