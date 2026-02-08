use dsrs_macros::Signature;

#[derive(Signature)]
struct FormatRenderConflict {
    #[input]
    #[format("json")]
    #[render(jinja = "{{ this }}")]
    context: String,

    #[output]
    answer: String,
}

fn main() {}
