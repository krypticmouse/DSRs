use dsrs_macros::Signature;

#[derive(Signature)]
struct BadRenderConflict {
    #[input]
    #[render(style = "json", template = "{{ value }}")]
    value: String,

    #[output]
    answer: String,
}

fn main() {}
