use dspy_rs::CompileExt;

#[derive(Clone, Debug, dspy_rs::Signature)]
struct SimpleSig {
    #[input]
    #[render(template = "override: {{ value.raw }}{{ ctx.suffix }}")]
    name: String,
    #[input]
    count: i32,
    #[output]
    result: String,
}

#[derive(serde::Serialize)]
struct TestCtx {
    suffix: &'static str,
}

#[test]
fn compiled_signature_renders_defaults_and_overrides() {
    let compiled = SimpleSig::compile();
    let compiled_again = SimpleSig::compile();
    let input = SimpleSigInput {
        name: "Ada".to_string(),
        count: 2,
    };

    let rendered = compiled
        .render_messages_with_ctx(&input, TestCtx { suffix: "!" })
        .expect("rendered messages");

    assert!(std::sync::Arc::ptr_eq(
        &compiled.world,
        &compiled_again.world
    ));

    assert!(rendered.system.contains("Your input fields are:"));
    assert!(rendered.system.contains("name"));
    assert!(rendered.system.contains("count"));
    assert!(rendered.system.contains("Your output fields are:"));
    assert!(rendered.system.contains("result"));

    assert!(rendered.user.contains("override: Ada!"));
    assert!(rendered.user.contains("2"));
}
