use dspy_rs::CompileExt;
use dspy_rs::{BamlType, Signature};

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

#[derive(Clone, Debug, BamlType)]
struct Payload {
    name: String,
    count: i64,
}

#[derive(Clone, Debug, Signature)]
struct JsonOverrideSig {
    #[input]
    #[render(style = "json")]
    payload: Payload,

    #[output]
    result: String,
}

fn extract_field(message: &str, field_name: &str) -> String {
    let start_marker = format!("[[ ## {field_name} ## ]]");
    let start = message
        .find(&start_marker)
        .unwrap_or_else(|| panic!("missing marker: {field_name}"));
    let after_marker = start + start_marker.len();
    let remaining = &message[after_marker..];
    let end = remaining.find("[[ ##").unwrap_or(remaining.len());
    remaining[..end].trim().to_string()
}

#[test]
fn compiled_signature_respects_style_override() {
    let compiled = JsonOverrideSig::compile();
    let input = JsonOverrideSigInput {
        payload: Payload {
            name: "Ada".to_string(),
            count: 1,
        },
    };

    let rendered = compiled.render_messages(&input).expect("rendered messages");
    let payload_block = extract_field(&rendered.user, "payload");

    assert!(payload_block.trim_start().starts_with('{'));
    assert!(payload_block.contains("\"name\""));
    assert!(!payload_block.contains("Payload {"));
}
