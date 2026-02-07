use dspy_rs::bamltype::compat::ToBamlValue;
use dspy_rs::{BamlType, BamlTypeTrait, BamlValue, ChatAdapter, Signature};

#[derive(Clone, Debug)]
#[BamlType]
struct Document {
    text: String,
}

#[derive(Signature, Clone, Debug)]
/// Provide an answer with supporting context.
struct FormatSig {
    #[input]
    question: String,

    #[input]
    #[format("yaml")]
    context: Vec<Document>,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Provide an answer with supporting context in JSON.
struct FormatJsonSig {
    #[input]
    question: String,

    #[input]
    #[format("json")]
    context: Vec<Document>,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Provide an answer with supporting context in TOON.
struct FormatToonSig {
    #[input]
    question: String,

    #[input]
    #[format("toon")]
    context: Vec<Document>,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Provide an answer with supporting context using the default format.
struct DefaultFormatSig {
    #[input]
    question: String,

    #[input]
    context: Vec<Document>,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Render a context field using Jinja.
struct RenderJinjaSig {
    #[input]
    question: String,

    #[input]
    #[alias("ctx")]
    #[render(
        jinja = "{{ this.text }} | {{ input.question }} | {{ input.ctx.text }} | {{ input.context.text }} | {{ field.name }} | {{ field.rust_name }}"
    )]
    context: Document,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Render with strict undefined vars.
struct RenderJinjaStrictSig {
    #[input]
    #[render(jinja = "{{ missing_var }}")]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
/// Render using field metadata and vars context.
struct RenderJinjaFieldMetaSig {
    #[input]
    #[alias("ctx")]
    #[render(
        jinja = "{{ field.name }}|{{ field.rust_name }}|{{ field.type }}|{{ vars is defined }}"
    )]
    context: Document,

    #[output]
    answer: String,
}

fn extract_field(message: &str, field_name: &str) -> String {
    let start_marker = format!("[[ ## {field_name} ## ]]");
    let start_pos = message
        .find(&start_marker)
        .unwrap_or_else(|| panic!("missing marker: {field_name}"));
    let after_marker = start_pos + start_marker.len();
    let remaining = &message[after_marker..];
    let end_pos = remaining.find("[[ ##").unwrap_or(remaining.len());
    remaining[..end_pos].trim().to_string()
}

fn extract_baml_field<'a>(value: &'a BamlValue, field_name: &str) -> &'a BamlValue {
    match value {
        BamlValue::Class(_, fields) | BamlValue::Map(fields) => fields
            .get(field_name)
            .unwrap_or_else(|| panic!("missing field: {field_name}")),
        other => panic!("unexpected input value: {other:?}"),
    }
}

#[test]
fn typed_input_format_yaml_renders_field_names() {
    let adapter = ChatAdapter;
    let input = FormatSigInput {
        question: "What is YAML?".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<FormatSig>(&input);
    let context_value = extract_field(&message, "context");
    let question_value = extract_field(&message, "question");

    assert!(context_value.contains("text: Hello"));
    assert_eq!(question_value, "What is YAML?");
}

#[test]
fn typed_input_format_json_is_parsable() {
    let adapter = ChatAdapter;
    let input = FormatJsonSigInput {
        question: "What is JSON?".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<FormatJsonSig>(&input);
    let context_value = extract_field(&message, "context");

    let parsed: serde_json::Value = serde_json::from_str(&context_value).expect("valid JSON");
    let first = parsed
        .as_array()
        .and_then(|items| items.first())
        .and_then(|value| value.as_object())
        .expect("expected array with object");
    assert_eq!(first.get("text").and_then(|v| v.as_str()), Some("Hello"));
}

#[test]
fn typed_input_format_toon_matches_formatter() {
    let adapter = ChatAdapter;
    let input = FormatToonSigInput {
        question: "What is TOON?".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<FormatToonSig>(&input);
    let context_value = extract_field(&message, "context");

    let baml_value = input.to_baml_value();
    let context_baml = extract_baml_field(&baml_value, "context");
    let output_format = <FormatToonSigInput as BamlTypeTrait>::baml_output_format();
    let expected = dspy_rs::bamltype::internal_baml_jinja::format_baml_value(
        context_baml,
        output_format,
        "toon",
    )
    .expect("formatting should succeed");

    assert_eq!(context_value, expected);
}

#[test]
fn typed_input_default_string_is_raw() {
    let adapter = ChatAdapter;
    let input = DefaultFormatSigInput {
        question: "Raw string".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<DefaultFormatSig>(&input);
    let question_value = extract_field(&message, "question");

    assert_eq!(question_value, "Raw string");
}

#[test]
fn typed_input_default_non_string_is_json() {
    let adapter = ChatAdapter;
    let input = DefaultFormatSigInput {
        question: "Default JSON".to_string(),
        context: vec![Document {
            text: "Hello".to_string(),
        }],
    };

    let message = adapter.format_user_message_typed::<DefaultFormatSig>(&input);
    let context_value = extract_field(&message, "context");
    let parsed: serde_json::Value = serde_json::from_str(&context_value).expect("valid JSON");
    let first = parsed
        .as_array()
        .and_then(|items| items.first())
        .and_then(|value| value.as_object())
        .expect("expected array with object");
    assert_eq!(first.get("text").and_then(|v| v.as_str()), Some("Hello"));
}

#[test]
fn typed_input_render_jinja_uses_context_values() {
    let adapter = ChatAdapter;
    let input = RenderJinjaSigInput {
        question: "Question".to_string(),
        context: Document {
            text: "Hello".to_string(),
        },
    };

    let message = adapter.format_user_message_typed::<RenderJinjaSig>(&input);
    let context_value = extract_field(&message, "ctx");

    assert_eq!(
        context_value,
        "Hello | Question | Hello | Hello | ctx | context"
    );
}

#[test]
fn typed_input_render_jinja_strict_undefined_returns_error_sentinel() {
    let adapter = ChatAdapter;
    let input = RenderJinjaStrictSigInput {
        question: "Question".to_string(),
    };

    let message = adapter.format_user_message_typed::<RenderJinjaStrictSig>(&input);
    let question_value = extract_field(&message, "question");

    assert_eq!(question_value, "<error>");
}

#[test]
fn typed_input_render_jinja_exposes_field_metadata_and_vars() {
    let adapter = ChatAdapter;
    let input = RenderJinjaFieldMetaSigInput {
        context: Document {
            text: "Hello".to_string(),
        },
    };

    let message = adapter.format_user_message_typed::<RenderJinjaFieldMetaSig>(&input);
    let context_value = extract_field(&message, "ctx");
    let parts: Vec<&str> = context_value.split('|').collect();

    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0], "ctx");
    assert_eq!(parts[1], "context");
    assert!(parts[2].contains("Document"));
    assert_eq!(parts[3].to_ascii_lowercase(), "true");
}
