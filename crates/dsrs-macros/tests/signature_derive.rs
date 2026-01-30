use dspy_rs::{FieldRendererSpec, Signature as SignatureTrait};

/// Test instruction
#[derive(dsrs_macros::Signature)]
struct TestSig {
    #[input]
    #[alias("question_text")]
    question: String,

    #[output]
    #[check("this != ''", label = "non_empty")]
    answer: String,
}

fn render_note(
    _pv: &dspy_rs::baml_bridge::prompt::PromptValue,
    _session: &dspy_rs::baml_bridge::prompt::RenderSession,
) -> dspy_rs::baml_bridge::prompt::RenderResult {
    Ok("note".to_string())
}

/// Render attributes for field specs.
#[derive(dsrs_macros::Signature)]
struct RenderSig {
    #[input]
    #[render(style = "json", max_list_items = 5, max_depth = 2)]
    context: Vec<String>,

    #[input]
    #[render(template = r#"- {{ value }}"#)]
    note: String,

    #[input]
    #[render(r#fn = "render_note")]
    detailed_note: String,

    #[output]
    answer: String,
}

#[test]
fn test_generates_input_struct() {
    let input = TestSigInput {
        question: "test".to_string(),
    };
    assert_eq!(input.question, "test");
}

#[test]
fn test_generates_signature_impl() {
    assert_eq!(
        <TestSig as SignatureTrait>::instruction(),
        "Test instruction"
    );

    let input_fields = <TestSig as SignatureTrait>::input_fields();
    assert_eq!(input_fields.len(), 1);
    assert_eq!(input_fields[0].name, "question_text");

    let output_fields = <TestSig as SignatureTrait>::output_fields();
    assert_eq!(output_fields.len(), 1);
    assert_eq!(output_fields[0].constraints.len(), 1);
    assert_eq!(output_fields[0].constraints[0].label, "non_empty");
}

#[test]
fn test_render_attributes() {
    let input_fields = <RenderSig as SignatureTrait>::input_fields();
    assert_eq!(input_fields.len(), 3);

    let context = &input_fields[0];
    assert_eq!(context.style, Some("json"));
    let settings = context.render_settings.expect("render settings");
    assert_eq!(settings.max_list_items, Some(5));
    assert_eq!(settings.max_depth, Some(2));
    assert_eq!(settings.max_string_chars, None);

    let note = &input_fields[1];
    match note.renderer {
        Some(FieldRendererSpec::Jinja { template }) => {
            assert_eq!(template, "- {{ value }}");
        }
        _ => panic!("expected Jinja renderer"),
    }

    let detailed = &input_fields[2];
    match detailed.renderer {
        Some(FieldRendererSpec::Func { f }) => {
            let expected: fn(
                &dspy_rs::baml_bridge::prompt::PromptValue,
                &dspy_rs::baml_bridge::prompt::RenderSession,
            ) -> dspy_rs::baml_bridge::prompt::RenderResult = render_note;
            assert_eq!(f as usize, expected as usize);
        }
        _ => panic!("expected func renderer"),
    }
}

#[test]
fn test_from_parts_into_parts() {
    let input = TestSigInput {
        question: "q".to_string(),
    };
    let output = __TestSigOutput {
        answer: "a".to_string(),
    };

    let full = TestSig::from_parts(input.clone(), output.clone());
    assert_eq!(full.question, "q");
    assert_eq!(full.answer, "a");

    let (input2, output2) = full.into_parts();
    assert_eq!(input2.question, "q");
    assert_eq!(output2.answer, "a");
}

#[test]
fn test_baml_type_impl() {
    let _ = <TestSig as dspy_rs::baml_bridge::BamlType>::baml_output_format();
}
