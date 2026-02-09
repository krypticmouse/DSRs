use bamltype::HoistClasses;
use dspy_rs::{BamlType, ChatAdapter, RenderOptions, Signature};

#[derive(Clone, Debug)]
#[BamlType]
struct AliasPayload {
    #[baml(alias = "fullName")]
    full_name: String,
}

#[derive(Clone, Debug)]
#[BamlType]
#[baml(rename_all = "camelCase")]
struct RenamedFieldsPayload {
    user_name: String,
    created_at: String,
}

#[derive(Clone, Debug)]
#[BamlType]
struct SkipDefaultPayload {
    content: String,
    #[baml(skip)]
    internal_id: i64,
    #[baml(default)]
    retries: i32,
}

#[derive(Clone, Debug)]
#[BamlType]
struct BigIdPayload {
    #[baml(int_repr = "string")]
    large_id: u64,
}

#[derive(Clone, Debug)]
#[BamlType]
#[baml(name = "UserProfile")]
struct NamedPayload {
    name: String,
}

#[derive(Signature, Clone, Debug)]
/// Contract test for docs-visible type behavior in prompts.
struct DocsTypeEffectsSig {
    #[input]
    question: String,

    #[output]
    alias_payload: AliasPayload,

    #[output]
    renamed_payload: RenamedFieldsPayload,

    #[output]
    skip_default_payload: SkipDefaultPayload,

    #[output]
    big_id_payload: BigIdPayload,

    #[output]
    named_payload: NamedPayload,
}

fn system_message() -> String {
    let adapter = ChatAdapter;
    adapter
        .format_system_message_typed::<DocsTypeEffectsSig>()
        .expect("system message")
}

fn extract_field_block(message: &str, field_name: &str) -> String {
    let marker = format!("[[ ## {field_name} ## ]]");
    let start = message
        .find(&marker)
        .unwrap_or_else(|| panic!("missing marker: {field_name}"));
    let after = start + marker.len();
    let remaining = &message[after..];
    let end = remaining.find("[[ ##").unwrap_or(remaining.len());
    remaining[..end].trim().to_string()
}

fn find_line<'a>(block: &'a str, needle: &str) -> &'a str {
    block
        .lines()
        .find(|line| line.contains(needle))
        .unwrap_or_else(|| panic!("missing line containing {needle:?} in:\n{block}"))
}

#[test]
fn alias_is_visible_to_model_schema() {
    let block = extract_field_block(&system_message(), "alias_payload");
    assert!(block.contains("fullName"));
    assert!(!block.contains("full_name"));
}

#[test]
fn rename_all_is_visible_to_model_schema() {
    let block = extract_field_block(&system_message(), "renamed_payload");
    assert!(block.contains("userName"));
    assert!(block.contains("createdAt"));
    assert!(!block.contains("user_name"));
    assert!(!block.contains("created_at"));
}

#[test]
fn skip_hides_field_and_default_marks_optional() {
    let block = extract_field_block(&system_message(), "skip_default_payload");
    assert!(block.contains("content"));
    assert!(!block.contains("internal_id"));

    let retries_line = find_line(&block, "retries");
    assert!(
        retries_line.contains('?') || retries_line.contains("null"),
        "expected optional retries marker in line: {retries_line}"
    );
}

#[test]
fn int_repr_changes_prompt_type_to_string() {
    let block = extract_field_block(&system_message(), "big_id_payload");
    let id_line = find_line(&block, "large_id");
    assert!(
        id_line.contains("string"),
        "expected string type for int_repr=\"string\": {id_line}"
    );
}

#[test]
fn name_changes_type_label_in_prompt_and_hoisted_render() {
    let block = extract_field_block(&system_message(), "named_payload");
    assert!(
        block.contains("Output field `named_payload` should be of type: UserProfile"),
        "expected renamed type label in prompt:\n{block}"
    );

    let rendered = <NamedPayload as BamlType>::baml_output_format()
        .render(RenderOptions::hoist_classes(HoistClasses::All))
        .expect("render")
        .unwrap_or_default();
    assert!(
        rendered.contains("UserProfile"),
        "expected renamed class in hoisted render:\n{rendered}"
    );
}

#[test]
fn parse_behavior_matches_skip_and_default_claims() {
    let parsed =
        bamltype::parse_llm_output::<SkipDefaultPayload>(r#"{ "content": "hello" }"#, true)
            .expect("parse");

    assert_eq!(parsed.value.content, "hello");
    assert_eq!(parsed.value.internal_id, 0);
    assert_eq!(parsed.value.retries, 0);
}
