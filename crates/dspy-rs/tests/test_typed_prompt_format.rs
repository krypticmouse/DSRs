use dspy_rs::{BamlType, ChatAdapter, Signature};

#[derive(Clone, Debug, BamlType)]
/// A citation reference.
struct Citation {
    /// Document identifier
    doc_id: String,
    /// Relevant quote
    quote: String,
}

#[derive(Clone, Debug, BamlType)]
/// Sentiment classification.
enum Sentiment {
    Positive,
    Neutral,
    Negative,
}

#[derive(Signature, Clone, Debug)]
/// Tests all output field types.
struct ComprehensiveSignature {
    #[input]
    query: String,

    #[output]
    answer: String,
    #[output]
    count: i32,
    #[output]
    score: f64,
    #[output]
    is_valid: bool,
    #[output]
    maybe_answer: Option<String>,
    #[output]
    keywords: Vec<String>,
    #[output]
    citations: Vec<Citation>,
    #[output]
    sentiment: Sentiment,
}

fn system_message() -> String {
    let adapter = ChatAdapter;
    adapter
        .format_system_message::<ComprehensiveSignature>()
        .expect("system message")
}

fn output_section(message: &str) -> &str {
    let marker = "Your output fields are:";
    let start = message
        .find(marker)
        .unwrap_or_else(|| panic!("missing output section"));
    &message[start + marker.len()..]
}

fn output_field_line(message: &str, field_name: &str) -> String {
    // Format is now: 1. `field_name` (type): description
    let needle = format!("`{field_name}`");
    for line in output_section(message).lines() {
        let trimmed = line.trim();
        if trimmed.contains(&needle) {
            return trimmed.to_string();
        }
    }
    panic!("missing output field: {field_name}");
}

fn output_schema_block(message: &str, field_name: &str) -> String {
    // In the restored format, schema info is in the field structure section
    // after the [[ ## field_name ## ]] marker
    let marker = format!("[[ ## {} ## ]]", field_name);
    let start = match message.find(&marker) {
        Some(pos) => pos + marker.len(),
        None => return String::new(),
    };

    let remaining = &message[start..];
    let mut lines = Vec::new();
    for line in remaining.lines() {
        let trimmed = line.trim();
        // Stop at next field marker or completed marker
        if trimmed.starts_with("[[ ## ") {
            break;
        }
        if !trimmed.is_empty() {
            lines.push(line.to_string());
        }
    }

    lines.join("\n").trim().to_string()
}

#[test]
fn test_primitive_types() {
    let message = system_message();
    let answer = output_field_line(&message, "answer");
    let count = output_field_line(&message, "count");
    let score = output_field_line(&message, "score");
    let is_valid = output_field_line(&message, "is_valid");

    // Format is now: N. `field` (type)
    assert!(answer.contains("`answer`") && answer.contains("string"));
    assert!(count.contains("`count`") && count.contains("int"));
    assert!(score.contains("`score`") && score.contains("float"));
    assert!(is_valid.contains("`is_valid`") && is_valid.contains("bool"));
}

#[test]
fn test_optional_renders_as_nullable() {
    let message = system_message();
    let maybe_answer = output_field_line(&message, "maybe_answer");
    let maybe_schema = output_schema_block(&message, "maybe_answer");

    assert!(maybe_answer.contains("string"));
    assert!(
        maybe_answer.contains("null") || maybe_schema.contains("null"),
        "expected optional type to mention null"
    );
}

#[test]
fn test_array_renders_with_brackets() {
    let message = system_message();
    let keywords = output_field_line(&message, "keywords");
    let citations = output_field_line(&message, "citations");
    let citations_schema = output_schema_block(&message, "citations");

    // Format shows list types
    assert!(keywords.contains("`keywords`"));
    assert!(keywords.contains("string") || keywords.contains("list"));
    assert!(citations.contains("`citations`"));
    assert!(citations.contains("Citation") || citations.contains("list"));
    assert!(citations_schema.contains("doc_id"));
    assert!(citations_schema.contains("quote"));
}

#[test]
fn test_schema_is_separate_from_type_line() {
    let message = system_message();
    let header = output_field_line(&message, "citations");
    let schema = output_schema_block(&message, "citations");

    // Type line has the field name
    assert!(header.contains("`citations`"));
    // Schema block has the field details
    assert!(schema.contains("doc_id") || schema.contains("Citation"));
}

#[test]
fn test_nested_struct_with_comments() {
    let message = system_message();
    let citations = output_schema_block(&message, "citations");

    assert!(citations.contains("Document identifier"));
    assert!(citations.contains("Relevant quote"));
}

#[test]
fn test_enum_rendering() {
    let message = system_message();
    let sentiment = output_schema_block(&message, "sentiment");

    assert!(sentiment.contains("Positive"));
    assert!(sentiment.contains("Neutral"));
    assert!(sentiment.contains("Negative"));
}

#[test]
fn test_no_answer_in_schema_block() {
    let message = system_message();
    assert!(!message.contains("Answer in this schema"));
}

#[test]
fn test_field_order_preserved() {
    let message = system_message();
    let fields = [
        "answer",
        "count",
        "score",
        "is_valid",
        "maybe_answer",
        "keywords",
        "citations",
        "sentiment",
    ];

    let mut last_pos = None;
    for field in fields {
        // Look for the field in the structure section with [[ ## field ## ]] markers
        let marker = format!("[[ ## {} ## ]]", field);
        let pos = message
            .find(&marker)
            .unwrap_or_else(|| panic!("missing marker: {field}"));
        if let Some(prev) = last_pos {
            assert!(pos > prev, "field {field} is out of order");
        }
        last_pos = Some(pos);
    }
}

#[test]
fn test_system_sections_present() {
    let message = system_message();
    assert!(message.contains("Your input fields are:"));
    assert!(message.contains("Your output fields are:"));
}

#[test]
fn test_system_scaffold_present() {
    // The restored format includes the full DSPy-style prompt structure
    let message = system_message();
    assert!(message.contains("Respond with the corresponding output fields"));
    assert!(message.contains("In adhering to this structure, your objective is:"));
    assert!(message.contains("[[ ## completed ## ]]"));
}
