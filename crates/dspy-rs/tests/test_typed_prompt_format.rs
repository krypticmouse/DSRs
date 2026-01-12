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
        .format_system_message_typed::<ComprehensiveSignature>()
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

#[test]
fn test_primitive_types() {
    let message = system_message();
    let answer = extract_field_block(&message, "answer");
    let count = extract_field_block(&message, "count");
    let score = extract_field_block(&message, "score");
    let is_valid = extract_field_block(&message, "is_valid");

    assert!(answer.contains("Output field `answer` should be of type: string"));
    assert!(count.contains("Output field `count` should be of type: int"));
    assert!(score.contains("Output field `score` should be of type: float"));
    assert!(is_valid.contains("Output field `is_valid` should be of type: bool"));
}

#[test]
fn test_optional_renders_as_nullable() {
    let message = system_message();
    let maybe_answer = extract_field_block(&message, "maybe_answer");

    assert!(maybe_answer.contains("string"));
    assert!(maybe_answer.contains("null"));
}

#[test]
fn test_array_renders_with_brackets() {
    let message = system_message();
    let keywords = extract_field_block(&message, "keywords");
    let citations = extract_field_block(&message, "citations");

    assert!(keywords.contains("string[]"));
    assert!(citations.contains("["));
    assert!(citations.contains("doc_id"));
    assert!(citations.contains("quote"));
}

#[test]
fn test_schema_is_separate_from_type_line() {
    let message = system_message();
    let citations = extract_field_block(&message, "citations");
    let header_line = citations
        .lines()
        .find(|line| line.starts_with("Output field `citations`"))
        .expect("type header line");

    assert!(header_line.contains("Citation[]"));
    assert!(!header_line.contains("//"));

    let schema_lines: Vec<&str> = citations.lines().collect();
    let header_index = schema_lines
        .iter()
        .position(|line| line.starts_with("Output field `citations`"))
        .expect("type header line");
    let mut cursor = header_index + 1;
    while cursor < schema_lines.len() && schema_lines[cursor].trim().is_empty() {
        cursor += 1;
    }
    if schema_lines
        .get(cursor)
        .is_some_and(|line| line.trim() == "Definitions (used below):")
    {
        cursor += 1;
        while cursor < schema_lines.len() {
            let trimmed = schema_lines[cursor].trim_start();
            if trimmed.starts_with('{') || trimmed.starts_with('[') {
                break;
            }
            cursor += 1;
        }
    }

    let schema_start = schema_lines
        .get(cursor)
        .expect("schema content line");
    assert!(
        schema_start.trim_start().starts_with('[')
            || schema_start.trim_start().starts_with('{')
    );
}

#[test]
fn test_nested_struct_with_comments() {
    let message = system_message();
    let citations = extract_field_block(&message, "citations");

    assert!(citations.contains("Document identifier"));
    assert!(citations.contains("Relevant quote"));
}

#[test]
fn test_enum_rendering() {
    let message = system_message();
    let sentiment = extract_field_block(&message, "sentiment");

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
        let marker = format!("[[ ## {field} ## ]]");
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
fn test_completed_marker_present() {
    let message = system_message();
    assert!(message.contains("[[ ## completed ## ]]"));
}

#[test]
fn test_objective_line_present() {
    let message = system_message();
    assert!(message.contains("In adhering to this structure, your objective is:"));
    assert!(message.contains("Tests all output field types."));
}

#[test]
fn test_response_instruction_line_present() {
    let message = system_message();
    let instruction_line = message
        .lines()
        .find(|line| line.starts_with("Respond with the corresponding output fields"))
        .expect("response instruction line");

    let markers = [
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
    for marker in markers {
        let token = format!("[[ ## {marker} ## ]]");
        let pos = instruction_line
            .find(&token)
            .unwrap_or_else(|| panic!("missing marker in response instruction: {marker}"));
        if let Some(prev) = last_pos {
            assert!(pos > prev, "marker {marker} is out of order");
        }
        last_pos = Some(pos);
    }

    assert!(instruction_line.contains("[[ ## completed ## ]]"));
}
