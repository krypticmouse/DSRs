#![cfg(feature = "rlm")]

use baml_bridge::{BamlType, DefaultJinjaRender, ToBamlValue};

/// A single entry in the REPL history.
#[derive(Debug, Clone, BamlType)]
pub struct REPLEntry {
    pub reasoning: String,
    pub code: String,
    pub output: String,
}

impl REPLEntry {
    /// Create a new REPL entry.
    pub fn new(code: String, output: String) -> Self {
        Self {
            reasoning: String::new(),
            code,
            output,
        }
    }

    /// Create a new REPL entry with reasoning.
    pub fn with_reasoning(reasoning: String, code: String, output: String) -> Self {
        Self {
            reasoning,
            code,
            output,
        }
    }
}


/// Immutable REPL history container.
///
/// This type follows an immutable pattern where `append` returns a new
/// history instance, preserving the original. This aligns with Python
/// DSPy's frozen model pattern.
#[derive(Debug, Clone, Default, BamlType)]
pub struct REPLHistory {
    pub entries: Vec<REPLEntry>,
    max_output_chars: usize,
}

impl REPLHistory {
    /// Default maximum output characters for truncation.
    pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 5000;

    /// Create a new empty REPL history.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            max_output_chars: Self::DEFAULT_MAX_OUTPUT_CHARS,
        }
    }

    /// Create a new empty REPL history with custom max output chars.
    pub fn with_max_output_chars(max_output_chars: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_output_chars,
        }
    }

    /// Immutable append - returns new history with the entry added.
    pub fn append(&self, code: String, output: String) -> Self {
        let mut entries = self.entries.clone();
        entries.push(REPLEntry::new(code, output));
        Self {
            entries,
            max_output_chars: self.max_output_chars,
        }
    }

    /// Immutable append with reasoning - returns new history with the entry added.
    pub fn append_with_reasoning(&self, reasoning: String, code: String, output: String) -> Self {
        let mut entries = self.entries.clone();
        entries.push(REPLEntry::with_reasoning(reasoning, code, output));
        Self {
            entries,
            max_output_chars: self.max_output_chars,
        }
    }

    /// Number of entries in the history.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the entries as a slice.
    pub fn entries(&self) -> &[REPLEntry] {
        &self.entries
    }

    /// Get the maximum output characters setting.
    pub fn max_output_chars(&self) -> usize {
        self.max_output_chars
    }

    /// Render the history to a string using the default template.
    pub fn render(&self) -> String {
        let ctx = minijinja::Value::from_serialize(RenderContext {
            max_output_chars: self.max_output_chars as i64,
        });
        let baml_value = self.to_baml_value();
        let template_ctx = minijinja::Value::from_serialize(TemplateContext {
            value: &baml_value,
            ctx: &ctx,
        });

        let mut env = crate::baml_bridge::jsonish::jinja_helpers::get_env();
        env.add_filter("format_count", format_count_filter);
        env.add_filter("slice_chars", slice_chars_filter);
        env.render_str(Self::DEFAULT_TEMPLATE, template_ctx)
            .unwrap_or_else(|err| format!("[Error rendering history: {}]", err))
    }
}

#[derive(serde::Serialize)]
struct RenderContext {
    max_output_chars: i64,
}

#[derive(serde::Serialize)]
struct TemplateContext<'a> {
    value: &'a baml_bridge::baml_types::BamlValue,
    ctx: &'a minijinja::Value,
}

fn format_count_filter(value: i64) -> String {
    if value <= 0 {
        return "0".to_string();
    }
    format_count(value as usize)
}

fn slice_chars_filter(value: String, length: i64) -> String {
    if length <= 0 {
        return String::new();
    }
    value.chars().take(length as usize).collect()
}


impl DefaultJinjaRender for REPLHistory {
    const DEFAULT_TEMPLATE: &'static str = r#"
{%- if value.entries | length == 0 -%}
You have not interacted with the REPL environment yet.
{%- else %}
{%- for entry in value.entries %}
=== Step {{ loop.index }} ===
{% if entry.reasoning %}
Reasoning: {{ entry.reasoning }}
{% endif %}
Code:
```python
{{ entry.code }}
```
{% set output_len = entry.output | length %}
Output ({{ output_len | format_count }} chars):
{% if output_len > ctx.max_output_chars %}
{{ entry.output | slice_chars(ctx.max_output_chars) }}
... (truncated to {{ ctx.max_output_chars | format_count }}/{{ output_len | format_count }} chars)
{% else %}
{{ entry.output }}
{% endif %}
{% endfor -%}
{%- endif -%}
"#;
}

/// Truncate text with a marker if it exceeds max_chars.
#[cfg(test)]
fn truncate_output(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!(
        "{truncated}\n... (truncated to {max_chars}/{total} chars)",
        max_chars = format_count(max_chars),
        total = format_count(total)
    )
}

fn format_count(value: usize) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (idx, ch) in digits.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    formatted.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use baml_bridge::baml_types::BamlValue;
    use baml_bridge::ToBamlValue;

    // ==================== REPLEntry Tests ====================

    #[test]
    fn repl_entry_new_has_empty_reasoning() {
        let entry = REPLEntry::new("code".into(), "output".into());
        assert_eq!(entry.reasoning, "");
        assert_eq!(entry.code, "code");
        assert_eq!(entry.output, "output");
    }

    #[test]
    fn repl_entry_with_reasoning_stores_all_fields() {
        let entry = REPLEntry::with_reasoning(
            "thinking".into(),
            "print(1)".into(),
            "1".into(),
        );
        assert_eq!(entry.reasoning, "thinking");
        assert_eq!(entry.code, "print(1)");
        assert_eq!(entry.output, "1");
    }

    #[test]
    fn repl_entry_to_baml_value_structure() {
        let entry = REPLEntry::with_reasoning(
            "reason".into(),
            "x = 42".into(),
            "done".into(),
        );
        let value = entry.to_baml_value();

        match value {
            BamlValue::Class(name, fields) => {
                assert!(name.ends_with("REPLEntry"));
                assert_eq!(fields.get("reasoning"), Some(&BamlValue::String("reason".into())));
                assert_eq!(fields.get("code"), Some(&BamlValue::String("x = 42".into())));
                assert_eq!(fields.get("output"), Some(&BamlValue::String("done".into())));
            }
            _ => panic!("Expected BamlValue::Class, got {:?}", value),
        }
    }

    // ==================== REPLHistory Immutability Tests ====================

    #[test]
    fn repl_history_new_is_empty() {
        let history = REPLHistory::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.entries().len(), 0);
    }

    #[test]
    fn repl_history_append_is_immutable() {
        let h1 = REPLHistory::new();
        let h2 = h1.append("a".into(), "1".into());
        let h3 = h2.append("b".into(), "2".into());

        // Original histories unchanged
        assert!(h1.is_empty());
        assert_eq!(h2.len(), 1);
        assert_eq!(h3.len(), 2);

        // Each has correct entries
        assert_eq!(h2.entries()[0].code, "a");
        assert_eq!(h3.entries()[0].code, "a");
        assert_eq!(h3.entries()[1].code, "b");
    }

    #[test]
    fn repl_history_append_with_reasoning_is_immutable() {
        let h1 = REPLHistory::new();
        let h2 = h1.append_with_reasoning("r1".into(), "a".into(), "1".into());
        let h3 = h2.append_with_reasoning("r2".into(), "b".into(), "2".into());

        assert!(h1.is_empty());
        assert_eq!(h2.len(), 1);
        assert_eq!(h3.len(), 2);
        assert_eq!(h3.entries()[0].reasoning, "r1");
        assert_eq!(h3.entries()[1].reasoning, "r2");
    }

    #[test]
    fn repl_history_with_max_output_chars_preserves_setting() {
        let history = REPLHistory::with_max_output_chars(100)
            .append("x".into(), "y".into())
            .append("z".into(), "w".into());

        assert_eq!(history.max_output_chars(), 100);
    }

    #[test]
    fn repl_history_default_max_output_chars() {
        let history = REPLHistory::new();
        assert_eq!(history.max_output_chars(), REPLHistory::DEFAULT_MAX_OUTPUT_CHARS);
        assert_eq!(history.max_output_chars(), 5000);
    }

    // ==================== REPLHistory ToBamlValue Tests ====================

    #[test]
    fn repl_history_to_baml_value_empty() {
        let history = REPLHistory::new();
        let value = history.to_baml_value();

        match value {
            BamlValue::Class(name, fields) => {
                assert!(name.ends_with("REPLHistory"));
                match fields.get("entries") {
                    Some(BamlValue::List(entries)) => assert!(entries.is_empty()),
                    _ => panic!("Expected empty list for entries"),
                }
            }
            _ => panic!("Expected BamlValue::Class"),
        }
    }

    #[test]
    fn repl_history_to_baml_value_with_entries() {
        let history = REPLHistory::new()
            .append("code1".into(), "out1".into())
            .append_with_reasoning("reason".into(), "code2".into(), "out2".into());

        let value = history.to_baml_value();

        match value {
            BamlValue::Class(name, fields) => {
                assert!(name.ends_with("REPLHistory"));

                let entries = match fields.get("entries") {
                    Some(BamlValue::List(e)) => e,
                    _ => panic!("Expected list for entries"),
                };
                assert_eq!(entries.len(), 2);

                // Verify first entry
                if let BamlValue::Class(_, e1) = &entries[0] {
                    assert_eq!(e1.get("code"), Some(&BamlValue::String("code1".into())));
                    assert_eq!(e1.get("reasoning"), Some(&BamlValue::String("".into())));
                }

                // Verify second entry has reasoning
                if let BamlValue::Class(_, e2) = &entries[1] {
                    assert_eq!(e2.get("code"), Some(&BamlValue::String("code2".into())));
                    assert_eq!(e2.get("reasoning"), Some(&BamlValue::String("reason".into())));
                }
            }
            _ => panic!("Expected BamlValue::Class"),
        }
    }

    // ==================== Rendering Tests ====================

    #[test]
    fn render_empty_history_message() {
        let history = REPLHistory::new();
        let rendered = history.render();

        assert_eq!(
            rendered.trim(),
            "You have not interacted with the REPL environment yet."
        );
    }

    #[test]
    fn render_single_entry_format() {
        let history = REPLHistory::new()
            .append("print('hello')".into(), "hello".into());
        let rendered = history.render();

        // Check structure
        assert!(rendered.contains("=== Step 1 ==="), "Missing step header");
        assert!(rendered.contains("Code:"), "Missing Code label");
        assert!(rendered.contains("```python"), "Missing python fence");
        assert!(rendered.contains("print('hello')"), "Missing code content");
        assert!(rendered.contains("```"), "Missing closing fence");
        assert!(rendered.contains("Output (5 chars):"), "Missing output header with char count");
        assert!(rendered.contains("hello"), "Missing output content");

        // Should NOT contain reasoning line when empty
        assert!(!rendered.contains("Reasoning:"), "Should not show empty reasoning");
    }

    #[test]
    fn render_entry_with_reasoning_shows_reasoning() {
        let history = REPLHistory::new()
            .append_with_reasoning(
                "Let me calculate the sum".into(),
                "1 + 1".into(),
                "2".into(),
            );
        let rendered = history.render();

        assert!(rendered.contains("Reasoning: Let me calculate the sum"));
    }

    #[test]
    fn render_multiple_entries_numbered_correctly() {
        let history = REPLHistory::new()
            .append("a = 1".into(), "".into())
            .append("b = 2".into(), "".into())
            .append("c = 3".into(), "".into())
            .append("d = 4".into(), "".into())
            .append("e = 5".into(), "".into());

        let rendered = history.render();

        assert!(rendered.contains("=== Step 1 ==="), "Missing Step 1");
        assert!(rendered.contains("=== Step 2 ==="), "Missing Step 2");
        assert!(rendered.contains("=== Step 3 ==="), "Missing Step 3");
        assert!(rendered.contains("=== Step 4 ==="), "Missing Step 4");
        assert!(rendered.contains("=== Step 5 ==="), "Missing Step 5");

        // Verify order: Step 1 should come before Step 2, etc.
        let pos1 = rendered.find("=== Step 1 ===").unwrap();
        let pos2 = rendered.find("=== Step 2 ===").unwrap();
        let pos3 = rendered.find("=== Step 3 ===").unwrap();
        assert!(pos1 < pos2, "Step 1 should come before Step 2");
        assert!(pos2 < pos3, "Step 2 should come before Step 3");
    }

    #[test]
    fn render_output_char_count_accurate() {
        let history = REPLHistory::new()
            .append("x".into(), "abc".into()); // 3 chars
        let rendered = history.render();
        assert!(rendered.contains("Output (3 chars):"));

        let history2 = REPLHistory::new()
            .append("x".into(), "".into()); // 0 chars
        let rendered2 = history2.render();
        assert!(rendered2.contains("Output (0 chars):"));

        // Unicode: "café" = 4 chars
        let history3 = REPLHistory::new()
            .append("x".into(), "café".into());
        let rendered3 = history3.render();
        assert!(rendered3.contains("Output (4 chars):"));
    }

    #[test]
    fn render_truncates_long_output() {
        let long_output = "x".repeat(100);
        let history = REPLHistory::with_max_output_chars(20)
            .append("code".into(), long_output.clone());

        let rendered = history.render();

        // Should NOT contain full output
        assert!(
            !rendered.contains(&long_output),
            "Long output should be truncated"
        );
        // Should contain truncated portion
        assert!(
            rendered.contains(&"x".repeat(20)),
            "Should contain truncated content"
        );
        // Should include truncation marker with counts
        assert!(rendered.contains("... (truncated to 20/100 chars)"));
    }

    #[test]
    fn render_preserves_short_output() {
        let short_output = "short";
        let history = REPLHistory::with_max_output_chars(1000)
            .append("code".into(), short_output.into());

        let rendered = history.render();
        assert!(rendered.contains(short_output));
    }

    #[test]
    fn render_multiline_code_preserved() {
        let code = "def foo():\n    return 42\n\nresult = foo()";
        let history = REPLHistory::new()
            .append(code.into(), "42".into());

        let rendered = history.render();
        assert!(rendered.contains("def foo():"));
        assert!(rendered.contains("    return 42"));
        assert!(rendered.contains("result = foo()"));
    }

    #[test]
    fn render_multiline_output_preserved() {
        let output = "line1\nline2\nline3";
        let history = REPLHistory::new()
            .append("print('test')".into(), output.into());

        let rendered = history.render();
        assert!(rendered.contains("line1\nline2\nline3"));
    }

    // ==================== Edge Cases ====================

    #[test]
    fn empty_code_renders() {
        let history = REPLHistory::new()
            .append("".into(), "output".into());
        let rendered = history.render();

        assert!(rendered.contains("```python\n\n```"));
    }

    #[test]
    fn empty_output_renders() {
        let history = REPLHistory::new()
            .append("pass".into(), "".into());
        let rendered = history.render();

        assert!(rendered.contains("Output (0 chars):"));
    }

    #[test]
    fn empty_reasoning_not_shown() {
        let history = REPLHistory::new()
            .append_with_reasoning("".into(), "x".into(), "y".into());
        let rendered = history.render();

        // Empty reasoning should not produce "Reasoning:" line
        assert!(!rendered.contains("Reasoning:"));
    }

    #[test]
    fn special_characters_in_code() {
        let code = r#"print("hello \"world\"")"#;
        let history = REPLHistory::new()
            .append(code.into(), "hello \"world\"".into());

        let rendered = history.render();
        assert!(rendered.contains(code));
    }

    #[test]
    fn unicode_in_all_fields() {
        let history = REPLHistory::new()
            .append_with_reasoning(
                "计算 café 的长度".into(),
                "len('日本語')".into(),
                "3".into(),
            );

        let rendered = history.render();
        assert!(rendered.contains("计算 café 的长度"));
        assert!(rendered.contains("len('日本語')"));
        assert!(rendered.contains("3"));
    }

    // ==================== Golden Test: Exact Format ====================

    #[test]
    fn render_format_golden_test() {
        // This test locks down the exact format to prevent regressions
        let history = REPLHistory::with_max_output_chars(5000)
            .append_with_reasoning(
                "First, set up variables".into(),
                "x = 1\ny = 2".into(),
                "".into(),
            )
            .append(
                "result = x + y\nprint(result)".into(),
                "3".into(),
            );

        let rendered = history.render();

        // Verify exact structure (allowing for whitespace variations)
        assert!(rendered.contains("=== Step 1 ==="));
        assert!(rendered.contains("Reasoning: First, set up variables"));
        assert!(rendered.contains("Code:\n```python\nx = 1\ny = 2\n```"));
        assert!(rendered.contains("Output (0 chars):"));

        assert!(rendered.contains("=== Step 2 ==="));
        assert!(rendered.contains("Code:\n```python\nresult = x + y\nprint(result)\n```"));
        assert!(rendered.contains("Output (1 chars):"));
        assert!(rendered.contains("\n3\n"));
    }

    // ==================== Helper Function Tests ====================

    #[test]
    fn format_count_edge_cases() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(1), "1");
        assert_eq!(format_count(12), "12");
        assert_eq!(format_count(123), "123");
        assert_eq!(format_count(1234), "1,234");
        assert_eq!(format_count(12345), "12,345");
        assert_eq!(format_count(123456), "123,456");
        assert_eq!(format_count(1234567), "1,234,567");
        assert_eq!(format_count(1000000), "1,000,000");
    }

    #[test]
    fn truncate_output_edge_cases() {
        // Zero max returns empty
        assert_eq!(truncate_output("hello", 0), "");

        // Exact length preserved
        assert_eq!(truncate_output("hello", 5), "hello");

        // One over gets truncated
        let result = truncate_output("hello!", 5);
        assert!(result.starts_with("hello"));
        assert!(result.contains("truncated"));
    }

    #[test]
    fn truncate_output_unicode_boundary() {
        // "héllo" = 5 chars, truncate at 3 should give "hél"
        let result = truncate_output("héllo", 3);
        assert!(result.starts_with("hél"));
        assert!(result.contains("truncated to 3/5 chars"));
    }

    // ==================== Template Format Tests ====================

    #[test]
    fn template_has_no_leading_whitespace_on_first_content_line() {
        // The template should start with `{%-` not `  {%-`
        // This ensures the raw string literal doesn't have unintended indentation
        let template = <REPLHistory as DefaultJinjaRender>::DEFAULT_TEMPLATE;
        let first_content_line = template
            .lines()
            .find(|line| !line.trim().is_empty())
            .expect("Template should have at least one non-empty line");

        let first_char = first_content_line.chars().next().unwrap();
        assert!(
            !first_char.is_whitespace(),
            "First non-empty line should not start with whitespace. Got: {:?}",
            first_content_line
        );
    }

    #[test]
    fn render_output_has_no_leading_whitespace() {
        // Verify rendered output doesn't start with unintended spaces
        let history = REPLHistory::new();
        let rendered = history.render();

        // The very first non-empty line should start without leading whitespace
        let first_content_line = rendered
            .lines()
            .find(|line| !line.trim().is_empty())
            .expect("Rendered output should have content");

        let first_char = first_content_line.chars().next().unwrap();
        assert!(
            !first_char.is_whitespace(),
            "Rendered output should not start with whitespace. Got: {:?}",
            first_content_line
        );
    }
}
