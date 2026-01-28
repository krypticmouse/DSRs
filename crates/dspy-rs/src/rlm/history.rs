#![cfg(feature = "rlm")]

#[derive(Debug, Clone)]
pub(crate) struct ReplHistoryEntry {
    code: String,
    output: String,
    reasoning: Option<String>,
}

impl ReplHistoryEntry {
    pub(crate) fn new(code: String, output: String) -> Self {
        Self {
            code,
            output,
            reasoning: None,
        }
    }

    pub(crate) fn with_reasoning(mut self, reasoning: String) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
}

pub(crate) fn render_history(entries: &[ReplHistoryEntry], max_output_chars: usize) -> String {
    let mut output = String::new();
    for (idx, entry) in entries.iter().enumerate() {
        let output_len = entry.output.chars().count();
        let truncated_output = truncate_history_output(&entry.output, max_output_chars);
        output.push_str(&format!("=== Step {} ===\n", idx + 1));
        if let Some(reasoning) = entry.reasoning.as_ref() {
            let reasoning = reasoning.trim();
            if !reasoning.is_empty() {
                output.push_str("Reasoning: ");
                output.push_str(reasoning);
                output.push('\n');
            }
        }
        output.push_str("Code:\n```python\n");
        output.push_str(&entry.code);
        output.push_str("\n```\n");
        output.push_str(&format!(
            "Output ({} chars):\n",
            format_count(output_len)
        ));
        output.push_str(&truncated_output);
        output.push_str("\n\n");
    }
    output.trim_end().to_string()
}

fn truncate_history_output(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_string();
    }
    let truncated: String = text.chars().take(max_chars).collect();
    format!(
        "{truncated}\n... (truncated to {}/{total} chars)",
        max_chars,
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

    #[test]
    fn render_history_truncates_with_marker() {
        let entry = ReplHistoryEntry::new("x = 1".to_string(), "a".repeat(10));
        let rendered = render_history(&[entry], 4);

        assert!(rendered.contains("=== Step 1 ==="));
        assert!(rendered.contains("Output (10 chars):"));
        assert!(rendered.contains("aaaa\n... (truncated to 4/10 chars)"));
    }
}
