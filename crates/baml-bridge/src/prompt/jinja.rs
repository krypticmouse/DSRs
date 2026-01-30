//! Jinja helpers for prompt rendering.

use minijinja::Environment;

/// Register prompt-specific filters.
pub fn register_prompt_filters(env: &mut Environment<'static>) {
    env.add_filter("truncate", filter_truncate);
    env.add_filter("slice_chars", filter_slice_chars);
    env.add_filter("format_count", filter_format_count);
}

fn filter_truncate(s: &str, n: usize) -> String {
    let length = s.chars().count();
    if length <= n {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(n.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

fn filter_slice_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn filter_format_count(n: i64) -> String {
    let sign = if n < 0 { "-" } else { "" };
    let s = n.abs().to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    format!("{sign}{}", result.chars().rev().collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::{filter_format_count, filter_slice_chars, filter_truncate};

    #[test]
    fn truncate_keeps_exact_length() {
        assert_eq!(filter_truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_adds_suffix_when_needed() {
        assert_eq!(filter_truncate("hello world", 8), "hello...");
    }

    #[test]
    fn slice_chars_handles_empty() {
        assert_eq!(filter_slice_chars("", 3), "");
    }

    #[test]
    fn format_count_handles_negative() {
        assert_eq!(filter_format_count(-12345), "-12,345");
    }
}
