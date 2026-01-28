use crate::describe::RlmDescribe;

/// A variable description for inclusion in RLM prompts.
#[derive(Debug, Clone)]
pub struct RlmVariable {
    pub name: String,
    pub type_desc: String,
    pub description: String,
    pub constraints: Vec<String>,
    pub total_length: usize,
    pub preview: String,
    pub properties: Vec<(String, String)>,
}

impl RlmVariable {
    /// Create a variable description from a Rust type implementing `RlmDescribe`.
    pub fn from_rust<T: RlmDescribe>(name: &str, value: &T) -> Self {
        let type_desc = T::describe_type();
        let value_desc = value.describe_value();
        let total_length = value_desc.chars().count();
        let preview = truncate_preview(&value_desc, 500);

        let properties = T::properties()
            .into_iter()
            .map(|prop| (prop.name.to_string(), prop.type_name.to_string()))
            .collect();

        Self {
            name: name.to_string(),
            type_desc,
            description: String::new(),
            constraints: Vec::new(),
            total_length,
            preview,
            properties,
        }
    }

    /// Add a human-readable description.
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Add stringified constraints.
    pub fn with_constraints(mut self, constraints: Vec<String>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Format this variable description for prompt inclusion.
    pub fn format(&self) -> String {
        let mut output = format!("Variable: `{}` (access it in your code)\n", self.name);
        let type_summary = self.type_desc.lines().next().unwrap_or("unknown");
        output.push_str(&format!("Type: {}\n", type_summary));

        if !self.description.is_empty() {
            output.push_str(&format!("Description: {}\n", self.description));
        }
        if !self.constraints.is_empty() {
            output.push_str(&format!("Constraints: {}\n", self.constraints.join("; ")));
        }

        output.push_str(&format!(
            "Total length: {} characters\n",
            self.total_length
        ));

        if !self.properties.is_empty() {
            output.push_str("Properties:\n");
            for (name, ret_type) in &self.properties {
                output.push_str(&format!("  .{} -> {}\n", name, ret_type));
            }
        }

        output.push_str("Preview:\n```\n");
        output.push_str(&self.preview);
        output.push_str("\n```\n");

        output
    }
}

fn truncate_preview(value: &str, max_chars: usize) -> String {
    let total_length = value.chars().count();
    if total_length <= max_chars {
        return value.to_string();
    }

    let mut preview: String = value.chars().take(max_chars).collect();
    preview.push_str("...");
    preview
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::describe::{RlmDescribe, RlmPropertyDesc};

    #[derive(Debug)]
    struct LongValue(String);

    impl RlmDescribe for LongValue {
        fn type_name() -> &'static str {
            "LongValue"
        }

        fn describe_value(&self) -> String {
            self.0.clone()
        }

        fn describe_type() -> String {
            "LongValue - long string".to_string()
        }
    }

    #[derive(Debug)]
    struct Example {
        label: String,
    }

    impl RlmDescribe for Example {
        fn type_name() -> &'static str {
            "Example"
        }

        fn properties() -> Vec<RlmPropertyDesc> {
            vec![RlmPropertyDesc::new("count", "usize")]
        }

        fn describe_value(&self) -> String {
            format!("Example(label={})", self.label)
        }

        fn describe_type() -> String {
            "Example - test type\nfields:\n  - label: String".to_string()
        }
    }

    #[test]
    fn from_rust_truncates_preview_and_tracks_length() {
        let value = LongValue("x".repeat(600));
        let variable = RlmVariable::from_rust("long", &value);

        assert_eq!(variable.total_length, 600);
        assert!(variable.preview.ends_with("..."));
        assert_eq!(variable.preview.chars().count(), 503);
    }

    #[test]
    fn format_includes_optional_sections_when_present() {
        let value = Example {
            label: "ok".to_string(),
        };
        let variable = RlmVariable::from_rust("example", &value)
            .with_description("Example value")
            .with_constraints(vec!["must be ok".to_string()]);

        let formatted = variable.format();
        assert!(formatted.contains("Description: Example value"));
        assert!(formatted.contains("Constraints: must be ok"));
        assert!(formatted.contains("Properties:"));
        assert!(formatted.contains(".count -> usize"));
    }

    #[test]
    fn format_skips_empty_sections() {
        let value = LongValue("short".to_string());
        let variable = RlmVariable::from_rust("long", &value);
        let formatted = variable.format();

        assert!(!formatted.contains("Description:"));
        assert!(!formatted.contains("Constraints:"));
        assert!(!formatted.contains("Properties:"));
    }
}
