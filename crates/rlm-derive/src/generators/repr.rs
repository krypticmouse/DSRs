//! `__repr__` generation for RLM types.
//!
//! Supports template strings with placeholders:
//! - `{self.field}` - Simple field access
//! - `{self.field:N}` - Truncated to N characters
//! - `{len(self.field)}` - Length of field (for Vec, String, etc.)
//!
//! Auto-generates repr for types without explicit template.

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

/// Parsed template segment for `__repr__` generation.
#[derive(Debug, Clone, PartialEq)]
pub enum ReprSegment {
    /// Literal text, output as-is.
    Literal(String),
    /// Field access: `{self.field}`
    Field {
        field_name: String,
    },
    /// Truncated field: `{self.field:N}`
    TruncatedField {
        field_name: String,
        max_chars: usize,
    },
    /// Length of field: `{len(self.field)}`
    FieldLen {
        field_name: String,
    },
}

/// Errors that can occur during template parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Unclosed brace in template.
    UnclosedBrace { position: usize },
    /// Invalid placeholder syntax.
    InvalidPlaceholder { content: String, reason: String },
    /// Invalid truncation specifier.
    InvalidTruncation { content: String },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnclosedBrace { position } => {
                write!(f, "unclosed '{{' at position {}", position)
            }
            ParseError::InvalidPlaceholder { content, reason } => {
                write!(f, "invalid placeholder '{{{}}}': {}", content, reason)
            }
            ParseError::InvalidTruncation { content } => {
                write!(f, "invalid truncation specifier in '{{{}}}'", content)
            }
        }
    }
}

/// Parse a repr template string into segments.
///
/// # Syntax
///
/// - Literal text is preserved as-is
/// - `{self.field}` becomes a field access
/// - `{self.field:N}` truncates to N characters
/// - `{len(self.field)}` gets the length
///
/// # Examples
///
/// ```ignore
/// let segments = parse_template("Trajectory({len(self.steps)} steps)")?;
/// assert_eq!(segments, vec![
///     ReprSegment::Literal("Trajectory(".into()),
///     ReprSegment::FieldLen { field_name: "steps".into() },
///     ReprSegment::Literal(" steps)".into()),
/// ]);
/// ```
pub fn parse_template(template: &str) -> Result<Vec<ReprSegment>, ParseError> {
    let mut segments = Vec::new();
    let mut chars = template.char_indices().peekable();
    let mut literal_start = 0;

    while let Some((i, c)) = chars.next() {
        if c == '{' {
            // Check for escape: {{
            if chars.peek().map(|(_, c)| *c) == Some('{') {
                // Flush literal up to first {
                if i > literal_start {
                    segments.push(ReprSegment::Literal(template[literal_start..i].to_string()));
                }
                chars.next(); // consume second {
                segments.push(ReprSegment::Literal("{".to_string()));
                literal_start = chars.peek().map(|(i, _)| *i).unwrap_or(template.len());
                continue;
            }

            // Find closing brace
            let brace_start = i;
            let mut brace_end = None;
            let mut depth = 1;

            for (j, ch) in chars.by_ref() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            brace_end = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            let brace_end = brace_end.ok_or(ParseError::UnclosedBrace { position: brace_start })?;

            // Flush literal before placeholder
            if brace_start > literal_start {
                segments.push(ReprSegment::Literal(
                    template[literal_start..brace_start].to_string(),
                ));
            }

            // Parse placeholder content
            let content = &template[brace_start + 1..brace_end];
            let segment = parse_placeholder(content)?;
            segments.push(segment);

            literal_start = brace_end + 1;
        } else if c == '}' {
            // Check for escape: }}
            if chars.peek().map(|(_, c)| *c) == Some('}') {
                // Flush literal up to first }
                if i > literal_start {
                    segments.push(ReprSegment::Literal(template[literal_start..i].to_string()));
                }
                chars.next(); // consume second }
                segments.push(ReprSegment::Literal("}".to_string()));
                literal_start = chars.peek().map(|(i, _)| *i).unwrap_or(template.len());
            }
            // Single } without matching { is just literal text
        }
    }

    // Flush remaining literal
    if literal_start < template.len() {
        segments.push(ReprSegment::Literal(template[literal_start..].to_string()));
    }

    Ok(segments)
}

/// Parse a single placeholder (content between { and }).
fn parse_placeholder(content: &str) -> Result<ReprSegment, ParseError> {
    let content = content.trim();

    // Check for len(self.field)
    if content.starts_with("len(") && content.ends_with(")") {
        let inner = &content[4..content.len() - 1].trim();
        let field_name = parse_self_field(inner)?;
        return Ok(ReprSegment::FieldLen { field_name });
    }

    // Check for self.field or self.field:N
    if content.contains(':') {
        // Truncated field
        let parts: Vec<&str> = content.splitn(2, ':').collect();
        let field_name = parse_self_field(parts[0].trim())?;
        let max_chars: usize = parts[1]
            .trim()
            .parse()
            .map_err(|_| ParseError::InvalidTruncation {
                content: content.to_string(),
            })?;
        Ok(ReprSegment::TruncatedField {
            field_name,
            max_chars,
        })
    } else {
        // Simple field access
        let field_name = parse_self_field(content)?;
        Ok(ReprSegment::Field { field_name })
    }
}

/// Parse `self.field_name` and extract the field name.
fn parse_self_field(s: &str) -> Result<String, ParseError> {
    let s = s.trim();
    if let Some(field) = s.strip_prefix("self.") {
        if field.is_empty() {
            return Err(ParseError::InvalidPlaceholder {
                content: s.to_string(),
                reason: "field name cannot be empty".to_string(),
            });
        }
        // Validate field name (simple check for now)
        if field.chars().all(|c| c.is_alphanumeric() || c == '_') {
            Ok(field.to_string())
        } else {
            Err(ParseError::InvalidPlaceholder {
                content: s.to_string(),
                reason: "invalid field name characters".to_string(),
            })
        }
    } else {
        Err(ParseError::InvalidPlaceholder {
            content: s.to_string(),
            reason: "expected 'self.' prefix".to_string(),
        })
    }
}

/// Generate code for a single repr segment.
fn generate_segment_code(segment: &ReprSegment) -> TokenStream {
    match segment {
        ReprSegment::Literal(text) => {
            quote! { result.push_str(#text); }
        }
        ReprSegment::Field { field_name } => {
            let field_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
            quote! {
                result.push_str(&format!("{}", self.#field_ident));
            }
        }
        ReprSegment::TruncatedField {
            field_name,
            max_chars,
        } => {
            let field_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
            quote! {
                let field_str = format!("{}", self.#field_ident);
                if field_str.len() > #max_chars {
                    result.push_str(&field_str[..#max_chars]);
                    result.push_str("...");
                } else {
                    result.push_str(&field_str);
                }
            }
        }
        ReprSegment::FieldLen { field_name } => {
            let field_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
            quote! {
                result.push_str(&format!("{}", self.#field_ident.len()));
            }
        }
    }
}

/// Generate `__repr__` method code from parsed segments.
pub fn generate_repr_from_segments(segments: &[ReprSegment]) -> TokenStream {
    let segment_code: Vec<TokenStream> = segments.iter().map(generate_segment_code).collect();

    quote! {
        fn __repr__(&self) -> String {
            let mut result = String::new();
            #(#segment_code)*
            result
        }
    }
}

/// Generate `__repr__` method code from a template string.
///
/// Returns the generated code and any parse errors as a compile error.
pub fn generate_repr_from_template(template: &str) -> Result<TokenStream, ParseError> {
    let segments = parse_template(template)?;
    Ok(generate_repr_from_segments(&segments))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_literal_only() {
        let segments = parse_template("Hello, world!").unwrap();
        assert_eq!(segments, vec![ReprSegment::Literal("Hello, world!".into())]);
    }

    #[test]
    fn test_parse_simple_field() {
        let segments = parse_template("{self.name}").unwrap();
        assert_eq!(
            segments,
            vec![ReprSegment::Field {
                field_name: "name".into()
            }]
        );
    }

    #[test]
    fn test_parse_truncated_field() {
        let segments = parse_template("{self.description:50}").unwrap();
        assert_eq!(
            segments,
            vec![ReprSegment::TruncatedField {
                field_name: "description".into(),
                max_chars: 50,
            }]
        );
    }

    #[test]
    fn test_parse_field_len() {
        let segments = parse_template("{len(self.items)}").unwrap();
        assert_eq!(
            segments,
            vec![ReprSegment::FieldLen {
                field_name: "items".into()
            }]
        );
    }

    #[test]
    fn test_parse_complex_template() {
        let template = "Trajectory({len(self.steps)} steps, session={self.session_id:12}...)";
        let segments = parse_template(template).unwrap();
        assert_eq!(
            segments,
            vec![
                ReprSegment::Literal("Trajectory(".into()),
                ReprSegment::FieldLen {
                    field_name: "steps".into()
                },
                ReprSegment::Literal(" steps, session=".into()),
                ReprSegment::TruncatedField {
                    field_name: "session_id".into(),
                    max_chars: 12,
                },
                ReprSegment::Literal("...)".into()),
            ]
        );
    }

    #[test]
    fn test_parse_escaped_braces() {
        let segments = parse_template("{{literal}}").unwrap();
        assert_eq!(
            segments,
            vec![
                ReprSegment::Literal("{".into()),
                ReprSegment::Literal("literal".into()),
                ReprSegment::Literal("}".into()),
            ]
        );
    }

    #[test]
    fn test_unclosed_brace_error() {
        let result = parse_template("Hello {self.name");
        assert!(matches!(result, Err(ParseError::UnclosedBrace { .. })));
    }

    #[test]
    fn test_missing_self_prefix() {
        let result = parse_template("{name}");
        assert!(matches!(result, Err(ParseError::InvalidPlaceholder { .. })));
    }

    #[test]
    fn test_invalid_truncation() {
        let result = parse_template("{self.name:abc}");
        assert!(matches!(result, Err(ParseError::InvalidTruncation { .. })));
    }
}
