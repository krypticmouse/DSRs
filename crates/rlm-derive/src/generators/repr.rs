//! `__repr__` generation for RLM types.
//!
//! Supports template strings with placeholders:
//! - `{self.field}` - Simple field access
//! - `{self.field:N}` - Truncated to N characters
//! - `{len(self.field)}` - Length of field (for Vec, String, etc.)
//!
//! Auto-generates repr for types without explicit template.

use proc_macro2::TokenStream;
use quote::quote;

use crate::attrs::RlmTypeAttrs;

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

/// Default truncation length for auto-generated repr.
pub const AUTO_REPR_TRUNCATE_LEN: usize = 40;

/// Generate auto-repr for a struct without an explicit template.
///
/// Strategy:
/// 1. Find the first String field and use it as a preview
/// 2. If no String field, show struct name with field count
/// 3. Truncate the preview to 40 characters
///
/// # Example output
///
/// - `Trajectory("session_abc123... (truncated)")` - first string field preview
/// - `Trajectory(3 fields)` - fallback when no string fields
pub fn generate_auto_repr(attrs: &RlmTypeAttrs) -> TokenStream {
    let struct_name = &attrs.ident;
    let struct_name_str = struct_name.to_string();

    // Find first String field for preview
    let string_field = attrs
        .fields()
        .find(|f| is_string_type(&f.ty));

    if let Some(field) = string_field {
        let field_name = field.ident.as_ref().expect("named field required");
        let max_len = AUTO_REPR_TRUNCATE_LEN;

        quote! {
            fn __repr__(&self) -> String {
                let preview = &self.#field_name;
                let chars: Vec<char> = preview.chars().collect();
                if chars.len() > #max_len {
                    let truncated: String = chars[..#max_len].iter().collect();
                    format!("{}(\"{}...\")", #struct_name_str, truncated)
                } else {
                    format!("{}(\"{}\")", #struct_name_str, preview)
                }
            }
        }
    } else {
        // Fallback: show field count
        let field_count = attrs.fields().count();

        quote! {
            fn __repr__(&self) -> String {
                format!("{}({} fields)", #struct_name_str, #field_count)
            }
        }
    }
}

/// Generate the `__repr__` method for a struct.
///
/// Uses the explicit template if provided via `#[rlm(repr = "...")]`,
/// otherwise generates an auto-repr.
pub fn generate_repr(attrs: &RlmTypeAttrs) -> Result<TokenStream, ParseError> {
    if let Some(ref template) = attrs.repr {
        generate_repr_from_template(template)
    } else {
        Ok(generate_auto_repr(attrs))
    }
}

/// Check if a type is `String`.
fn is_string_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "String";
        }
    }
    false
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

    #[test]
    fn test_is_string_type() {
        let string_ty: syn::Type = syn::parse_quote!(String);
        assert!(is_string_type(&string_ty));

        let i32_ty: syn::Type = syn::parse_quote!(i32);
        assert!(!is_string_type(&i32_ty));

        let vec_ty: syn::Type = syn::parse_quote!(Vec<String>);
        assert!(!is_string_type(&vec_ty));
    }
}
