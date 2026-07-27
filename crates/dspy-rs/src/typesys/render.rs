//! Prompt rendering for the in-house type model, replacing BAML's jinja
//! `OutputFormatContent::render` and the adapter's backtick-token pipeline.
//!
//! Two entry points:
//! - [`type_name`] — the short inline type label (`string`, `int`, `string[]`, `Citation[]`,
//!   `string or null`) shown in field descriptions and the `should be of type:` line.
//! - [`schema_block`] — the expanded block for structured types (class field layouts, enum
//!   value lists). Primitive types return their [`type_name`] so the adapter can skip the block.

use super::schema::{ClassDef, FieldType, OutputSchema};

/// Renders the short, inline type label for a field.
///
/// `schema` resolves class/enum internal names to their rendered names; pass `None` when
/// no registry is available (input fields), in which case the last `::` path segment is used.
pub fn type_name(field_type: &FieldType, schema: Option<&OutputSchema>) -> String {
    match field_type {
        FieldType::String => "string".to_string(),
        FieldType::Int => "int".to_string(),
        FieldType::Float => "float".to_string(),
        FieldType::Bool => "bool".to_string(),
        FieldType::Literal(value) => format!("\"{value}\""),
        FieldType::List(inner) => format!("{}[]", type_name(inner, schema)),
        FieldType::Optional(inner) => format!("{} or null", type_name(inner, schema)),
        FieldType::Map(key, value) => {
            format!("map<{}, {}>", type_name(key, schema), type_name(value, schema))
        }
        FieldType::Class(name) | FieldType::Enum(name) => resolve_name(name, schema),
        FieldType::Union(items) => items
            .iter()
            .map(|item| type_name(item, schema))
            .collect::<Vec<_>>()
            .join(" or "),
    }
}

fn resolve_name(token: &str, schema: Option<&OutputSchema>) -> String {
    match schema {
        Some(schema) => schema.rendered_name(token),
        None => token.rsplit("::").next().unwrap_or(token).to_string(),
    }
}

/// Renders the expanded schema block for a field type.
///
/// For primitive/optional-primitive/map types this returns [`type_name`] (the adapter then
/// skips emitting a redundant block). For classes, enums, and lists thereof it renders a
/// structured, indented block that names each field/value with doc comments.
pub fn schema_block(field_type: &FieldType, schema: &OutputSchema) -> String {
    match field_type {
        FieldType::Class(name) => schema
            .classes
            .get(name)
            .map(|class| render_class(class, schema))
            .unwrap_or_else(|| type_name(field_type, Some(schema))),
        FieldType::Enum(name) => schema
            .enums
            .get(name)
            .map(render_enum)
            .unwrap_or_else(|| type_name(field_type, Some(schema))),
        FieldType::List(inner) => match inner.as_ref() {
            FieldType::Class(_) | FieldType::Enum(_) => {
                let inner_block = schema_block(inner, schema);
                format!("[\n{}\n]", indent(&inner_block, 2))
            }
            _ => type_name(field_type, Some(schema)),
        },
        FieldType::Optional(inner) => match inner.as_ref() {
            FieldType::Class(_) | FieldType::Enum(_) | FieldType::List(_) => {
                schema_block(inner, schema)
            }
            _ => type_name(field_type, Some(schema)),
        },
        _ => type_name(field_type, Some(schema)),
    }
}

fn render_class(class: &ClassDef, schema: &OutputSchema) -> String {
    let mut lines = vec!["{".to_string()];
    for field in &class.fields {
        if let Some(docs) = &field.docs {
            for line in docs.lines() {
                lines.push(format!("  // {}", line.trim()));
            }
        }
        let rendered_type = match field.field_type.peel() {
            FieldType::Class(_) | FieldType::Enum(_) => {
                // Inline nested structured types one level so the block is self-describing.
                let nested = schema_block(&field.field_type, schema);
                indent_inline(&nested)
            }
            _ => type_name(&field.field_type, Some(schema)),
        };
        lines.push(format!("  {}: {},", field.rendered_name, rendered_type));
    }
    lines.push("}".to_string());
    lines.join("\n")
}

fn render_enum(enm: &super::schema::EnumDef) -> String {
    let mut lines = vec!["one of:".to_string()];
    for value in &enm.values {
        if let Some(docs) = &value.docs {
            for line in docs.lines() {
                lines.push(format!("// {}", line.trim()));
            }
        }
        lines.push(format!("- {}", value.rendered_name));
    }
    lines.join("\n")
}

fn indent(text: &str, spaces: usize) -> String {
    let pad = " ".repeat(spaces);
    text.lines()
        .map(|line| {
            if line.is_empty() {
                line.to_string()
            } else {
                format!("{pad}{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Indents every line except the first by two spaces, so a nested block reads correctly
/// when placed after a `field:` prefix.
fn indent_inline(text: &str) -> String {
    let mut out = String::new();
    for (idx, line) in text.lines().enumerate() {
        if idx == 0 {
            out.push_str(line);
        } else {
            out.push('\n');
            out.push_str("  ");
            out.push_str(line);
        }
    }
    out
}
