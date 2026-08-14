//! Tolerant coercion of raw LM field text into `serde_json::Value`, guided by [`FieldType`].
//!
//! Replaces BAML's `jsonish` parser with a focused, dependency-light coercer that handles
//! the quirks DSRs actually relies on: markdown code fences, bulleted/numbered lists as
//! arrays, loose bool/number spellings, and JSON objects for nested structs.

use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value};

use super::schema::{FieldType, TypeTable};

/// A non-fatal observation made while coercing a value (e.g. a code fence was stripped).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flag {
    /// A ```` ```lang ```` code fence was stripped before parsing.
    StrippedCodeFence,
    /// A bulleted/numbered list was parsed into an array.
    ParsedListFromText,
    /// A scalar was coerced from a different textual representation.
    CoercedFromString,
    /// Extra text around a JSON object/array was ignored.
    ExtraTextIgnored,
}

impl std::fmt::Display for Flag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Flag::StrippedCodeFence => "stripped code fence",
            Flag::ParsedListFromText => "parsed list from text",
            Flag::CoercedFromString => "coerced from string",
            Flag::ExtraTextIgnored => "extra text ignored",
        };
        f.write_str(s)
    }
}

/// The result of coercing raw text into a typed value.
#[derive(Debug, Clone)]
pub struct Coerced {
    pub value: Value,
    pub flags: Vec<Flag>,
}

/// Coerces `raw` into a `serde_json::Value` matching `field_type`.
pub fn coerce(raw: &str, field_type: &FieldType, schema: &TypeTable) -> Result<Coerced> {
    let mut flags = Vec::new();
    let value = coerce_inner(raw, field_type, schema, &mut flags)?;
    Ok(Coerced { value, flags })
}

fn coerce_inner(
    raw: &str,
    field_type: &FieldType,
    schema: &TypeTable,
    flags: &mut Vec<Flag>,
) -> Result<Value> {
    match field_type {
        FieldType::String => Ok(Value::String(
            raw.trim_end_matches(['\n', '\r']).to_string(),
        )),
        FieldType::Int => coerce_int(raw, flags),
        FieldType::Float => coerce_float(raw, flags),
        FieldType::Bool => coerce_bool(raw, flags),
        FieldType::Literal(expected) => {
            let cleaned = strip_quotes(raw.trim());
            if cleaned == *expected {
                Ok(Value::String(expected.clone()))
            } else {
                bail!("expected literal `{expected}`, got `{cleaned}`")
            }
        }
        FieldType::Optional(inner) => {
            if is_nullish(raw) {
                Ok(Value::Null)
            } else {
                coerce_inner(raw, inner, schema, flags)
            }
        }
        FieldType::List(inner) => coerce_list(raw, inner, schema, flags),
        FieldType::Map(_, value_type) => coerce_map(raw, value_type, schema, flags),
        FieldType::Class(name) => coerce_class(raw, name, schema, flags),
        FieldType::Enum(name) => coerce_enum(raw, name, schema),
        FieldType::Union(items) => {
            let mut last_err = None;
            for item in items {
                match coerce_inner(raw, item, schema, flags) {
                    Ok(value) => return Ok(value),
                    Err(err) => last_err = Some(err),
                }
            }
            Err(last_err.unwrap_or_else(|| anyhow!("empty union")))
        }
    }
}

fn coerce_int(raw: &str, flags: &mut Vec<Flag>) -> Result<Value> {
    let stripped = strip_quotes(raw.trim());
    let trimmed = stripped.trim();
    if let Ok(v) = trimmed.parse::<i64>() {
        return Ok(Value::from(v));
    }

    // Fractions like "8/10" are common LM rating outputs; evaluate and round to nearest.
    if let Some((num, den)) = trimmed.split_once('/')
        && let (Ok(n), Ok(d)) = (num.trim().parse::<f64>(), den.trim().parse::<f64>())
        && d != 0.0
    {
        flags.push(Flag::CoercedFromString);
        return Ok(Value::from((n / d).round() as i64));
    }

    // Thousands separators ("1,000") — only when the token is strictly numeric, so prose
    // like "rate this 8 out of 10" is rejected rather than silently mined for digits.
    if trimmed
        .chars()
        .all(|c| c.is_ascii_digit() || c == ',' || c == '-' || c == '+')
    {
        let cleaned: String = trimmed.chars().filter(|c| *c != ',').collect();
        if let Ok(v) = cleaned.parse::<i64>() {
            flags.push(Flag::CoercedFromString);
            return Ok(Value::from(v));
        }
    }

    // A bare decimal ("8.0", "0.8") rounds to the nearest integer.
    if let Ok(v) = trimmed.parse::<f64>() {
        flags.push(Flag::CoercedFromString);
        return Ok(Value::from(v.round() as i64));
    }

    bail!("could not parse `{trimmed}` as int")
}

fn coerce_float(raw: &str, flags: &mut Vec<Flag>) -> Result<Value> {
    let trimmed = strip_quotes(raw.trim());
    if let Ok(v) = trimmed.parse::<f64>() {
        return serde_json::Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| anyhow!("non-finite float `{trimmed}`"));
    }
    let cleaned: String = trimmed
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == 'e' || *c == 'E')
        .collect();
    if let Ok(v) = cleaned.parse::<f64>() {
        flags.push(Flag::CoercedFromString);
        return serde_json::Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| anyhow!("non-finite float `{cleaned}`"));
    }
    bail!("could not parse `{trimmed}` as float")
}

fn coerce_bool(raw: &str, flags: &mut Vec<Flag>) -> Result<Value> {
    let trimmed = strip_quotes(raw.trim()).to_ascii_lowercase();
    match trimmed.as_str() {
        "true" => Ok(Value::Bool(true)),
        "false" => Ok(Value::Bool(false)),
        "yes" | "y" | "1" => {
            flags.push(Flag::CoercedFromString);
            Ok(Value::Bool(true))
        }
        "no" | "n" | "0" => {
            flags.push(Flag::CoercedFromString);
            Ok(Value::Bool(false))
        }
        other => bail!("could not parse `{other}` as bool"),
    }
}

fn coerce_list(
    raw: &str,
    inner: &FieldType,
    schema: &TypeTable,
    flags: &mut Vec<Flag>,
) -> Result<Value> {
    let cleaned = strip_code_fence(raw, flags);
    let trimmed = cleaned.trim();

    // Prefer a real JSON array when present.
    if trimmed.starts_with('[') {
        if let Some(json) = extract_json(trimmed) {
            if let Value::Array(items) = json {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(coerce_json_value(item, inner, schema, flags)?);
                }
                return Ok(Value::Array(out));
            }
        }
    }

    // Fall back to bulleted / numbered / newline-separated list items.
    let items = split_list_items(trimmed);
    if items.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }
    flags.push(Flag::ParsedListFromText);
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(coerce_inner(&item, inner, schema, flags)?);
    }
    Ok(Value::Array(out))
}

fn coerce_map(
    raw: &str,
    value_type: &FieldType,
    schema: &TypeTable,
    flags: &mut Vec<Flag>,
) -> Result<Value> {
    let cleaned = strip_code_fence(raw, flags);
    let json = extract_json(cleaned.trim())
        .ok_or_else(|| anyhow!("could not parse `{}` as an object", cleaned.trim()))?;
    let Value::Object(obj) = json else {
        bail!("expected object for map, got {cleaned}");
    };
    let mut out = Map::new();
    for (key, value) in obj {
        out.insert(key, coerce_json_value(value, value_type, schema, flags)?);
    }
    Ok(Value::Object(out))
}

fn coerce_class(
    raw: &str,
    class_name: &str,
    schema: &TypeTable,
    flags: &mut Vec<Flag>,
) -> Result<Value> {
    let class = schema
        .classes
        .get(class_name)
        .ok_or_else(|| anyhow!("unknown class `{class_name}`"))?;
    let cleaned = strip_code_fence(raw, flags);
    let json = extract_json(cleaned.trim())
        .ok_or_else(|| anyhow!("could not parse `{}` as an object", cleaned.trim()))?;
    let Value::Object(obj) = json else {
        bail!("expected object for class `{class_name}`, got {cleaned}");
    };

    let mut out = Map::new();
    for field in &class.fields {
        let raw_value = obj
            .get(&field.rendered_name)
            .or_else(|| obj.get(&field.name))
            .cloned();
        match raw_value {
            Some(value) => {
                out.insert(
                    field.name.clone(),
                    coerce_json_value(value, &field.field_type, schema, flags)?,
                );
            }
            None if field.field_type.is_optional() => {
                out.insert(field.name.clone(), Value::Null);
            }
            None => bail!(
                "missing field `{}` for class `{class_name}`",
                field.rendered_name
            ),
        }
    }
    Ok(Value::Object(out))
}

fn coerce_enum(raw: &str, enum_name: &str, schema: &TypeTable) -> Result<Value> {
    let enm = schema
        .enums
        .get(enum_name)
        .ok_or_else(|| anyhow!("unknown enum `{enum_name}`"))?;
    let needle = strip_quotes(raw.trim());
    let needle_lower = needle.to_ascii_lowercase();
    for value in &enm.values {
        if value.rendered_name == needle
            || value.name == needle
            || value.rendered_name.to_ascii_lowercase() == needle_lower
            || value.name.to_ascii_lowercase() == needle_lower
        {
            // serde deserializes unit enum variants from their Rust name.
            return Ok(Value::String(value.name.clone()));
        }
    }
    bail!("`{needle}` is not a valid `{}` variant", enm.rendered_name)
}

/// Coerces an already-parsed JSON value into the target type. Used for list items and
/// nested object fields where we already have structured JSON.
fn coerce_json_value(
    value: Value,
    field_type: &FieldType,
    schema: &TypeTable,
    flags: &mut Vec<Flag>,
) -> Result<Value> {
    match field_type {
        FieldType::String => match value {
            Value::String(s) => Ok(Value::String(s)),
            other => Ok(Value::String(json_scalar_to_string(&other))),
        },
        FieldType::Int | FieldType::Float | FieldType::Bool => {
            if matches!(field_type, FieldType::Int) && value.is_i64() {
                return Ok(value);
            }
            if matches!(field_type, FieldType::Float) && value.is_number() {
                return Ok(value);
            }
            if matches!(field_type, FieldType::Bool) && value.is_boolean() {
                return Ok(value);
            }
            // Fall back to textual coercion.
            match &value {
                Value::String(s) => coerce_inner(s, field_type, schema, flags),
                other => coerce_inner(&json_scalar_to_string(other), field_type, schema, flags),
            }
        }
        FieldType::Optional(inner) => {
            if value.is_null() {
                Ok(Value::Null)
            } else {
                coerce_json_value(value, inner, schema, flags)
            }
        }
        FieldType::List(inner) => {
            let Value::Array(items) = value else {
                bail!("expected array, got {value}");
            };
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(coerce_json_value(item, inner, schema, flags)?);
            }
            Ok(Value::Array(out))
        }
        FieldType::Map(_, value_type) => {
            let Value::Object(obj) = value else {
                bail!("expected object, got {value}");
            };
            let mut out = Map::new();
            for (key, entry) in obj {
                out.insert(key, coerce_json_value(entry, value_type, schema, flags)?);
            }
            Ok(Value::Object(out))
        }
        FieldType::Class(name) => {
            let text = value.to_string();
            coerce_class(&text, name, schema, flags)
        }
        FieldType::Enum(name) => {
            let text = match value {
                Value::String(s) => s,
                other => json_scalar_to_string(&other),
            };
            coerce_enum(&text, name, schema)
        }
        FieldType::Literal(expected) => {
            let text = match &value {
                Value::String(s) => s.clone(),
                other => json_scalar_to_string(other),
            };
            if text == *expected {
                Ok(Value::String(expected.clone()))
            } else {
                bail!("expected literal `{expected}`, got `{text}`")
            }
        }
        FieldType::Union(items) => {
            let mut last_err = None;
            for item in items {
                match coerce_json_value(value.clone(), item, schema, flags) {
                    Ok(v) => return Ok(v),
                    Err(err) => last_err = Some(err),
                }
            }
            Err(last_err.unwrap_or_else(|| anyhow!("empty union")))
        }
    }
}

fn json_scalar_to_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn is_nullish(raw: &str) -> bool {
    let trimmed = strip_quotes(raw.trim()).to_ascii_lowercase();
    trimmed.is_empty()
        || trimmed == "null"
        || trimmed == "none"
        || trimmed == "~"
        || trimmed == "nil"
}

fn strip_quotes(text: &str) -> String {
    let bytes = text.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return text[1..text.len() - 1].to_string();
        }
    }
    text.to_string()
}

/// Strips a leading/trailing markdown code fence (``` or ```lang) if present.
fn strip_code_fence(raw: &str, flags: &mut Vec<Flag>) -> String {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // Drop the optional language tag on the first line.
        let after_lang = match rest.find('\n') {
            Some(idx) => &rest[idx + 1..],
            None => rest,
        };
        let body = after_lang.strip_suffix("```").unwrap_or(after_lang);
        flags.push(Flag::StrippedCodeFence);
        return body.trim().to_string();
    }
    trimmed.to_string()
}

/// Extracts the first balanced JSON object/array from `text` and parses it, tolerating
/// surrounding prose.
fn extract_json(text: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(text) {
        return Some(value);
    }
    let open = text.find(['{', '['])?;
    let open_char = text.as_bytes()[open] as char;
    let close_char = if open_char == '{' { '}' } else { ']' };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in text[open..].char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            c if c == open_char => depth += 1,
            c if c == close_char => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &text[open..open + idx + ch.len_utf8()];
                    return serde_json::from_str::<Value>(candidate).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Splits free-form text into list items: bulleted (`-`, `*`, `+`), numbered (`1.`), or
/// one-per-line. Falls back to comma separation for single-line input.
fn split_list_items(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();

    let looks_like_list = lines.iter().any(|line| {
        line.starts_with("- ")
            || line.starts_with("* ")
            || line.starts_with("+ ")
            || line
                .split_once(['.', ')'])
                .map(|(prefix, _)| prefix.chars().all(|c| c.is_ascii_digit()) && !prefix.is_empty())
                .unwrap_or(false)
    });

    if looks_like_list {
        return lines
            .iter()
            .map(|line| strip_bullet(line))
            .filter(|item| !item.is_empty())
            .collect();
    }

    if lines.len() > 1 {
        return lines.iter().map(|l| l.to_string()).collect();
    }

    // Single line: comma-separated fallback.
    match lines.first() {
        Some(line) if line.contains(',') => line
            .split(',')
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
        Some(line) => vec![line.to_string()],
        None => Vec::new(),
    }
}

fn strip_bullet(line: &str) -> String {
    let line = line.trim();
    for prefix in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return strip_quotes(rest.trim());
        }
    }
    // Numbered: `12. item` or `12) item`.
    if let Some((prefix, rest)) = line.split_once(['.', ')']) {
        if !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit()) {
            return strip_quotes(rest.trim());
        }
    }
    strip_quotes(line)
}
