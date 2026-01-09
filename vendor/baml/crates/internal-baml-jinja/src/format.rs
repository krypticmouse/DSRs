use std::collections::HashMap;

use baml_types::{BamlValue, StreamingMode};
use serde_json::Value as JsonValue;

use crate::output_format::types::{Class, OutputFormatContent};

pub fn format_baml_value(
    value: &BamlValue,
    output_format: &OutputFormatContent,
    format: &str,
) -> Result<String, String> {
    let json_value = baml_value_to_json(value, output_format);
    match format.to_ascii_lowercase().as_str() {
        "json" => serde_json::to_string(&json_value).map_err(|e| e.to_string()),
        "yaml" => serde_yaml::to_string(&json_value).map_err(|e| e.to_string()),
        "toon" => Ok(toon::encode(&json_value, None)),
        other => Err(format!(
            "Unsupported format type '{other}'. Supported types: 'yaml', 'json', 'toon'",
        )),
    }
}

fn baml_value_to_json(value: &BamlValue, output_format: &OutputFormatContent) -> JsonValue {
    match value {
        BamlValue::String(s) => JsonValue::String(s.clone()),
        BamlValue::Int(i) => JsonValue::Number((*i).into()),
        BamlValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null),
        BamlValue::Bool(b) => JsonValue::Bool(*b),
        BamlValue::Null => JsonValue::Null,
        BamlValue::Media(media) => serde_json::to_value(media).unwrap_or(JsonValue::Null),
        BamlValue::List(items) => JsonValue::Array(
            items
                .iter()
                .map(|item| baml_value_to_json(item, output_format))
                .collect(),
        ),
        BamlValue::Map(map) => {
            let mut json_map = serde_json::Map::new();
            for (key, value) in map.iter() {
                json_map.insert(key.to_string(), baml_value_to_json(value, output_format));
            }
            JsonValue::Object(json_map)
        }
        BamlValue::Enum(enum_name, variant) => {
            let rendered =
                enum_alias(output_format, enum_name, variant).unwrap_or_else(|| variant.clone());
            JsonValue::String(rendered)
        }
        BamlValue::Class(class_name, fields) => {
            let aliases = class_aliases(output_format, class_name);
            let mut json_map = serde_json::Map::new();
            for (key, value) in fields.iter() {
                let alias = aliases
                    .as_ref()
                    .and_then(|aliases| aliases.get(key))
                    .map(String::as_str)
                    .unwrap_or(key);
                json_map.insert(alias.to_string(), baml_value_to_json(value, output_format));
            }
            JsonValue::Object(json_map)
        }
    }
}

fn class_aliases(
    output_format: &OutputFormatContent,
    class_name: &str,
) -> Option<HashMap<String, String>> {
    let class = find_class(output_format, class_name)?;
    let mut aliases = HashMap::new();
    for (name, _, _, _) in &class.fields {
        aliases.insert(
            name.real_name().to_string(),
            name.rendered_name().to_string(),
        );
    }
    Some(aliases)
}

fn enum_alias(
    output_format: &OutputFormatContent,
    enum_name: &str,
    variant: &str,
) -> Option<String> {
    let enm = output_format.enums.get(enum_name)?;
    enm.values
        .iter()
        .find(|(name, _)| name.real_name() == variant)
        .map(|(name, _)| name.rendered_name().to_string())
}

fn find_class<'a>(output_format: &'a OutputFormatContent, class_name: &str) -> Option<&'a Class> {
    let key = (class_name.to_string(), StreamingMode::NonStreaming);
    if let Some(class) = output_format.classes.get(&key) {
        return Some(class);
    }
    output_format
        .classes
        .iter()
        .find(|((name, _), _)| name == class_name)
        .map(|(_, class)| class)
}
