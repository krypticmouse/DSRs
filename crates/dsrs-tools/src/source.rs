//! Ephemeral tool definitions: what an LLM (or a human) hands the runtime.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::RegisterError;

/// The raw material for an ephemeral tool, before it has passed validation.
///
/// # Source contract
///
/// `js_source` must be a single JavaScript **expression that evaluates to a
/// function** taking one argument (the parsed JSON args object) and returning
/// a JSON-serializable value (or a promise of one):
///
/// ```js
/// (args) => args.x + args.y
/// ```
///
/// Helpers and state go inside an IIFE that returns the tool function:
///
/// ```js
/// (() => {
///     const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
///     return (args) => clamp(args.value, args.lo, args.hi);
/// })()
/// ```
///
/// Named `function` declarations work too (they are wrapped in parentheses and
/// become function expressions). A single trailing `;` is tolerated.
///
/// # Self-test contract
///
/// `self_test`, if present, is a JavaScript program evaluated in the sandbox
/// with the global `tool` bound to the compiled tool function. It fails if it
/// throws or if its completion value is `false`; anything else passes:
///
/// ```js
/// if (tool({x: 2, y: 3}) !== 5) throw new Error("2+3 should be 5");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSource {
    /// Unique tool name, `[A-Za-z0-9_-]{1,64}`.
    pub name: String,
    /// Natural-language description shown to the model.
    pub description: String,
    /// JSON Schema for the tool arguments (an object schema).
    pub params: Value,
    /// JavaScript source per the contract above.
    pub js_source: String,
    /// Optional self-test program; tools with a failing self-test are never
    /// registered.
    pub self_test: Option<String>,
}

impl ToolSource {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        params: Value,
        js_source: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            params,
            js_source: js_source.into(),
            self_test: None,
        }
    }

    /// Attach a self-test program (see the type-level docs for the contract).
    pub fn with_self_test(mut self, self_test: impl Into<String>) -> Self {
        self.self_test = Some(self_test.into());
        self
    }

    /// Structural validation of name and params schema. Cheap and synchronous;
    /// the compile/self-test stages need a sandbox and live on the executor.
    pub fn validate_shape(&self) -> Result<(), RegisterError> {
        validate_tool_name(&self.name)?;
        validate_params_schema(&self.params)?;
        Ok(())
    }

    /// The names of required arguments declared by the params schema.
    pub fn required_params(&self) -> Vec<String> {
        self.params
            .get("required")
            .and_then(Value::as_array)
            .map(|reqs| {
                reqs.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    }
}

pub(crate) fn validate_tool_name(name: &str) -> Result<(), RegisterError> {
    let invalid = |reason: &str| RegisterError::InvalidName {
        name: name.to_string(),
        reason: reason.to_string(),
    };
    if name.is_empty() {
        return Err(invalid("name is empty"));
    }
    if name.len() > 64 {
        return Err(invalid("name is longer than 64 characters"));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(invalid(
            "name may only contain ASCII letters, digits, `_` and `-`",
        ));
    }
    Ok(())
}

/// Structural well-formedness checks on a params JSON schema. This is not a
/// full JSON Schema validator; it catches the malformed shapes LLMs commonly
/// emit so they can be repaired before the tool is ever advertised.
pub(crate) fn validate_params_schema(schema: &Value) -> Result<(), RegisterError> {
    let invalid = |reason: String| RegisterError::InvalidSchema { reason };

    let Some(obj) = schema.as_object() else {
        return Err(invalid(format!(
            "params schema must be a JSON object, got {}",
            json_type_name(schema)
        )));
    };
    if let Some(ty) = obj.get("type")
        && ty.as_str() != Some("object")
    {
        return Err(invalid(format!(
            "params schema `type` must be \"object\", got {ty}"
        )));
    }
    let properties = match obj.get("properties") {
        None => None,
        Some(props) => Some(props.as_object().ok_or_else(|| {
            invalid(format!(
                "`properties` must be an object, got {}",
                json_type_name(props)
            ))
        })?),
    };
    if let Some(props) = properties {
        for (key, prop) in props {
            if !prop.is_object() {
                return Err(invalid(format!(
                    "property `{key}` must be a schema object, got {}",
                    json_type_name(prop)
                )));
            }
        }
    }
    if let Some(required) = obj.get("required") {
        let Some(required) = required.as_array() else {
            return Err(invalid(format!(
                "`required` must be an array of strings, got {}",
                json_type_name(required)
            )));
        };
        for entry in required {
            let Some(name) = entry.as_str() else {
                return Err(invalid(format!(
                    "`required` entries must be strings, got {}",
                    json_type_name(entry)
                )));
            };
            if let Some(props) = properties
                && !props.contains_key(name)
            {
                return Err(invalid(format!(
                    "`required` names `{name}` which is not in `properties`"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
