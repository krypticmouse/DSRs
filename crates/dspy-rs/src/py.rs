use anyhow::anyhow;
use baml_bridge::baml_types::{
    BamlMap, BamlValue, ConstraintLevel, ResponseCheck, TypeIR,
};
use baml_bridge::jsonish::deserializer::coercer::run_user_checks;
use baml_bridge::{BamlConvertError, BamlParseError};
use pyo3::types::{PyDict, PyDictMethods};
use pyo3::{Bound, Python};

use crate::Signature;

pub fn missing_output_fields<S: Signature>(kwargs: &Bound<'_, PyDict>) -> Vec<String> {
    let mut missing = Vec::new();
    for field in S::output_fields() {
        let present = kwargs.contains(field.name).unwrap_or(false);
        if !present {
            missing.push(field.name.to_string());
        }
    }
    missing
}

pub fn kwargs_to_baml_value<S: Signature>(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
) -> Result<BamlValue, BamlParseError> {
    let output_format = S::output_format_content();
    let mut fields = BamlMap::new();

    for field in S::output_fields() {
        let value = kwargs
            .get_item(field.name)
            .map_err(py_err_to_parse)?
            .ok_or_else(|| missing_field_error(field.name))?;
        let field_type = (field.type_ir)();
        let baml_value = baml_bridge::py::py_to_baml_value(py, &value, &field_type, output_format)?;
        fields.insert(field.rust_name.to_string(), baml_value);
    }

    if let Some(class_name) = output_class_name(output_format) {
        Ok(BamlValue::Class(class_name, fields))
    } else {
        Ok(BamlValue::Map(fields))
    }
}

pub fn collect_checks_for_output<S: Signature>(
    value: &BamlValue,
) -> Result<Vec<ResponseCheck>, BamlParseError> {
    let fields = match value {
        BamlValue::Class(_, fields) | BamlValue::Map(fields) => fields,
        other => {
            return Err(BamlParseError::Convert(BamlConvertError::new(
                Vec::new(),
                "object",
                format!("{other:?}"),
                "expected an object",
            )))
        }
    };

    let mut checks = Vec::new();
    let mut failed = Vec::new();

    for field in S::output_fields() {
        let Some(value) = fields.get(field.rust_name) else {
            return Err(missing_field_error(field.rust_name));
        };
        let field_type = (field.type_ir)();
        let results = run_user_checks(value, &field_type).map_err(BamlParseError::from)?;
        for (constraint, ok) in results {
            if constraint.level == ConstraintLevel::Assert && !ok {
                failed.push(ResponseCheck {
                    name: constraint
                        .label
                        .clone()
                        .unwrap_or_else(|| "assert".to_string()),
                    expression: constraint.expression.0.clone(),
                    status: "failed".to_string(),
                });
            }
            if let Some(check) = ResponseCheck::from_check_result((constraint, ok)) {
                checks.push(check);
            }
        }
    }

    if !failed.is_empty() {
        return Err(BamlParseError::ConstraintAssertsFailed { failed });
    }

    Ok(checks)
}

fn output_class_name(output_format: &crate::OutputFormatContent) -> Option<String> {
    let mut current = output_format.target.clone();
    loop {
        match current {
            TypeIR::Class { name, .. } => return Some(name),
            TypeIR::RecursiveTypeAlias { name, .. } => {
                if let Some(next) = output_format.structural_recursive_aliases.get(&name) {
                    current = next.clone();
                    continue;
                }
                return None;
            }
            _ => return None,
        }
    }
}

fn missing_field_error(field: &str) -> BamlParseError {
    BamlParseError::Convert(BamlConvertError::new(
        vec![field.to_string()],
        "field",
        "missing",
        format!("missing required field {field}"),
    ))
}

fn py_err_to_parse(err: pyo3::PyErr) -> BamlParseError {
    BamlParseError::Jsonish(anyhow!(err.to_string()))
}
