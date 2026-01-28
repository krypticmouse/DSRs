#![cfg(feature = "rlm")]

use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

use baml_bridge::BamlParseError;

use crate::{BamlValue, ConstraintResult, FieldMeta, Flag, ResponseCheck, Signature};

/// Result of a SUBMIT call.
#[derive(Debug, Clone)]
pub enum SubmitResult<O> {
    /// Successful submission with validated output.
    Success {
        output: O,
        metas: IndexMap<String, FieldMeta>,
    },
    /// Validation errors that can be retried.
    ValidationError {
        message: String,
        errors: Vec<String>,
    },
    /// Hard assertion failures that block completion.
    AssertionFailed {
        field: String,
        label: String,
        expression: String,
    },
}

/// Type-erased result for cross-thread communication.
pub type SubmitResultDyn = Result<(BamlValue, IndexMap<String, FieldMeta>), SubmitError>;

#[derive(Debug, Clone)]
pub enum SubmitError {
    ValidationError { message: String, errors: Vec<String> },
    AssertionFailed { label: String, expression: String },
}

pub struct ParsedDyn {
    pub baml_value: BamlValue,
    pub flags: Vec<Flag>,
    pub checks: Vec<ResponseCheck>,
}

/// PyO3-compatible SUBMIT handler.
#[pyclass]
#[derive(Clone)]
pub struct SubmitHandler {
    parse_fn: Arc<
        dyn for<'py> Fn(Python<'py>, &Bound<'py, PyDict>) -> Result<ParsedDyn, BamlParseError>
            + Send
            + Sync,
    >,
    result_tx: Arc<Mutex<Option<SubmitResultDyn>>>,
    schema_description: String,
    output_fields: Vec<String>,
}

impl SubmitHandler {
    pub fn new<S: Signature>() -> (Self, Arc<Mutex<Option<SubmitResultDyn>>>) {
        let result_tx = Arc::new(Mutex::new(None));
        let schema_description = generate_schema_description::<S>();
        let output_fields = S::output_fields()
            .iter()
            .map(|field| field.name.to_string())
            .collect::<Vec<_>>();

        let parse_fn: Arc<
            dyn for<'py> Fn(Python<'py>, &Bound<'py, PyDict>) -> Result<ParsedDyn, BamlParseError>
                + Send
                + Sync,
        > = Arc::new(|py, kwargs| {
            let baml_value = crate::py::kwargs_to_baml_value::<S>(py, kwargs)?;
            let checks = crate::py::collect_checks_for_output::<S>(&baml_value)?;
            Ok(ParsedDyn {
                baml_value,
                flags: Vec::new(),
                checks,
            })
        });

        let handler = Self {
            parse_fn,
            result_tx: result_tx.clone(),
            schema_description,
            output_fields,
        };

        (handler, result_tx)
    }
}

#[pymethods]
impl SubmitHandler {
    /// SUBMIT(field1=value1, field2=value2, ...)
    #[pyo3(signature = (**kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let kwargs = kwargs.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "SUBMIT requires keyword arguments. Usage: SUBMIT(field1=value1, field2=value2)",
            )
        })?;

        let mut missing = Vec::new();
        for field in &self.output_fields {
            let present = kwargs
                .contains(field.as_str())
                .map_err(py_err_to_value)?;
            if !present {
                missing.push(field.clone());
            }
        }

        if !missing.is_empty() {
            let message = "Missing output fields".to_string();
            let errors = vec![format!("Missing fields: {:?}", missing)];
            *self.result_tx.lock().unwrap() = Some(Err(SubmitError::ValidationError {
                message: message.clone(),
                errors: errors.clone(),
            }));
            return Ok(format!(
                "[Error] Missing output fields: {:?}. Use SUBMIT({})",
                missing,
                self.output_fields.join(", ")
            ));
        }

        let parsed_result = (self.parse_fn)(py, kwargs);

        match parsed_result {
            Ok(parsed) => {
                let raw_text = serde_json::to_string(&parsed.baml_value)
                    .unwrap_or_else(|_| "<unserializable>".to_string());
                let metas = build_field_metas(&parsed, &raw_text);
                *self.result_tx.lock().unwrap() =
                    Some(Ok((parsed.baml_value.clone(), metas)));

                let warnings: Vec<String> = parsed
                    .checks
                    .iter()
                    .filter(|check| check.status != "succeeded")
                    .map(|check| format!("  - {} ({})", check.name, check.expression))
                    .collect();

                if warnings.is_empty() {
                    Ok("✓ SUBMIT successful! All validations passed.".to_string())
                } else {
                    Ok(format!(
                        "✓ SUBMIT successful with warnings:\n{}\n(These are soft constraints - output accepted but noted)",
                        warnings.join("\n")
                    ))
                }
            }
            Err(BamlParseError::ConstraintAssertsFailed { failed }) => {
                let failure = failed.first().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "SUBMIT assertion failed with no details",
                    )
                })?;
                *self.result_tx.lock().unwrap() = Some(Err(SubmitError::AssertionFailed {
                    label: failure.name.clone(),
                    expression: failure.expression.clone(),
                }));

                Ok(format!(
                    "[Error] Assertion '{}' failed: {}\nPlease fix and try again.",
                    failure.name, failure.expression
                ))
            }
            Err(err) => {
                let errors = format_parse_errors(&err);
                *self.result_tx.lock().unwrap() = Some(Err(SubmitError::ValidationError {
                    message: err.to_string(),
                    errors: errors.clone(),
                }));

                let joined = errors.join("\n");
                if self.schema_description.is_empty() {
                    Ok(joined)
                } else {
                    Ok(format!(
                        "{}\n\nExpected schema:\n{}",
                        joined, self.schema_description
                    ))
                }
            }
        }
    }

    fn schema(&self) -> String {
        self.schema_description.clone()
    }
}

fn build_field_metas(
    parsed: &ParsedDyn,
    raw_json: &str,
) -> IndexMap<String, FieldMeta> {
    let mut metas = IndexMap::new();
    let mut meta = FieldMeta {
        raw_text: raw_json.to_string(),
        flags: Vec::new(),
        checks: Vec::new(),
    };

    meta.flags.extend(parsed.flags.iter().cloned());

    for check in &parsed.checks {
        meta.checks.push(ConstraintResult {
            label: check.name.clone(),
            expression: check.expression.clone(),
            passed: check.status == "succeeded",
        });
    }

    metas.insert("_root".to_string(), meta);
    metas
}

fn format_parse_errors(err: &BamlParseError) -> Vec<String> {
    match err {
        BamlParseError::Convert(err) => vec![format_type_error(err)],
        BamlParseError::Jsonish(err) => vec![format!("[Error] {}", err)],
        BamlParseError::ConstraintAssertsFailed { failed } => failed
            .iter()
            .map(|check| {
                format!("[Error] Assertion '{}' failed: {}", check.name, check.expression)
            })
            .collect(),
    }
}

fn format_type_error(err: &baml_bridge::BamlConvertError) -> String {
    let expected = err
        .message
        .strip_prefix("expected ")
        .unwrap_or(err.expected);
    format!(
        "[Type Error] {}: expected {}, got {}: {}",
        err.path_string(),
        expected,
        err.got,
        err.message
    )
}

fn generate_schema_description<S: Signature>() -> String {
    let fields: Vec<String> = S::output_fields()
        .iter()
        .map(|field| field.name.to_string())
        .collect();

    if fields.is_empty() {
        return String::new();
    }

    format!("SUBMIT({})", fields.join(", "))
}

fn py_err_to_value(err: pyo3::PyErr) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}
