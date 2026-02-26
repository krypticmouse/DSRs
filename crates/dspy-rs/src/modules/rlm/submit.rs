use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use bamltype::BamlParseError;
use indexmap::IndexMap;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};

use crate::{
    BamlValue, ConstraintKind, ConstraintResult, FieldMeta, Flag, ResponseCheck, Signature,
    SignatureSchema,
};

/// Type-erased SUBMIT result used by the outer loop controller.
pub type SubmitResultDyn = Result<(BamlValue, IndexMap<String, FieldMeta>), SubmitError>;

/// Shared storage slot written by SUBMIT and consumed by the RLM loop.
pub type SubmitSlot = Arc<Mutex<Option<SubmitResultDyn>>>;

#[derive(Debug, Clone)]
pub enum SubmitError {
    ValidationError {
        message: String,
        errors: Vec<String>,
    },
    AssertionFailed {
        label: String,
        expression: String,
    },
}

pub struct ParsedDyn {
    pub baml_value: BamlValue,
    pub flags: Vec<Flag>,
    pub checks: Vec<ResponseCheck>,
}

type ParseFn = dyn for<'py> Fn(Python<'py>, &Bound<'py, PyDict>) -> Result<ParsedDyn, BamlParseError>
    + Send
    + Sync;

pyo3::create_exception!(
    dspy_rs_rlm,
    SubmitTerminated,
    PyException,
    "Raised to terminate REPL execution after a successful SUBMIT."
);

pub const SUBMIT_STDOUT_ATTR: &str = "__dsrs_stdout__";

pub fn is_submit_terminated(err: &PyErr, py: Python<'_>) -> bool {
    err.is_instance_of::<SubmitTerminated>(py)
}

pub fn clear_submit_slot(slot: &SubmitSlot) {
    *slot.lock().expect("submit slot lock poisoned") = None;
}

pub fn take_submit_result(slot: &SubmitSlot) -> Option<SubmitResultDyn> {
    slot.lock().expect("submit slot lock poisoned").take()
}

#[pyclass]
#[derive(Clone)]
pub struct SubmitHandler {
    parse_fn: Arc<ParseFn>,
    schema: Arc<SignatureSchema>,
    slot: SubmitSlot,
    schema_description: String,
    output_fields_lm: Vec<String>,
    output_fields_set: HashSet<String>,
}

impl SubmitHandler {
    pub fn new<S: Signature>(slot: SubmitSlot) -> Self {
        let schema = Arc::new(S::schema().clone());
        let schema_description = generate_schema_description(schema.as_ref());
        let output_fields_lm = schema
            .output_fields()
            .iter()
            .map(|field| field.lm_name.to_string())
            .collect::<Vec<_>>();
        let output_fields_set = output_fields_lm.iter().cloned().collect::<HashSet<_>>();

        let parse_fn: Arc<ParseFn> = Arc::new(|py, kwargs| {
            let baml_value = super::py_bridge::kwargs_to_baml_value::<S>(py, kwargs)?;
            let checks = super::py_bridge::collect_checks_for_output::<S>(&baml_value)?;
            Ok(ParsedDyn {
                baml_value,
                flags: Vec::new(),
                checks,
            })
        });

        Self {
            parse_fn,
            schema,
            slot,
            schema_description,
            output_fields_lm,
            output_fields_set,
        }
    }

    pub fn with_new_slot<S: Signature>() -> (Self, SubmitSlot) {
        let slot = Arc::new(Mutex::new(None));
        (Self::new::<S>(Arc::clone(&slot)), slot)
    }
}

#[pymethods]
impl SubmitHandler {
    #[pyo3(signature = (**kwargs))]
    fn __call__(&self, py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        let kwargs = kwargs.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "SUBMIT requires keyword arguments. Usage: SUBMIT(field1=value1, field2=value2)",
            )
        })?;

        let mut unexpected = Vec::new();
        for (key, _) in kwargs.iter() {
            let key = key.extract::<String>().map_err(py_err_to_value)?;
            if !self.output_fields_set.contains(&key) {
                unexpected.push(key);
            }
        }
        unexpected.sort();

        let mut missing = Vec::new();
        for field in &self.output_fields_lm {
            let present = kwargs.contains(field.as_str()).map_err(py_err_to_value)?;
            if !present {
                missing.push(field.clone());
            }
        }

        if !missing.is_empty() || !unexpected.is_empty() {
            let usage = format_submit_usage(&self.output_fields_lm);
            let mut errors = Vec::new();
            if !missing.is_empty() {
                errors.push(format!("missing fields: {:?}", missing));
            }
            if !unexpected.is_empty() {
                errors.push(format!("unexpected fields: {:?}", unexpected));
            }
            errors.push(format!("use SUBMIT({usage})"));

            let message = match (missing.is_empty(), unexpected.is_empty()) {
                (false, true) => "Missing output fields".to_string(),
                (true, false) => "Unexpected output fields".to_string(),
                (false, false) => "Invalid output fields".to_string(),
                (true, true) => unreachable!(),
            };

            let user_message = format_submit_error("Validation failed", &errors, None);
            *self.slot.lock().expect("submit slot lock poisoned") =
                Some(Err(SubmitError::ValidationError { message, errors }));
            return Ok(user_message);
        }

        let parsed_result = (self.parse_fn)(py, kwargs);

        match parsed_result {
            Ok(parsed) => {
                let raw_text = serde_json::to_string(&parsed.baml_value)
                    .unwrap_or_else(|_| "<unserializable>".to_string());
                let metas = build_field_metas(&parsed, &raw_text);
                *self.slot.lock().expect("submit slot lock poisoned") =
                    Some(Ok((parsed.baml_value.clone(), metas)));

                Err(SubmitTerminated::new_err("SUBMIT accepted"))
            }
            Err(BamlParseError::ConstraintAssertsFailed { failed }) => {
                let failure = failed.first().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "SUBMIT assertion failed with no details",
                    )
                })?;

                *self.slot.lock().expect("submit slot lock poisoned") =
                    Some(Err(SubmitError::AssertionFailed {
                        label: failure.name.clone(),
                        expression: failure.expression.clone(),
                    }));

                Ok(format_submit_error(
                    "Assertion failed",
                    &[format!(
                        "'{}': {} (please fix and try again)",
                        failure.name, failure.expression
                    )],
                    None,
                ))
            }
            Err(err) => {
                let errors = format_parse_errors(kwargs, &self.schema, &err);
                *self.slot.lock().expect("submit slot lock poisoned") =
                    Some(Err(SubmitError::ValidationError {
                        message: err.to_string(),
                        errors: errors.clone(),
                    }));

                Ok(format_submit_error(
                    "Validation failed",
                    &errors,
                    if self.schema_description.is_empty() {
                        None
                    } else {
                        Some(self.schema_description.as_str())
                    },
                ))
            }
        }
    }

    pub fn schema(&self) -> String {
        self.schema_description.clone()
    }
}

fn build_field_metas(parsed: &ParsedDyn, raw_json: &str) -> IndexMap<String, FieldMeta> {
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

fn format_parse_errors(
    kwargs: &Bound<'_, PyDict>,
    schema: &SignatureSchema,
    err: &BamlParseError,
) -> Vec<String> {
    match err {
        BamlParseError::Convert(err) => vec![format_convert_error(kwargs, schema, err)],
        BamlParseError::Jsonish(err) => vec![err.to_string()],
        BamlParseError::ConstraintAssertsFailed { failed } => failed
            .iter()
            .map(|check| format!("assertion '{}' failed: {}", check.name, check.expression))
            .collect(),
    }
}

fn format_convert_error(
    kwargs: &Bound<'_, PyDict>,
    schema: &SignatureSchema,
    err: &crate::BamlConvertError,
) -> String {
    if err.expected == "field" && err.got == "missing" {
        return format!("missing required field: {}", err.path_string());
    }

    let expected = err
        .message
        .strip_prefix("expected ")
        .unwrap_or(err.expected)
        .trim();

    let field_path = err.path_string();
    let value_repr = first_path_value_repr(kwargs, schema, &err.path);

    match value_repr {
        Some(value_repr) => format!(
            "field '{}' expected {}, got {} {}",
            field_path, expected, err.got, value_repr
        ),
        None => format!(
            "field '{}' expected {}, got {}",
            field_path, expected, err.got
        ),
    }
}

fn format_submit_error(summary: &str, details: &[String], schema: Option<&str>) -> String {
    let mut message = format!("SubmitError: {summary}");
    if !details.is_empty() {
        message.push('\n');
        for detail in details {
            message.push_str("  - ");
            message.push_str(detail);
            message.push('\n');
        }
        message.pop();
    }
    if let Some(schema) = schema {
        message.push_str("\n\nExpected schema:\n");
        message.push_str(schema);
    }
    message
}

fn first_path_value_repr(
    kwargs: &Bound<'_, PyDict>,
    schema: &SignatureSchema,
    path: &[String],
) -> Option<String> {
    let first = path.first()?;

    let lm_name = schema
        .output_fields()
        .iter()
        .find_map(|field| {
            if field.rust_name == *first || field.lm_name == first {
                Some(field.lm_name)
            } else {
                None
            }
        })
        .unwrap_or(first.as_str());

    let value = kwargs.get_item(lm_name).ok().flatten()?;
    value
        .repr()
        .ok()
        .and_then(|repr| repr.extract::<String>().ok())
}

fn format_submit_usage(fields: &[String]) -> String {
    fields
        .iter()
        .map(|field| format!("{field}={field}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn generate_schema_description(schema: &SignatureSchema) -> String {
    let fields = schema.output_fields();
    if fields.is_empty() {
        return String::new();
    }

    let mut desc = String::new();
    desc.push_str("SUBMIT(");
    desc.push_str(
        &fields
            .iter()
            .map(|field| field.lm_name)
            .collect::<Vec<_>>()
            .join(", "),
    );
    desc.push_str(") where:\n");

    for field in fields {
        let type_name = format_type_name(&field.type_ir);
        desc.push_str(&format!("  {}: {}", field.lm_name, type_name));

        if !field.docs.is_empty() {
            desc.push_str(&format!("  # {}", field.docs));
        }
        desc.push('\n');

        for constraint in field.constraints {
            let kind = match constraint.kind {
                ConstraintKind::Check => "check",
                ConstraintKind::Assert => "ASSERT",
            };
            if constraint.label.is_empty() {
                desc.push_str(&format!("    [{kind}] {}\n", constraint.expression));
            } else {
                desc.push_str(&format!(
                    "    [{kind}] {}: {}\n",
                    constraint.label, constraint.expression
                ));
            }
        }
    }

    desc.trim_end().to_string()
}

fn py_err_to_value(err: pyo3::PyErr) -> pyo3::PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn format_type_name(type_ir: &crate::TypeIR) -> String {
    let raw = type_ir.diagnostic_repr().to_string();
    simplify_type_name(&raw)
        .replace("class ", "")
        .replace("enum ", "")
        .replace(" | ", " or ")
        .trim()
        .to_string()
}

fn simplify_type_name(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '`' {
                    break;
                }
                token.push(next);
            }
            let simplified = token.rsplit("::").next().unwrap_or(&token);
            result.push_str(simplified);
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyDict;

    use super::*;
    use crate::Signature;

    #[derive(Signature, Clone, Debug)]
    struct SubmitSig {
        #[input]
        question: String,

        #[output]
        answer: String,

        #[output]
        score: f64,
    }

    #[derive(Signature, Clone, Debug)]
    struct SubmitAssertSig {
        #[input]
        question: String,

        #[output]
        #[assert("this > 0", label = "positive")]
        score: i64,
    }

    #[test]
    fn submit_success_writes_slot_and_raises_terminated() {
        Python::attach(|py| {
            let (handler, slot) = SubmitHandler::with_new_slot::<SubmitSig>();
            let kwargs = PyDict::new(py);
            kwargs.set_item("answer", "ok").expect("set answer");
            kwargs.set_item("score", 0.9).expect("set score");

            let err = handler
                .__call__(py, Some(&kwargs))
                .expect_err("successful submit must raise SubmitTerminated");
            assert!(is_submit_terminated(&err, py));

            let stored = take_submit_result(&slot).expect("slot must be populated");
            assert!(stored.is_ok());
        });
    }

    #[test]
    fn missing_field_returns_validation_error() {
        Python::attach(|py| {
            let (handler, slot) = SubmitHandler::with_new_slot::<SubmitSig>();
            let kwargs = PyDict::new(py);
            kwargs.set_item("answer", "ok").expect("set answer");

            let message = handler
                .__call__(py, Some(&kwargs))
                .expect("missing field should return recoverable message");
            assert!(message.contains("SubmitError: Validation failed"));

            let stored = take_submit_result(&slot).expect("slot must be populated");
            match stored {
                Err(SubmitError::ValidationError { errors, .. }) => {
                    assert!(errors.iter().any(|err| err.contains("missing fields")));
                }
                other => panic!("unexpected stored result: {other:?}"),
            }
        });
    }

    #[test]
    fn type_mismatch_returns_detailed_field_error() {
        Python::attach(|py| {
            let (handler, slot) = SubmitHandler::with_new_slot::<SubmitSig>();
            let kwargs = PyDict::new(py);
            kwargs.set_item("answer", "ok").expect("set answer");
            kwargs.set_item("score", "oops").expect("set score");

            let message = handler
                .__call__(py, Some(&kwargs))
                .expect("type mismatch should be recoverable");
            assert!(message.contains("field 'score'"));
            assert!(message.contains("expected"));
            assert!(message.contains("got"));

            let stored = take_submit_result(&slot).expect("slot must be populated");
            assert!(matches!(stored, Err(SubmitError::ValidationError { .. })));
        });
    }

    #[test]
    fn assertion_failure_is_recorded() {
        Python::attach(|py| {
            let (handler, slot) = SubmitHandler::with_new_slot::<SubmitAssertSig>();
            let kwargs = PyDict::new(py);
            kwargs.set_item("score", -1).expect("set score");

            let message = handler
                .__call__(py, Some(&kwargs))
                .expect("assertion failure should be recoverable");
            assert!(message.contains("SubmitError: Assertion failed"));

            let stored = take_submit_result(&slot).expect("slot must be populated");
            match stored {
                Err(SubmitError::AssertionFailed { label, .. }) => {
                    assert_eq!(label, "positive");
                }
                other => panic!("unexpected stored result: {other:?}"),
            }
        });
    }

    #[test]
    fn clear_submit_slot_removes_previous_value() {
        let (handler, slot) = SubmitHandler::with_new_slot::<SubmitSig>();
        drop(handler);

        *slot.lock().expect("lock") = Some(Err(SubmitError::ValidationError {
            message: "x".to_string(),
            errors: vec!["y".to_string()],
        }));

        clear_submit_slot(&slot);
        assert!(slot.lock().expect("lock").is_none());
    }
}
