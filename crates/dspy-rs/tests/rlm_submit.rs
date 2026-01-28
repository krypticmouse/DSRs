#![cfg(feature = "rlm")]

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use dspy_rs::baml_bridge::baml_types::BamlMap;
use dspy_rs::{BamlValue, RlmType};
use dspy_rs::rlm::submit::{SubmitError, SubmitHandler};

fn call_submit(
    py: Python<'_>,
    handler: &SubmitHandler,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<String> {
    let py_handler = Py::new(py, handler.clone())?;
    let result = py_handler.bind(py).call((), Some(kwargs))?;
    result.extract()
}

fn output_fields<'a>(value: &'a BamlValue) -> &'a BamlMap<String, BamlValue> {
    match value {
        BamlValue::Map(fields) | BamlValue::Class(_, fields) => fields,
        other => panic!("expected object output, got {other:?}"),
    }
}

#[derive(dspy_rs::Signature, Clone, Debug)]
struct SubmitBasic {
    #[input]
    prompt: String,

    #[output]
    answer: String,

    #[output]
    numbers: Vec<i64>,
}

#[derive(dspy_rs::Signature, Clone, Debug)]
struct SubmitChecks {
    #[input]
    prompt: String,

    #[output]
    #[check("this != ''", label = "non_empty")]
    verdict: String,

    #[output]
    #[assert("this != 'nope'", label = "not_nope")]
    status: String,
}

#[dspy_rs::rlm_type]
#[derive(Clone, Debug)]
struct Payload {
    value: String,
    count: i64,
}

#[derive(dspy_rs::Signature, Clone, Debug)]
struct SubmitPayload {
    #[input]
    prompt: String,

    #[output]
    payload: Payload,
}

#[derive(dspy_rs::Signature, Clone, Debug)]
struct SubmitCoercion {
    #[input]
    prompt: String,

    #[output]
    count: i64,
}

#[test]
fn submit_returns_typed_output() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitBasic>();
        let kwargs = PyDict::new(py);
        kwargs.set_item("answer", "ok")?;
        let list = PyList::new(py, [1_i64, 2_i64])?;
        kwargs.set_item("numbers", list)?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(message.starts_with("✓ SUBMIT successful"));

        let result = result_tx.lock().unwrap().clone().expect("result set");
        let (baml_value, _metas) = result.expect("submit ok");
        let fields = output_fields(&baml_value);
        let answer = fields.get("answer").expect("answer field");
        assert!(matches!(answer, BamlValue::String(value) if value == "ok"));
        Ok(())
    })
}

#[test]
fn submit_missing_fields_error() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitBasic>();
        let kwargs = PyDict::new(py);
        kwargs.set_item("answer", "ok")?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert_eq!(
            message,
            "[Error] Missing output fields: [\"numbers\"]. Use SUBMIT(answer, numbers)"
        );

        let result = result_tx.lock().unwrap().clone().expect("result set");
        match result {
            Err(SubmitError::ValidationError { message, errors }) => {
                assert_eq!(message, "Missing output fields");
                assert_eq!(errors, vec!["Missing fields: [\"numbers\"]".to_string()]);
            }
            other => panic!("expected validation error, got {other:?}"),
        }

        Ok(())
    })
}

#[test]
fn submit_type_error_for_wrong_shape() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitBasic>();
        let locals = PyDict::new(py);
        let code = CString::new(concat!(
            "class Weird:\n",
            "    __slots__ = ()\n",
            "weird = Weird()\n",
        ))
        .expect("valid python code");
        py.run(&code, None, Some(&locals))?;
        let weird = locals.get_item("weird").expect("weird instance");

        let kwargs = PyDict::new(py);
        kwargs.set_item("answer", "ok")?;
        kwargs.set_item("numbers", weird)?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(
            message.contains("[Type Error]"),
            "unexpected message: {message}"
        );

        let result = result_tx.lock().unwrap().clone().expect("result set");
        assert!(matches!(result, Err(SubmitError::ValidationError { .. })));
        Ok(())
    })
}

#[test]
fn submit_records_soft_checks() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitChecks>();
        let kwargs = PyDict::new(py);
        kwargs.set_item("verdict", "")?;
        kwargs.set_item("status", "ok")?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(message.contains("warnings"));

        let result = result_tx.lock().unwrap().clone().expect("result set");
        let (_baml_value, metas) = result.expect("submit ok");
        let meta = metas.get("_root").expect("root meta");
        assert!(meta.checks.iter().any(|check| !check.passed));
        Ok(())
    })
}

#[test]
fn submit_blocks_assert_failures() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitChecks>();
        let kwargs = PyDict::new(py);
        kwargs.set_item("verdict", "fine")?;
        kwargs.set_item("status", "nope")?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(message.starts_with("[Error] Assertion 'not_nope' failed"));

        let result = result_tx.lock().unwrap().clone().expect("result set");
        match result {
            Err(SubmitError::AssertionFailed { label, expression }) => {
                assert_eq!(label, "not_nope");
                assert_eq!(expression, "this != 'nope'");
            }
            other => panic!("expected assert failure, got {other:?}"),
        }

        Ok(())
    })
}

#[test]
fn submit_uses_baml_normalization() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitPayload>();
        let kwargs = PyDict::new(py);
        let payload = Py::new(
            py,
            Payload {
                value: "hello".to_string(),
                count: 3,
            },
        )?;
        kwargs.set_item("payload", payload)?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(message.starts_with("✓ SUBMIT successful"));

        let result = result_tx.lock().unwrap().clone().expect("result set");
        let (baml_value, _metas) = result.expect("submit ok");
        let fields = output_fields(&baml_value);
        let payload_value = fields.get("payload").expect("payload field");
        assert!(matches!(payload_value, BamlValue::Class(_, _)));
        Ok(())
    })
}

#[test]
fn submit_normalizes_dataclasses_and_optional_extras() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitPayload>();
        let locals = PyDict::new(py);
        let code = CString::new(concat!(
            "from dataclasses import dataclass\n",
            "@dataclass\n",
            "class Payload:\n",
            "    value: str\n",
            "    count: int\n",
            "payload = Payload(value='hi', count=7)\n",
        ))
        .expect("valid python code");
        py.run(&code, None, Some(&locals))?;
        let payload = locals
            .get_item("payload")
            .expect("payload instance");

        let kwargs = PyDict::new(py);
        kwargs.set_item("payload", payload)?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(message.starts_with("✓ SUBMIT successful"));

        let result = result_tx.lock().unwrap().clone().expect("result set");
        assert!(matches!(result, Ok(_)));

        if PyModule::import(py, "attr").is_ok() {
            let locals = PyDict::new(py);
            let code = CString::new(concat!(
                "import attr\n",
                "@attr.define\n",
                "class Payload:\n",
                "    value: str\n",
                "    count: int\n",
                "payload = Payload(value='yo', count=2)\n",
            ))
            .expect("valid python code");
            py.run(&code, None, Some(&locals))?;
            let payload = locals.get_item("payload").expect("attrs payload");
            let kwargs = PyDict::new(py);
            kwargs.set_item("payload", payload)?;
            let message = call_submit(py, &handler, &kwargs)?;
            assert!(message.starts_with("✓ SUBMIT successful"));
        }

        if PyModule::import(py, "pydantic").is_ok() {
            let locals = PyDict::new(py);
            let code = CString::new(concat!(
                "from pydantic import BaseModel\n",
                "class Payload(BaseModel):\n",
                "    value: str\n",
                "    count: int\n",
                "payload = Payload(value='hey', count=5)\n",
            ))
            .expect("valid python code");
            py.run(&code, None, Some(&locals))?;
            let payload = locals.get_item("payload").expect("pydantic payload");
            let kwargs = PyDict::new(py);
            kwargs.set_item("payload", payload)?;
            let message = call_submit(py, &handler, &kwargs)?;
            assert!(message.starts_with("✓ SUBMIT successful"));
        }

        Ok(())
    })
}

#[test]
fn submit_allows_string_coercion() -> PyResult<()> {
    Python::with_gil(|py| {
        let (handler, result_tx) = SubmitHandler::new::<SubmitCoercion>();
        let kwargs = PyDict::new(py);
        kwargs.set_item("count", "42")?;

        let message = call_submit(py, &handler, &kwargs)?;
        assert!(message.starts_with("✓ SUBMIT successful"));

        let result = result_tx.lock().unwrap().clone().expect("result set");
        let (baml_value, _metas) = result.expect("submit ok");
        let fields = output_fields(&baml_value);
        let count = fields.get("count").expect("count field");
        assert!(matches!(count, BamlValue::Int(42)));
        Ok(())
    })
}
