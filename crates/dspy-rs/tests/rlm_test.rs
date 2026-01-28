#![cfg(feature = "rlm")]

use dspy_rs::baml_bridge::{py as baml_py, ToBamlValue};
use dspy_rs::rlm::{RlmConfig, TypedRlm};
use dspy_rs::{rlm_type, BamlType, RlmType, Signature};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyIterator, PyList};
use rig::prelude::*;
use rig::providers::openai;

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
#[rlm(repr = "Item({self.name}, {self.value})")]
struct Item {
    name: String,
    value: i32,
}

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
#[rlm(repr = "Bag({len(self.items)} items)", iter = "items", index = "items")]
struct Bag {
    items: Vec<Item>,
}

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
struct Pantry {
    #[rlm(desc = "Owner of the pantry")]
    owner: String,
    items: Vec<Item>,
    featured: Option<Item>,
}

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
struct ToolCall {
    name: String,
}

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
struct Step {
    source: String,
    tool_calls: Option<Vec<ToolCall>>,
}

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
struct Trace {
    #[rlm(
        desc = "All steps in the trace",
        filter_property = "user_steps",
        filter_value = "user",
        filter_field = "source"
    )]
    steps: Vec<Step>,

    #[rlm(flatten_property = "all_tool_calls", flatten_parent = "steps")]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Sum the values of items.
struct SumItems {
    #[input]
    items: Vec<Item>,

    #[output]
    total: i32,
}

#[tokio::test]
async fn typed_rlm_construction_compiles() {
    let client = openai::CompletionsClient::new("test-key").expect("client builds");
    let agent = client.agent(openai::GPT_4O_MINI).build();
    let _rlm = TypedRlm::<SumItems>::new(agent, RlmConfig::default());
}

#[test]
fn rlm_type_getters_repr_and_baml() -> PyResult<()> {
    Python::attach(|py| {
        let item = Item {
            name: "apples".to_string(),
            value: 5,
        };
        let py_item = Py::new(py, item)?;
        let bound = py_item.bind(py);
        let any = bound.as_any();

        let name: String = any.getattr("name")?.extract()?;
        let value: i32 = any.getattr("value")?.extract()?;
        assert_eq!(name, "apples");
        assert_eq!(value, 5);

        let repr_obj = any.repr()?;
        let repr = repr_obj.to_str()?;
        assert_eq!(repr, "Item(apples, 5)");

        let baml = any.call_method0("__baml__")?;
        let dict = baml.cast::<PyDict>()?;
        let name_item = dict.get_item("name")?.expect("name field");
        let value_item = dict.get_item("value")?.expect("value field");
        let name: String = name_item.extract()?;
        let value: i32 = value_item.extract()?;
        assert_eq!(name, "apples");
        assert_eq!(value, 5);

        Ok(())
    })
}

#[test]
fn rlm_type_len_iter_getitem() -> PyResult<()> {
    Python::attach(|py| {
        let bag = Bag {
            items: vec![
                Item {
                    name: "apples".to_string(),
                    value: 5,
                },
                Item {
                    name: "bananas".to_string(),
                    value: 7,
                },
            ],
        };
        let py_bag = Py::new(py, bag)?;
        let bound = py_bag.bind(py);
        let any = bound.as_any();

        assert_eq!(any.len()?, 2);

        let first = any.get_item(0)?;
        let first_name: String = first.getattr("name")?.extract()?;
        assert_eq!(first_name, "apples");

        let iter_obj = any.call_method0("__iter__")?;
        let iter = PyIterator::from_object(&iter_obj)?;
        let mut names = Vec::new();
        for item in iter {
            let item = item?;
            let name: String = item.getattr("name")?.extract()?;
            names.push(name);
        }
        assert_eq!(names, vec!["apples".to_string(), "bananas".to_string()]);

        Ok(())
    })
}

#[test]
fn rlm_type_baml_roundtrip_and_schema() -> PyResult<()> {
    Python::attach(|py| {
        let pantry = Pantry {
            owner: "Darin".to_string(),
            items: vec![
                Item {
                    name: "apples".to_string(),
                    value: 5,
                },
                Item {
                    name: "bananas".to_string(),
                    value: 7,
                },
            ],
            featured: None,
        };
        let py_pantry = Py::new(py, pantry.clone())?;
        let bound = py_pantry.bind(py);
        let any = bound.as_any();

        let baml = any.call_method0("__baml__")?;
        let type_ir = <Pantry as BamlType>::baml_type_ir();
        let output_format = <Pantry as BamlType>::baml_output_format();
        let parsed = baml_py::py_to_baml_value(py, &baml, &type_ir, output_format)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let expected = pantry.to_baml_value();
        assert_eq!(parsed, expected);

        let roundtrip = <Pantry as BamlType>::try_from_baml_value(parsed)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        assert_eq!(roundtrip, pantry);

        let schema_obj = any.call_method0("__rlm_schema__")?;
        let schema = schema_obj.cast::<PyDict>()?;
        let owner_meta = schema.get_item("owner")?.expect("owner field");
        let (owner_type, owner_desc): (String, String) = owner_meta.extract()?;
        assert_eq!(owner_type, "String");
        assert_eq!(owner_desc, "Owner of the pantry");

        let items_meta = schema.get_item("items")?.expect("items field");
        let (items_type, _items_desc): (String, String) = items_meta.extract()?;
        assert!(items_type.contains("Vec"));

        let featured_meta = schema.get_item("featured")?.expect("featured field");
        let (featured_type, _featured_desc): (String, String) = featured_meta.extract()?;
        assert!(featured_type.contains("Option"));

        Ok(())
    })
}

#[test]
fn rlm_type_filter_and_flatten_properties() -> PyResult<()> {
    Python::attach(|py| {
        let trace = Trace {
            steps: vec![
                Step {
                    source: "user".to_string(),
                    tool_calls: Some(vec![ToolCall {
                        name: "search".to_string(),
                    }]),
                },
                Step {
                    source: "agent".to_string(),
                    tool_calls: Some(vec![
                        ToolCall {
                            name: "summarize".to_string(),
                        },
                        ToolCall {
                            name: "finalize".to_string(),
                        },
                    ]),
                },
                Step {
                    source: "user".to_string(),
                    tool_calls: None,
                },
            ],
            tool_calls: Some(vec![ToolCall {
                name: "ignored".to_string(),
            }]),
        };
        let py_trace = Py::new(py, trace)?;
        let bound = py_trace.bind(py);
        let any = bound.as_any();

        let user_steps_any = any.getattr("user_steps")?;
        let user_steps = user_steps_any.cast::<PyList>()?;
        assert_eq!(user_steps.len(), 2);
        for item in user_steps.iter() {
            let source: String = item.getattr("source")?.extract()?;
            assert_eq!(source, "user");
        }

        let all_tool_calls_any = any.getattr("all_tool_calls")?;
        let all_tool_calls = all_tool_calls_any.cast::<PyList>()?;
        let mut names = Vec::new();
        for item in all_tool_calls.iter() {
            let name: String = item.getattr("name")?.extract()?;
            names.push(name);
        }
        assert_eq!(
            names,
            vec![
                "search".to_string(),
                "summarize".to_string(),
                "finalize".to_string()
            ]
        );

        Ok(())
    })
}

#[test]
fn rlm_describe_is_stable_and_includes_properties() {
    use dspy_rs::rlm_core::describe::RlmDescribe;

    let first = <Trace as RlmDescribe>::describe_type();
    let second = <Trace as RlmDescribe>::describe_type();

    assert_eq!(first, second);
    assert!(first.starts_with("type Trace"));
    assert!(first.contains("properties:"));
    assert!(first.contains("user_steps"));
    assert!(first.contains("all_tool_calls"));
}
