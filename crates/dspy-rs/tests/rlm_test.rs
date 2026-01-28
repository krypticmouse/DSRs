#![cfg(feature = "rlm")]

use dspy_rs::rlm::{RlmConfig, TypedRlm};
use dspy_rs::{rlm_type, RlmType, Signature};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyIterator};
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
