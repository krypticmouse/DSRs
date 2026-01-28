/*
Demonstrate Python integration for #[rlm_type] values.

Run with:
```
cargo run --example python_types --features rlm
```
*/

use anyhow::Result;
use dspy_rs::{rlm_type, RlmType};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[rlm_type]
#[derive(Clone, Debug)]
#[rlm(repr = "Person({self.name}, {self.age})")]
struct Person {
    #[rlm(desc = "Full name")]
    name: String,
    #[rlm(desc = "Age in years")]
    age: i32,
}

fn main() -> Result<()> {
    Python::attach(|py| -> PyResult<()> {
        let person = Person {
            name: "Ada Lovelace".to_string(),
            age: 36,
        };
        let py_person = Py::new(py, person)?;
        let any = py_person.bind(py);

        let repr = any.repr()?.to_str()?.to_string();
        println!("repr: {repr}");

        let schema_obj = any.call_method0("__rlm_schema__")?;
        let schema = schema_obj.cast::<PyDict>()?;
        let schema_repr = schema.repr()?.to_str()?.to_string();
        println!("__rlm_schema__: {schema_repr}");

        let baml_obj = any.call_method0("__baml__")?;
        let baml_repr = baml_obj.repr()?.to_str()?.to_string();
        println!("__baml__: {baml_repr}");

        Ok(())
    })?;

    Ok(())
}
