use derive_builder::Builder;
use indexmap::IndexMap;
use std::fmt;

use crate::field::{In, Out};

#[derive(Builder, Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub name: String,
    pub instruction: String,

    pub input_fields: IndexMap<String, In>,
    pub output_fields: IndexMap<String, Out>,
}

impl Signature {
    pub fn insert_input(&mut self, field_name: String, field: In, index: usize) {
        self.input_fields.insert_before(index, field_name, field);
    }

    pub fn insert_output(&mut self, field_name: String, field: Out, index: usize) {
        self.output_fields.insert_before(index, field_name, field);
    }

    pub fn append_input(&mut self, field_name: String, field: In) {
        self.input_fields.insert_before(0, field_name, field);
    }

    pub fn append_output(&mut self, field_name: String, field: Out) {
        self.output_fields.insert_before(0, field_name, field);
    }

    pub fn prepend_input(&mut self, field_name: String, field: In) {
        self.input_fields.insert_before(0, field_name, field);
    }

    pub fn prepend_output(&mut self, field_name: String, field: Out) {
        self.output_fields.insert_before(0, field_name, field);
    }

    pub fn builder() -> SignatureBuilder {
        SignatureBuilder::default()
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let input_str = self
            .input_fields
            .iter()
            .map(|(key, value)| format!("{key}: {value:?}"))
            .collect::<Vec<String>>()
            .join(",\n\t");
        let output_str = self
            .output_fields
            .iter()
            .map(|(key, value)| format!("{key}: {value:?}"))
            .collect::<Vec<String>>()
            .join(",\n\t");

        write!(
            f,
            "Signature(\n\tdescription: {},\n\tinput_fields: {},\n\toutput_fields: {}\n)",
            self.instruction, input_str, output_str
        )
    }
}

impl From<&str> for Signature {
    fn from(signature: &str) -> Self {
        let fields = signature.split("->").collect::<Vec<&str>>();
        let input_fields: Vec<&str> = fields[0].split(",").map(|s| s.trim()).collect();
        let output_fields: Vec<&str> = fields[1].split(",").map(|s| s.trim()).collect();
        let default_desc: String = format!(
            "Given a inputs {}, return outputs {}",
            input_fields.join(", "),
            output_fields.join(", ")
        );

        let input_fields_map = IndexMap::from_iter(
            input_fields
                .iter()
                .map(|field| (field.to_string(), In::default())),
        );

        let output_fields_map = IndexMap::from_iter(
            output_fields
                .iter()
                .map(|field| (field.to_string(), Out::default())),
        );

        Self {
            name: signature.to_string(),
            instruction: default_desc,
            input_fields: input_fields_map,
            output_fields: output_fields_map,
        }
    }
}
