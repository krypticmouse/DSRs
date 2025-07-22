use derive_builder::Builder;
use indexmap::IndexMap;
use std::fmt;

use crate::signature::field::Field;

#[derive(Builder, Debug, Clone, PartialEq, Eq)]
pub struct Signature<'a> {
    pub name: &'a str,
    pub instruction: String,

    pub input_fields: IndexMap<String, Field<'a>>,
    pub output_fields: IndexMap<String, Field<'a>>,
}

impl<'a> Signature<'a> {
    pub fn insert(&mut self, field_name: String, field: Field<'a>, index: usize) {
        match &field {
            Field::In(_) => {
                self.input_fields.insert_before(index, field_name, field);
            }
            Field::Out(_) => {
                self.output_fields.insert_before(index, field_name, field);
            }
        }
    }

    pub fn append(&mut self, field_name: String, field: Field<'a>) {
        match &field {
            Field::In(_) => {
                self.input_fields.insert_before(0, field_name, field);
            }
            Field::Out(_) => {
                self.output_fields.insert_before(0, field_name, field);
            }
        }
    }

    pub fn prepend(&mut self, field_name: String, field: Field<'a>) {
        let index = self.input_fields.len();

        match &field {
            Field::In(_) => {
                self.input_fields.insert_before(index, field_name, field);
            }
            Field::Out(_) => {
                self.output_fields.insert_before(index, field_name, field);
            }
        }
    }

    pub fn builder() -> SignatureBuilder<'a> {
        SignatureBuilder::default()
    }
}

impl<'a> fmt::Display for Signature<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let input_str = self
            .input_fields
            .iter()
            .map(|(key, value)| format!("{key}: {value}"))
            .collect::<Vec<String>>()
            .join(",\n\t");
        let output_str = self
            .output_fields
            .iter()
            .map(|(key, value)| format!("{key}: {value}"))
            .collect::<Vec<String>>()
            .join(",\n\t");

        write!(
            f,
            "Signature(\n\tdescription: {},\n\tinput_fields: {},\n\toutput_fields: {}\n)",
            self.instruction, input_str, output_str
        )
    }
}

impl<'a> From<&'a str> for Signature<'a> {
    fn from(signature: &'a str) -> Self {
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
                .map(|field| (field.to_string(), Field::In(""))),
        );

        let output_fields_map = IndexMap::from_iter(
            output_fields
                .iter()
                .map(|field| (field.to_string(), Field::Out(""))),
        );

        Self {
            name: signature,
            instruction: default_desc,
            input_fields: input_fields_map,
            output_fields: output_fields_map,
        }
    }
}
