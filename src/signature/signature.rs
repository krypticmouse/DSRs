use derive_builder::Builder;
use indexmap::IndexMap;
use std::fmt;

use crate::signature::field::Field;

#[derive(Builder, Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    #[builder(default = "String::new()")]
    pub instruction: String,

    pub input_fields: IndexMap<String, Field>,
    pub output_fields: IndexMap<String, Field>,
}

impl Signature {
    pub fn insert(&mut self, field_name: String, field: Field, index: usize) {
        match &field {
            Field::InputField { .. } => {
                self.input_fields.insert_before(index, field_name, field);
            }
            Field::OutputField { .. } => {
                self.output_fields.insert_before(index, field_name, field);
            }
        }
    }

    pub fn append(&mut self, field_name: String, field: Field) {
        match &field {
            Field::InputField { .. } => {
                self.input_fields.insert_before(0, field_name, field);
            }
            Field::OutputField { .. } => {
                self.output_fields.insert_before(0, field_name, field);
            }
        }
    }

    pub fn prepend(&mut self, field_name: String, field: Field) {
        let index = self.input_fields.len();

        match &field {
            Field::InputField { .. } => {
                self.input_fields.insert_before(index, field_name, field);
            }
            Field::OutputField { .. } => {
                self.output_fields.insert_before(index, field_name, field);
            }
        }
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

impl From<String> for Signature {
    fn from(signature: String) -> Self {
        let fields = signature.split("->").collect::<Vec<&str>>();
        let input_fields: Vec<&str> = fields[0].split(",").map(|s| s.trim()).collect();
        let output_fields: Vec<&str> = fields[1].split(",").map(|s| s.trim()).collect();
        let default_desc: String = format!(
            "Given a inputs {}, return outputs {}",
            input_fields.join(", "),
            output_fields.join(", ")
        );

        let input_fields_map = IndexMap::from_iter(input_fields.iter().map(|field| {
            (
                field.to_string(),
                Field::InputField {
                    prefix: String::new(),
                    desc: String::new(),
                    format: None,
                    output_type: String::new(),
                },
            )
        }));

        let output_fields_map = IndexMap::from_iter(output_fields.iter().map(|field| {
            (
                (*field).to_string(),
                Field::OutputField {
                    prefix: String::new(),
                    desc: String::new(),
                    format: None,
                    output_type: String::new(),
                },
            )
        }));

        Self {
            instruction: default_desc,
            input_fields: input_fields_map,
            output_fields: output_fields_map,
        }
    }
}
