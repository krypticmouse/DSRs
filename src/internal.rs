use crate::field::Field;
use indexmap::IndexMap;

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub enum FieldType {
    #[default]
    Input,
    Output,
}

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct MetaField {
    pub desc: String,
    pub schema: String,
    pub data_type: String,
    pub __dsrs_field_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaSignature {
    pub name: String,
    pub instruction: String,

    pub input_fields: IndexMap<String, MetaField>,
    pub output_fields: IndexMap<String, MetaField>,
}

impl MetaSignature {
    pub fn update_instruction(&mut self, instruction: String) {
        self.instruction = instruction;
    }

    pub fn insert(&mut self, field_name: String, field: impl Field, index: usize) {
        let meta_field = MetaField {
            desc: field.desc(),
            schema: field.schema(),
            data_type: field.data_type(),
            __dsrs_field_type: field.field_type(),
        };

        if field.field_type() == "Input" {
            self.input_fields
                .insert_before(index, field_name, meta_field);
        } else {
            self.output_fields
                .insert_before(index, field_name, meta_field);
        }
    }

    pub fn append(&mut self, field_name: String, field: impl Field) {
        let meta_field = MetaField {
            desc: field.desc(),
            schema: field.schema(),
            data_type: field.data_type(),
            __dsrs_field_type: field.field_type(),
        };

        if field.field_type() == "Input" {
            self.input_fields.insert_before(0, field_name, meta_field);
        } else {
            self.output_fields.insert_before(0, field_name, meta_field);
        }
    }

    pub fn prepend(&mut self, field_name: String, field: impl Field) {
        let meta_field = MetaField {
            desc: field.desc(),
            schema: field.schema(),
            data_type: field.data_type(),
            __dsrs_field_type: field.field_type(),
        };

        if field.field_type() == "Input" {
            self.input_fields.insert_before(0, field_name, meta_field);
        } else {
            self.output_fields.insert_before(0, field_name, meta_field);
        }
    }
}
