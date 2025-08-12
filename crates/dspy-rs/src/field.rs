use schemars::{JsonSchema, schema_for};
use std::{fmt::Debug, marker::PhantomData};

pub trait Field {
    fn desc(&self) -> String;
    fn field_type(&self) -> String;
    fn schema(&self) -> String;
    fn data_type(&self) -> String;
}

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct In<T: JsonSchema> {
    phantom: PhantomData<T>,
    pub desc: String,
}

impl<T: JsonSchema> Field for In<T> {
    fn desc(&self) -> String {
        self.desc.clone()
    }

    fn field_type(&self) -> String {
        "Input".to_string()
    }

    fn schema(&self) -> String {
        let json_value = serde_json::to_value(schema_for!(T)).unwrap();

        if let Some(properties) = json_value
            .as_object()
            .and_then(|obj| obj.get("properties"))
            .and_then(|props| props.as_object())
        {
            serde_json::to_string(&properties).unwrap_or_else(|_| "".to_string())
        } else {
            "".to_string()
        }
    }

    fn data_type(&self) -> String {
        std::any::type_name::<T>().to_string()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct Out<T: JsonSchema> {
    phantom: PhantomData<T>,
    pub desc: String,
}

impl<T: JsonSchema> Field for Out<T> {
    fn desc(&self) -> String {
        self.desc.clone()
    }

    fn field_type(&self) -> String {
        "Output".to_string()
    }

    fn schema(&self) -> String {
        let json_value = serde_json::to_value(schema_for!(T)).unwrap();

        if let Some(properties) = json_value
            .as_object()
            .and_then(|obj| obj.get("properties"))
            .and_then(|props| props.as_object())
        {
            serde_json::to_string(&properties).unwrap_or_else(|_| "".to_string())
        } else {
            "".to_string()
        }
    }

    fn data_type(&self) -> String {
        std::any::type_name::<T>().to_string()
    }
}
