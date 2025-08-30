use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::LmUsage;

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Prediction {
    pub data: HashMap<String, serde_json::Value>,
    pub lm_usage: LmUsage,
}

impl Prediction {
    pub fn new(data: HashMap<String, serde_json::Value>, lm_usage: LmUsage) -> Self {
        Self { data, lm_usage }
    }

    pub fn get(&self, key: &str, default: Option<&str>) -> serde_json::Value {
        self.data
            .get(key)
            .unwrap_or(&default.unwrap_or_default().to_string().into())
            .clone()
    }

    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<serde_json::Value> {
        self.data.values().cloned().collect()
    }

    pub fn set_lm_usage(&mut self, lm_usage: LmUsage) -> Self {
        self.lm_usage = lm_usage;
        self.clone()
    }
}
