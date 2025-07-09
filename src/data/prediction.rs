use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Default, Debug, Clone, PartialEq, Eq)]
pub struct LmUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Prediction {
    pub data: HashMap<String, String>,
    pub lm_usage: LmUsage,
}

impl Prediction {
    pub fn new(data: HashMap<String, String>) -> Self {
        Self {
            data,
            lm_usage: LmUsage::default(),
        }
    }

    pub fn set_lm_usage(&mut self, lm_usage: LmUsage) {
        self.lm_usage = lm_usage;
    }

    pub fn get(&self, key: &str, default: Option<&str>) -> String {
        self.data
            .get(key)
            .unwrap_or(&default.unwrap_or_default().to_string())
            .clone()
    }

    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<String> {
        self.data.values().cloned().collect()
    }
}
