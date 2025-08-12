// a module for special types
// right now most of these are just placeholders
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, JsonSchema, Clone)]
pub struct History;
#[derive(Serialize, JsonSchema, Clone)]
pub struct Tool;
#[derive(Deserialize, JsonSchema, Clone)]
pub struct ToolCall;
