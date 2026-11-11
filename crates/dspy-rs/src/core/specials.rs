//! Typed signature fields for conversation history and caller-managed tools.
use crate::typesys::{FieldType, Schema};
use facet::Facet;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Facet)]
pub struct History {
    pub messages: Vec<HistoryMessage>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Facet)]
pub struct HistoryMessage {
    pub role: String,
    pub content: String,
}

/// A serializable tool declaration. Executable tools remain host-owned; a
/// signature with ToolCalls returns requests for the caller to execute.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Facet)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the argument object, encoded as JSON text.
    pub parameters: String,
}

impl Tool {
    pub fn from_definition(definition: &rig::completion::ToolDefinition) -> Self {
        Self {
            name: definition.name.clone(),
            description: definition.description.clone(),
            parameters: definition.parameters.to_string(),
        }
    }
    pub fn definition(&self) -> Result<rig::completion::ToolDefinition, serde_json::Error> {
        Ok(rig::completion::ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: serde_json::from_str(&self.parameters)?,
        })
    }
}

/// Renamed to avoid colliding with rig::message::ToolCall.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Facet)]
pub struct ToolInvocation {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Facet)]
pub struct ToolCalls {
    pub calls: Vec<ToolInvocation>,
}

impl ToolCalls {
    pub fn from_native(calls: &[rig::message::ToolCall]) -> Self {
        Self {
            calls: calls
                .iter()
                .map(|call| ToolInvocation {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.to_string(),
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialField {
    History,
    Tools,
    ToolCalls,
}

pub(crate) fn special_field(ty: &FieldType) -> Option<SpecialField> {
    let FieldType::Class(name) = ty.peel() else {
        return None;
    };
    if *name == History::internal_name() {
        Some(SpecialField::History)
    } else if *name == Tool::internal_name() {
        Some(SpecialField::Tools)
    } else if *name == ToolCalls::internal_name() {
        Some(SpecialField::ToolCalls)
    } else {
        None
    }
}
