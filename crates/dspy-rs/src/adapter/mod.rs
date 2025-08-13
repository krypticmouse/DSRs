pub mod chat_adapter;

pub use chat_adapter::*;

use crate::core::Adapter;

#[derive(Clone)]
pub enum ConcreteAdapter {
    Chat(ChatAdapter),
}

impl Default for ConcreteAdapter {
    fn default() -> Self {
        ConcreteAdapter::Chat(ChatAdapter::default())
    }
}

impl Adapter for ConcreteAdapter {
    fn call<S: crate::core::Signature>(
        &self,
        lm: &mut crate::core::LM,
        signature: &S,
        inputs: &S::Inputs,
    ) -> impl Future<Output = anyhow::Result<S::Outputs>> {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.call(lm, signature, inputs),
        }
    }

    fn format<S: crate::core::Signature>(
        &self,
        signature: &S,
        inputs: &S::Inputs,
    ) -> crate::core::Chat {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.format(signature, inputs),
        }
    }

    fn parse<S: crate::core::Signature>(
        &self,
        signature: &S,
        response: crate::core::Message,
    ) -> anyhow::Result<S::Outputs> {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.parse(signature, response),
        }
    }

    fn format_field_description(&self, signature: &impl crate::core::Signature) -> String {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.format_field_description(signature),
        }
    }

    fn format_field_structure(&self, signature: &impl crate::core::Signature) -> String {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.format_field_structure(signature),
        }
    }

    fn format_system_message(&self, signature: &impl crate::core::Signature) -> String {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.format_system_message(signature),
        }
    }

    fn format_task_description(&self, signature: &impl crate::core::Signature) -> String {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.format_task_description(signature),
        }
    }

    fn format_user_message(
        &self,
        signature: &impl crate::core::Signature,
        inputs: crate::data::example::Example,
    ) -> String {
        match self {
            ConcreteAdapter::Chat(adapter) => adapter.format_user_message(signature, inputs),
        }
    }
}
