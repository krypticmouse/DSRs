#![cfg(feature = "rlm")]

use dspy_rs::rlm::{RlmConfig, TypedRlm};
use dspy_rs::{rlm_type, RlmType, Signature};
use rig::prelude::*;
use rig::providers::openai;

#[rlm_type]
#[derive(Clone, Debug, PartialEq)]
#[rlm(repr = "Item({self.name}, {self.value})")]
struct Item {
    name: String,
    value: i32,
}

#[derive(Signature, Clone, Debug, PartialEq)]
/// Sum the values of items.
struct SumItems {
    #[input]
    items: Vec<Item>,

    #[output]
    total: i32,
}

#[tokio::test]
async fn typed_rlm_construction_compiles() {
    let client = openai::CompletionsClient::new("test-key").expect("client builds");
    let agent = client.agent(openai::GPT_4O_MINI).build();
    let _rlm = TypedRlm::<SumItems>::new(agent, RlmConfig::default());
}
