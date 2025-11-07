use indexmap::IndexMap;
use rig::tool::ToolDyn;
use std::sync::Arc;

use crate::core::{MetaSignature, Optimizable};
use crate::{ChatAdapter, Example, GLOBAL_SETTINGS, LM, Prediction, adapter::Adapter};

pub struct Predict {
    pub signature: Box<dyn MetaSignature>,
    pub tools: Vec<Arc<dyn ToolDyn>>,
}

impl Predict {
    pub fn new(signature: impl MetaSignature + 'static) -> Self {
        Self {
            signature: Box::new(signature),
            tools: vec![],
        }
    }

    pub fn new_with_tools(
        signature: impl MetaSignature + 'static,
        tools: Vec<Box<dyn ToolDyn>>,
    ) -> Self {
        Self {
            signature: Box::new(signature),
            tools: tools.into_iter().map(Arc::from).collect(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<Box<dyn ToolDyn>>) -> Self {
        self.tools = tools.into_iter().map(Arc::from).collect();
        self
    }

    pub fn add_tool(mut self, tool: Box<dyn ToolDyn>) -> Self {
        self.tools.push(Arc::from(tool));
        self
    }
}

impl super::Predictor for Predict {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        let (adapter, lm) = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            (settings.adapter.clone(), Arc::clone(&settings.lm))
        }; // guard is dropped here
        adapter
            .call(lm, self.signature.as_ref(), inputs, self.tools.clone())
            .await
    }

    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>,
    ) -> anyhow::Result<Prediction> {
        ChatAdapter
            .call(lm, self.signature.as_ref(), inputs, self.tools.clone())
            .await
    }
}

impl Optimizable for Predict {
    fn get_signature(&self) -> &dyn MetaSignature {
        self.signature.as_ref()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        IndexMap::new()
    }

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        let _ = self.signature.update_instruction(instruction);
        Ok(())
    }
}
