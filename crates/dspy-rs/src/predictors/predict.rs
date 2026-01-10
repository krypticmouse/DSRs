use indexmap::IndexMap;
use rig::tool::ToolDyn;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::adapter::Adapter;
use crate::core::{MetaSignature, Optimizable, Signature};
use crate::{
    CallResult, Chat, ChatAdapter, Example, GLOBAL_SETTINGS, LmError, LM, PredictError, Prediction,
};
use crate::baml_bridge::ToBamlValue;

pub struct Predict<S: Signature> {
    tools: Vec<Arc<dyn ToolDyn>>,
    demos: Vec<S>,
    instruction_override: Option<String>,
    _marker: PhantomData<S>,
}

impl<S: Signature> Predict<S> {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            demos: Vec::new(),
            instruction_override: None,
            _marker: PhantomData,
        }
    }

    pub fn builder() -> PredictBuilder<S> {
        PredictBuilder::new()
    }

    pub async fn call(&self, input: S::Input) -> Result<S, PredictError>
    where
        S: Clone,
        S::Input: ToBamlValue,
        S::Output: ToBamlValue,
    {
        Ok(self.call_with_meta(input).await?.output)
    }

    pub async fn call_with_meta(&self, input: S::Input) -> Result<CallResult<S>, PredictError>
    where
        S: Clone,
        S::Input: ToBamlValue,
        S::Output: ToBamlValue,
    {
        let lm = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            Arc::clone(&settings.lm)
        };

        let chat_adapter = ChatAdapter;
        let system = chat_adapter
            .format_system_message_typed_with_instruction::<S>(
                self.instruction_override.as_deref(),
            )
            .map_err(|err| PredictError::Lm {
                source: LmError::Provider {
                    provider: "internal".to_string(),
                    message: err.to_string(),
                    source: None,
                },
            })?;
        let user = chat_adapter.format_user_message_typed::<S>(&input);

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system);
        for demo in &self.demos {
            let (demo_user, demo_assistant) = chat_adapter.format_demo_typed::<S>(demo.clone());
            chat.push("user", &demo_user);
            chat.push("assistant", &demo_assistant);
        }
        chat.push("user", &user);

        let response = lm
            .call(chat, self.tools.clone())
            .await
            .map_err(|err| PredictError::Lm {
                source: LmError::Provider {
                    provider: lm.model.clone(),
                    message: err.to_string(),
                    source: None,
                },
            })?;

        let raw_response = response.output.content().to_string();
        let lm_usage = response.usage.clone();
        let (typed_output, field_metas) =
            chat_adapter
                .parse_response_typed::<S>(&response.output)
                .map_err(|err| PredictError::Parse {
                    source: err,
                    raw_response: raw_response.clone(),
                    lm_usage: lm_usage.clone(),
                })?;

        let output = S::from_parts(input, typed_output);

        let node_id = if crate::trace::is_tracing() {
            crate::trace::record_node(
                crate::trace::NodeType::Predict {
                    signature_name: std::any::type_name::<S>().to_string(),
                },
                vec![],
                None,
            )
        } else {
            None
        };

        Ok(CallResult::new(
            output,
            raw_response,
            lm_usage,
            response.tool_calls,
            response.tool_executions,
            node_id,
            field_metas,
        ))
    }
}

impl<S: Signature> Default for Predict<S> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PredictBuilder<S: Signature> {
    tools: Vec<Arc<dyn ToolDyn>>,
    demos: Vec<S>,
    instruction_override: Option<String>,
    _marker: PhantomData<S>,
}

impl<S: Signature> PredictBuilder<S> {
    fn new() -> Self {
        Self {
            tools: Vec::new(),
            demos: Vec::new(),
            instruction_override: None,
            _marker: PhantomData,
        }
    }

    pub fn demo(mut self, demo: S) -> Self {
        self.demos.push(demo);
        self
    }

    pub fn with_demos(mut self, demos: impl IntoIterator<Item = S>) -> Self {
        self.demos.extend(demos);
        self
    }

    pub fn add_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Arc<dyn ToolDyn>>) -> Self {
        self.tools.extend(tools);
        self
    }

    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction_override = Some(instruction.into());
        self
    }

    pub fn build(self) -> Predict<S> {
        Predict {
            tools: self.tools,
            demos: self.demos,
            instruction_override: self.instruction_override,
            _marker: PhantomData,
        }
    }
}

pub struct LegacyPredict {
    pub signature: Arc<dyn MetaSignature>,
    pub tools: Vec<Arc<dyn ToolDyn>>,
}

impl LegacyPredict {
    pub fn new(signature: impl MetaSignature + 'static) -> Self {
        Self {
            signature: Arc::new(signature),
            tools: vec![],
        }
    }

    pub fn new_with_tools(
        signature: impl MetaSignature + 'static,
        tools: Vec<Box<dyn ToolDyn>>,
    ) -> Self {
        Self {
            signature: Arc::new(signature),
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

impl super::Predictor for LegacyPredict {
    async fn forward(&self, inputs: Example) -> anyhow::Result<Prediction> {
        let trace_node_id = if crate::trace::is_tracing() {
            let input_id = if let Some(id) = inputs.node_id {
                id
            } else {
                crate::trace::record_node(
                    crate::trace::NodeType::Root,
                    vec![],
                    Some(inputs.clone()),
                )
                .unwrap_or(0)
            };

            crate::trace::record_node(
                crate::trace::NodeType::Predict {
                    signature_name: "LegacyPredict".to_string(),
                },
                vec![input_id],
                None,
            )
        } else {
            None
        };

        let (adapter, lm) = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            (settings.adapter.clone(), Arc::clone(&settings.lm))
        }; // guard is dropped here
        let mut prediction = adapter
            .call(lm, self.signature.as_ref(), inputs, self.tools.clone())
            .await?;

        if let Some(id) = trace_node_id {
            prediction.node_id = Some(id);
            crate::trace::record_output(id, prediction.clone());
        }

        Ok(prediction)
    }

    async fn forward_with_config(
        &self,
        inputs: Example,
        lm: Arc<LM>,
    ) -> anyhow::Result<Prediction> {
        let trace_node_id = if crate::trace::is_tracing() {
            let input_id = if let Some(id) = inputs.node_id {
                id
            } else {
                crate::trace::record_node(
                    crate::trace::NodeType::Root,
                    vec![],
                    Some(inputs.clone()),
                )
                .unwrap_or(0)
            };

            crate::trace::record_node(
                crate::trace::NodeType::Predict {
                    signature_name: "LegacyPredict".to_string(),
                },
                vec![input_id],
                None,
            )
        } else {
            None
        };

        let mut prediction = ChatAdapter
            .call(lm, self.signature.as_ref(), inputs, self.tools.clone())
            .await?;

        if let Some(id) = trace_node_id {
            prediction.node_id = Some(id);
            crate::trace::record_output(id, prediction.clone());
        }

        Ok(prediction)
    }
}

impl Optimizable for LegacyPredict {
    fn get_signature(&self) -> &dyn MetaSignature {
        self.signature.as_ref()
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        IndexMap::new()
    }

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        if let Some(sig) = Arc::get_mut(&mut self.signature) {
            sig.update_instruction(instruction)?;
            Ok(())
        } else {
            // If Arc is shared, we might need to clone it first?
            // But Optimizable usually assumes exclusive access for modification.
            // If we are optimizing, we should have ownership or mutable access.
            // If tracing is active, `LegacyPredict` instances might be shared in Graph, but here we are modifying the instance.
            // If we can't get mut, it means it's shared.
            // We can clone-on-write? But MetaSignature is a trait object, so we can't easily clone it unless we implement Clone for Box<dyn MetaSignature>.
            // However, we changed it to Arc.
            // If we are running optimization, we probably shouldn't be tracing or the graph is already built.
            // For now, let's error or assume we can clone if we had a way.
            // But actually, we can't clone `dyn MetaSignature` easily without more boilerplate.
            // Let's assume unique ownership for optimization.
            anyhow::bail!(
                "Cannot update signature instruction: Signature is shared (Arc has multiple strong references)"
            )
        }
    }
}
