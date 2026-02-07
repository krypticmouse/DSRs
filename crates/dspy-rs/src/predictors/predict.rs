use anyhow::Result;
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use tracing::{debug, trace};

use crate::adapter::Adapter;
use crate::bamltype::baml_types::BamlMap;
use crate::bamltype::compat::{BamlValueConvert, ToBamlValue};
use crate::core::{FieldSpec, InputRenderSpec, MetaSignature, Module, Optimizable, Signature};
use crate::{
    BamlValue, CallResult, Chat, ChatAdapter, Example, GLOBAL_SETTINGS, LM, LmError, LmUsage,
    PredictError, Prediction,
};

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

    #[tracing::instrument(
        name = "dsrs.predict.call_with_meta",
        level = "debug",
        skip(self, input),
        fields(
            signature = std::any::type_name::<S>(),
            demo_count = self.demos.len(),
            tool_count = self.tools.len(),
            instruction_override = self.instruction_override.is_some(),
            tracing_graph = crate::trace::is_tracing()
        )
    )]
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
            .format_system_message_typed_with_instruction::<S>(self.instruction_override.as_deref())
            .map_err(|err| PredictError::Lm {
                source: LmError::Provider {
                    provider: "internal".to_string(),
                    message: err.to_string(),
                    source: None,
                },
            })?;
        let user = chat_adapter.format_user_message_typed::<S>(&input);
        trace!(
            system_len = system.len(),
            user_len = user.len(),
            "typed prompt formatted"
        );

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system);
        for demo in &self.demos {
            let (demo_user, demo_assistant) = chat_adapter.format_demo_typed::<S>(demo.clone());
            chat.push("user", &demo_user);
            chat.push("assistant", &demo_assistant);
        }
        chat.push("user", &user);
        trace!(message_count = chat.len(), "chat constructed");

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
        debug!(
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            tool_calls = response.tool_calls.len(),
            "lm response received"
        );

        let raw_response = response.output.content().to_string();
        let lm_usage = response.usage.clone();
        let (typed_output, field_metas) =
            match chat_adapter.parse_response_typed::<S>(&response.output) {
                Ok(parsed) => parsed,
                Err(err) => {
                    let fields = err.fields();
                    debug!(
                        failed_fields = fields.len(),
                        fields = ?fields,
                        raw_response_len = raw_response.len(),
                        "typed parse failed"
                    );
                    return Err(PredictError::Parse {
                        source: err,
                        raw_response: raw_response.clone(),
                        lm_usage: lm_usage.clone(),
                    });
                }
            };
        let checks_total = field_metas
            .values()
            .map(|meta| meta.checks.len())
            .sum::<usize>();
        let checks_failed = field_metas
            .values()
            .flat_map(|meta| meta.checks.iter())
            .filter(|check| !check.passed)
            .count();
        let flagged_fields = field_metas
            .values()
            .filter(|meta| !meta.flags.is_empty())
            .count();
        debug!(
            output_fields = field_metas.len(),
            checks_total, checks_failed, flagged_fields, "typed parse completed"
        );

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

        if let Some(id) = node_id {
            match prediction_from_output::<S>(&typed_output, lm_usage.clone(), Some(id)) {
                Ok(prediction) => {
                    crate::trace::record_output(id, prediction);
                    trace!(node_id = id, "recorded typed predictor output");
                }
                Err(err) => {
                    debug!(error = %err, "failed to build typed prediction for trace output");
                }
            }
        }

        let output = S::from_parts(input, typed_output);

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

fn field_specs_to_value(fields: &[FieldSpec], field_type: &'static str) -> Value {
    let mut result = serde_json::Map::new();
    for field in fields {
        let type_repr = (field.type_ir)().diagnostic_repr().to_string();
        let mut meta = serde_json::Map::new();
        meta.insert("type".to_string(), json!(type_repr));
        meta.insert("desc".to_string(), json!(field.description));
        meta.insert("schema".to_string(), json!(""));
        meta.insert("__dsrs_field_type".to_string(), json!(field_type));
        match field.input_render {
            InputRenderSpec::Default => {}
            InputRenderSpec::Format(format) => {
                meta.insert("format".to_string(), json!(format));
            }
            InputRenderSpec::Jinja(template) => {
                meta.insert("render".to_string(), json!({ "jinja": template }));
            }
        }
        result.insert(field.rust_name.to_string(), Value::Object(meta));
    }
    Value::Object(result)
}

fn baml_map_from_example_keys(
    data: &HashMap<String, Value>,
    keys: &[String],
) -> Result<BamlMap<String, BamlValue>> {
    let mut map = BamlMap::new();
    for key in keys {
        if let Some(value) = data.get(key) {
            let baml_value =
                BamlValue::try_from(value.clone()).map_err(|err| anyhow::anyhow!(err))?;
            map.insert(key.clone(), baml_value);
        }
    }
    Ok(map)
}

fn input_keys_for_signature<S: Signature>(example: &Example) -> Vec<String> {
    if example.input_keys.is_empty() {
        S::input_fields()
            .iter()
            .map(|field| field.rust_name.to_string())
            .collect()
    } else {
        example.input_keys.clone()
    }
}

fn output_keys_for_signature<S: Signature>(example: &Example) -> Vec<String> {
    if example.output_keys.is_empty() {
        S::output_fields()
            .iter()
            .map(|field| field.rust_name.to_string())
            .collect()
    } else {
        example.output_keys.clone()
    }
}

fn input_from_example<S: Signature>(example: &Example) -> Result<S::Input>
where
    S::Input: BamlValueConvert,
{
    let keys = input_keys_for_signature::<S>(example);
    let map = baml_map_from_example_keys(&example.data, &keys)?;
    let baml_value = BamlValue::Map(map);
    S::Input::try_from_baml_value(baml_value, Vec::new()).map_err(|err| anyhow::anyhow!(err))
}

fn output_from_example<S: Signature>(example: &Example) -> Result<S::Output>
where
    S::Output: BamlValueConvert,
{
    let keys = output_keys_for_signature::<S>(example);
    let map = baml_map_from_example_keys(&example.data, &keys)?;
    let baml_value = BamlValue::Map(map);
    S::Output::try_from_baml_value(baml_value, Vec::new()).map_err(|err| anyhow::anyhow!(err))
}

fn signature_from_example<S: Signature>(example: Example) -> Result<S>
where
    S::Input: BamlValueConvert,
    S::Output: BamlValueConvert,
{
    let input = input_from_example::<S>(&example)?;
    let output = output_from_example::<S>(&example)?;
    Ok(S::from_parts(input, output))
}

fn example_from_signature<S: Signature>(signature: S) -> Result<Example>
where
    S::Input: ToBamlValue,
    S::Output: ToBamlValue,
{
    let (input, output) = signature.into_parts();
    let input_value = serde_json::to_value(input.to_baml_value())?;
    let output_value = serde_json::to_value(output.to_baml_value())?;

    let input_map = input_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected object for signature input"))?
        .clone();
    let output_map = output_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected object for signature output"))?
        .clone();

    let input_keys = input_map.keys().cloned().collect();
    let output_keys = output_map.keys().cloned().collect();

    let mut data = HashMap::new();
    data.extend(input_map);
    data.extend(output_map);

    Ok(Example::new(data, input_keys, output_keys))
}

fn prediction_from_output<S: Signature>(
    output: &S::Output,
    lm_usage: LmUsage,
    node_id: Option<usize>,
) -> Result<Prediction>
where
    S::Output: ToBamlValue,
{
    let output_value = serde_json::to_value(output.to_baml_value())?;
    let output_map = output_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("expected object for signature output"))?;

    let data = output_map
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<HashMap<String, Value>>();
    let mut prediction = Prediction::new(data, lm_usage);
    prediction.node_id = node_id;
    Ok(prediction)
}

impl<S> Module for Predict<S>
where
    S: Signature + Clone + ToBamlValue,
    S::Input: ToBamlValue + BamlValueConvert,
    S::Output: ToBamlValue + BamlValueConvert,
{
    #[tracing::instrument(
        name = "dsrs.module.forward",
        level = "debug",
        skip(self, inputs),
        fields(
            signature = std::any::type_name::<S>(),
            input_keys = inputs.input_keys.len(),
            output_keys = inputs.output_keys.len()
        )
    )]
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let typed_input = input_from_example::<S>(&inputs).map_err(|err| {
            debug!(error = %err, "typed input conversion failed");
            err
        })?;
        let call_result = self.call_with_meta(typed_input).await.map_err(|err| {
            debug!(error = %err, "predict call_with_meta failed");
            anyhow::anyhow!(err)
        })?;
        let (_, output) = call_result.output.into_parts();
        let prediction =
            prediction_from_output::<S>(&output, call_result.lm_usage, call_result.node_id)?;
        debug!(
            output_fields = prediction.data.len(),
            "typed module forward complete"
        );
        Ok(prediction)
    }

    #[tracing::instrument(
        name = "dsrs.module.forward_untyped",
        level = "debug",
        skip(self, input),
        fields(signature = std::any::type_name::<S>())
    )]
    async fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> std::result::Result<BamlValue, PredictError> {
        let typed_input =
            S::Input::try_from_baml_value(input.clone(), Vec::new()).map_err(|err| {
                debug!(error = %err, "untyped input conversion failed");
                PredictError::Conversion {
                    source: err.into(),
                    parsed: input,
                }
            })?;
        let output = self.call(typed_input).await?;
        debug!("typed module forward_untyped complete");
        Ok(output.to_baml_value())
    }
}

impl<S> MetaSignature for Predict<S>
where
    S: Signature + Clone,
    S::Input: BamlValueConvert + ToBamlValue,
    S::Output: BamlValueConvert + ToBamlValue,
{
    fn demos(&self) -> Vec<Example> {
        self.demos
            .iter()
            .cloned()
            .map(|demo| {
                example_from_signature(demo).expect("typed Predict demo conversion should succeed")
            })
            .collect()
    }

    fn set_demos(&mut self, demos: Vec<Example>) -> Result<()> {
        self.demos = demos
            .into_iter()
            .map(signature_from_example::<S>)
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    fn instruction(&self) -> String {
        self.instruction_override
            .clone()
            .unwrap_or_else(|| S::instruction().to_string())
    }

    fn input_fields(&self) -> Value {
        field_specs_to_value(S::input_fields(), "input")
    }

    fn output_fields(&self) -> Value {
        field_specs_to_value(S::output_fields(), "output")
    }

    fn update_instruction(&mut self, instruction: String) -> Result<()> {
        self.instruction_override = Some(instruction);
        Ok(())
    }

    fn append(&mut self, _name: &str, _value: Value) -> Result<()> {
        Err(anyhow::anyhow!(
            "Typed signatures cannot be extended at runtime"
        ))
    }
}

impl<S> Optimizable for Predict<S>
where
    S: Signature + Clone,
    S::Input: BamlValueConvert + ToBamlValue,
    S::Output: BamlValueConvert + ToBamlValue,
{
    fn get_signature(&self) -> &dyn MetaSignature {
        self
    }

    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        IndexMap::new()
    }

    fn update_signature_instruction(&mut self, instruction: String) -> anyhow::Result<()> {
        self.instruction_override = Some(instruction);
        Ok(())
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
    #[tracing::instrument(
        name = "dsrs.legacy_predict.forward",
        level = "debug",
        skip(self, inputs),
        fields(
            tool_count = self.tools.len(),
            tracing_graph = crate::trace::is_tracing()
        )
    )]
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
        debug!(
            prompt_tokens = prediction.lm_usage.prompt_tokens,
            completion_tokens = prediction.lm_usage.completion_tokens,
            total_tokens = prediction.lm_usage.total_tokens,
            "legacy predictor call complete"
        );

        if let Some(id) = trace_node_id {
            prediction.node_id = Some(id);
            crate::trace::record_output(id, prediction.clone());
            trace!(node_id = id, "recorded legacy predictor output");
        }

        Ok(prediction)
    }

    #[tracing::instrument(
        name = "dsrs.legacy_predict.forward_with_config",
        level = "debug",
        skip(self, inputs, lm),
        fields(
            tool_count = self.tools.len(),
            tracing_graph = crate::trace::is_tracing()
        )
    )]
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
        debug!(
            prompt_tokens = prediction.lm_usage.prompt_tokens,
            completion_tokens = prediction.lm_usage.completion_tokens,
            total_tokens = prediction.lm_usage.total_tokens,
            "legacy predictor call_with_config complete"
        );

        if let Some(id) = trace_node_id {
            prediction.node_id = Some(id);
            crate::trace::record_output(id, prediction.clone());
            trace!(node_id = id, "recorded legacy predictor output");
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
