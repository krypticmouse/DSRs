use anyhow::Result;
use bamltype::baml_types::BamlMap;
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use tracing::{debug, trace};

use crate::adapter::Adapter;
use crate::core::{MetaSignature, Module, Optimizable, Signature};
use crate::{
    BamlType, BamlValue, CallMetadata, CallOutcome, CallOutcomeError, CallOutcomeErrorKind, Chat,
    ChatAdapter, Example, FieldSchema, GLOBAL_SETTINGS, LM, LmError, LmUsage, PredictError,
    Prediction,
};

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct Demo<S: Signature> {
    pub input: S::Input,
    pub output: S::Output,
}

impl<S: Signature> Demo<S> {
    pub fn new(input: S::Input, output: S::Output) -> Self {
        Self { input, output }
    }
}

#[derive(facet::Facet)]
#[facet(crate = facet, opaque)]
pub struct Predict<S: Signature> {
    #[facet(skip, opaque)]
    tools: Vec<Arc<dyn ToolDyn>>,
    #[facet(skip, opaque)]
    demos: Vec<Demo<S>>,
    instruction_override: Option<String>,
    #[facet(skip, opaque)]
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

    #[tracing::instrument(
        name = "dsrs.predict.call",
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
    pub async fn call(&self, input: S::Input) -> CallOutcome<S::Output>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let lm = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            Arc::clone(&settings.lm)
        };

        let chat_adapter = ChatAdapter;
        let system = match chat_adapter
            .format_system_message_typed_with_instruction::<S>(self.instruction_override.as_deref())
        {
            Ok(system) => system,
            Err(err) => {
                let metadata = CallMetadata::default();
                return CallOutcome::err(
                    CallOutcomeErrorKind::Lm(LmError::Provider {
                        provider: "internal".to_string(),
                        message: err.to_string(),
                        source: None,
                    }),
                    metadata,
                );
            }
        };

        let user = chat_adapter.format_user_message_typed::<S>(&input);
        trace!(
            system_len = system.len(),
            user_len = user.len(),
            "typed prompt formatted"
        );

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system);
        for demo in &self.demos {
            let demo_user = chat_adapter.format_user_message_typed::<S>(&demo.input);
            let demo_assistant = chat_adapter.format_assistant_message_typed::<S>(&demo.output);
            chat.push("user", &demo_user);
            chat.push("assistant", &demo_assistant);
        }
        chat.push("user", &user);
        trace!(message_count = chat.len(), "chat constructed");

        let response = match lm.call(chat, self.tools.clone()).await {
            Ok(response) => response,
            Err(err) => {
                let metadata = CallMetadata::default();
                return CallOutcome::err(
                    CallOutcomeErrorKind::Lm(LmError::Provider {
                        provider: lm.model.clone(),
                        message: err.to_string(),
                        source: None,
                    }),
                    metadata,
                );
            }
        };
        debug!(
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            tool_calls = response.tool_calls.len(),
            "lm response received"
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

        let raw_response = response.output.content().to_string();
        let lm_usage = response.usage.clone();

        let (typed_output, field_metas) = match chat_adapter.parse_response_typed::<S>(&response.output)
        {
            Ok(parsed) => parsed,
            Err(err) => {
                let failed_fields = err.fields();
                debug!(
                    failed_fields = failed_fields.len(),
                    fields = ?failed_fields,
                    raw_response_len = raw_response.len(),
                    "typed parse failed"
                );
                let metadata = CallMetadata::new(
                    raw_response,
                    lm_usage,
                    response.tool_calls,
                    response.tool_executions,
                    node_id,
                    IndexMap::new(),
                );
                return CallOutcome::err(CallOutcomeErrorKind::Parse(err), metadata);
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
            checks_total,
            checks_failed,
            flagged_fields,
            "typed parse completed"
        );

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

        let metadata = CallMetadata::new(
            raw_response,
            lm_usage,
            response.tool_calls,
            response.tool_executions,
            node_id,
            field_metas,
        );

        CallOutcome::ok(typed_output, metadata)
    }
}

impl<S: Signature> Default for Predict<S> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PredictBuilder<S: Signature> {
    tools: Vec<Arc<dyn ToolDyn>>,
    demos: Vec<Demo<S>>,
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

    pub fn demo(mut self, demo: Demo<S>) -> Self {
        self.demos.push(demo);
        self
    }

    pub fn with_demos(mut self, demos: impl IntoIterator<Item = Demo<S>>) -> Self {
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

fn schema_fields_to_value(fields: &[FieldSchema], field_type: &'static str) -> Value {
    let mut result = serde_json::Map::new();
    for field in fields {
        let type_repr = field.type_ir.diagnostic_repr().to_string();
        let mut meta = serde_json::Map::new();
        meta.insert("type".to_string(), json!(type_repr));
        meta.insert("desc".to_string(), json!(field.docs));
        meta.insert("schema".to_string(), json!(""));
        meta.insert("__dsrs_field_type".to_string(), json!(field_type));
        if let Some(format) = field.format {
            meta.insert("format".to_string(), json!(format));
        }
        result.insert(field.lm_name.to_string(), Value::Object(meta));
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
        S::schema()
            .input_fields()
            .iter()
            .map(|field| field.rust_name.clone())
            .collect()
    } else {
        example.input_keys.clone()
    }
}

fn output_keys_for_signature<S: Signature>(example: &Example) -> Vec<String> {
    if example.output_keys.is_empty() {
        S::schema()
            .output_fields()
            .iter()
            .map(|field| field.rust_name.clone())
            .collect()
    } else {
        example.output_keys.clone()
    }
}

fn input_from_example<S: Signature>(example: &Example) -> Result<S::Input>
where
    S::Input: BamlType,
{
    let keys = input_keys_for_signature::<S>(example);
    let map = baml_map_from_example_keys(&example.data, &keys)?;
    let baml_value = BamlValue::Map(map);
    S::Input::try_from_baml_value(baml_value).map_err(|err| anyhow::anyhow!(err))
}

fn output_from_example<S: Signature>(example: &Example) -> Result<S::Output>
where
    S::Output: BamlType,
{
    let keys = output_keys_for_signature::<S>(example);
    let map = baml_map_from_example_keys(&example.data, &keys)?;
    let baml_value = BamlValue::Map(map);
    S::Output::try_from_baml_value(baml_value).map_err(|err| anyhow::anyhow!(err))
}

fn demo_from_example<S: Signature>(example: Example) -> Result<Demo<S>>
where
    S::Input: BamlType,
    S::Output: BamlType,
{
    let input = input_from_example::<S>(&example)?;
    let output = output_from_example::<S>(&example)?;
    Ok(Demo::new(input, output))
}

fn example_from_demo<S: Signature>(demo: &Demo<S>) -> Result<Example>
where
    S::Input: BamlType,
    S::Output: BamlType,
{
    let input_value = serde_json::to_value(demo.input.to_baml_value())?;
    let output_value = serde_json::to_value(demo.output.to_baml_value())?;

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
    S::Output: BamlType,
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

fn predict_error_from_outcome(kind: CallOutcomeErrorKind, metadata: CallMetadata) -> PredictError {
    CallOutcomeError { metadata, kind }.into_predict_error()
}

impl<S> Module for Predict<S>
where
    S: Signature + Clone,
    S::Input: BamlType,
    S::Output: BamlType,
{
    type Input = S::Input;
    type Output = S::Output;

    #[tracing::instrument(
        name = "dsrs.module.forward",
        level = "debug",
        skip(self, input),
        fields(
            signature = std::any::type_name::<S>(),
            typed = true
        )
    )]
    async fn forward(&self, input: S::Input) -> CallOutcome<S::Output> {
        self.call(input).await
    }
}

impl<S> Predict<S>
where
    S: Signature + Clone,
    S::Input: BamlType,
    S::Output: BamlType,
{
    #[tracing::instrument(
        name = "dsrs.predict.forward_untyped",
        level = "debug",
        skip(self, input),
        fields(signature = std::any::type_name::<S>())
    )]
    pub async fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> CallOutcome<BamlValue> {
        let typed_input = match S::Input::try_from_baml_value(input.clone()) {
            Ok(typed_input) => typed_input,
            Err(err) => {
                debug!(error = %err, "untyped input conversion failed");
                return CallOutcome::err(
                    CallOutcomeErrorKind::Conversion(err.into(), input),
                    CallMetadata::default(),
                );
            }
        };
        let (result, metadata) = self.call(typed_input).await.into_parts();
        let output = match result {
            Ok(output) => output,
            Err(kind) => return CallOutcome::err(kind, metadata),
        };
        debug!("typed predict forward_untyped complete");
        CallOutcome::ok(output.to_baml_value(), metadata)
    }
}

impl<S> MetaSignature for Predict<S>
where
    S: Signature + Clone,
    S::Input: BamlType,
    S::Output: BamlType,
{
    fn demos(&self) -> Vec<Example> {
        self.demos
            .iter()
            .map(|demo| example_from_demo::<S>(demo).expect("typed Predict demo conversion should succeed"))
            .collect()
    }

    fn set_demos(&mut self, demos: Vec<Example>) -> Result<()> {
        self.demos = demos
            .into_iter()
            .map(demo_from_example::<S>)
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    fn instruction(&self) -> String {
        self.instruction_override
            .clone()
            .unwrap_or_else(|| S::instruction().to_string())
    }

    fn input_fields(&self) -> Value {
        schema_fields_to_value(S::schema().input_fields(), "input")
    }

    fn output_fields(&self) -> Value {
        schema_fields_to_value(S::schema().output_fields(), "output")
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
    S::Input: BamlType,
    S::Output: BamlType,
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
