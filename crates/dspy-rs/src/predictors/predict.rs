use anyhow::Result;
use bamltype::baml_types::BamlMap;
use rig::tool::ToolDyn;
use serde_json::Value;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::Arc;
use tracing::{debug, trace};

use crate as dsrs;
use crate::core::{DynPredictor, Module, PredictAccessorFns, PredictState, Signature};
use crate::data::example::Example as RawExample;
use crate::{
    BamlType, BamlValue, CallMetadata, Chat, ChatAdapter, GLOBAL_SETTINGS, LmError, LmUsage,
    PredictError, Predicted, Prediction, Role, SignatureSchema, ToolLoopMode,
};

/// A typed input/output pair for few-shot prompting.
///
/// Demos are formatted as user/assistant exchanges in the prompt, showing the LM
/// what good responses look like. The types enforce that demos match the signature —
/// you can't accidentally pass a `QAOutput` demo to a `Predict<SummarizeSig>`.
///
/// ```
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// let example = Example::<QA>::new(
///     QAInput { question: "What is 2+2?".into() },
///     QAOutput { answer: "4".into() },
/// );
/// ```
#[derive(Clone, Debug, facet::Facet)]
#[facet(crate = facet)]
pub struct Example<S: Signature> {
    pub input: S::Input,
    pub output: S::Output,
}

impl<S: Signature> Example<S> {
    pub fn new(input: S::Input, output: S::Output) -> Self {
        Self { input, output }
    }
}

fn predict_dyn_visit<S>(
    value: *mut (),
    visitor: &mut dyn FnMut(&mut dyn DynPredictor) -> ControlFlow<()>,
) -> ControlFlow<()>
where
    S: Signature,
{
    // SAFETY: this function is only called through the shape-local
    // `dsrs::predict_accessor` payload attached to a shape with strict
    // `Predict` identity (`type_identifier` + `module_path`).
    let typed = unsafe { &mut *(value.cast::<Predict<S>>()) };
    visitor(typed)
}

type VisitPredictorMutFn =
    fn(*mut (), &mut dyn FnMut(&mut dyn DynPredictor) -> ControlFlow<()>) -> ControlFlow<()>;

trait PredictAccessorProvider {
    const VISIT_MUT: VisitPredictorMutFn;
}

impl<S> PredictAccessorProvider for S
where
    S: Signature,
{
    const VISIT_MUT: VisitPredictorMutFn = predict_dyn_visit::<S>;
}

/// The leaf module. The only thing in the system that actually calls the LM.
///
/// One `Predict` = one prompt template = one LM call. It takes a [`Signature`]'s fields
/// and instruction, formats them into a prompt (with any demos and tools), calls the
/// configured LM, and parses the response back into `S::Output`. Every other module —
/// [`ChainOfThought`](crate::ChainOfThought), `ReAct`, custom pipelines — ultimately
/// delegates to one or more `Predict` leaves.
///
/// This is also the unit of optimization. When an optimizer tunes your program, it's
/// adjusting `Predict` leaves: their demos (few-shot examples) and instructions.
/// The optimizer's Facet walker discovers leaves automatically from struct fields —
/// no `#[parameter]` annotations or manual traversal needed.
///
/// # Optimizer discovery
///
/// `Predict<S>` encodes shape-local discovery payloads:
/// - strict shape identity (`type_identifier` + `module_path`) identifies the leaf
/// - `dsrs::predict_accessor` stores the typed mutable accessor visitor
///
/// The optimizer walker consumes these through `visit_named_predictors_mut`.
/// There is no runtime registration side effect in `new()` or `build()`.
///
/// ```no_run
/// # async fn example() -> Result<(), dspy_rs::PredictError> {
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// // Minimal
/// let predict = Predict::<QA>::new();
/// let result = predict.call(QAInput { question: "What is 2+2?".into() }).await?;
/// println!("{}", result.answer);
///
/// // With demos and custom instruction
/// let predict = Predict::<QA>::builder()
///     .demo(Example::new(
///         QAInput { question: "What is 1+1?".into() },
///         QAOutput { answer: "2".into() },
///     ))
///     .instruction("Answer in one word.")
///     .build();
/// # Ok(())
/// # }
/// ```
#[derive(facet::Facet)]
#[facet(crate = facet, opaque)]
#[facet(dsrs::predict_accessor = &PredictAccessorFns {
    visit_mut: <S as PredictAccessorProvider>::VISIT_MUT,
})]
pub struct Predict<S: Signature> {
    #[facet(skip, opaque)]
    tools: Vec<Arc<dyn ToolDyn>>,
    #[facet(skip, opaque)]
    demos: Vec<Example<S>>,
    instruction_override: Option<String>,
    #[facet(skip, opaque)]
    _marker: PhantomData<S>,
}

impl<S: Signature> Predict<S> {
    /// Creates a new `Predict` with no demos, no instruction override, and no tools.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            demos: Vec::new(),
            instruction_override: None,
            _marker: PhantomData,
        }
    }

    /// Returns a builder for configuring demos, instruction, and tools.
    pub fn builder() -> PredictBuilder<S> {
        PredictBuilder::new()
    }

    /// Calls the LM with this predictor's signature, demos, and tools.
    ///
    /// Convenience wrapper around [`forward`](Predict::forward) with `history = None`.
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
    pub async fn call(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        self.forward(input, None).await
    }

    /// Canonical typed predict path.
    ///
    /// - `history = None` starts a new conversation (system + demos + input).
    /// - `history = Some(chat)` continues a prior conversation by appending the
    ///   typed `input` as the next user turn.
    ///
    /// Returns the parsed prediction. Updated chat history is available via
    /// [`Predicted::chat`](crate::Predicted::chat).
    pub async fn forward(
        &self,
        input: S::Input,
        history: Option<Chat>,
    ) -> Result<Predicted<S::Output>, PredictError>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let chat = self.compose_chat(&input, history)?;
        self.execute_chat(chat).await
    }

    #[allow(clippy::result_large_err)]
    fn compose_chat(&self, input: &S::Input, history: Option<Chat>) -> Result<Chat, PredictError>
    where
        S::Input: BamlType,
    {
        let chat_adapter = ChatAdapter;
        let user = chat_adapter.format_user_message_typed::<S>(input);
        trace!(
            user_len = user.len(),
            continuing = history.is_some(),
            "typed input formatted"
        );

        if let Some(mut chat) = history {
            chat.push(Role::User, &user);
            trace!(message_count = chat.len(), "chat continued");
            return Ok(chat);
        }

        let system = match chat_adapter
            .format_system_message_typed_with_instruction::<S>(self.instruction_override.as_deref())
        {
            Ok(system) => system,
            Err(err) => {
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "internal".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };

        trace!(
            system_len = system.len(),
            user_len = user.len(),
            "typed prompt initialized"
        );

        let mut chat = Chat::new(vec![]);
        chat.push(Role::System, &system);
        for demo in &self.demos {
            let demo_user = chat_adapter.format_user_message_typed::<S>(&demo.input);
            let demo_assistant = chat_adapter.format_assistant_message_typed::<S>(&demo.output);
            chat.push(Role::User, &demo_user);
            chat.push(Role::Assistant, &demo_assistant);
        }
        chat.push(Role::User, &user);
        trace!(message_count = chat.len(), "chat constructed");
        Ok(chat)
    }

    async fn execute_chat(&self, chat: Chat) -> Result<Predicted<S::Output>, PredictError>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        let lm = {
            let guard = GLOBAL_SETTINGS.read().unwrap();
            let settings = guard.as_ref().unwrap();
            Arc::clone(&settings.lm)
        };

        let response = match lm.call(chat, self.tools.clone(), ToolLoopMode::Auto).await {
            Ok(response) => response,
            Err(err) => {
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: lm.model.clone(),
                        message: err.to_string(),
                        source: None,
                    },
                });
            }
        };
        debug!(
            prompt_tokens = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens = response.usage.total_tokens,
            tool_calls = response.tool_calls.len(),
            "lm response received"
        );

        let crate::core::lm::LMResponse {
            output,
            usage,
            chat,
            tool_calls,
            tool_executions,
        } = response;

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

        let chat_adapter = ChatAdapter;
        let raw_response = output.content().to_string();
        let lm_usage = usage.clone();

        let (typed_output, field_metas) = match chat_adapter.parse_response_typed::<S>(&output) {
            Ok(parsed) => parsed,
            Err(err) => {
                let failed_fields = err.fields();
                debug!(
                    failed_fields = failed_fields.len(),
                    fields = ?failed_fields,
                    raw_response_len = raw_response.len(),
                    "typed parse failed"
                );
                return Err(PredictError::Parse {
                    source: err,
                    raw_response,
                    lm_usage,
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
            tool_calls,
            tool_executions,
            node_id,
            field_metas,
        );

        Ok(Predicted::new(typed_output, metadata, chat))
    }
}

impl<S: Signature> Default for Predict<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`Predict`] with demos, tools, and instruction override.
///
/// ```ignore
/// let predict = Predict::<QA>::builder()
///     .demo(demo1)
///     .demo(demo2)
///     .instruction("Answer in one word.")
///     .add_tool(my_tool)
///     .build();
/// ```
pub struct PredictBuilder<S: Signature> {
    tools: Vec<Arc<dyn ToolDyn>>,
    demos: Vec<Example<S>>,
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

    /// Adds a single demo (few-shot example) to the predictor.
    pub fn demo(mut self, demo: Example<S>) -> Self {
        self.demos.push(demo);
        self
    }

    /// Adds multiple demos from an iterator.
    pub fn with_demos(mut self, demos: impl IntoIterator<Item = Example<S>>) -> Self {
        self.demos.extend(demos);
        self
    }

    /// Adds a tool the LM can invoke during this call.
    pub fn add_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Adds multiple tools from an iterator.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Arc<dyn ToolDyn>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Overrides the signature's default instruction for this predictor.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction_override = Some(instruction.into());
        self
    }

    /// Builds the [`Predict`].
    pub fn build(self) -> Predict<S> {
        Predict {
            tools: self.tools,
            demos: self.demos,
            instruction_override: self.instruction_override,
            _marker: PhantomData,
        }
    }
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

fn input_keys_for_signature<S: Signature>(example: &RawExample) -> Vec<String> {
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

fn output_keys_for_signature<S: Signature>(example: &RawExample) -> Vec<String> {
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

fn input_from_raw_example<S: Signature>(example: &RawExample) -> Result<S::Input>
where
    S::Input: BamlType,
{
    let keys = input_keys_for_signature::<S>(example);
    let map = baml_map_from_example_keys(&example.data, &keys)?;
    let baml_value = BamlValue::Map(map);
    S::Input::try_from_baml_value(baml_value).map_err(|err| anyhow::anyhow!(err))
}

fn output_from_raw_example<S: Signature>(example: &RawExample) -> Result<S::Output>
where
    S::Output: BamlType,
{
    let keys = output_keys_for_signature::<S>(example);
    let map = baml_map_from_example_keys(&example.data, &keys)?;
    let baml_value = BamlValue::Map(map);
    S::Output::try_from_baml_value(baml_value).map_err(|err| anyhow::anyhow!(err))
}

fn typed_example_from_raw<S: Signature>(example: RawExample) -> Result<Example<S>>
where
    S::Input: BamlType,
    S::Output: BamlType,
{
    let input = input_from_raw_example::<S>(&example)?;
    let output = output_from_raw_example::<S>(&example)?;
    Ok(Example::new(input, output))
}

fn raw_example_from_typed<S: Signature>(example: &Example<S>) -> Result<RawExample>
where
    S::Input: BamlType,
    S::Output: BamlType,
{
    let input_value = serde_json::to_value(example.input.to_baml_value())?;
    let output_value = serde_json::to_value(example.output.to_baml_value())?;

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

    Ok(RawExample::new(data, input_keys, output_keys))
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
    async fn forward(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError> {
        Predict::forward(self, input, None).await
    }
}

impl<S> DynPredictor for Predict<S>
where
    S: Signature,
    S::Input: BamlType,
    S::Output: BamlType,
{
    fn schema(&self) -> &SignatureSchema {
        S::schema()
    }

    fn instruction(&self) -> String {
        self.instruction_override
            .clone()
            .unwrap_or_else(|| S::instruction().to_string())
    }

    fn set_instruction(&mut self, instruction: String) {
        self.instruction_override = Some(instruction);
    }

    fn demos_as_examples(&self) -> Vec<RawExample> {
        self.demos
            .iter()
            .map(|example| {
                raw_example_from_typed::<S>(example)
                    .expect("typed Predict demo conversion should succeed")
            })
            .collect()
    }

    fn set_demos_from_examples(&mut self, demos: Vec<RawExample>) -> Result<()> {
        self.demos = demos
            .into_iter()
            .map(typed_example_from_raw::<S>)
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    fn dump_state(&self) -> PredictState {
        PredictState {
            demos: self.demos_as_examples(),
            instruction_override: self.instruction_override.clone(),
        }
    }

    fn load_state(&mut self, state: PredictState) -> Result<()> {
        self.set_demos_from_examples(state.demos)?;
        self.instruction_override = state.instruction_override;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(crate::Signature, Clone, Debug)]
    struct PredictConversionSig {
        #[input]
        prompt: String,

        #[output]
        answer: String,
    }

    fn typed_row(prompt: &str, answer: &str) -> Example<PredictConversionSig> {
        Example::new(
            PredictConversionSigInput {
                prompt: prompt.to_string(),
            },
            PredictConversionSigOutput {
                answer: answer.to_string(),
            },
        )
    }

    #[test]
    fn typed_and_raw_example_round_trip_preserves_fields() {
        let typed = typed_row("question", "response");
        let raw = raw_example_from_typed::<PredictConversionSig>(&typed)
            .expect("typed example should convert to raw example");

        assert_eq!(raw.input_keys, vec!["prompt".to_string()]);
        assert_eq!(raw.output_keys, vec!["answer".to_string()]);
        assert_eq!(raw.data.get("prompt"), Some(&json!("question")));
        assert_eq!(raw.data.get("answer"), Some(&json!("response")));

        let round_trip = typed_example_from_raw::<PredictConversionSig>(raw)
            .expect("raw example should convert back to typed example");
        assert_eq!(round_trip.input.prompt, "question");
        assert_eq!(round_trip.output.answer, "response");
    }

    #[test]
    fn typed_example_from_raw_uses_schema_keys_when_key_lists_missing() {
        let raw = RawExample::new(
            HashMap::from([
                ("prompt".to_string(), json!("schema-input")),
                ("answer".to_string(), json!("schema-output")),
            ]),
            Vec::new(),
            Vec::new(),
        );

        let typed = typed_example_from_raw::<PredictConversionSig>(raw)
            .expect("schema key fallback should parse typed example");
        assert_eq!(typed.input.prompt, "schema-input");
        assert_eq!(typed.output.answer, "schema-output");
    }

    #[test]
    fn dyn_predictor_set_demos_from_examples_round_trips_raw_rows() {
        let typed = typed_row("demo-input", "demo-output");
        let raw = raw_example_from_typed::<PredictConversionSig>(&typed)
            .expect("typed demo should convert to raw demo");
        let mut predictor = Predict::<PredictConversionSig>::new();

        DynPredictor::set_demos_from_examples(&mut predictor, vec![raw])
            .expect("predictor should accept raw demos");

        let demos = DynPredictor::demos_as_examples(&predictor);
        assert_eq!(demos.len(), 1);
        assert_eq!(demos[0].data.get("prompt"), Some(&json!("demo-input")));
        assert_eq!(demos[0].data.get("answer"), Some(&json!("demo-output")));
    }
}
