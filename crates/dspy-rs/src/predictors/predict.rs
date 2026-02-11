use anyhow::Result;
use bamltype::baml_types::BamlMap;
use rig::tool::ToolDyn;
use serde_json::Value;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use tracing::{debug, trace};

use crate::core::{DynPredictor, Module, PredictState, Signature, register_predict_accessor};
use crate::{
    BamlType, BamlValue, CallMetadata, Chat, ChatAdapter, Example, GLOBAL_SETTINGS, LmError,
    LmUsage, PredictError, Predicted, Prediction, SignatureSchema,
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
/// let demo = Demo::<QA>::new(
///     QAInput { question: "What is 2+2?".into() },
///     QAOutput { answer: "4".into() },
/// );
/// ```
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

fn predict_dyn_accessor<S>(value: *mut ()) -> *mut dyn DynPredictor
where
    S: Signature,
{
    // SAFETY: this function is only called via `register_predict_accessor` for
    // `Predict<S>`'s own shape, so `value` points at a valid `Predict<S>`.
    let typed = unsafe { &mut *(value.cast::<Predict<S>>()) };
    let dyn_ref: &mut dyn DynPredictor = typed;
    dyn_ref as *mut dyn DynPredictor
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
/// # Construction side effect
///
/// `new()` and `builder().build()` register an accessor function in a global registry.
/// This is a workaround — ideally the type system would handle it, but Facet doesn't
/// yet support shape-local typed attr payloads on generic containers. If you construct
/// a `Predict<S>` without going through `new()`/`build()` (e.g. via unsafe or manual
/// field init), [`named_parameters`](crate::named_parameters) will error when it finds
/// the unregistered leaf.
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
///     .demo(Demo::new(
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
        register_predict_accessor(
            <Self as facet::Facet<'static>>::SHAPE,
            predict_dyn_accessor::<S>,
        );
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
    pub async fn call(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError>
    where
        S::Input: BamlType,
        S::Output: BamlType,
    {
        self.forward(input).await
    }

    pub async fn forward(&self, input: S::Input) -> Result<Predicted<S::Output>, PredictError>
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
                return Err(PredictError::Lm {
                    source: LmError::Provider {
                        provider: "internal".to_string(),
                        message: err.to_string(),
                        source: None,
                    },
                });
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

        let (typed_output, field_metas) =
            match chat_adapter.parse_response_typed::<S>(&response.output) {
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

        Ok(Predicted::new(typed_output, metadata))
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
        register_predict_accessor(
            <Predict<S> as facet::Facet<'static>>::SHAPE,
            predict_dyn_accessor::<S>,
        );
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
        Predict::forward(self, input).await
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

    fn demos_as_examples(&self) -> Vec<Example> {
        self.demos
            .iter()
            .map(|demo| {
                example_from_demo::<S>(demo).expect("typed Predict demo conversion should succeed")
            })
            .collect()
    }

    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()> {
        self.demos = demos
            .into_iter()
            .map(demo_from_example::<S>)
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
