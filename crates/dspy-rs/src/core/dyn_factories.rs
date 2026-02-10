use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use bamltype::baml_types::{BamlMap, BamlValue};
use bamltype::build_type_ir_from_shape;
use facet::Facet;
use rig::message::{ToolCall, ToolFunction};
use rig::tool::ToolDyn;

use crate::core::{
    DynModule, DynPredictor, PredictState, StrategyConfig, StrategyConfigSchema, StrategyError,
    StrategyFactory, StrategyFactoryRegistration,
};
use crate::{
    CallMetadata, Chat, ChatAdapter, ConversionError, Example, GLOBAL_SETTINGS, LmError,
    PredictError, Predicted, SignatureSchema,
};

#[derive(Clone)]
pub struct SchemaPredictor {
    schema: SignatureSchema,
    demos: Vec<Example>,
    instruction_override: Option<String>,
    tools: Vec<Arc<dyn ToolDyn>>,
}

impl SchemaPredictor {
    pub fn new(schema: SignatureSchema) -> Self {
        Self {
            schema,
            demos: Vec::new(),
            instruction_override: None,
            tools: Vec::new(),
        }
    }

    fn input_from_example(&self, example: &Example) -> Result<BamlValue> {
        baml_value_from_example_keys(&example.data, &example.input_keys)
    }

    fn output_from_example(&self, example: &Example) -> Result<BamlValue> {
        baml_value_from_example_keys(&example.data, &example.output_keys)
    }
}

#[async_trait::async_trait]
impl DynPredictor for SchemaPredictor {
    fn schema(&self) -> &SignatureSchema {
        &self.schema
    }

    fn instruction(&self) -> String {
        self.instruction_override
            .clone()
            .unwrap_or_else(|| self.schema.instruction().to_string())
    }

    fn set_instruction(&mut self, instruction: String) {
        self.instruction_override = Some(instruction);
    }

    fn demos_as_examples(&self) -> Vec<Example> {
        self.demos.clone()
    }

    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()> {
        self.demos = demos;
        Ok(())
    }

    fn dump_state(&self) -> PredictState {
        PredictState {
            demos: self.demos.clone(),
            instruction_override: self.instruction_override.clone(),
        }
    }

    fn load_state(&mut self, state: PredictState) -> Result<()> {
        self.demos = state.demos;
        self.instruction_override = state.instruction_override;
        Ok(())
    }

    async fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        let lm = {
            let guard = GLOBAL_SETTINGS.read().expect("settings lock poisoned");
            let settings = guard.as_ref().expect("settings not configured");
            Arc::clone(&settings.lm)
        };

        let chat_adapter = ChatAdapter;
        let system = chat_adapter
            .build_system(&self.schema, self.instruction_override.as_deref())
            .map_err(|err| PredictError::Lm {
                source: LmError::Provider {
                    provider: "internal".to_string(),
                    message: err.to_string(),
                    source: None,
                },
            })?;

        let user = chat_adapter.format_input_baml(&self.schema, &input);

        let mut chat = Chat::new(vec![]);
        chat.push("system", &system);
        for demo in &self.demos {
            let demo_input =
                self.input_from_example(demo)
                    .map_err(|err| PredictError::Conversion {
                        source: crate::ConversionError::TypeMismatch {
                            expected: "BamlValue",
                            actual: err.to_string(),
                        },
                        parsed: BamlValue::Null,
                    })?;
            let demo_output =
                self.output_from_example(demo)
                    .map_err(|err| PredictError::Conversion {
                        source: crate::ConversionError::TypeMismatch {
                            expected: "BamlValue",
                            actual: err.to_string(),
                        },
                        parsed: BamlValue::Null,
                    })?;
            let demo_user = chat_adapter.format_input_baml(&self.schema, &demo_input);
            let demo_assistant = chat_adapter.format_output_baml(&self.schema, &demo_output);
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
        let (output, field_metas) = chat_adapter
            .parse_output_baml_with_meta(&self.schema, &response.output)
            .map_err(|source| PredictError::Parse {
                source,
                raw_response: raw_response.clone(),
                lm_usage: lm_usage.clone(),
            })?;

        let metadata = CallMetadata::new(
            raw_response,
            lm_usage,
            response.tool_calls,
            response.tool_executions,
            None,
            field_metas,
        );

        Ok(Predicted::new(output, metadata))
    }
}

pub struct PredictDynModule {
    schema: SignatureSchema,
    predictor: SchemaPredictor,
}

impl PredictDynModule {
    pub fn new(schema: SignatureSchema) -> Self {
        Self {
            predictor: SchemaPredictor::new(schema.clone()),
            schema,
        }
    }
}

#[async_trait::async_trait]
impl DynModule for PredictDynModule {
    fn schema(&self) -> &SignatureSchema {
        &self.schema
    }

    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)> {
        vec![("predictor", &self.predictor)]
    }

    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)> {
        vec![("predictor", &mut self.predictor)]
    }

    async fn forward(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        self.predictor.forward_untyped(input).await
    }
}

pub struct ChainOfThoughtDynModule {
    schema: SignatureSchema,
    predictor: SchemaPredictor,
}

impl ChainOfThoughtDynModule {
    pub fn new(schema: SignatureSchema) -> Self {
        Self {
            predictor: SchemaPredictor::new(schema.clone()),
            schema,
        }
    }
}

#[async_trait::async_trait]
impl DynModule for ChainOfThoughtDynModule {
    fn schema(&self) -> &SignatureSchema {
        &self.schema
    }

    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)> {
        vec![("predictor", &self.predictor)]
    }

    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)> {
        vec![("predictor", &mut self.predictor)]
    }

    async fn forward(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        self.predictor.forward_untyped(input).await
    }
}

pub struct ReActDynModule {
    schema: SignatureSchema,
    action: SchemaPredictor,
    extract: SchemaPredictor,
    max_steps: usize,
    tools: Vec<Arc<dyn ToolDyn>>,
}

impl ReActDynModule {
    pub fn new(schema: SignatureSchema, max_steps: usize, tools: Vec<Arc<dyn ToolDyn>>) -> Self {
        let action_schema = react_action_schema(&schema);
        let extract_schema = react_extract_schema(&schema);
        Self {
            action: SchemaPredictor::new(action_schema),
            extract: SchemaPredictor::new(extract_schema),
            schema,
            max_steps,
            tools,
        }
    }

    async fn render_tool_manifest(&self) -> String {
        if self.tools.is_empty() {
            return "Available tools: (none)".to_string();
        }

        let mut lines = vec!["Available tools:".to_string()];
        for tool in &self.tools {
            let definition = tool.definition(String::new()).await;
            lines.push(format!("- {}: {}", definition.name, definition.description));
        }

        lines.join("\n")
    }

    async fn execute_tool(&self, name: &str, args: String) -> String {
        let normalized = name.trim();

        for tool in &self.tools {
            let candidate = tool.name();
            if candidate.eq_ignore_ascii_case(normalized)
                || normalized.contains(&candidate)
                || candidate.contains(normalized)
            {
                return match tool.call(args).await {
                    Ok(result) => result,
                    Err(err) => format!("tool_error: {err}"),
                };
            }
        }

        if let Some(first_tool) = self.tools.first() {
            return match first_tool.call(args).await {
                Ok(result) => result,
                Err(err) => format!("tool_error: {err}"),
            };
        }

        format!("tool_not_found: {name}")
    }

    fn is_terminal_action(action: &str) -> bool {
        action.eq_ignore_ascii_case("finish")
            || action.eq_ignore_ascii_case("final")
            || action.eq_ignore_ascii_case("done")
    }

    fn format_trace_entry(
        step: usize,
        thought: &str,
        action: &str,
        action_input: &str,
        observation: Option<&str>,
    ) -> String {
        let observation_text = observation.unwrap_or("<none>");
        format!(
            "Step {step}\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation_text}"
        )
    }
}

#[async_trait::async_trait]
impl DynModule for ReActDynModule {
    fn schema(&self) -> &SignatureSchema {
        &self.schema
    }

    fn predictors(&self) -> Vec<(&str, &dyn DynPredictor)> {
        vec![("action", &self.action), ("extract", &self.extract)]
    }

    fn predictors_mut(&mut self) -> Vec<(&str, &mut dyn DynPredictor)> {
        vec![("action", &mut self.action), ("extract", &mut self.extract)]
    }

    async fn forward(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError> {
        let serialized_input = serde_json::to_string(&input)
            .unwrap_or_else(|_| "<input serialization failed>".to_string());

        let tool_manifest = self.render_tool_manifest().await;
        let mut trajectory_text = tool_manifest.clone();
        trajectory_text.push_str("\n\n");

        let mut tool_calls = Vec::<ToolCall>::new();
        let mut tool_executions = vec![tool_manifest];

        for step in 0..self.max_steps {
            let action_input = baml_class([
                ("input", BamlValue::String(serialized_input.clone())),
                ("trajectory", BamlValue::String(trajectory_text.clone())),
            ]);

            let action_predicted = self.action.forward_untyped(action_input).await?;
            let (action_output, mut action_metadata) = action_predicted.into_parts();
            tool_calls.append(&mut action_metadata.tool_calls);
            tool_executions.append(&mut action_metadata.tool_executions);

            let thought = required_string_output(&self.action, &action_output, "thought")?;
            let action = required_string_output(&self.action, &action_output, "action")?;
            let action_input =
                required_string_output(&self.action, &action_output, "action_input")?;

            let action_name = action
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();

            if Self::is_terminal_action(&action_name) {
                let trace =
                    Self::format_trace_entry(step + 1, &thought, &action_name, &action_input, None);
                tool_executions.push(trace.clone());
                trajectory_text.push_str(&format!(
                    "Step {}\nThought: {}\nFinal: {}\n\n",
                    step + 1,
                    thought,
                    action_input
                ));
                break;
            }

            let observation = self.execute_tool(&action_name, action_input.clone()).await;
            tool_calls.push(ToolCall {
                id: format!("react-step-{}", step + 1),
                call_id: None,
                function: ToolFunction {
                    name: action_name.clone(),
                    arguments: serde_json::json!(action_input),
                },
            });
            tool_executions.push(Self::format_trace_entry(
                step + 1,
                &thought,
                &action_name,
                &action_input,
                Some(&observation),
            ));

            trajectory_text.push_str(&format!(
                "Step {}\nThought: {}\nAction: {}\nAction Input: {}\nObservation: {}\n\n",
                step + 1,
                thought,
                action_name,
                action_input,
                observation
            ));
        }

        let extract_input = baml_class([
            ("input", BamlValue::String(serialized_input)),
            ("trajectory", BamlValue::String(trajectory_text)),
        ]);

        let extract_predicted = self.extract.forward_untyped(extract_input).await?;
        let (output, mut metadata) = extract_predicted.into_parts();
        metadata.tool_calls.extend(tool_calls);
        metadata.tool_executions.extend(tool_executions);

        Ok(Predicted::new(output, metadata))
    }
}

pub struct PredictFactory;
pub struct ChainOfThoughtFactory;
pub struct ReActFactory;

impl StrategyFactory for PredictFactory {
    fn name(&self) -> &'static str {
        "predict"
    }

    fn config_schema(&self) -> StrategyConfigSchema {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        })
    }

    fn create(
        &self,
        base_schema: &SignatureSchema,
        _config: StrategyConfig,
    ) -> std::result::Result<Box<dyn DynModule>, StrategyError> {
        Ok(Box::new(PredictDynModule::new(base_schema.clone())))
    }
}

impl StrategyFactory for ChainOfThoughtFactory {
    fn name(&self) -> &'static str {
        "chain_of_thought"
    }

    fn config_schema(&self) -> StrategyConfigSchema {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "additionalProperties": true,
        })
    }

    fn create(
        &self,
        base_schema: &SignatureSchema,
        _config: StrategyConfig,
    ) -> std::result::Result<Box<dyn DynModule>, StrategyError> {
        let mut output_fields = Vec::with_capacity(base_schema.output_fields().len() + 1);
        output_fields.push(crate::FieldSchema {
            lm_name: "reasoning",
            rust_name: "reasoning".to_string(),
            docs: String::new(),
            type_ir: build_type_ir_from_shape(<String as Facet<'static>>::SHAPE),
            shape: <String as Facet<'static>>::SHAPE,
            path: crate::FieldPath::new(["reasoning"]),
            constraints: &[],
            format: None,
        });
        output_fields.extend(base_schema.output_fields().iter().cloned());
        let schema = base_schema.with_fields(base_schema.input_fields().to_vec(), output_fields);
        Ok(Box::new(ChainOfThoughtDynModule::new(schema)))
    }
}

impl StrategyFactory for ReActFactory {
    fn name(&self) -> &'static str {
        "react"
    }

    fn config_schema(&self) -> StrategyConfigSchema {
        serde_json::json!({
            "type": "object",
            "properties": {
                "max_steps": { "type": "integer", "minimum": 1 }
            },
            "additionalProperties": true,
        })
    }

    fn create(
        &self,
        base_schema: &SignatureSchema,
        config: StrategyConfig,
    ) -> std::result::Result<Box<dyn DynModule>, StrategyError> {
        let object = config
            .as_object()
            .ok_or_else(|| StrategyError::InvalidConfig {
                strategy: self.name(),
                reason: "config must be a JSON object".to_string(),
            })?;

        let max_steps = match object.get("max_steps") {
            None => 4usize,
            Some(value) => {
                let parsed = value.as_u64().ok_or_else(|| StrategyError::InvalidConfig {
                    strategy: self.name(),
                    reason: "`max_steps` must be an integer >= 1".to_string(),
                })?;
                if parsed == 0 {
                    return Err(StrategyError::InvalidConfig {
                        strategy: self.name(),
                        reason: "`max_steps` must be >= 1".to_string(),
                    });
                }
                parsed as usize
            }
        };

        Ok(Box::new(ReActDynModule::new(
            base_schema.clone(),
            max_steps,
            Vec::new(),
        )))
    }
}

inventory::submit! {
    StrategyFactoryRegistration { factory: &PredictFactory }
}

inventory::submit! {
    StrategyFactoryRegistration { factory: &ChainOfThoughtFactory }
}

inventory::submit! {
    StrategyFactoryRegistration { factory: &ReActFactory }
}

fn react_action_schema(base_schema: &SignatureSchema) -> SignatureSchema {
    let string_shape = <String as Facet<'static>>::SHAPE;
    let string_type = build_type_ir_from_shape(string_shape);
    let output_format = Arc::new(base_schema.output_format().clone());

    SignatureSchema::from_parts(
        "Given input and trajectory, choose the next action and its input.",
        vec![
            crate::FieldSchema {
                lm_name: "input",
                rust_name: "input".to_string(),
                docs: String::new(),
                type_ir: string_type.clone(),
                shape: string_shape,
                path: crate::FieldPath::new(["input"]),
                constraints: &[],
                format: None,
            },
            crate::FieldSchema {
                lm_name: "trajectory",
                rust_name: "trajectory".to_string(),
                docs: String::new(),
                type_ir: string_type.clone(),
                shape: string_shape,
                path: crate::FieldPath::new(["trajectory"]),
                constraints: &[],
                format: None,
            },
        ],
        vec![
            crate::FieldSchema {
                lm_name: "thought",
                rust_name: "thought".to_string(),
                docs: String::new(),
                type_ir: string_type.clone(),
                shape: string_shape,
                path: crate::FieldPath::new(["thought"]),
                constraints: &[],
                format: None,
            },
            crate::FieldSchema {
                lm_name: "action",
                rust_name: "action".to_string(),
                docs: String::new(),
                type_ir: string_type.clone(),
                shape: string_shape,
                path: crate::FieldPath::new(["action"]),
                constraints: &[],
                format: None,
            },
            crate::FieldSchema {
                lm_name: "action_input",
                rust_name: "action_input".to_string(),
                docs: String::new(),
                type_ir: string_type,
                shape: string_shape,
                path: crate::FieldPath::new(["action_input"]),
                constraints: &[],
                format: None,
            },
        ],
        output_format,
    )
}

fn react_extract_schema(base_schema: &SignatureSchema) -> SignatureSchema {
    let string_shape = <String as Facet<'static>>::SHAPE;
    let string_type = build_type_ir_from_shape(string_shape);

    SignatureSchema::from_parts(
        base_schema.instruction(),
        vec![
            crate::FieldSchema {
                lm_name: "input",
                rust_name: "input".to_string(),
                docs: String::new(),
                type_ir: string_type.clone(),
                shape: string_shape,
                path: crate::FieldPath::new(["input"]),
                constraints: &[],
                format: None,
            },
            crate::FieldSchema {
                lm_name: "trajectory",
                rust_name: "trajectory".to_string(),
                docs: String::new(),
                type_ir: string_type,
                shape: string_shape,
                path: crate::FieldPath::new(["trajectory"]),
                constraints: &[],
                format: None,
            },
        ],
        base_schema.output_fields().to_vec(),
        Arc::new(base_schema.output_format().clone()),
    )
}

fn required_string_output(
    predictor: &SchemaPredictor,
    output: &BamlValue,
    field: &'static str,
) -> std::result::Result<String, PredictError> {
    let field_schema = predictor
        .schema()
        .output_field_by_rust(field)
        .or_else(|| {
            predictor
                .schema()
                .output_fields()
                .iter()
                .find(|candidate| candidate.lm_name == field)
        })
        .ok_or_else(|| PredictError::Conversion {
            source: ConversionError::TypeMismatch {
                expected: field,
                actual: "missing output field metadata".to_string(),
            },
            parsed: output.clone(),
        })?;

    let value = predictor
        .schema()
        .navigate_field(field_schema.path(), output)
        .ok_or_else(|| PredictError::Conversion {
            source: ConversionError::TypeMismatch {
                expected: field,
                actual: "missing output value".to_string(),
            },
            parsed: output.clone(),
        })?;

    match value {
        BamlValue::String(s) => Ok(s.clone()),
        _ => Err(PredictError::Conversion {
            source: ConversionError::TypeMismatch {
                expected: field,
                actual: format!("{value:?}"),
            },
            parsed: output.clone(),
        }),
    }
}

fn baml_class<const N: usize>(fields: [(&str, BamlValue); N]) -> BamlValue {
    let mut map = BamlMap::new();
    for (key, value) in fields {
        map.insert(key.to_string(), value);
    }
    BamlValue::Class("DynamicInput".to_string(), map)
}

fn baml_value_from_example_keys(
    data: &HashMap<String, serde_json::Value>,
    keys: &[String],
) -> Result<BamlValue> {
    let mut map = BamlMap::new();
    for key in keys {
        if let Some(value) = data.get(key) {
            let baml_value =
                BamlValue::try_from(value.clone()).map_err(|err| anyhow::anyhow!(err))?;
            map.insert(key.clone(), baml_value);
        }
    }
    Ok(BamlValue::Class("DynamicExample".to_string(), map))
}
