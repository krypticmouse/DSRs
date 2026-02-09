use std::future::Future;
use std::sync::Arc;

use facet::Facet;
use rig::completion::ToolDefinition;
use rig::message::{ToolCall, ToolFunction};
use rig::tool::{ToolDyn, ToolError};
use rig::wasm_compat::WasmBoxedFuture;

use crate::core::{Module, Signature};
use crate::predictors::{Predict, PredictBuilder};
use crate::{BamlType, CallOutcome};

/// ReAct action-step schema.
#[derive(dsrs_macros::Signature, Clone, Debug)]
struct ReActActionStep {
    #[input]
    input: String,

    #[input]
    trajectory: String,

    #[output]
    thought: String,

    #[output]
    action: String,

    #[output]
    action_input: String,
}

type ReActActionStepOutput = __ReActActionStepOutput;

/// ReAct extraction-step schema.
#[derive(dsrs_macros::Signature, Clone, Debug)]
struct ReActExtractStep<O>
where
    O: BamlType + for<'a> Facet<'a> + Send + Sync + 'static,
{
    #[input]
    input: String,

    #[input]
    trajectory: String,

    #[output]
    output: O,
}

type ReActExtractStepOutput<O> = __ReActExtractStepOutput<O>;

#[derive(facet::Facet)]
#[facet(crate = facet)]
pub struct ReAct<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
{
    #[facet(opaque)]
    action: Predict<ReActActionStep>,
    #[facet(opaque)]
    extract: Predict<ReActExtractStep<S::Output>>,
    #[facet(skip, opaque)]
    tools: Vec<Arc<dyn ToolDyn>>,
    #[facet(skip)]
    max_steps: usize,
}

impl<S> ReAct<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
{
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> ReActBuilder<S> {
        ReActBuilder::new()
    }

    pub async fn call(&self, input: S::Input) -> CallOutcome<S::Output> {
        self.forward(input).await
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
}

impl<S> Default for ReAct<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Module for ReAct<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
{
    type Input = S::Input;
    type Output = S::Output;

    async fn forward(&self, input: S::Input) -> CallOutcome<S::Output> {
        let serialized_input = serde_json::to_string(&input.to_baml_value())
            .unwrap_or_else(|_| "<input serialization failed>".to_string());

        let mut trajectory = self.render_tool_manifest().await;
        trajectory.push_str("\n\n");

        let mut tool_calls = Vec::new();
        let mut tool_executions = Vec::new();

        for step in 0..self.max_steps {
            let action_input = ReActActionStepInput {
                input: serialized_input.clone(),
                trajectory: trajectory.clone(),
            };

            let (action_result, mut action_metadata) = self.action.call(action_input).await.into_parts();
            tool_calls.append(&mut action_metadata.tool_calls);
            tool_executions.append(&mut action_metadata.tool_executions);

            let ReActActionStepOutput {
                thought,
                action,
                action_input,
            } = match action_result {
                Ok(output) => output,
                Err(err) => return CallOutcome::err(err, action_metadata),
            };

            let action_name = action
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();

            if Self::is_terminal_action(&action_name) {
                trajectory.push_str(&format!(
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
            tool_executions.push(observation.clone());

            trajectory.push_str(&format!(
                "Step {}\nThought: {}\nAction: {}\nAction Input: {}\nObservation: {}\n\n",
                step + 1,
                thought,
                action_name,
                action_input,
                observation
            ));
        }

        let extract_input = ReActExtractStepInput {
            input: serialized_input,
            trajectory,
            __phantom: std::marker::PhantomData,
        };

        let (extract_result, mut extract_metadata) = self.extract.call(extract_input).await.into_parts();
        extract_metadata.tool_calls.extend(tool_calls);
        extract_metadata.tool_executions.extend(tool_executions);

        match extract_result {
            Ok(output) => {
                let output: ReActExtractStepOutput<S::Output> = output;
                CallOutcome::ok(output.output, extract_metadata)
            }
            Err(err) => CallOutcome::err(err, extract_metadata),
        }
    }
}

pub struct ReActBuilder<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
{
    action: PredictBuilder<ReActActionStep>,
    extract: PredictBuilder<ReActExtractStep<S::Output>>,
    tools: Vec<Arc<dyn ToolDyn>>,
    max_steps: usize,
}

impl<S> ReActBuilder<S>
where
    S: Signature,
    S::Input: BamlType + Clone,
    S::Output: BamlType,
{
    fn new() -> Self {
        Self {
            action: Predict::builder(),
            extract: Predict::builder(),
            tools: Vec::new(),
            max_steps: 4,
        }
    }

    pub fn action_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.action = self.action.instruction(instruction);
        self
    }

    pub fn extract_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.extract = self.extract.instruction(instruction);
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
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

    pub fn tool<F, Fut>(mut self, name: impl Into<String>, description: impl Into<String>, tool_fn: F) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = String> + Send + 'static,
    {
        self.tools.push(Arc::new(PlainAsyncTool {
            name: name.into(),
            description: description.into(),
            handler: tool_fn,
        }));
        self
    }

    pub fn build(self) -> ReAct<S> {
        ReAct {
            action: self.action.build(),
            extract: self.extract.build(),
            tools: self.tools,
            max_steps: self.max_steps,
        }
    }
}

struct PlainAsyncTool<F> {
    name: String,
    description: String,
    handler: F,
}

impl<F, Fut> ToolDyn for PlainAsyncTool<F>
where
    F: Fn(String) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = String> + Send + 'static,
{
    fn name(&self) -> String {
        self.name.clone()
    }

    fn definition<'a>(&'a self, _prompt: String) -> WasmBoxedFuture<'a, ToolDefinition> {
        Box::pin(async move {
            ToolDefinition {
                name: self.name.clone(),
                description: self.description.clone(),
                parameters: serde_json::json!({
                    "type": "object",
                    "additionalProperties": true
                }),
            }
        })
    }

    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move { Ok((self.handler)(args).await) })
    }
}
