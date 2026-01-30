use std::marker::PhantomData;
use std::sync::Arc;

use serde::Serialize;

use crate::baml_bridge::prompt::{PromptWorld, RenderError, RenderSettings};
use crate::baml_bridge::{BamlTypeInternal, Registry};
use crate::TypeIR;

use super::{SigMeta, Signature};

/// Default system template - describes input/output fields.
pub const DEFAULT_SYSTEM_TEMPLATE: &str = r#"
Your input fields are:
{% for f in sig.inputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% endfor %}

Your output fields are:
{% for f in sig.outputs -%}
- {{ f.llm_name }}: {{ f.type_name }}
{% if f.schema %}
{{ f.schema }}
{% endif %}
{% endfor %}
"#;

/// Default user template - renders actual input values.
pub const DEFAULT_USER_TEMPLATE: &str = r#"
{% for f in sig.inputs %}
[[ ## {{ f.llm_name }} ## ]]
{{ inputs[f.rust_name] }}

{% endfor %}
"#;

/// A compiled signature ready for prompt rendering.
pub struct CompiledSignature<S: Signature> {
    /// The prompt world with types and renderers.
    pub world: Arc<PromptWorld>,
    /// System message template (compiled name in env).
    pub system_template: String,
    /// User message template (compiled name in env).
    pub user_template: String,
    /// Signature metadata for templates.
    pub sig_meta: SigMeta,
    pub(crate) _phantom: PhantomData<S>,
}

/// Rendered prompt messages.
pub struct RenderedMessages {
    pub system: String,
    pub user: String,
}

pub fn register_default_templates(
    world: &mut PromptWorld,
) -> Result<(), Box<RenderError>> {
    world
        .jinja
        .add_template_owned("sig::system", DEFAULT_SYSTEM_TEMPLATE.to_string())
        .map_err(|err| {
            Box::new(
                RenderError::template_error(
                    "<signature>",
                    "signature",
                    "default",
                    "template",
                    "sig::system",
                    err.line().map(|line| (line, 0)),
                    err.to_string(),
                )
                .with_cause(err),
            )
        })?;

    world
        .jinja
        .add_template_owned("sig::user", DEFAULT_USER_TEMPLATE.to_string())
        .map_err(|err| {
            Box::new(
                RenderError::template_error(
                    "<signature>",
                    "signature",
                    "default",
                    "template",
                    "sig::user",
                    err.line().map(|line| (line, 0)),
                    err.to_string(),
                )
                .with_cause(err),
            )
        })?;

    Ok(())
}

/// Extension trait for compiling signatures.
pub trait CompileExt: Signature + Sized {
    /// Compile this signature for prompt rendering.
    fn compile() -> CompiledSignature<Self>;
}

impl<T: Signature> CompileExt for T {
    fn compile() -> CompiledSignature<Self> {
        let mut registry = Registry::new();
        <Self::Input as BamlTypeInternal>::register(&mut registry);
        <Self::Output as BamlTypeInternal>::register(&mut registry);

        let (output_format, renderer_seed) = registry.build_with_renderers(TypeIR::string());
        let mut world = PromptWorld::from_registry(
            output_format,
            renderer_seed,
            RenderSettings::default(),
        )
        .expect("failed to build prompt world");
        register_default_templates(&mut world)
            .expect("failed to register default signature templates");

        CompiledSignature {
            world: Arc::new(world),
            system_template: "sig::system".to_string(),
            user_template: "sig::user".to_string(),
            sig_meta: SigMeta::from_signature::<Self>(),
            _phantom: PhantomData,
        }
    }
}

impl<S: Signature> CompiledSignature<S> {
    /// Render messages with default settings.
    pub fn render_messages(
        &self,
        input: &S::Input,
    ) -> Result<RenderedMessages, Box<RenderError>> {
        self.render_messages_with_ctx(input, ())
    }

    /// Render messages with custom context.
    pub fn render_messages_with_ctx<C: Serialize>(
        &self,
        _input: &S::Input,
        _ctx: C,
    ) -> Result<RenderedMessages, Box<RenderError>> {
        todo!("render_messages_with_ctx implemented in dsrs-n9u.40")
    }
}
