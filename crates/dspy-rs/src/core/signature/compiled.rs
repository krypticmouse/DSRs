use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock};

use serde::Serialize;

use crate::baml_bridge::prompt::{
    PromptPath, PromptValue, PromptWorld, RenderError, RenderSession, RenderSettings,
    RendererOverride,
};
use crate::baml_bridge::{BamlTypeInternal, Registry, ToBamlValue};
use crate::utils::SyncCache;
use crate::{BamlValue, TypeIR};
use minijinja::value::Value;

use super::{FieldRenderSettings, FieldRendererSpec, FieldSpec, SigMeta, Signature};

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

impl<S: Signature> Clone for CompiledSignature<S> {
    fn clone(&self) -> Self {
        Self {
            world: self.world.clone(),
            system_template: self.system_template.clone(),
            user_template: self.user_template.clone(),
            sig_meta: self.sig_meta.clone(),
            _phantom: PhantomData,
        }
    }
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

fn register_field_templates<S: Signature>(
    world: &mut PromptWorld,
) -> Result<(), Box<RenderError>> {
    for field in S::input_fields() {
        let Some(FieldRendererSpec::Jinja { template }) = field.renderer else {
            continue;
        };
        let template_name = field_template_name(field);
        world
            .jinja
            .add_template_owned(template_name.clone(), template.to_string())
            .map_err(|err| {
                Box::new(
                    RenderError::template_error(
                        "<signature>",
                        "signature",
                        "default",
                        "template",
                        template_name,
                        err.line().map(|line| (line, 0)),
                        err.to_string(),
                    )
                    .with_cause(err),
                )
            })?;
    }
    Ok(())
}

fn field_template_name(field: &FieldSpec) -> String {
    format!("sig::input:{}:template", field.rust_name)
}

fn apply_field_settings(
    base: &RenderSettings,
    field_settings: &FieldRenderSettings,
) -> RenderSettings {
    let mut settings = base.clone();
    if let Some(max) = field_settings.max_string_chars {
        settings.max_string_chars = max;
    }
    if let Some(max) = field_settings.max_list_items {
        settings.max_list_items = max;
    }
    if let Some(max) = field_settings.max_map_entries {
        settings.max_map_entries = max;
    }
    if let Some(max) = field_settings.max_depth {
        settings.max_depth = max;
    }
    settings
}

fn extract_input_field(
    input_value: &BamlValue,
    field: &FieldSpec,
) -> Result<BamlValue, Box<RenderError>> {
    match input_value {
        BamlValue::Class(_, fields) => fields.get(field.rust_name).cloned().ok_or_else(|| {
            Box::new(RenderError::new(
                format!("inputs.{}", field.rust_name),
                "input",
                "default",
                "compiled_signature",
                format!("missing input field '{}'", field.rust_name),
            ))
        }),
        BamlValue::Map(fields) => fields.get(field.rust_name).cloned().ok_or_else(|| {
            Box::new(RenderError::new(
                format!("inputs.{}", field.rust_name),
                "input",
                "default",
                "compiled_signature",
                format!("missing input field '{}'", field.rust_name),
            ))
        }),
        other => Err(Box::new(RenderError::new(
            "inputs",
            format!("{other:?}"),
            "default",
            "compiled_signature",
            "input value is not a class or map",
        ))),
    }
}

fn render_signature_template(
    world: &PromptWorld,
    template_name: &str,
    ctx: &Value,
    role: &str,
) -> Result<String, Box<RenderError>> {
    let template = world.jinja.get_template(template_name).map_err(|err| {
        Box::new(
            RenderError::template_error(
                role,
                "signature",
                "default",
                "template",
                template_name,
                err.line().map(|line| (line, 0)),
                err.to_string(),
            )
            .with_cause(err),
        )
    })?;

    template.render(ctx).map_err(|err| {
        Box::new(
            RenderError::template_error(
                role,
                "signature",
                "default",
                "template",
                template_name,
                err.line().map(|line| (line, 0)),
                err.to_string(),
            )
            .with_cause(err),
        )
    })
}

static COMPILED_SIGNATURE_CACHE: LazyLock<
    SyncCache<TypeId, Arc<dyn std::any::Any + Send + Sync>>,
> = LazyLock::new(SyncCache::default);

fn compile_signature_inner<S: Signature>() -> CompiledSignature<S> {
    let mut registry = Registry::new();
    <S::Input as BamlTypeInternal>::register(&mut registry);
    <S::Output as BamlTypeInternal>::register(&mut registry);

    let (output_format, renderer_seed) = registry.build_with_renderers(TypeIR::string());
    let mut world =
        PromptWorld::from_registry(output_format, renderer_seed, RenderSettings::default())
            .expect("failed to build prompt world");
    register_default_templates(&mut world)
        .expect("failed to register default signature templates");
    register_field_templates::<S>(&mut world).expect("failed to register field templates");

    CompiledSignature {
        world: Arc::new(world),
        system_template: "sig::system".to_string(),
        user_template: "sig::user".to_string(),
        sig_meta: SigMeta::from_signature::<S>(),
        _phantom: PhantomData,
    }
}

/// Extension trait for compiling signatures.
pub trait CompileExt: Signature + Sized {
    /// Compile this signature for prompt rendering.
    fn compile() -> CompiledSignature<Self>;
}

impl<T: Signature> CompileExt for T {
    fn compile() -> CompiledSignature<Self> {
        let type_id = TypeId::of::<Self>();
        let cached = COMPILED_SIGNATURE_CACHE.get_or_insert_with(type_id, || {
            Arc::new(compile_signature_inner::<Self>())
                as Arc<dyn std::any::Any + Send + Sync>
        });
        cached
            .downcast_ref::<CompiledSignature<Self>>()
            .expect("cached signature has wrong type")
            .clone()
    }
}

impl<S: Signature> CompiledSignature<S> {
    /// Render system message without input values.
    pub fn render_system_message(&self) -> Result<String, Box<RenderError>> {
        self.render_system_message_with_ctx(())
    }

    /// Render system message with custom context.
    pub fn render_system_message_with_ctx<C: Serialize>(
        &self,
        ctx: C,
    ) -> Result<String, Box<RenderError>> {
        let session = RenderSession::new(self.world.settings.clone()).with_ctx(ctx);
        let ctx_value = Value::from_iter([
            ("sig".to_string(), Value::from_serialize(&self.sig_meta)),
            (
                "inputs".to_string(),
                Value::from_iter(std::iter::empty::<(String, Value)>()),
            ),
            ("ctx".to_string(), session.ctx.clone()),
        ]);

        render_signature_template(&self.world, &self.system_template, &ctx_value, "system")
    }

    /// Render messages with default settings.
    pub fn render_messages(
        &self,
        input: &S::Input,
    ) -> Result<RenderedMessages, Box<RenderError>>
    where
        S::Input: ToBamlValue,
    {
        self.render_messages_with_ctx(input, ())
    }

    /// Render messages with custom context.
    pub fn render_messages_with_ctx<C: Serialize>(
        &self,
        input: &S::Input,
        ctx: C,
    ) -> Result<RenderedMessages, Box<RenderError>>
    where
        S::Input: ToBamlValue,
    {
        let input_value = input.to_baml_value();
        let session = Arc::new(RenderSession::new(self.world.settings.clone()).with_ctx(ctx));

        let mut inputs = Vec::new();
        for field in S::input_fields() {
            let field_value = extract_input_field(&input_value, field)?;
            let is_string = matches!(field_value, BamlValue::String(_));
            let field_ty = (field.type_ir)();
            let path = PromptPath::new()
                .push_field("inputs")
                .push_field(field.rust_name);

            let field_session = if let Some(settings) = field.render_settings.as_ref() {
                let mut adjusted = (*session).clone();
                adjusted.settings = apply_field_settings(&session.settings, settings);
                Arc::new(adjusted)
            } else {
                session.clone()
            };

            let mut pv = PromptValue::new(
                field_value,
                field_ty,
                self.world.clone(),
                field_session,
                path,
            );

            let override_renderer = match field.renderer {
                Some(FieldRendererSpec::Jinja { template }) => Some(RendererOverride::Template {
                    source: template,
                    compiled_name: Some(field_template_name(field)),
                }),
                Some(FieldRendererSpec::Func { f }) => Some(RendererOverride::Func { f }),
                None => {
                    if let Some(style) = field.style {
                        Some(RendererOverride::style(style))
                    } else if !is_string {
                        Some(RendererOverride::style("json"))
                    } else {
                        None
                    }
                }
            };

            if let Some(override_renderer) = override_renderer {
                pv = pv.with_override(override_renderer);
            }

            inputs.push((field.rust_name.to_string(), pv.as_jinja_value()));
        }

        let ctx_value = Value::from_iter([
            ("sig".to_string(), Value::from_serialize(&self.sig_meta)),
            ("inputs".to_string(), Value::from_iter(inputs)),
            ("ctx".to_string(), session.ctx.clone()),
        ]);

        let system =
            render_signature_template(&self.world, &self.system_template, &ctx_value, "system")?;
        let user =
            render_signature_template(&self.world, &self.user_template, &ctx_value, "user")?;

        Ok(RenderedMessages { system, user })
    }
}
