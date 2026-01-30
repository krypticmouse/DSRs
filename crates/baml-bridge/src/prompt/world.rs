//! Prompt rendering world for typed prompt infrastructure.

use std::sync::Arc;

use baml_types::{StreamingMode, TypeIR};
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum};
use minijinja::{Environment, UndefinedBehavior};

use super::jinja::register_prompt_filters;
use super::renderer::{
    CompiledRenderer, RenderError, RenderResult, RenderSettings, RendererDb, RendererDbSeed,
    RendererOverride, TypeKey,
};
use minijinja::value::Value;
use super::value::{default_union_resolver, UnionResolver};
use internal_baml_jinja::types::OutputFormatContent;

/// Type database extracted from OutputFormatContent.
#[derive(Debug, Clone)]
pub struct TypeDb {
    pub enums: Arc<IndexMap<String, Enum>>,
    pub classes: Arc<IndexMap<(String, StreamingMode), Class>>,
    pub structural_recursive_aliases: Arc<IndexMap<String, TypeIR>>,
    pub recursive_classes: Arc<IndexSet<String>>,
}

impl TypeDb {
    /// Look up a class by name and streaming mode.
    pub fn find_class(&self, name: &str, mode: StreamingMode) -> Option<&Class> {
        self.classes.get(&(name.to_string(), mode))
    }

    /// Get a field's type from a class.
    pub fn class_field_type(
        &self,
        name: &str,
        mode: StreamingMode,
        field: &str,
    ) -> Option<TypeIR> {
        let class = self.find_class(name, mode)?;
        class
            .fields
            .iter()
            .find(|(field_name, _, _, _)| {
                field_name.real_name() == field || field_name.rendered_name() == field
            })
            .map(|(_, r#type, _, _)| r#type.clone())
    }

    /// Resolve a recursive type alias.
    pub fn resolve_recursive_alias(&self, name: &str) -> Option<&TypeIR> {
        self.structural_recursive_aliases.get(name)
    }

    /// Check if a class is recursive.
    pub fn is_recursive(&self, name: &str) -> bool {
        self.recursive_classes.contains(name)
    }

    /// Look up an enum by name.
    pub fn find_enum(&self, name: &str) -> Option<&Enum> {
        self.enums.get(name)
    }
}

#[derive(Debug, Clone)]
pub struct PromptWorld {
    pub types: TypeDb,
    pub renderers: RendererDb,
    pub jinja: Environment<'static>,
    pub settings: RenderSettings,
    pub union_resolver: UnionResolver,
}

impl PromptWorld {
    #[allow(clippy::result_large_err)]
    pub fn from_registry(
        output_format: OutputFormatContent,
        renderer_seed: RendererDbSeed,
        settings: RenderSettings,
    ) -> Result<Self, RenderError> {
        let types = TypeDb {
            enums: output_format.enums.clone(),
            classes: output_format.classes.clone(),
            structural_recursive_aliases: output_format.structural_recursive_aliases.clone(),
            recursive_classes: output_format.recursive_classes.clone(),
        };

        let mut jinja = crate::jsonish::jinja_helpers::get_env();
        jinja.set_undefined_behavior(UndefinedBehavior::Strict);
        register_prompt_filters(&mut jinja);

        let renderers = RendererDb::compile_from_seed(renderer_seed, &mut jinja)?;

        Ok(Self {
            types,
            renderers,
            jinja,
            settings,
            union_resolver: default_union_resolver,
        })
    }

    /// Main rendering entry point for typed prompt values.
    #[allow(clippy::result_large_err)]
    pub fn render_prompt_value(
        &self,
        pv: &super::PromptValue,
        style_override: Option<&str>,
    ) -> RenderResult {
        let style = self.resolve_style(pv, style_override);
        let rendered = self.try_render_chain(pv, &style)?;
        Ok(self.apply_budget_truncation(rendered, pv))
    }

    fn resolve_style(&self, pv: &super::PromptValue, override_style: Option<&str>) -> String {
        if let Some(style) = override_style {
            return style.to_string();
        }

        if let Some(RendererOverride::Style { style }) = pv.override_renderer.as_ref() {
            return style.to_string();
        }

        "default".to_string()
    }

    #[allow(clippy::result_large_err)]
    fn try_render_chain(&self, pv: &super::PromptValue, style: &str) -> RenderResult {
        if let Some(result) = self.try_field_override(pv, style)? {
            return Ok(result);
        }

        if let Some(result) = self.try_type_renderer(pv, style)? {
            return Ok(result);
        }

        if let Some(result) = self.try_builtin_style(pv, style)? {
            return Ok(result);
        }

        Ok(self.render_structural(pv))
    }

    #[allow(clippy::result_large_err)]
    fn try_field_override(
        &self,
        pv: &super::PromptValue,
        style: &str,
    ) -> Result<Option<String>, RenderError> {
        match pv.override_renderer.as_ref() {
            Some(RendererOverride::Func { f }) => f(pv, &pv.session).map(Some),
            Some(RendererOverride::Template {
                source,
                compiled_name,
            }) => {
                let renderer = "field_override:template";
                let rendered = if let Some(name) = compiled_name.as_deref() {
                    self.render_named_template(pv, style, renderer, name)?
                } else {
                    self.render_inline_template(pv, style, renderer, source)?
                };
                Ok(Some(rendered))
            }
            Some(RendererOverride::Style { .. }) | None => Ok(None),
        }
    }

    #[allow(clippy::result_large_err)]
    fn try_type_renderer(
        &self,
        pv: &super::PromptValue,
        style: &str,
    ) -> Result<Option<String>, RenderError> {
        let type_key = match pv.resolved_ty() {
            TypeIR::Class { name, mode, .. } => TypeKey::Class { name, mode },
            TypeIR::Enum { name, .. } => TypeKey::Enum { name },
            _ => return Ok(None),
        };

        let renderer = match self.renderers.find(&type_key, style) {
            Some(renderer) => renderer,
            None => return Ok(None),
        };

        let label = format!("type:{}:{}", type_key, style);
        match renderer {
            CompiledRenderer::Func { f } => f(pv, &pv.session).map(Some),
            CompiledRenderer::Jinja { template_name } => self
                .render_named_template(pv, style, &label, template_name)
                .map(Some),
        }
    }

    #[allow(clippy::result_large_err)]
    fn try_builtin_style(
        &self,
        pv: &super::PromptValue,
        style: &str,
    ) -> Result<Option<String>, RenderError> {
        match style {
            "json" | "yaml" | "toon" => Ok(Some(format!("<{style}> {}", pv.value()))),
            _ => Ok(None),
        }
    }

    fn render_structural(&self, pv: &super::PromptValue) -> String {
        format!("<structural> {}", pv.value())
    }

    fn apply_budget_truncation(&self, rendered: String, pv: &super::PromptValue) -> String {
        let max = pv.session.settings.max_total_chars;
        let len = rendered.chars().count();
        if len <= max {
            rendered
        } else {
            rendered.chars().take(max).collect()
        }
    }

    #[allow(clippy::result_large_err)]
    fn render_named_template(
        &self,
        pv: &super::PromptValue,
        style: &str,
        renderer: &str,
        template_name: &str,
    ) -> RenderResult {
        let ctx = self.render_context(pv);
        let template = self.jinja.get_template(template_name).map_err(|err| {
            RenderError::template_error(
                pv.path.to_string(),
                pv.ty().diagnostic_repr().to_string(),
                style.to_string(),
                renderer.to_string(),
                template_name.to_string(),
                err.line().map(|line| (line, 0)),
                err.to_string(),
            )
            .with_cause(err)
        })?;

        template.render(ctx).map_err(|err| {
            RenderError::template_error(
                pv.path.to_string(),
                pv.ty().diagnostic_repr().to_string(),
                style.to_string(),
                renderer.to_string(),
                template_name.to_string(),
                err.line().map(|line| (line, 0)),
                err.to_string(),
            )
            .with_cause(err)
        })
    }

    #[allow(clippy::result_large_err)]
    fn render_inline_template(
        &self,
        pv: &super::PromptValue,
        style: &str,
        renderer: &str,
        source: &str,
    ) -> RenderResult {
        let ctx = self.render_context(pv);
        self.jinja.render_str(source, ctx).map_err(|err| {
            RenderError::template_error(
                pv.path.to_string(),
                pv.ty().diagnostic_repr().to_string(),
                style.to_string(),
                renderer.to_string(),
                "<inline>",
                err.line().map(|line| (line, 0)),
                err.to_string(),
            )
            .with_cause(err)
        })
    }

    fn render_context(&self, pv: &super::PromptValue) -> Value {
        Value::from_iter([
            ("value".to_string(), pv.as_jinja_value()),
            ("ctx".to_string(), pv.session.ctx.clone()),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::PromptWorld;
    use crate::prompt::renderer::{
        CompiledRenderer, RenderError, RenderSession, RenderSettings, RendererDb, RendererKey,
        RendererOverride,
    };
    use crate::prompt::value::default_union_resolver;
    use crate::prompt::{PromptPath, PromptValue, TypeDb};
    use baml_types::{BamlValue, StreamingMode, TypeIR};
    use indexmap::{IndexMap, IndexSet};
    use internal_baml_jinja::types::{Class, Enum};
    use std::sync::Arc;

    #[test]
    fn resolve_style_prefers_override_param() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string());
        assert_eq!(world.resolve_style(&pv, Some("json")), "json");
    }

    #[test]
    fn resolve_style_uses_field_override_when_present() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string())
            .with_override(RendererOverride::Style { style: "toon" });
        assert_eq!(world.resolve_style(&pv, None), "toon");
    }

    #[test]
    fn render_prompt_value_prefers_field_override() {
        let mut renderers = RendererDb::new();
        renderers.insert(
            RendererKey::for_class("Widget", StreamingMode::NonStreaming, "default"),
            CompiledRenderer::Func { f: |_, _| Ok("type".to_string()) },
        );
        let world = make_world(renderers, RenderSettings::default());

        let pv = make_prompt_value(
            &world,
            BamlValue::Class("Widget".to_string(), IndexMap::new()),
            TypeIR::class("Widget"),
        )
        .with_override(RendererOverride::Func { f: |_, _| Ok("field".to_string()) });

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "field");
    }

    #[test]
    fn render_prompt_value_uses_type_renderer() {
        let mut renderers = RendererDb::new();
        renderers.insert(
            RendererKey::for_enum("Choice", "default"),
            CompiledRenderer::Func { f: |_, _| Ok("type".to_string()) },
        );
        let world = make_world(renderers, RenderSettings::default());

        let pv = make_prompt_value(
            &world,
            BamlValue::Enum("Choice".to_string(), "Yes".to_string()),
            TypeIR::r#enum("Choice"),
        );

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "type");
    }

    #[test]
    fn render_prompt_value_uses_builtin_style() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string());

        let rendered = world.render_prompt_value(&pv, Some("json")).unwrap();
        assert_eq!(rendered, "<json> String(\"hi\")");
    }

    #[test]
    fn render_prompt_value_falls_back_structural() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string());

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "<structural> String(\"hi\")");
    }

    #[test]
    fn render_prompt_value_truncates_budget() {
        let settings = RenderSettings {
            max_total_chars: 4,
            ..RenderSettings::default()
        };
        let world = make_world(RendererDb::new(), settings);
        let pv = make_prompt_value(&world, BamlValue::String("hello".to_string()), TypeIR::string());

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "<str");
    }

    #[test]
    fn render_prompt_value_propagates_errors() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string())
            .with_override(RendererOverride::Func { f: |pv, _| {
                Err(RenderError::new(
                    pv.path.to_string(),
                    pv.ty().diagnostic_repr().to_string(),
                    "default",
                    "field_override",
                    "boom",
                ))
            }});

        let err = world.render_prompt_value(&pv, None).unwrap_err();
        assert_eq!(err.message, "boom");
    }

    fn make_world(renderers: RendererDb, settings: RenderSettings) -> PromptWorld {
        PromptWorld {
            types: TypeDb {
                enums: Arc::new(IndexMap::<String, Enum>::new()),
                classes: Arc::new(IndexMap::<(String, StreamingMode), Class>::new()),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
            renderers,
            jinja: crate::jsonish::jinja_helpers::get_env(),
            settings,
            union_resolver: default_union_resolver,
        }
    }

    fn make_prompt_value(world: &PromptWorld, value: BamlValue, ty: TypeIR) -> PromptValue {
        PromptValue::new(
            value,
            ty,
            Arc::new(world.clone()),
            Arc::new(RenderSession::new(world.settings.clone())),
            PromptPath::new(),
        )
    }
}
