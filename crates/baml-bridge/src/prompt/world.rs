//! Prompt rendering world for typed prompt infrastructure.

use std::sync::Arc;

use baml_types::{BamlValue, StreamingMode, TypeIR};
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
            Some(RendererOverride::Style { style: override_style }) => {
                if style == *override_style || style == "default" {
                    return self.try_type_renderer(pv, override_style);
                }
                Ok(None)
            }
            Some(RendererOverride::Template { compiled_name, .. }) => {
                if style != "default" && style != "template" {
                    return Ok(None);
                }

                let template_name = compiled_name.as_deref().ok_or_else(|| {
                    RenderError::new(
                        pv.path.to_string(),
                        pv.ty().diagnostic_repr().to_string(),
                        style.to_string(),
                        "field_override:template",
                        "field template not compiled",
                    )
                })?;

                let rendered =
                    self.render_named_template(pv, style, "field_override:template", template_name)?;
                Ok(Some(rendered))
            }
            Some(RendererOverride::Func { f }) => {
                if style != "default" && style != "func" {
                    return Ok(None);
                }
                f(pv, &pv.session).map(Some)
            }
            None => Ok(None),
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
            "json" | "yaml" | "toon" => self.render_format_style(pv, style).map(Some),
            _ => Ok(None),
        }
    }

    #[allow(clippy::result_large_err)]
    fn render_format_style(
        &self,
        pv: &super::PromptValue,
        fmt: &str,
    ) -> Result<String, RenderError> {
        let view = self.build_output_format_view(pv.ty());
        internal_baml_jinja::format_baml_value(pv.value(), &view, fmt).map_err(|err| {
            RenderError::new(
                pv.path.to_string(),
                pv.ty().diagnostic_repr().to_string(),
                fmt.to_string(),
                format!("builtin:{fmt}"),
                format!("format style render failed: {err}"),
            )
        })
    }

    fn build_output_format_view(&self, target: &TypeIR) -> OutputFormatContent {
        OutputFormatContent {
            enums: self.types.enums.clone(),
            classes: self.types.classes.clone(),
            recursive_classes: self.types.recursive_classes.clone(),
            structural_recursive_aliases: self.types.structural_recursive_aliases.clone(),
            target: target.clone(),
        }
    }

    fn render_structural(&self, pv: &super::PromptValue) -> String {
        let mut buf = String::new();
        let resolved = pv.resolved_ty();
        self.render_structural_inner(pv, &resolved, pv.session.depth, &mut buf);
        buf
    }

    fn render_structural_inner(
        &self,
        pv: &super::PromptValue,
        resolved_ty: &TypeIR,
        depth: usize,
        buf: &mut String,
    ) {
        let settings = &pv.session.settings;

        if matches!(pv.ty(), TypeIR::Union(_, _)) && !pv.is_union_resolved() {
            self.render_union_ambiguous(pv, buf);
            return;
        }

        if depth >= settings.max_depth {
            let label = self.type_display_name(resolved_ty);
            buf.push_str(&format!("{label} {{ ... }}"));
            return;
        }

        match pv.value() {
            BamlValue::String(s) => self.render_string(s, buf, settings),
            BamlValue::Int(i) => buf.push_str(&i.to_string()),
            BamlValue::Float(f) => buf.push_str(&f.to_string()),
            BamlValue::Bool(b) => buf.push_str(if *b { "true" } else { "false" }),
            BamlValue::Null => buf.push_str("null"),
            BamlValue::List(items) => self.render_list_structural(pv, items, depth, buf),
            BamlValue::Map(map) => self.render_map_structural(pv, map, depth, buf),
            BamlValue::Class(name, fields) => {
                self.render_class_structural(pv, resolved_ty, name, fields, depth, buf)
            }
            BamlValue::Enum(_, variant) => self.render_enum_structural(resolved_ty, variant, buf),
            BamlValue::Media(media) => {
                buf.push_str(&format!("<media: {}>", media.media_type));
            }
        }
    }

    fn render_string(&self, s: &str, buf: &mut String, settings: &RenderSettings) {
        let max = settings.max_string_chars;
        if s.chars().count() > max {
            let truncated: String = s.chars().take(max).collect();
            buf.push_str(&truncated);
            buf.push_str("... (truncated)");
        } else {
            buf.push_str(s);
        }
    }

    fn render_list_structural(
        &self,
        pv: &super::PromptValue,
        items: &[BamlValue],
        depth: usize,
        buf: &mut String,
    ) {
        let settings = &pv.session.settings;
        let capped = items.len().min(settings.max_list_items);

        buf.push('[');
        for (idx, item) in items.iter().take(capped).enumerate() {
            if idx > 0 {
                buf.push_str(", ");
            }
            if let Some(child) = pv.child_index(idx) {
                let resolved = child.resolved_ty();
                self.render_structural_inner(&child, &resolved, depth + 1, buf);
            } else {
                buf.push_str(&format!("{}", item));
            }
        }

        if items.len() > capped {
            if capped > 0 {
                buf.push_str(", ");
            }
            let remaining = items.len() - capped;
            buf.push_str(&format!("... (+{remaining} more)"));
        }

        buf.push(']');
    }

    fn render_map_structural(
        &self,
        pv: &super::PromptValue,
        map: &IndexMap<String, BamlValue>,
        depth: usize,
        buf: &mut String,
    ) {
        let settings = &pv.session.settings;
        let mut keys: Vec<&String> = map.keys().collect();
        keys.sort();

        let capped = keys.len().min(settings.max_map_entries);
        buf.push('{');
        for (idx, key) in keys.iter().take(capped).enumerate() {
            if idx > 0 {
                buf.push_str(", ");
            }
            buf.push_str(key);
            buf.push_str(": ");
            if let Some(child) = pv.child_map_value(key) {
                let resolved = child.resolved_ty();
                self.render_structural_inner(&child, &resolved, depth + 1, buf);
            } else if let Some(value) = map.get(*key) {
                buf.push_str(&format!("{}", value));
            }
        }

        if keys.len() > capped {
            if capped > 0 {
                buf.push_str(", ");
            }
            let remaining = keys.len() - capped;
            buf.push_str(&format!("... (+{remaining} more)"));
        }

        buf.push('}');
    }

    fn render_class_structural(
        &self,
        pv: &super::PromptValue,
        resolved_ty: &TypeIR,
        class_name: &str,
        fields: &IndexMap<String, BamlValue>,
        depth: usize,
        buf: &mut String,
    ) {
        let settings = &pv.session.settings;
        let label = match resolved_ty {
            TypeIR::Class { name, .. } => name.as_str(),
            _ => class_name,
        };

        let mut ordered: Vec<String> = Vec::new();
        let mut used_keys = IndexSet::new();

        let class = match resolved_ty {
            TypeIR::Class { name, mode, .. } => self.types.find_class(name, *mode),
            _ => self.types.find_class(class_name, baml_types::StreamingMode::NonStreaming),
        };

        if let Some(class) = class {
            for (field_name, ..) in &class.fields {
                let real = field_name.real_name();
                let rendered = field_name.rendered_name();
                if fields.contains_key(real) || fields.contains_key(rendered) {
                    ordered.push(rendered.to_string());
                    used_keys.insert(real.to_string());
                    used_keys.insert(rendered.to_string());
                }
            }
        }

        let mut extras: Vec<String> = fields
            .keys()
            .filter(|key| !used_keys.contains(*key))
            .cloned()
            .collect();
        extras.sort();
        ordered.extend(extras);

        let capped = ordered.len().min(settings.max_map_entries);
        buf.push_str(label);
        buf.push_str(" {");
        for (idx, key) in ordered.iter().take(capped).enumerate() {
            if idx > 0 {
                buf.push_str(", ");
            }
            buf.push_str(key);
            buf.push_str(": ");
            if let Some(child) = pv.child_field(key) {
                let resolved = child.resolved_ty();
                self.render_structural_inner(&child, &resolved, depth + 1, buf);
            } else if let Some(value) = fields.get(key) {
                buf.push_str(&format!("{}", value));
            }
        }

        if ordered.len() > capped {
            if capped > 0 {
                buf.push_str(", ");
            }
            let remaining = ordered.len() - capped;
            buf.push_str(&format!("... (+{remaining} more)"));
        }

        buf.push('}');
    }

    fn render_enum_structural(&self, resolved_ty: &TypeIR, variant: &str, buf: &mut String) {
        let rendered = match resolved_ty {
            TypeIR::Enum { name, .. } => self
                .types
                .find_enum(name)
                .and_then(|enum_type| {
                    enum_type
                        .values
                        .iter()
                        .find(|(enum_name, _)| {
                            enum_name.real_name() == variant
                                || enum_name.rendered_name() == variant
                        })
                        .map(|(enum_name, _)| enum_name.rendered_name().to_string())
                })
                .unwrap_or_else(|| variant.to_string()),
            _ => variant.to_string(),
        };

        buf.push_str(&rendered);
    }

    fn render_union_ambiguous(&self, pv: &super::PromptValue, buf: &mut String) {
        let TypeIR::Union(union, _) = pv.ty() else {
            return;
        };
        let max = pv.session.settings.max_union_branches_shown;
        let mut labels: Vec<String> = union
            .iter_skip_null()
            .into_iter()
            .map(|ty| self.type_display_name(ty))
            .collect();

        if labels.is_empty() {
            buf.push_str("one of: <unknown>");
            return;
        }

        buf.push_str("one of: ");
        if labels.len() > max {
            let remaining = labels.len() - max;
            labels.truncate(max);
            buf.push_str(&labels.join(" | "));
            buf.push_str(&format!(" | ... (+{remaining} more)"));
        } else {
            buf.push_str(&labels.join(" | "));
        }
    }

    fn type_display_name(&self, ty: &TypeIR) -> String {
        match ty {
            TypeIR::Class { name, .. } => name.to_string(),
            TypeIR::Enum { name, .. } => name.to_string(),
            _ => ty.diagnostic_repr().to_string(),
        }
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
    use baml_types::{
        ir_type::UnionConstructor, BamlMedia, BamlMediaType, BamlValue, StreamingMode, TypeIR,
    };
    use indexmap::{IndexMap, IndexSet};
    use internal_baml_jinja::types::{Class, Enum, Name};
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
        assert_eq!(rendered, "\"hi\"");
    }

    #[test]
    fn render_prompt_value_falls_back_structural() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string());

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "hi");
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
        assert_eq!(rendered, "hell");
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

    #[test]
    fn field_override_style_delegates_to_type_renderer() {
        let mut renderers = RendererDb::new();
        renderers.insert(
            RendererKey::for_class("Widget", StreamingMode::NonStreaming, "json"),
            CompiledRenderer::Func { f: |_, _| Ok("type-json".to_string()) },
        );
        let world = make_world(renderers, RenderSettings::default());
        let pv = make_prompt_value(
            &world,
            BamlValue::Class("Widget".to_string(), IndexMap::new()),
            TypeIR::class("Widget"),
        )
        .with_override(RendererOverride::Style { style: "json" });

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "type-json");
    }

    #[test]
    fn field_override_style_ignores_mismatch() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string())
            .with_override(RendererOverride::Style { style: "json" });

        assert!(world.try_field_override(&pv, "yaml").unwrap().is_none());
    }

    #[test]
    fn field_override_template_requires_compiled_name() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string())
            .with_override(RendererOverride::Template {
                source: "{{ value.raw }}",
                compiled_name: None,
            });

        let err = world.try_field_override(&pv, "default").unwrap_err();
        assert_eq!(err.message, "field template not compiled");
    }

    #[test]
    fn field_override_template_executes_compiled_template() {
        let mut world = make_world(RendererDb::new(), RenderSettings::default());
        world
            .jinja
            .add_template_owned("field_template".to_string(), "{{ value.raw }}".to_string())
            .expect("template add");
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string())
            .with_override(RendererOverride::Template {
                source: "{{ value.raw }}",
                compiled_name: Some("field_template".to_string()),
            });

        let rendered = world.render_prompt_value(&pv, None).unwrap();
        assert_eq!(rendered, "hi");
    }

    #[test]
    fn builtin_format_uses_value_type_target() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(
            &world,
            BamlValue::String("hi".to_string()),
            TypeIR::string(),
        );

        let view = world.build_output_format_view(pv.ty());
        assert_eq!(view.target, TypeIR::string());
    }

    #[test]
    fn field_override_func_respects_style_filter() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let pv = make_prompt_value(&world, BamlValue::String("hi".to_string()), TypeIR::string())
            .with_override(RendererOverride::Func { f: |_, _| Ok("func".to_string()) });

        assert!(world.try_field_override(&pv, "json").unwrap().is_none());
        assert_eq!(
            world.try_field_override(&pv, "func").unwrap(),
            Some("func".to_string())
        );
    }

    #[test]
    fn structural_renders_primitives() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let cases = vec![
            (BamlValue::String("hi".to_string()), TypeIR::string(), "hi"),
            (BamlValue::Int(42), TypeIR::int(), "42"),
            (BamlValue::Float(1.5), TypeIR::float(), "1.5"),
            (BamlValue::Bool(true), TypeIR::bool(), "true"),
            (BamlValue::Null, TypeIR::null(), "null"),
        ];

        for (value, ty, expected) in cases {
            let pv = make_prompt_value(&world, value, ty);
            assert_eq!(world.render_structural(&pv), expected);
        }
    }

    #[test]
    fn structural_renders_enum_alias() {
        let enum_type = build_enum(
            "Choice",
            vec![("Yes", Some("y")), ("No", None)],
        );
        let world = make_world_with_types(
            RendererDb::new(),
            RenderSettings::default(),
            vec![enum_type],
            vec![],
        );
        let pv = make_prompt_value(
            &world,
            BamlValue::Enum("Choice".to_string(), "Yes".to_string()),
            TypeIR::r#enum("Choice"),
        );

        assert_eq!(world.render_structural(&pv), "y");
    }

    #[test]
    fn structural_renders_media() {
        let world = make_world(RendererDb::new(), RenderSettings::default());
        let media = BamlMedia::url(BamlMediaType::Image, "http://example.com".to_string(), None);
        let pv = make_prompt_value(&world, BamlValue::Media(media), TypeIR::image());

        assert_eq!(world.render_structural(&pv), "<media: image>");
    }

    #[test]
    fn structural_truncates_strings() {
        let settings = RenderSettings {
            max_string_chars: 3,
            ..RenderSettings::default()
        };
        let world = make_world(RendererDb::new(), settings);
        let pv = make_prompt_value(&world, BamlValue::String("hello".to_string()), TypeIR::string());

        assert_eq!(world.render_structural(&pv), "hel... (truncated)");
    }

    #[test]
    fn structural_caps_lists() {
        let settings = RenderSettings {
            max_list_items: 2,
            ..RenderSettings::default()
        };
        let world = make_world(RendererDb::new(), settings);
        let pv = make_prompt_value(
            &world,
            BamlValue::List(vec![
                BamlValue::String("a".to_string()),
                BamlValue::String("b".to_string()),
                BamlValue::String("c".to_string()),
            ]),
            TypeIR::list(TypeIR::string()),
        );

        assert_eq!(world.render_structural(&pv), "[a, b, ... (+1 more)]");
    }

    #[test]
    fn structural_caps_maps_sorted() {
        let settings = RenderSettings {
            max_map_entries: 1,
            ..RenderSettings::default()
        };
        let world = make_world(RendererDb::new(), settings);
        let pv = make_prompt_value(
            &world,
            BamlValue::Map(IndexMap::from([
                ("b".to_string(), BamlValue::Int(1)),
                ("a".to_string(), BamlValue::Int(2)),
            ])),
            TypeIR::map(TypeIR::string(), TypeIR::int()),
        );

        assert_eq!(world.render_structural(&pv), "{a: 2, ... (+1 more)}");
    }

    #[test]
    fn structural_renders_class_in_schema_order() {
        let class = build_class(
            "Widget",
            vec![
                ("z".to_string(), TypeIR::int(), None),
                ("a".to_string(), TypeIR::int(), None),
            ],
        );
        let world = make_world_with_types(
            RendererDb::new(),
            RenderSettings::default(),
            vec![],
            vec![class],
        );
        let pv = make_prompt_value(
            &world,
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([
                    ("a".to_string(), BamlValue::Int(1)),
                    ("z".to_string(), BamlValue::Int(2)),
                ]),
            ),
            TypeIR::class("Widget"),
        );

        assert_eq!(world.render_structural(&pv), "Widget {z: 2, a: 1}");
    }

    #[test]
    fn structural_depth_limit_shows_type() {
        let settings = RenderSettings {
            max_depth: 0,
            ..RenderSettings::default()
        };
        let class = build_class(
            "Widget",
            vec![("name".to_string(), TypeIR::string(), None)],
        );
        let world = make_world_with_types(RendererDb::new(), settings, vec![], vec![class]);
        let pv = make_prompt_value(
            &world,
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([("name".to_string(), BamlValue::String("hi".to_string()))]),
            ),
            TypeIR::class("Widget"),
        );

        assert_eq!(world.render_structural(&pv), "Widget { ... }");
    }

    #[test]
    fn structural_renders_union_ambiguous() {
        let class_foo = build_class("Foo", vec![]);
        let class_bar = build_class("Bar", vec![]);
        let world = make_world_with_types(
            RendererDb::new(),
            RenderSettings::default(),
            vec![],
            vec![class_foo, class_bar],
        );
        let pv = make_prompt_value(
            &world,
            BamlValue::Class("Foo".to_string(), IndexMap::new()),
            TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]),
        );

        assert_eq!(world.render_structural(&pv), "one of: Foo | Bar");
    }

    #[test]
    fn structural_renders_union_resolved() {
        let class_foo = build_class(
            "Foo",
            vec![("a".to_string(), TypeIR::int(), None)],
        );
        let class_bar = build_class(
            "Bar",
            vec![
                ("a".to_string(), TypeIR::int(), None),
                ("b".to_string(), TypeIR::int(), None),
            ],
        );
        let world = make_world_with_types(
            RendererDb::new(),
            RenderSettings::default(),
            vec![],
            vec![class_foo, class_bar],
        );
        let pv = make_prompt_value(
            &world,
            BamlValue::Class(
                "Bar".to_string(),
                IndexMap::from([
                    ("a".to_string(), BamlValue::Int(1)),
                    ("b".to_string(), BamlValue::Int(2)),
                ]),
            ),
            TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]),
        );

        assert_eq!(world.render_structural(&pv), "Bar {a: 1, b: 2}");
    }

    fn make_world(renderers: RendererDb, settings: RenderSettings) -> PromptWorld {
        make_world_with_types(renderers, settings, vec![], vec![])
    }

    fn make_world_with_types(
        renderers: RendererDb,
        settings: RenderSettings,
        enums: Vec<Enum>,
        classes: Vec<Class>,
    ) -> PromptWorld {
        let mut enum_map = IndexMap::new();
        for enum_type in enums {
            enum_map.insert(enum_type.name.real_name().to_string(), enum_type);
        }

        let mut class_map = IndexMap::new();
        for class in classes {
            class_map.insert(
                (class.name.real_name().to_string(), class.namespace),
                class,
            );
        }

        PromptWorld {
            types: TypeDb {
                enums: Arc::new(enum_map),
                classes: Arc::new(class_map),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
            renderers,
            jinja: crate::jsonish::jinja_helpers::get_env(),
            settings,
            union_resolver: default_union_resolver,
        }
    }

    fn build_enum(name: &str, variants: Vec<(&str, Option<&str>)>) -> Enum {
        Enum {
            name: Name::new(name.to_string()),
            description: None,
            values: variants
                .into_iter()
                .map(|(variant, alias)| {
                    (
                        Name::new_with_alias(
                            variant.to_string(),
                            alias.map(|alias| alias.to_string()),
                        ),
                        None,
                    )
                })
                .collect(),
            constraints: Vec::new(),
        }
    }

    fn build_class(name: &str, fields: Vec<(String, TypeIR, Option<String>)>) -> Class {
        Class {
            name: Name::new(name.to_string()),
            description: None,
            namespace: StreamingMode::NonStreaming,
            fields: fields
                .into_iter()
                .map(|(field_name, field_type, alias)| {
                    (
                        Name::new_with_alias(field_name, alias),
                        field_type,
                        None,
                        false,
                    )
                })
                .collect(),
            constraints: Vec::new(),
            streaming_behavior: baml_types::type_meta::base::StreamingBehavior::default(),
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
