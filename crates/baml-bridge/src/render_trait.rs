//! Trait for types with default Jinja rendering templates.

use baml_types::BamlValue;
use serde::Serialize;

use crate::ToBamlValue;

/// Types that have a default Jinja rendering template.
///
/// This trait allows types to specify how they should be rendered when used
/// in prompts. The template receives `value` (the BamlValue serialized from
/// the type) and `ctx` (render context with configuration).
///
/// # Example
///
/// ```ignore
/// impl DefaultJinjaRender for REPLHistory {
///     const DEFAULT_TEMPLATE: &'static str = r#"
/// {% if value.is_empty -%}
/// You have not interacted with the REPL environment yet.
/// {%- else -%}
/// {% for entry in value.entries -%}
/// === Step {{ loop.index }} ===
/// Code:
/// ```python
/// {{ entry.code }}
/// ```
/// Output: {{ entry.output }}
/// {% endfor -%}
/// {%- endif %}
/// "#;
/// }
/// ```
pub trait DefaultJinjaRender: ToBamlValue {
    /// The default Jinja template for rendering this type.
    /// Template receives `value` (the BamlValue) and `ctx` (render context).
    const DEFAULT_TEMPLATE: &'static str;

    /// Render this value using its default template.
    ///
    /// Returns the rendered string, or an error if rendering fails.
    fn render_default(&self, ctx: &minijinja::Value) -> Result<String, minijinja::Error> {
        let env = crate::jsonish::jinja_helpers::get_env();
        let baml_value = self.to_baml_value();
        let template_ctx = minijinja::Value::from_serialize(RenderContext {
            value: &baml_value,
            ctx,
        });
        env.render_str(Self::DEFAULT_TEMPLATE, template_ctx)
    }
}

#[derive(Serialize)]
struct RenderContext<'a> {
    value: &'a BamlValue,
    ctx: &'a minijinja::Value,
}
