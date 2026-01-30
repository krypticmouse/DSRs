//! Jinja helpers for prompt rendering.

use std::sync::Arc;

use minijinja::{
    value::{Enumerator, Object, ObjectRepr, Value},
    Environment,
};

use baml_types::{BamlValue, TypeIR};

use super::PromptValue;

/// Jinja object wrapper for typed prompt values.
pub struct JinjaPromptValue {
    pv: PromptValue,
}

impl PromptValue {
    /// Convert to a Jinja Value for template use.
    pub fn as_jinja_value(&self) -> Value {
        Value::from_object(JinjaPromptValue { pv: self.clone() })
    }
}

impl std::fmt::Debug for JinjaPromptValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JinjaPromptValue({:?} at {})", self.pv.ty(), self.pv.path)
    }
}

impl std::fmt::Display for JinjaPromptValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<PromptValue>")
    }
}

impl Object for JinjaPromptValue {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        match self.pv.resolved_ty() {
            TypeIR::Class { .. } | TypeIR::Map(_, _, _) => ObjectRepr::Map,
            TypeIR::List(_, _) => ObjectRepr::Seq,
            _ => match self.pv.value() {
                BamlValue::Class(_, _) | BamlValue::Map(_) => ObjectRepr::Map,
                BamlValue::List(_) => ObjectRepr::Seq,
                _ => ObjectRepr::Plain,
            },
        }
    }

    fn get_value(self: &Arc<Self>, _key: &Value) -> Option<Value> {
        None
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        Enumerator::Empty
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        Some(0)
    }
}

/// Register prompt-specific filters.
pub fn register_prompt_filters(env: &mut Environment<'static>) {
    env.add_filter("truncate", filter_truncate);
    env.add_filter("slice_chars", filter_slice_chars);
    env.add_filter("format_count", filter_format_count);
}

fn filter_truncate(s: &str, n: usize) -> String {
    let length = s.chars().count();
    if length <= n {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(n.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

fn filter_slice_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn filter_format_count(n: i64) -> String {
    let sign = if n < 0 { "-" } else { "" };
    let s = n.abs().to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    format!("{sign}{}", result.chars().rev().collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::{
        filter_format_count, filter_slice_chars, filter_truncate, JinjaPromptValue, PromptValue,
    };
    use crate::prompt::renderer::{RenderSession, RenderSettings, RendererDb};
    use crate::prompt::value::default_union_resolver;
    use crate::prompt::world::{PromptWorld, TypeDb};
    use crate::prompt::PromptPath;
    use baml_types::{BamlValue, TypeIR};
    use indexmap::{IndexMap, IndexSet};
    use internal_baml_jinja::types::{Class, Enum};
    use minijinja::value::{Object, ObjectRepr};
    use std::sync::Arc;

    #[test]
    fn truncate_keeps_exact_length() {
        assert_eq!(filter_truncate("hello", 5), "hello");
    }

    #[test]
    fn truncate_adds_suffix_when_needed() {
        assert_eq!(filter_truncate("hello world", 8), "hello...");
    }

    #[test]
    fn slice_chars_handles_empty() {
        assert_eq!(filter_slice_chars("", 3), "");
    }

    #[test]
    fn format_count_handles_negative() {
        assert_eq!(filter_format_count(-12345), "-12,345");
    }

    #[test]
    fn repr_prefers_class_type() {
        let pv = make_prompt_value(BamlValue::String("x".to_string()), TypeIR::class("Widget"));
        let obj = Arc::new(JinjaPromptValue { pv });

        assert_eq!(obj.repr(), ObjectRepr::Map);
    }

    #[test]
    fn repr_prefers_list_type() {
        let pv = make_prompt_value(BamlValue::String("x".to_string()), TypeIR::list(TypeIR::string()));
        let obj = Arc::new(JinjaPromptValue { pv });

        assert_eq!(obj.repr(), ObjectRepr::Seq);
    }

    #[test]
    fn repr_falls_back_to_value_shape() {
        let pv = make_prompt_value(
            BamlValue::Map(IndexMap::from([("key".to_string(), BamlValue::Bool(true))])),
            TypeIR::string(),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        assert_eq!(obj.repr(), ObjectRepr::Map);
    }

    #[test]
    fn repr_plain_for_primitives() {
        let pv = make_prompt_value(BamlValue::String("x".to_string()), TypeIR::string());
        let obj = Arc::new(JinjaPromptValue { pv });

        assert_eq!(obj.repr(), ObjectRepr::Plain);
    }

    fn make_prompt_value(value: BamlValue, ty: TypeIR) -> PromptValue {
        let world = PromptWorld {
            types: TypeDb {
                enums: Arc::new(IndexMap::<String, Enum>::new()),
                classes: Arc::new(IndexMap::<(String, baml_types::StreamingMode), Class>::new()),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
            renderers: RendererDb::new(),
            jinja: crate::jsonish::jinja_helpers::get_env(),
            settings: RenderSettings::default(),
            union_resolver: default_union_resolver,
        };

        PromptValue::new(
            value,
            ty,
            Arc::new(world),
            Arc::new(RenderSession::new(RenderSettings::default())),
            PromptPath::new(),
        )
    }
}
