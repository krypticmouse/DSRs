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

impl JinjaPromptValue {
    fn render_method(&self) -> Value {
        Value::from("<render>")
    }

    fn raw_value(&self) -> Value {
        Value::from_serialize(self.pv.value())
    }

    fn full_length(&self) -> usize {
        match self.pv.value() {
            BamlValue::Class(_, fields) => fields.len(),
            BamlValue::Map(map) => map.len(),
            BamlValue::List(items) => items.len(),
            _ => 0,
        }
    }

    fn is_class_like(&self) -> bool {
        match self.pv.resolved_ty() {
            TypeIR::Class { .. } => true,
            _ => matches!(self.pv.value(), BamlValue::Class(_, _)),
        }
    }

    fn is_map_like(&self) -> bool {
        match self.pv.resolved_ty() {
            TypeIR::Map(_, _, _) => true,
            _ => matches!(self.pv.value(), BamlValue::Map(_)),
        }
    }

    fn is_list_like(&self) -> bool {
        match self.pv.resolved_ty() {
            TypeIR::List(_, _) => true,
            _ => matches!(self.pv.value(), BamlValue::List(_)),
        }
    }

    fn list_len(&self) -> usize {
        match self.pv.value() {
            BamlValue::List(items) => items.len(),
            _ => 0,
        }
    }

    fn map_keys_sorted(&self) -> Vec<String> {
        let mut keys: Vec<String> = match self.pv.value() {
            BamlValue::Map(map) => map.keys().cloned().collect(),
            BamlValue::Class(_, fields) => fields.keys().cloned().collect(),
            _ => Vec::new(),
        };
        keys.sort();
        keys
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

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        if let Some(k) = key.as_str() {
            match k {
                "render" => return Some(self.render_method()),
                "raw" => return Some(self.raw_value()),
                "__type__" => {
                    return Some(Value::from(self.pv.ty().diagnostic_repr().to_string()))
                }
                "__path__" => return Some(Value::from(self.pv.path.to_string())),
                "__full_len__" => return Some(Value::from(self.full_length())),
                _ => {}
            }

            if self.is_class_like() {
                return self.pv.child_field(k).map(|child| child.as_jinja_value());
            }

            if self.is_map_like() {
                return self
                    .pv
                    .child_map_value(k)
                    .map(|child| child.as_jinja_value());
            }
        }

        if let Some(idx) = key.as_usize() {
            if self.is_list_like() {
                return self
                    .pv
                    .child_index(idx)
                    .map(|child| child.as_jinja_value());
            }
        }

        None
    }

    fn enumerate(self: &Arc<Self>) -> Enumerator {
        let max_list_items = self.pv.session.settings.max_list_items;
        let max_map_entries = self.pv.session.settings.max_map_entries;

        match self.pv.resolved_ty() {
            TypeIR::List(_, _) => {
                let capped_len = self.list_len().min(max_list_items);
                Enumerator::Seq(capped_len)
            }
            TypeIR::Class { name, mode, .. } => {
                if let Some(class) = self.pv.world.types.find_class(&name, mode) {
                    let keys: Vec<Value> = class
                        .fields
                        .iter()
                        .take(max_map_entries)
                        .map(|(field_name, _, _, _)| {
                            Value::from(field_name.real_name().to_string())
                        })
                        .collect();
                    Enumerator::Values(keys)
                } else {
                    Enumerator::Empty
                }
            }
            TypeIR::Map(_, _, _) => {
                let keys: Vec<Value> = self
                    .map_keys_sorted()
                    .into_iter()
                    .take(max_map_entries)
                    .map(Value::from)
                    .collect();
                Enumerator::Values(keys)
            }
            _ => {
                if self.is_list_like() {
                    let capped_len = self.list_len().min(max_list_items);
                    Enumerator::Seq(capped_len)
                } else if self.is_class_like() || self.is_map_like() {
                    let keys: Vec<Value> = self
                        .map_keys_sorted()
                        .into_iter()
                        .take(max_map_entries)
                        .map(Value::from)
                        .collect();
                    Enumerator::Values(keys)
                } else {
                    Enumerator::Empty
                }
            }
        }
    }

    fn enumerator_len(self: &Arc<Self>) -> Option<usize> {
        match self.enumerate() {
            Enumerator::Seq(len) => Some(len),
            Enumerator::Values(values) => Some(values.len()),
            _ => None,
        }
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
    use minijinja::value::{Enumerator, Object, ObjectRepr, Value};
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

    #[test]
    fn get_value_exposes_reserved_keys() {
        let path = PromptPath::new().push_field("root");
        let pv = make_prompt_value_with_path(
            BamlValue::List(vec![BamlValue::String("a".to_string())]),
            TypeIR::list(TypeIR::string()),
            path.clone(),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        let expected_type = TypeIR::list(TypeIR::string())
            .diagnostic_repr()
            .to_string();
        let ty_value = obj.get_value(&Value::from("__type__")).unwrap();
        assert_eq!(ty_value.as_str(), Some(expected_type.as_str()));

        let path_value = obj.get_value(&Value::from("__path__")).unwrap();
        assert_eq!(path_value.as_str(), Some(path.to_string().as_str()));

        let len_value = obj.get_value(&Value::from("__full_len__")).unwrap();
        assert_eq!(len_value.as_usize(), Some(1));

        assert!(obj.get_value(&Value::from("render")).is_some());
    }

    #[test]
    fn get_value_raw_returns_untyped_value() {
        let pv = make_prompt_value(BamlValue::String("hello".to_string()), TypeIR::string());
        let obj = Arc::new(JinjaPromptValue { pv });

        let raw = obj.get_value(&Value::from("raw")).unwrap();
        assert_eq!(raw.as_str(), Some("hello"));
    }

    #[test]
    fn get_value_reads_class_fields() {
        let pv = make_prompt_value(
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([("name".to_string(), BamlValue::String("ok".to_string()))]),
            ),
            TypeIR::class("Widget"),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        let value = obj.get_value(&Value::from("name")).unwrap();
        let child = value.downcast_object_ref::<JinjaPromptValue>().unwrap();
        assert_eq!(child.pv.value(), &BamlValue::String("ok".to_string()));
        assert_eq!(child.pv.path.to_string(), "name");
    }

    #[test]
    fn get_value_reads_map_keys() {
        let pv = make_prompt_value(
            BamlValue::Map(IndexMap::from([("flag".to_string(), BamlValue::Bool(true))])),
            TypeIR::map(TypeIR::string(), TypeIR::bool()),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        let value = obj.get_value(&Value::from("flag")).unwrap();
        let child = value.downcast_object_ref::<JinjaPromptValue>().unwrap();
        assert_eq!(child.pv.value(), &BamlValue::Bool(true));
    }

    #[test]
    fn get_value_reads_list_indices() {
        let pv = make_prompt_value(
            BamlValue::List(vec![
                BamlValue::String("a".to_string()),
                BamlValue::String("b".to_string()),
            ]),
            TypeIR::list(TypeIR::string()),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        let value = obj.get_value(&Value::from(1)).unwrap();
        let child = value.downcast_object_ref::<JinjaPromptValue>().unwrap();
        assert_eq!(child.pv.value(), &BamlValue::String("b".to_string()));
    }

    #[test]
    fn get_value_returns_none_for_missing() {
        let pv = make_prompt_value(BamlValue::String("x".to_string()), TypeIR::string());
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(obj.get_value(&Value::from("missing")).is_none());
    }

    #[test]
    fn enumerate_caps_list_length() {
        let pv = make_prompt_value_with_settings(
            BamlValue::List(vec![
                BamlValue::String("a".to_string()),
                BamlValue::String("b".to_string()),
                BamlValue::String("c".to_string()),
            ]),
            TypeIR::list(TypeIR::string()),
            RenderSettings {
                max_list_items: 2,
                ..RenderSettings::default()
            },
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        assert_eq!(obj.enumerator_len(), Some(2));
        match obj.enumerate() {
            Enumerator::Seq(len) => assert_eq!(len, 2),
            _ => panic!("expected seq enumerator"),
        }
    }

    #[test]
    fn enumerate_sorts_and_caps_map_keys() {
        let pv = make_prompt_value_with_settings(
            BamlValue::Map(IndexMap::from([
                ("b".to_string(), BamlValue::Bool(true)),
                ("a".to_string(), BamlValue::Bool(false)),
                ("c".to_string(), BamlValue::Bool(true)),
            ])),
            TypeIR::map(TypeIR::string(), TypeIR::bool()),
            RenderSettings {
                max_map_entries: 2,
                ..RenderSettings::default()
            },
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        match obj.enumerate() {
            Enumerator::Values(values) => {
                let keys: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
                assert_eq!(keys, vec!["a", "b"]);
            }
            _ => panic!("expected values enumerator"),
        }
        assert_eq!(obj.enumerator_len(), Some(2));
    }

    #[test]
    fn enumerate_uses_schema_field_order_for_classes() {
        let mut class_fields = Vec::new();
        class_fields.push(("z".to_string(), TypeIR::string()));
        class_fields.push(("a".to_string(), TypeIR::string()));
        class_fields.push(("m".to_string(), TypeIR::string()));

        let world = make_world_with_class("Widget", class_fields);
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([
                    ("a".to_string(), BamlValue::String("1".to_string())),
                    ("m".to_string(), BamlValue::String("2".to_string())),
                    ("z".to_string(), BamlValue::String("3".to_string())),
                ]),
            ),
            TypeIR::class("Widget"),
            world,
            RenderSettings {
                max_map_entries: 2,
                ..RenderSettings::default()
            },
            PromptPath::new(),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        match obj.enumerate() {
            Enumerator::Values(values) => {
                let keys: Vec<&str> = values.iter().filter_map(|v| v.as_str()).collect();
                assert_eq!(keys, vec!["z", "a"]);
            }
            _ => panic!("expected values enumerator"),
        }
    }

    fn make_prompt_value(value: BamlValue, ty: TypeIR) -> PromptValue {
        make_prompt_value_with_path(value, ty, PromptPath::new())
    }

    fn make_prompt_value_with_settings(
        value: BamlValue,
        ty: TypeIR,
        settings: RenderSettings,
    ) -> PromptValue {
        make_prompt_value_with_world_and_settings(
            value,
            ty,
            make_world_empty(),
            settings,
            PromptPath::new(),
        )
    }

    fn make_prompt_value_with_path(
        value: BamlValue,
        ty: TypeIR,
        path: PromptPath,
    ) -> PromptValue {
        make_prompt_value_with_world_and_settings(
            value,
            ty,
            make_world_empty(),
            RenderSettings::default(),
            path,
        )
    }

    fn make_prompt_value_with_world_and_settings(
        value: BamlValue,
        ty: TypeIR,
        world: PromptWorld,
        settings: RenderSettings,
        path: PromptPath,
    ) -> PromptValue {
        PromptValue::new(
            value,
            ty,
            Arc::new(world),
            Arc::new(RenderSession::new(settings)),
            path,
        )
    }

    fn make_world_empty() -> PromptWorld {
        PromptWorld {
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
        }
    }

    fn make_world_with_class(name: &str, fields: Vec<(String, TypeIR)>) -> PromptWorld {
        let class = Class {
            name: internal_baml_jinja::types::Name::new(name.to_string()),
            description: None,
            namespace: baml_types::StreamingMode::NonStreaming,
            fields: fields
                .into_iter()
                .map(|(field_name, field_type)| {
                    (
                        internal_baml_jinja::types::Name::new(field_name),
                        field_type,
                        None,
                        false,
                    )
                })
                .collect(),
            constraints: Vec::new(),
            streaming_behavior: baml_types::type_meta::base::StreamingBehavior::default(),
        };

        PromptWorld {
            types: TypeDb {
                enums: Arc::new(IndexMap::<String, Enum>::new()),
                classes: Arc::new(IndexMap::from([(
                    (name.to_string(), baml_types::StreamingMode::NonStreaming),
                    class,
                )])),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
            renderers: RendererDb::new(),
            jinja: crate::jsonish::jinja_helpers::get_env(),
            settings: RenderSettings::default(),
            union_resolver: default_union_resolver,
        }
    }
}
