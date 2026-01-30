//! Jinja helpers for prompt rendering.

use std::sync::Arc;

use minijinja::{
    value::{Enumerator, Object, ObjectRepr, Value},
    Environment, Error, Output, State, UndefinedBehavior,
};

use baml_types::{BamlMediaContent, BamlValue, TypeIR};

use super::PromptValue;

/// Callable object for value.render('style') syntax.
pub struct JinjaRenderMethod {
    pv: PromptValue,
}

impl std::fmt::Debug for JinjaRenderMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JinjaRenderMethod({})", self.pv.path)
    }
}

impl std::fmt::Display for JinjaRenderMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<render method>")
    }
}

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
        Value::from_object(JinjaRenderMethod { pv: self.pv.clone() })
    }

    fn raw_value(&self) -> Value {
        baml_value_to_jinja(self.pv.value())
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

impl Object for JinjaRenderMethod {
    fn call(self: &Arc<Self>, _state: &State<'_, '_>, args: &[Value]) -> Result<Value, Error> {
        let style = args.first().and_then(Value::as_str).ok_or_else(|| {
            Error::new(
                minijinja::ErrorKind::InvalidOperation,
                "render() requires a style string argument",
            )
        })?;

        let rendered = self
            .pv
            .world
            .render_prompt_value(&self.pv, Some(style))
            .map_err(|err| Error::new(minijinja::ErrorKind::InvalidOperation, err.message))?;

        Ok(Value::from(rendered))
    }
}

fn baml_value_to_jinja(value: &BamlValue) -> Value {
    match value {
        BamlValue::String(s) => Value::from(s.as_str()),
        BamlValue::Int(i) => Value::from(*i),
        BamlValue::Float(f) => Value::from(*f),
        BamlValue::Bool(b) => Value::from(*b),
        BamlValue::Null => Value::from(()),
        BamlValue::List(items) => {
            Value::from_iter(items.iter().map(baml_value_to_jinja))
        }
        BamlValue::Map(map) => Value::from_iter(
            map.iter()
                .map(|(k, v)| (k.clone(), baml_value_to_jinja(v))),
        ),
        BamlValue::Class(_, fields) => Value::from_iter(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), baml_value_to_jinja(v))),
        ),
        BamlValue::Enum(_, variant) => Value::from(variant.as_str()),
        BamlValue::Media(media) => {
            let mut entries = Vec::new();
            entries.push(("type".to_string(), Value::from(media.media_type.to_string())));
            if let Some(mime_type) = &media.mime_type {
                entries.push(("mime_type".to_string(), Value::from(mime_type.as_str())));
            }
            match &media.content {
                BamlMediaContent::File(file) => {
                    entries.push((
                        "file".to_string(),
                        Value::from(file.relpath.to_string_lossy().to_string()),
                    ));
                }
                BamlMediaContent::Url(url) => {
                    entries.push(("url".to_string(), Value::from(url.url.as_str())));
                }
                BamlMediaContent::Base64(base64) => {
                    entries.push(("base64".to_string(), Value::from(base64.base64.as_str())));
                }
            }
            Value::from_iter(entries)
        }
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

    fn is_true(self: &Arc<Self>) -> bool {
        match self.pv.value() {
            BamlValue::Null => false,
            BamlValue::Bool(b) => *b,
            BamlValue::String(s) => !s.is_empty(),
            BamlValue::Int(i) => *i != 0,
            BamlValue::Float(f) => *f != 0.0 && !f.is_nan(),
            BamlValue::List(items) => !items.is_empty(),
            BamlValue::Map(map) => !map.is_empty(),
            BamlValue::Class(_, fields) => !fields.is_empty(),
            BamlValue::Enum(_, variant) => !variant.is_empty(),
            BamlValue::Media(_) => true,
        }
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

/// Configure the Jinja environment for prompt rendering.
pub fn configure_prompt_env(env: &mut Environment<'static>) {
    env.set_undefined_behavior(UndefinedBehavior::Strict);
    register_prompt_filters(env);
    env.set_formatter(prompt_formatter);
}

fn prompt_formatter(
    out: &mut Output<'_>,
    state: &State<'_, '_>,
    value: &Value,
) -> Result<(), Error> {
    if let Some(obj) = value.as_object() {
        if let Some(jpv) = obj.downcast_ref::<JinjaPromptValue>() {
            let rendered = jpv
                .pv
                .world
                .render_prompt_value(&jpv.pv, None)
                .map_err(|err| Error::new(minijinja::ErrorKind::InvalidOperation, err.message))?;
            out.write_str(&rendered)?;
            return Ok(());
        }
    }

    if value.is_none() {
        out.write_str("null")?;
        return Ok(());
    }

    minijinja::escape_formatter(out, state, value)
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
        configure_prompt_env, filter_format_count, filter_slice_chars, filter_truncate,
        JinjaPromptValue, PromptValue,
    };
    use crate::prompt::renderer::{RenderSession, RenderSettings, RendererDb};
    use crate::prompt::value::default_union_resolver;
    use crate::prompt::world::{PromptWorld, TypeDb};
    use crate::prompt::PromptPath;
    use baml_types::{
        ir_type::UnionConstructor, BamlMedia, BamlMediaType, BamlValue, TypeIR,
    };
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
    fn formatter_renders_prompt_value_via_pipeline() {
        let mut world = make_world_empty();
        world.jinja.add_template("entry", "{{ value }}").unwrap();

        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::String("hi".to_string()),
            TypeIR::string(),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );
        let ctx = Value::from_iter([("value".to_string(), pv.as_jinja_value())]);

        let rendered = pv.world.jinja.get_template("entry").unwrap().render(ctx).unwrap();
        assert_eq!(rendered, "hi");
    }

    #[test]
    fn render_method_uses_style_override() {
        let mut world = make_world_empty();
        world
            .jinja
            .add_template("entry", "{{ value.render('json') }}")
            .unwrap();

        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::String("hi".to_string()),
            TypeIR::string(),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );
        let ctx = Value::from_iter([("value".to_string(), pv.as_jinja_value())]);

        let rendered = pv.world.jinja.get_template("entry").unwrap().render(ctx).unwrap();
        assert_eq!(rendered, "\"hi\"");
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
    fn raw_null_is_none() {
        let pv = make_prompt_value(BamlValue::Null, TypeIR::null());
        let obj = Arc::new(JinjaPromptValue { pv });

        let raw = obj.get_value(&Value::from("raw")).unwrap();
        assert!(raw.is_none());
    }

    #[test]
    fn raw_class_exposes_fields() {
        let pv = make_prompt_value(
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([("name".to_string(), BamlValue::String("ok".to_string()))]),
            ),
            TypeIR::class("Widget"),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        let raw = obj.get_value(&Value::from("raw")).unwrap();
        let field = raw.get_item(&Value::from("name")).unwrap();
        assert_eq!(field.as_str(), Some("ok"));
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
    fn get_value_reads_rendered_field_names() {
        let world = make_world_with_class_alias(
            "Widget",
            vec![("real".to_string(), TypeIR::string(), Some("alias".to_string()))],
        );
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([("real".to_string(), BamlValue::String("ok".to_string()))]),
            ),
            TypeIR::class("Widget"),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        let value = obj.get_value(&Value::from("alias")).unwrap();
        let child = value.downcast_object_ref::<JinjaPromptValue>().unwrap();
        assert_eq!(child.pv.value(), &BamlValue::String("ok".to_string()));
        assert_eq!(child.pv.path.to_string(), "alias");
    }

    #[test]
    fn get_value_returns_none_for_missing_class_field() {
        let pv = make_prompt_value(
            BamlValue::Class("Widget".to_string(), IndexMap::new()),
            TypeIR::class("Widget"),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(obj.get_value(&Value::from("missing")).is_none());
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
    fn get_value_list_index_respects_budget() {
        let pv = make_prompt_value_with_settings(
            BamlValue::List(vec![
                BamlValue::String("a".to_string()),
                BamlValue::String("b".to_string()),
            ]),
            TypeIR::list(TypeIR::string()),
            RenderSettings {
                max_list_items: 1,
                ..RenderSettings::default()
            },
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(obj.get_value(&Value::from(1)).is_none());
    }

    #[test]
    fn get_value_list_index_on_non_list_returns_none() {
        let pv = make_prompt_value(BamlValue::String("x".to_string()), TypeIR::string());
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(obj.get_value(&Value::from(0)).is_none());
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

    #[test]
    fn render_method_supports_json_style() {
        let mut world = make_world_with_class(
            "Widget",
            vec![("name".to_string(), TypeIR::string())],
        );
        world
            .jinja
            .add_template_owned(
                "render_json".to_string(),
                "{{ value.render(\"json\") }}".to_string(),
            )
            .expect("template add");
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([("name".to_string(), BamlValue::String("Ada".to_string()))]),
            ),
            TypeIR::class("Widget"),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );

        let ctx = Value::from_iter([("value".to_string(), pv.as_jinja_value())]);
        let rendered = pv
            .world
            .jinja
            .get_template("render_json")
            .expect("template get")
            .render(ctx)
            .expect("rendered");

        assert!(rendered.trim_start().starts_with('{'));
        assert!(rendered.contains("\"name\""));
    }

    #[test]
    fn render_method_uses_nested_schema_target() {
        let inner = build_class("Inner", vec![("label".to_string(), TypeIR::string())]);
        let outer = build_class(
            "Outer",
            vec![
                ("inner".to_string(), TypeIR::class("Inner")),
                ("extra".to_string(), TypeIR::string()),
            ],
        );
        let mut world = make_world_with_classes(vec![inner, outer]);
        world
            .jinja
            .add_template_owned(
                "render_nested".to_string(),
                "{{ value.inner.render(\"json\") }}".to_string(),
            )
            .expect("template add");

        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class(
                "Outer".to_string(),
                IndexMap::from([
                    (
                        "inner".to_string(),
                        BamlValue::Class(
                            "Inner".to_string(),
                            IndexMap::from([(
                                "label".to_string(),
                                BamlValue::String("Ada".to_string()),
                            )]),
                        ),
                    ),
                    ("extra".to_string(), BamlValue::String("skip".to_string())),
                ]),
            ),
            TypeIR::class("Outer"),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );

        let ctx = Value::from_iter([("value".to_string(), pv.as_jinja_value())]);
        let rendered = pv
            .world
            .jinja
            .get_template("render_nested")
            .expect("template get")
            .render(ctx)
            .expect("rendered");

        assert!(rendered.contains("\"label\""));
        assert!(!rendered.contains("\"extra\""));
    }

    #[test]
    fn render_errors_on_missing_field_in_template() {
        let mut world = make_world_with_class(
            "Widget",
            vec![("name".to_string(), TypeIR::string())],
        );
        world
            .jinja
            .add_template_owned("missing".to_string(), "{{ value.missing }}".to_string())
            .expect("template add");
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class(
                "Widget".to_string(),
                IndexMap::from([("name".to_string(), BamlValue::String("Ada".to_string()))]),
            ),
            TypeIR::class("Widget"),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );

        let ctx = Value::from_iter([("value".to_string(), pv.as_jinja_value())]);
        let err = pv
            .world
            .jinja
            .get_template("missing")
            .expect("template get")
            .render(ctx)
            .expect_err("expected render error");

        let message = err.to_string();
        assert!(
            message.contains("missing") || message.contains("undefined"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn union_resolution_is_deterministic() {
        let world = make_world_with_classes(vec![
            build_class("Foo", vec![("a".to_string(), TypeIR::int())]),
            build_class("Bar", vec![("b".to_string(), TypeIR::int())]),
        ]);
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class(
                "Foo".to_string(),
                IndexMap::from([("a".to_string(), BamlValue::Int(1))]),
            ),
            TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );

        let rendered = pv
            .world
            .render_prompt_value(&pv, None)
            .expect("rendered");
        assert!(rendered.contains("Foo {a: 1}"));
        assert!(!rendered.contains("one of:"));
    }

    #[test]
    fn ambiguous_union_field_access_errors() {
        let world = make_world_with_classes(vec![build_class("Foo", vec![]), build_class("Bar", vec![])]);
        let mut world = world;
        world
            .jinja
            .add_template_owned("ambiguous".to_string(), "{{ value.anything }}".to_string())
            .expect("template add");
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class("Foo".to_string(), IndexMap::new()),
            TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );

        let ctx = Value::from_iter([("value".to_string(), pv.as_jinja_value())]);
        let err = pv
            .world
            .jinja
            .get_template("ambiguous")
            .expect("template get")
            .render(ctx)
            .expect_err("expected render error");

        let message = err.to_string();
        assert!(
            message.contains("undefined") || message.contains("anything"),
            "unexpected error message: {message}"
        );
    }

    #[test]
    fn get_value_returns_none_for_ambiguous_union_missing_field() {
        let world = make_world_with_classes(vec![
            build_class("Foo", vec![]),
            build_class("Bar", vec![]),
        ]);
        let pv = make_prompt_value_with_world_and_settings(
            BamlValue::Class("Foo".to_string(), IndexMap::new()),
            TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]),
            world,
            RenderSettings::default(),
            PromptPath::new(),
        );
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(obj.get_value(&Value::from("missing")).is_none());
    }

    #[test]
    fn is_true_handles_primitives() {
        let cases = vec![
            (BamlValue::Null, TypeIR::null(), false),
            (BamlValue::Bool(false), TypeIR::bool(), false),
            (BamlValue::Bool(true), TypeIR::bool(), true),
            (BamlValue::String(String::new()), TypeIR::string(), false),
            (BamlValue::String("ok".to_string()), TypeIR::string(), true),
            (BamlValue::Int(0), TypeIR::int(), false),
            (BamlValue::Int(3), TypeIR::int(), true),
            (BamlValue::Float(0.0), TypeIR::float(), false),
            (BamlValue::Float(1.25), TypeIR::float(), true),
            (BamlValue::Float(f64::NAN), TypeIR::float(), false),
        ];

        for (value, ty, expected) in cases {
            let pv = make_prompt_value(value, ty);
            let obj = Arc::new(JinjaPromptValue { pv });
            assert_eq!(obj.is_true(), expected);
        }
    }

    #[test]
    fn is_true_handles_containers() {
        let cases = vec![
            (
                BamlValue::List(Vec::new()),
                TypeIR::list(TypeIR::string()),
                false,
            ),
            (
                BamlValue::Map(IndexMap::new()),
                TypeIR::map(TypeIR::string(), TypeIR::bool()),
                false,
            ),
            (
                BamlValue::Class("Widget".to_string(), IndexMap::new()),
                TypeIR::class("Widget"),
                false,
            ),
            (
                BamlValue::List(vec![BamlValue::String("x".to_string())]),
                TypeIR::list(TypeIR::string()),
                true,
            ),
            (
                BamlValue::Map(IndexMap::from([("flag".to_string(), BamlValue::Bool(true))])),
                TypeIR::map(TypeIR::string(), TypeIR::bool()),
                true,
            ),
            (
                BamlValue::Class(
                    "Widget".to_string(),
                    IndexMap::from([("name".to_string(), BamlValue::String("ok".to_string()))]),
                ),
                TypeIR::class("Widget"),
                true,
            ),
        ];

        for (value, ty, expected) in cases {
            let pv = make_prompt_value(value, ty);
            let obj = Arc::new(JinjaPromptValue { pv });
            assert_eq!(obj.is_true(), expected);
        }
    }

    #[test]
    fn is_true_prefers_value_over_type() {
        let pv = make_prompt_value(BamlValue::Null, TypeIR::list(TypeIR::string()));
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(!obj.is_true());
    }

    #[test]
    fn is_true_media_is_truthy() {
        let media = BamlMedia::url(BamlMediaType::Image, "https://example.com".to_string(), None);
        let pv = make_prompt_value(BamlValue::Media(media), TypeIR::image());
        let obj = Arc::new(JinjaPromptValue { pv });

        assert!(obj.is_true());
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
            jinja: {
                let mut jinja = crate::jsonish::jinja_helpers::get_env();
                configure_prompt_env(&mut jinja);
                jinja
            },
            settings: RenderSettings::default(),
            union_resolver: default_union_resolver,
        }
    }

    fn make_world_with_class(name: &str, fields: Vec<(String, TypeIR)>) -> PromptWorld {
        make_world_with_class_alias(
            name,
            fields
                .into_iter()
                .map(|(field, ty)| (field, ty, None))
                .collect(),
        )
    }

    fn make_world_with_class_alias(
        name: &str,
        fields: Vec<(String, TypeIR, Option<String>)>,
    ) -> PromptWorld {
        make_world_with_classes(vec![build_class_with_alias(name, fields)])
    }

    fn make_world_with_classes(classes: Vec<Class>) -> PromptWorld {
        let mut class_map = IndexMap::new();
        for class in classes {
            let key = (
                class.name.real_name().to_string(),
                baml_types::StreamingMode::NonStreaming,
            );
            class_map.insert(key, class);
        }

        PromptWorld {
            types: TypeDb {
                enums: Arc::new(IndexMap::<String, Enum>::new()),
                classes: Arc::new(class_map),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
            renderers: RendererDb::new(),
            jinja: {
                let mut jinja = crate::jsonish::jinja_helpers::get_env();
                configure_prompt_env(&mut jinja);
                jinja
            },
            settings: RenderSettings::default(),
            union_resolver: default_union_resolver,
        }
    }

    fn build_class(name: &str, fields: Vec<(String, TypeIR)>) -> Class {
        build_class_with_alias(
            name,
            fields.into_iter().map(|(field, ty)| (field, ty, None)).collect(),
        )
    }

    fn build_class_with_alias(
        name: &str,
        fields: Vec<(String, TypeIR, Option<String>)>,
    ) -> Class {
        Class {
            name: internal_baml_jinja::types::Name::new(name.to_string()),
            description: None,
            namespace: baml_types::StreamingMode::NonStreaming,
            fields: fields
                .into_iter()
                .map(|(field_name, field_type, alias)| {
                    (
                        internal_baml_jinja::types::Name::new_with_alias(field_name, alias),
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
}
