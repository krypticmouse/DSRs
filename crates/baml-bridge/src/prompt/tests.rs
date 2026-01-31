use super::{JinjaPromptValue, PromptPath, PromptValue, PromptWorld};
use crate::prompt::renderer::{
    RenderSession, RenderSettings, RendererDb, RendererKey, RendererOverride, RendererSpec,
};
use crate::prompt::value::default_union_resolver;
use crate::prompt::world::TypeDb;
use crate::Registry;
use baml_types::{BamlValue, StreamingMode, TypeIR};
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum, Name};
use minijinja::value::Value;
use serde_json::json;
use std::sync::Arc;

#[test]
fn prompt_value_child_field_traversal_smoke() {
    let world = make_world_with_class(
        "Widget",
        vec![("name".to_string(), TypeIR::string()), ("count".to_string(), TypeIR::int())],
    );
    let value = BamlValue::Class(
        "Widget".to_string(),
        IndexMap::from([
            ("name".to_string(), BamlValue::String("test".to_string())),
            ("count".to_string(), BamlValue::Int(42)),
        ]),
    );
    let path = PromptPath::new().push_field("root");
    let prompt = PromptValue::new(
        value,
        TypeIR::class("Widget"),
        Arc::new(world),
        make_session(),
        path,
    );

    let child = prompt.child_field("name").expect("child field");
    assert_eq!(child.value(), &BamlValue::String("test".to_string()));
    assert_eq!(child.ty(), &TypeIR::string());
    assert_eq!(child.path.to_string(), "root.name");
}

#[test]
fn prompt_value_child_index_traversal_smoke() {
    let value = BamlValue::List(vec![
        BamlValue::String("a".to_string()),
        BamlValue::String("b".to_string()),
        BamlValue::String("c".to_string()),
    ]);
    let path = PromptPath::new().push_field("root");
    let prompt = PromptValue::new(
        value,
        TypeIR::list(TypeIR::string()),
        Arc::new(make_world_empty()),
        make_session(),
        path,
    );

    let child = prompt.child_index(1).expect("child index");
    assert_eq!(child.value(), &BamlValue::String("b".to_string()));
    assert_eq!(child.path.to_string(), "root[1]");
}

#[test]
fn jinja_prompt_value_get_value_smoke() {
    let world = make_world_with_class(
        "Widget",
        vec![("name".to_string(), TypeIR::string()), ("count".to_string(), TypeIR::int())],
    );
    let value = BamlValue::Class(
        "Widget".to_string(),
        IndexMap::from([
            ("name".to_string(), BamlValue::String("test".to_string())),
            ("count".to_string(), BamlValue::Int(42)),
        ]),
    );
    let path = PromptPath::new().push_field("root");
    let pv = PromptValue::new(
        value,
        TypeIR::class("Widget"),
        Arc::new(world),
        make_session(),
        path,
    );
    let obj = pv.as_jinja_value();

    let root_type = obj.get_item(&Value::from("__type__")).unwrap();
    let expected_root_type = TypeIR::class("Widget").diagnostic_repr().to_string();
    assert_eq!(root_type.as_str(), Some(expected_root_type.as_str()));

    let root_path = obj.get_item(&Value::from("__path__")).unwrap();
    assert_eq!(root_path.as_str(), Some("root"));

    let child_value = obj.get_item(&Value::from("name")).unwrap();
    assert!(child_value
        .downcast_object_ref::<JinjaPromptValue>()
        .is_some());

    let raw = child_value.get_item(&Value::from("raw")).unwrap();
    assert_eq!(raw.as_str(), Some("test"));

    let child_type = child_value.get_item(&Value::from("__type__")).unwrap();
    let expected_child_type = TypeIR::string().diagnostic_repr().to_string();
    assert_eq!(child_type.as_str(), Some(expected_child_type.as_str()));

    let child_path = child_value.get_item(&Value::from("__path__")).unwrap();
    assert_eq!(child_path.as_str(), Some("root.name"));
}

#[test]
fn prompt_value_renders_structural_by_default() {
    let world = Arc::new(make_world_with_class(
        "SimpleRender",
        vec![("name".to_string(), TypeIR::string()), ("count".to_string(), TypeIR::int())],
    ));
    let value = BamlValue::Class(
        "SimpleRender".to_string(),
        IndexMap::from([
            ("name".to_string(), BamlValue::String("Ada".to_string())),
            ("count".to_string(), BamlValue::Int(2)),
        ]),
    );
    let pv = PromptValue::new(
        value,
        TypeIR::class("SimpleRender"),
        world.clone(),
        make_session(),
        PromptPath::new().push_field("value"),
    );

    let rendered = world
        .render_prompt_value(&pv, None)
        .expect("rendered");

    assert_eq!(rendered, "SimpleRender {name: Ada, count: 2}");
}

#[test]
fn prompt_value_uses_type_level_renderer() {
    let class = Class {
        name: Name::new("RenderWithTemplate".to_string()),
        description: None,
        namespace: StreamingMode::NonStreaming,
        fields: vec![(
            Name::new("name".to_string()),
            TypeIR::string(),
            None,
            false,
            None,
        )],
        constraints: Vec::new(),
        streaming_behavior: baml_types::type_meta::base::StreamingBehavior::default(),
    };
    let mut reg = Registry::new();
    reg.register_class(class);
    reg.register_renderer(
        RendererKey::for_class("RenderWithTemplate", StreamingMode::NonStreaming, "default"),
        RendererSpec::Jinja {
            source: "Name: {{ value.name }}",
        },
    );
    let (output_format, renderers) = reg.build_with_renderers(TypeIR::class("RenderWithTemplate"));
    let world = Arc::new(
        PromptWorld::from_registry(output_format, renderers, RenderSettings::default())
            .expect("prompt world"),
    );
    let value = BamlValue::Class(
        "RenderWithTemplate".to_string(),
        IndexMap::from([("name".to_string(), BamlValue::String("Ada".to_string()))]),
    );
    let pv = PromptValue::new(
        value,
        TypeIR::class("RenderWithTemplate"),
        world.clone(),
        make_session(),
        PromptPath::new().push_field("value"),
    );

    let rendered = world
        .render_prompt_value(&pv, None)
        .expect("rendered");

    assert_eq!(rendered, "Name: Ada");
}

#[test]
fn prompt_value_passes_ctx_to_templates() {
    let mut world = make_world_with_class(
        "RenderWithTemplate",
        vec![("name".to_string(), TypeIR::string())],
    );
    world
        .jinja
        .add_template_owned("ctx_template".to_string(), "{{ ctx.max_output_chars }}".to_string())
        .expect("template add");
    let value = BamlValue::Class(
        "RenderWithTemplate".to_string(),
        IndexMap::from([("name".to_string(), BamlValue::String("Ada".to_string()))]),
    );
    let session = Arc::new(
        RenderSession::new(RenderSettings::default()).with_ctx(json!({ "max_output_chars": 42 })),
    );
    let pv = PromptValue::new(
        value,
        TypeIR::class("RenderWithTemplate"),
        Arc::new(world),
        session,
        PromptPath::new().push_field("value"),
    )
    .with_override(RendererOverride::Template {
        source: "{{ ctx.max_output_chars }}",
        compiled_name: Some("ctx_template".to_string()),
    });

    let rendered = pv
        .world
        .render_prompt_value(&pv, None)
        .expect("rendered");
    assert_eq!(rendered, "42");
}

#[test]
fn prompt_value_budget_truncation_returns_output() {
    let world = Arc::new(make_world_with_class(
        "SimpleRender",
        vec![("name".to_string(), TypeIR::string()), ("count".to_string(), TypeIR::int())],
    ));
    let value = BamlValue::Class(
        "SimpleRender".to_string(),
        IndexMap::from([
            ("name".to_string(), BamlValue::String("Ada".to_string())),
            ("count".to_string(), BamlValue::Int(2)),
        ]),
    );
    let session = Arc::new(RenderSession {
        settings: RenderSettings {
            max_total_chars: 20,
            ..RenderSettings::default()
        },
        ..RenderSession::new(RenderSettings::default())
    });
    let pv = PromptValue::new(
        value,
        TypeIR::class("SimpleRender"),
        world.clone(),
        session,
        PromptPath::new().push_field("value"),
    );

    let rendered = pv
        .world
        .render_prompt_value(&pv, None)
        .expect("rendered");

    assert!(
        rendered.ends_with("... (truncated)"),
        "expected truncation, got: {rendered}"
    );
}

fn make_session() -> Arc<RenderSession> {
    Arc::new(RenderSession::new(RenderSettings::default()))
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
        name: Name::new(name.to_string()),
        description: None,
        namespace: baml_types::StreamingMode::NonStreaming,
        fields: fields
            .into_iter()
            .map(|(field_name, field_type)| {
                (
                    Name::new(field_name),
                    field_type,
                    None,
                    false,
                    None,
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
