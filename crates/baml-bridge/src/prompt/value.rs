//! Prompt value wrappers for typed rendering.

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use baml_types::{
    ir_type::UnionTypeGeneric,
    type_meta, BamlValue, LiteralValue, TypeIR, TypeValue,
};
use indexmap::IndexMap;

use super::renderer::{RenderSession, RendererOverride};
use super::PromptWorld;

#[derive(Debug, Clone, Default)]
pub struct PromptPath {
    segments: Vec<PathSegment>,
}

#[derive(Debug, Clone)]
enum PathSegment {
    Field(String),
    Index(usize),
    MapKey(String),
}

impl PromptPath {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_field(&self, name: impl Into<String>) -> Self {
        let mut new = self.clone();
        new.segments.push(PathSegment::Field(name.into()));
        new
    }

    pub fn push_index(&self, idx: usize) -> Self {
        let mut new = self.clone();
        new.segments.push(PathSegment::Index(idx));
        new
    }

    pub fn push_map_key(&self, key: impl Into<String>) -> Self {
        let mut new = self.clone();
        new.segments.push(PathSegment::MapKey(key.into()));
        new
    }
}

impl fmt::Display for PromptPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for segment in &self.segments {
            match segment {
                PathSegment::Field(name) => {
                    if first {
                        write!(f, "{name}")?;
                    } else {
                        write!(f, ".{name}")?;
                    }
                }
                PathSegment::Index(idx) => {
                    write!(f, "[{idx}]")?;
                }
                PathSegment::MapKey(key) => {
                    let escaped = key.replace('\\', "\\\\").replace('"', "\\\"");
                    write!(f, "[\"{escaped}\"]")?;
                }
            }
            first = false;
        }
        Ok(())
    }
}

/// Result of union resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnionResolution {
    /// Successfully resolved to a single branch type.
    Resolved(TypeIR),
    /// Could not determine - multiple candidates equally likely.
    Ambiguous { candidates: Vec<TypeIR> },
}

/// Function type for resolving unions.
pub type UnionResolver =
    fn(value: &BamlValue, union: &UnionTypeGeneric<type_meta::IR>, world: &PromptWorld)
        -> UnionResolution;

/// Default union resolver using scoring heuristics.
pub fn default_union_resolver(
    value: &BamlValue,
    union: &UnionTypeGeneric<type_meta::IR>,
    world: &PromptWorld,
) -> UnionResolution {
    if matches!(value, BamlValue::Null) {
        return if union.is_optional() {
            UnionResolution::Resolved(TypeIR::null())
        } else {
            UnionResolution::Ambiguous {
                candidates: union.iter_skip_null().into_iter().cloned().collect(),
            }
        };
    }

    let candidates = union.iter_skip_null();
    if candidates.is_empty() {
        return UnionResolution::Ambiguous {
            candidates: Vec::new(),
        };
    }

    let mut best_score = 0usize;
    let mut best = Vec::new();

    for candidate in candidates {
        let resolved = resolve_recursive_alias(candidate, world).unwrap_or_else(|| candidate.clone());
        let score = score_candidate(value, &resolved, world);
        if score > best_score {
            best_score = score;
            best.clear();
            best.push(resolved);
        } else if score == best_score {
            best.push(resolved);
        }
    }

    match best.len() {
        0 => UnionResolution::Ambiguous {
            candidates: Vec::new(),
        },
        1 => UnionResolution::Resolved(best.remove(0)),
        _ => {
            if best_score == 0 {
                UnionResolution::Ambiguous { candidates: best }
            } else {
                UnionResolution::Resolved(best.remove(0))
            }
        }
    }
}

fn resolve_recursive_alias(candidate: &TypeIR, world: &PromptWorld) -> Option<TypeIR> {
    match candidate {
        TypeIR::RecursiveTypeAlias { name, .. } => world.types.resolve_recursive_alias(name).cloned(),
        _ => None,
    }
}

fn score_candidate(value: &BamlValue, candidate: &TypeIR, world: &PromptWorld) -> usize {
    match candidate {
        TypeIR::Union(union, _) => score_union(value, union, world),
        TypeIR::RecursiveTypeAlias { .. } => 0,
        _ => score_value_against_type(value, candidate, world),
    }
}

fn score_union(value: &BamlValue, union: &UnionTypeGeneric<type_meta::IR>, world: &PromptWorld) -> usize {
    union
        .iter_skip_null()
        .into_iter()
        .map(|candidate| score_candidate(value, candidate, world))
        .max()
        .unwrap_or(0)
}

fn score_value_against_type(value: &BamlValue, candidate: &TypeIR, world: &PromptWorld) -> usize {
    match (value, candidate) {
        (BamlValue::Class(_, fields), TypeIR::Class { name, mode, .. }) => {
            world
                .types
                .find_class(name, *mode)
                .map(|class| score_class_fields(fields, class))
                .unwrap_or(0)
        }
        (BamlValue::String(value), TypeIR::Enum { name, .. }) => {
            score_enum_value(name, value, world)
        }
        (BamlValue::Enum(_, value), TypeIR::Enum { name, .. }) => {
            score_enum_value(name, value, world)
        }
        (BamlValue::String(value), TypeIR::Literal(lit, _)) => score_literal(value, lit),
        (BamlValue::Enum(_, value), TypeIR::Literal(lit, _)) => score_literal(value, lit),
        (BamlValue::String(_), TypeIR::Primitive(TypeValue::String, _)) => 10,
        (BamlValue::Int(value), TypeIR::Literal(lit, _)) => match lit {
            LiteralValue::Int(lit) if lit == value => 100,
            _ => 0,
        },
        (BamlValue::Bool(value), TypeIR::Literal(lit, _)) => match lit {
            LiteralValue::Bool(lit) if lit == value => 100,
            _ => 0,
        },
        (BamlValue::Int(_), TypeIR::Primitive(TypeValue::Int, _)) => 10,
        (BamlValue::Float(_), TypeIR::Primitive(TypeValue::Float, _)) => 10,
        (BamlValue::Bool(_), TypeIR::Primitive(TypeValue::Bool, _)) => 10,
        (BamlValue::List(_), TypeIR::List(_, _)) => 5,
        (BamlValue::Map(_), TypeIR::Map(_, _, _)) => 5,
        (BamlValue::Media(media), TypeIR::Primitive(TypeValue::Media(kind), _)) => {
            if &media.media_type == kind {
                10
            } else {
                0
            }
        }
        (BamlValue::Enum(_, _), TypeIR::Primitive(TypeValue::String, _)) => 2,
        _ => 0,
    }
}

fn score_literal(value: &str, lit: &LiteralValue) -> usize {
    match lit {
        LiteralValue::String(lit) if lit == value => 100,
        _ => 0,
    }
}

fn score_enum_value(enum_name: &str, value: &str, world: &PromptWorld) -> usize {
    world
        .types
        .find_enum(enum_name)
        .and_then(|enum_type| {
            enum_type.values.iter().find(|(name, _, _)| {
                name.real_name() == value || name.rendered_name() == value
            })
        })
        .map(|_| 90)
        .unwrap_or(0)
}

fn score_class_fields(fields: &IndexMap<String, BamlValue>, class: &internal_baml_jinja::types::Class) -> usize {
    fields
        .keys()
        .filter(|key| {
            class.fields.iter().any(|(name, _, _, _, _)| {
                name.real_name() == key.as_str() || name.rendered_name() == key.as_str()
            })
        })
        .count()
}

/// Inner storage for PromptValue with memoized union resolution.
#[derive(Debug)]
struct PromptValueInner {
    value: BamlValue,
    ty: TypeIR,
    union_resolution: OnceLock<UnionResolution>,
}

/// A typed value in the prompt rendering system.
#[derive(Clone)]
pub struct PromptValue {
    inner: Arc<PromptValueInner>,
    /// Reference to the type universe.
    pub world: Arc<PromptWorld>,
    /// Per-render session (settings, ctx, depth).
    pub session: Arc<RenderSession>,
    /// Per-field renderer override (if any).
    pub override_renderer: Option<RendererOverride>,
    /// Path for error reporting (e.g., `inputs.history.entries[3]`).
    pub path: PromptPath,
}

impl PromptValue {
    pub fn new(
        value: BamlValue,
        ty: TypeIR,
        world: Arc<PromptWorld>,
        session: Arc<RenderSession>,
        path: PromptPath,
    ) -> Self {
        Self {
            inner: Arc::new(PromptValueInner {
                value,
                ty,
                union_resolution: OnceLock::new(),
            }),
            world,
            session,
            override_renderer: None,
            path,
        }
    }

    pub fn value(&self) -> &BamlValue {
        &self.inner.value
    }

    pub fn ty(&self) -> &TypeIR {
        &self.inner.ty
    }

    pub fn with_override(mut self, override_renderer: RendererOverride) -> Self {
        self.override_renderer = Some(override_renderer);
        self
    }

    /// Navigate to a class field by name.
    /// Supports both real_name and rendered_name.
    /// Applies field-level FieldRenderSpec if present.
    pub fn child_field(&self, field: &str) -> Option<PromptValue> {
        let (class_name, fields) = match self.value() {
            BamlValue::Class(name, fields) => (name.as_str(), fields),
            _ => return None,
        };

        let resolved_ty = self.resolved_ty();
        let class = match self.resolve_alias(&resolved_ty) {
            TypeIR::Class { name, mode, .. } => self.world.types.find_class(name, *mode),
            _ => self
                .world
                .types
                .classes
                .iter()
                .find(|((name, _), _)| name == class_name)
                .map(|(_, class)| class),
        };

        let mut candidate_keys = vec![field];
        let mut child_type = None;
        let mut render_spec = None;

        if let Some(class) = class {
            if let Some((field_name, field_type, _, _, field_render_spec)) =
                class.fields.iter().find(|(name, ..)| {
                    name.real_name() == field || name.rendered_name() == field
                })
            {
                candidate_keys = vec![field_name.real_name()];
                if field_name.rendered_name() != field_name.real_name() {
                    candidate_keys.push(field_name.rendered_name());
                }
                child_type = Some(field_type.clone());
                render_spec = field_render_spec.as_ref();
            }
        }

        let child_value = candidate_keys
            .into_iter()
            .find_map(|key| fields.get(key))?;
        let child_type = child_type.unwrap_or_else(|| Self::infer_type_from_value(child_value));

        // Build child session with depth increment and optional spec settings
        let child_session = match render_spec {
            Some(spec) => {
                let mut session = self.session.push_depth();
                session.settings = session.settings.with_field_spec(spec);
                session
            }
            None => self.session.push_depth(),
        };

        let mut child = PromptValue::new(
            child_value.clone(),
            child_type,
            self.world.clone(),
            Arc::new(child_session),
            self.path.push_field(field),
        );

        // Apply renderer override from spec (template takes precedence over style)
        if let Some(spec) = render_spec {
            if let Some(template) = spec.template {
                child = child.with_override(RendererOverride::template(template));
            } else if let Some(style) = spec.style {
                child = child.with_override(RendererOverride::style(style));
            }
        }

        Some(child)
    }

    /// Navigate to a list element by index.
    /// Respects max_list_items budget.
    pub fn child_index(&self, idx: usize) -> Option<PromptValue> {
        if idx >= self.session.settings.max_list_items {
            return None;
        }

        let items = match self.value() {
            BamlValue::List(items) => items,
            _ => return None,
        };

        let child_value = items.get(idx)?;
        let resolved_ty = self.resolved_ty();
        let child_type = match self.resolve_alias(&resolved_ty) {
            TypeIR::List(inner, _) => inner.as_ref().clone(),
            _ => Self::infer_type_from_value(child_value),
        };

        Some(PromptValue::new(
            child_value.clone(),
            child_type,
            self.world.clone(),
            Arc::new(self.session.push_depth()),
            self.path.push_index(idx),
        ))
    }

    /// Navigate to a map value by key.
    pub fn child_map_value(&self, key: &str) -> Option<PromptValue> {
        let map = match self.value() {
            BamlValue::Map(map) => map,
            _ => return None,
        };

        let child_value = map.get(key)?;
        let resolved_ty = self.resolved_ty();
        let child_type = match self.resolve_alias(&resolved_ty) {
            TypeIR::Map(_, value_type, _) => value_type.as_ref().clone(),
            _ => Self::infer_type_from_value(child_value),
        };

        Some(PromptValue::new(
            child_value.clone(),
            child_type,
            self.world.clone(),
            Arc::new(self.session.push_depth()),
            self.path.push_map_key(key),
        ))
    }

    /// Infer type from value when schema info is unavailable.
    fn infer_type_from_value(value: &BamlValue) -> TypeIR {
        match value {
            BamlValue::String(_) => TypeIR::string(),
            BamlValue::Int(_) => TypeIR::int(),
            BamlValue::Float(_) => TypeIR::float(),
            BamlValue::Bool(_) => TypeIR::bool(),
            BamlValue::Null => TypeIR::null(),
            BamlValue::List(items) => {
                let inner = items
                    .first()
                    .map(Self::infer_type_from_value)
                    .unwrap_or_else(TypeIR::string);
                TypeIR::list(inner)
            }
            BamlValue::Map(map) => {
                let value_type = map
                    .values()
                    .next()
                    .map(Self::infer_type_from_value)
                    .unwrap_or_else(TypeIR::string);
                TypeIR::map(TypeIR::string(), value_type)
            }
            BamlValue::Media(media) => match media.media_type {
                baml_types::BamlMediaType::Image => TypeIR::image(),
                baml_types::BamlMediaType::Audio => TypeIR::audio(),
                baml_types::BamlMediaType::Pdf => TypeIR::pdf(),
                baml_types::BamlMediaType::Video => TypeIR::video(),
            },
            BamlValue::Enum(name, _) => TypeIR::r#enum(name),
            BamlValue::Class(name, _) => TypeIR::class(name),
        }
    }

    fn resolve_alias<'a>(&'a self, ty: &'a TypeIR) -> &'a TypeIR {
        match ty {
            TypeIR::RecursiveTypeAlias { name, .. } => {
                self.world.types.resolve_recursive_alias(name).unwrap_or(ty)
            }
            _ => ty,
        }
    }

    fn union_resolution(&self) -> Option<&UnionResolution> {
        match &self.inner.ty {
            TypeIR::Union(union, _) => Some(
                self.inner
                    .union_resolution
                    .get_or_init(|| (self.world.union_resolver)(&self.inner.value, union, &self.world)),
            ),
            _ => None,
        }
    }

    pub fn resolved_ty(&self) -> TypeIR {
        match self.union_resolution() {
            Some(UnionResolution::Resolved(ty)) => ty.clone(),
            Some(UnionResolution::Ambiguous { .. }) => self.inner.ty.clone(),
            None => self.inner.ty.clone(),
        }
    }

    pub fn is_union_resolved(&self) -> bool {
        match self.union_resolution() {
            Some(UnionResolution::Resolved(_)) | None => true,
            Some(UnionResolution::Ambiguous { .. }) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{default_union_resolver, PromptPath, PromptValue, UnionResolution};
    use crate::prompt::renderer::{RenderSession, RenderSettings, RendererDb};
    use crate::prompt::world::{PromptWorld, TypeDb};
    use baml_types::{
        ir_type::{UnionConstructor, UnionTypeGeneric},
        type_meta, BamlValue, TypeIR,
    };
    use indexmap::{IndexMap, IndexSet};
    use internal_baml_jinja::types::{Class, Enum, Name};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    static UNION_RESOLUTION_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn counting_union_resolver(
        value: &BamlValue,
        union: &UnionTypeGeneric<type_meta::IR>,
        world: &PromptWorld,
    ) -> UnionResolution {
        UNION_RESOLUTION_CALLS.fetch_add(1, Ordering::SeqCst);
        default_union_resolver(value, union, world)
    }

    #[test]
    fn formats_field_and_index_path() {
        let path = PromptPath::new()
            .push_field("inputs")
            .push_field("history")
            .push_field("entries")
            .push_index(3)
            .push_field("output");

        assert_eq!(path.to_string(), "inputs.history.entries[3].output");
    }

    #[test]
    fn formats_map_key_path() {
        let path = PromptPath::new().push_field("meta").push_map_key("key");

        assert_eq!(path.to_string(), "meta[\"key\"]");
    }

    #[test]
    fn resolves_union_by_class_field_overlap() {
        let world = make_world(
            vec![],
            vec![
                class("Foo", &["a", "b"]),
                class("Bar", &["a"]),
            ],
        );

        let value = BamlValue::Class(
            "Foo".to_string(),
            IndexMap::from([
                ("a".to_string(), BamlValue::String("x".to_string())),
                ("b".to_string(), BamlValue::String("y".to_string())),
            ]),
        );

        let union_type = TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]);
        let union = match &union_type {
            TypeIR::Union(union, _) => union,
            _ => panic!("expected union type"),
        };

        let resolved = default_union_resolver(&value, union, &world);
        assert_eq!(resolved, UnionResolution::Resolved(TypeIR::class("Foo")));
    }

    #[test]
    fn resolves_union_by_enum_match() {
        let world = make_world(
            vec![enum_type("Choice", &["Yes", "No"])],
            vec![],
        );

        let value = BamlValue::String("Yes".to_string());
        let union_type = TypeIR::union(vec![TypeIR::r#enum("Choice"), TypeIR::string()]);
        let union = match &union_type {
            TypeIR::Union(union, _) => union,
            _ => panic!("expected union type"),
        };

        let resolved = default_union_resolver(&value, union, &world);
        assert_eq!(resolved, UnionResolution::Resolved(TypeIR::r#enum("Choice")));
    }

    #[test]
    fn returns_ambiguous_on_tie_with_no_signal() {
        let world = make_world(
            vec![],
            vec![class("Foo", &[]), class("Bar", &[])],
        );

        let value = BamlValue::Class("Foo".to_string(), IndexMap::new());
        let union_type = TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]);
        let union = match &union_type {
            TypeIR::Union(union, _) => union,
            _ => panic!("expected union type"),
        };

        let resolved = default_union_resolver(&value, union, &world);
        assert_eq!(
            resolved,
            UnionResolution::Ambiguous {
                candidates: vec![TypeIR::class("Foo"), TypeIR::class("Bar")]
            }
        );
    }

    #[test]
    fn resolves_null_for_optional_union() {
        let world = make_world(vec![], vec![]);
        let value = BamlValue::Null;
        let union_type = TypeIR::union(vec![TypeIR::string(), TypeIR::null()]);
        let union = match &union_type {
            TypeIR::Union(union, _) => union,
            _ => panic!("expected union type"),
        };

        let resolved = default_union_resolver(&value, union, &world);
        assert_eq!(resolved, UnionResolution::Resolved(TypeIR::null()));
    }

    #[test]
    fn resolved_ty_memoizes_union_resolution() {
        UNION_RESOLUTION_CALLS.store(0, Ordering::SeqCst);
        let mut world = make_world(vec![], vec![]);
        world.union_resolver = counting_union_resolver;
        let session = Arc::new(RenderSession::new(RenderSettings::default()));
        let value = BamlValue::String("hi".to_string());
        let union_type = TypeIR::union(vec![TypeIR::string(), TypeIR::int()]);

        let prompt = PromptValue::new(
            value,
            union_type,
            Arc::new(world),
            session,
            PromptPath::new(),
        );

        let first = prompt.resolved_ty();
        let second = prompt.resolved_ty();

        assert_eq!(first, TypeIR::string());
        assert_eq!(second, TypeIR::string());
        assert_eq!(UNION_RESOLUTION_CALLS.load(Ordering::SeqCst), 1);
        assert!(prompt.is_union_resolved());
    }

    #[test]
    fn resolved_ty_returns_union_when_ambiguous() {
        let world = make_world(
            vec![],
            vec![class("Foo", &[]), class("Bar", &[])],
        );
        let session = Arc::new(RenderSession::new(RenderSettings::default()));
        let value = BamlValue::Class("Foo".to_string(), IndexMap::new());
        let union_type = TypeIR::union(vec![TypeIR::class("Foo"), TypeIR::class("Bar")]);

        let prompt = PromptValue::new(
            value,
            union_type.clone(),
            Arc::new(world),
            session,
            PromptPath::new(),
        );

        assert_eq!(prompt.resolved_ty(), union_type);
        assert!(!prompt.is_union_resolved());
    }

    #[test]
    fn child_field_supports_rendered_name() {
        let world = make_world(
            vec![],
            vec![class_with_alias("Widget", &[("real", Some("alias"))])],
        );
        let session = Arc::new(RenderSession::new(RenderSettings::default()));
        let value = BamlValue::Class(
            "Widget".to_string(),
            IndexMap::from([(
                "real".to_string(),
                BamlValue::String("ok".to_string()),
            )]),
        );

        let prompt = PromptValue::new(
            value,
            TypeIR::class("Widget"),
            Arc::new(world),
            session,
            PromptPath::new(),
        );

        let child = prompt.child_field("alias").expect("child field");
        assert_eq!(child.value(), &BamlValue::String("ok".to_string()));
        assert_eq!(child.ty(), &TypeIR::string());
        assert_eq!(child.path.to_string(), "alias");
        assert!(child.override_renderer.is_none());
    }

    #[test]
    fn child_field_falls_back_to_value_inference() {
        let world = make_world(vec![], vec![]);
        let session = Arc::new(RenderSession::new(RenderSettings::default()));
        let value = BamlValue::Class(
            "Unknown".to_string(),
            IndexMap::from([("flag".to_string(), BamlValue::Bool(true))]),
        );

        let prompt = PromptValue::new(
            value,
            TypeIR::class("Unknown"),
            Arc::new(world),
            session,
            PromptPath::new(),
        );

        let child = prompt.child_field("flag").expect("child field");
        assert_eq!(child.value(), &BamlValue::Bool(true));
        assert_eq!(child.ty(), &TypeIR::bool());
    }

    #[test]
    fn child_index_respects_budget_and_schema() {
        let world = make_world(vec![], vec![]);
        let settings = RenderSettings {
            max_list_items: 1,
            ..RenderSettings::default()
        };
        let session = Arc::new(RenderSession::new(settings));
        let value = BamlValue::List(vec![
            BamlValue::String("a".to_string()),
            BamlValue::String("b".to_string()),
        ]);

        let prompt = PromptValue::new(
            value,
            TypeIR::list(TypeIR::string()),
            Arc::new(world),
            session,
            PromptPath::new(),
        );

        assert!(prompt.child_index(1).is_none());
        let child = prompt.child_index(0).expect("child index");
        assert_eq!(child.ty(), &TypeIR::string());
        assert_eq!(child.path.to_string(), "[0]");
    }

    #[test]
    fn child_map_value_infers_type_without_schema() {
        let world = make_world(vec![], vec![]);
        let session = Arc::new(RenderSession::new(RenderSettings::default()));
        let value = BamlValue::Map(IndexMap::from([(
            "flag".to_string(),
            BamlValue::Bool(true),
        )]));

        let prompt = PromptValue::new(
            value,
            TypeIR::string(),
            Arc::new(world),
            session,
            PromptPath::new(),
        );

        let child = prompt.child_map_value("flag").expect("child map value");
        assert_eq!(child.ty(), &TypeIR::bool());
        assert_eq!(child.path.to_string(), "[\"flag\"]");
    }

    fn make_world(enums: Vec<Enum>, classes: Vec<Class>) -> PromptWorld {
        let mut enum_map = IndexMap::new();
        for enum_type in enums {
            enum_map.insert(enum_type.name.real_name().to_string(), enum_type);
        }

        let mut class_map = IndexMap::new();
        for class_type in classes {
            class_map.insert(
                (class_type.name.real_name().to_string(), class_type.namespace),
                class_type,
            );
        }

        let jinja = crate::jsonish::jinja_helpers::get_env();
        PromptWorld {
            types: TypeDb {
                enums: Arc::new(enum_map),
                classes: Arc::new(class_map),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
            renderers: RendererDb::new(),
            jinja,
            settings: RenderSettings::default(),
            union_resolver: default_union_resolver,
        }
    }

    fn class(name: &str, fields: &[&str]) -> Class {
        Class {
            name: Name::new(name.to_string()),
            description: None,
            namespace: baml_types::StreamingMode::NonStreaming,
            fields: fields
                .iter()
                .map(|field_name| {
                    (
                        Name::new(field_name.to_string()),
                        TypeIR::string(),
                        None,
                        false,
                        None,
                    )
                })
                .collect(),
            constraints: Vec::new(),
            streaming_behavior: type_meta::base::StreamingBehavior::default(),
        }
    }

    fn class_with_alias(name: &str, fields: &[(&str, Option<&str>)]) -> Class {
        Class {
            name: Name::new(name.to_string()),
            description: None,
            namespace: baml_types::StreamingMode::NonStreaming,
            fields: fields
                .iter()
                .map(|(field_name, alias)| {
                    (
                        Name::new_with_alias(
                            field_name.to_string(),
                            alias.map(|alias| alias.to_string()),
                        ),
                        TypeIR::string(),
                        None,
                        false,
                        None,
                    )
                })
                .collect(),
            constraints: Vec::new(),
            streaming_behavior: type_meta::base::StreamingBehavior::default(),
        }
    }

    fn enum_type(name: &str, values: &[&str]) -> Enum {
        Enum {
            name: Name::new(name.to_string()),
            description: None,
            values: values
                .iter()
                .map(|value| (Name::new(value.to_string()), None, None))
                .collect(),
            constraints: Vec::new(),
        }
    }
}
