//! Prompt value wrappers for typed rendering.

use std::fmt;

use baml_types::{
    ir_type::UnionTypeGeneric,
    type_meta, BamlValue, LiteralValue, TypeIR, TypeValue,
};
use indexmap::IndexMap;

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
            enum_type.values.iter().find(|(name, _)| {
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
            class.fields.iter().any(|(name, _, _, _)| {
                name.real_name() == key.as_str() || name.rendered_name() == key.as_str()
            })
        })
        .count()
}

#[derive(Debug, Clone)]
pub struct PromptValue;

#[cfg(test)]
mod tests {
    use super::{default_union_resolver, PromptPath, UnionResolution};
    use crate::prompt::world::{PromptWorld, TypeDb};
    use baml_types::{ir_type::UnionConstructor, type_meta, BamlValue, TypeIR};
    use indexmap::{IndexMap, IndexSet};
    use internal_baml_jinja::types::{Class, Enum, Name};
    use std::sync::Arc;

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

        PromptWorld {
            types: TypeDb {
                enums: Arc::new(enum_map),
                classes: Arc::new(class_map),
                structural_recursive_aliases: Arc::new(IndexMap::new()),
                recursive_classes: Arc::new(IndexSet::new()),
            },
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
                .map(|value| (Name::new(value.to_string()), None))
                .collect(),
            constraints: Vec::new(),
        }
    }
}
