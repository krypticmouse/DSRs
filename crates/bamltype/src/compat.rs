//! Compatibility layer providing baml-bridge-compatible traits and helpers.

use std::collections::{HashMap, HashSet};

use baml_types::{BamlMap, BamlValue, Constraint, StreamingMode, TypeIR, type_meta};
use facet::Facet;
use indexmap::{IndexMap, IndexSet};
use internal_baml_jinja::types::{Class, Enum, OutputFormatContent};

use crate::BamlSchema;
use crate::convert;
use crate::schema_builder::build_type_ir_from_shape;

/// Error during BamlValue ↔ Rust conversion (baml-bridge compatible).
#[derive(Debug, Clone)]
pub struct BamlConvertError {
    pub path: Vec<String>,
    pub expected: &'static str,
    pub got: String,
    pub message: String,
}

impl BamlConvertError {
    pub fn new(
        path: Vec<String>,
        expected: &'static str,
        got: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            path,
            expected,
            got: got.into(),
            message: message.into(),
        }
    }

    pub fn with_path(mut self, segment: impl Into<String>) -> Self {
        self.path.push(segment.into());
        self
    }

    pub fn path_string(&self) -> String {
        if self.path.is_empty() {
            "<root>".to_string()
        } else {
            self.path.join(".")
        }
    }
}

impl std::fmt::Display for BamlConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (expected {}, got {}) at {}",
            self.message,
            self.expected,
            self.got,
            self.path_string()
        )
    }
}

impl std::error::Error for BamlConvertError {}

impl From<convert::ConvertError> for BamlConvertError {
    fn from(err: convert::ConvertError) -> Self {
        match err {
            convert::ConvertError::Adapter(inner) => inner,
            other => Self {
                path: Vec::new(),
                expected: "compatible type",
                got: other.to_string(),
                message: other.to_string(),
            },
        }
    }
}

/// Registry for schema elements (baml-bridge compatible).
#[derive(Debug, Default)]
pub struct Registry {
    enums: IndexMap<String, Enum>,
    classes: IndexMap<(String, StreamingMode), Class>,
    class_deps: IndexMap<String, IndexSet<String>>,
    structural_recursive_aliases: IndexMap<String, TypeIR>,
    registered: HashSet<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_type(&mut self, name: &str) -> bool {
        if self.registered.contains(name) {
            return false;
        }
        self.registered.insert(name.to_string());
        true
    }

    pub fn register_enum(&mut self, r#enum: Enum) {
        let name = r#enum.name.real_name().to_string();
        self.enums.entry(name).or_insert(r#enum);
    }

    pub fn register_class(&mut self, class: Class) {
        let name = class.name.real_name().to_string();
        let mode = class.namespace;
        let key = (name.clone(), mode);
        let entry = self.classes.entry(key).or_insert(class);

        let deps = self.class_deps.entry(name).or_default();
        for (_, field_type, _, _) in &entry.fields {
            collect_class_refs(field_type, deps);
        }
    }

    pub fn register_structural_alias(&mut self, name: String, alias: TypeIR) {
        self.structural_recursive_aliases.insert(name, alias);
    }

    pub fn build(self, target: TypeIR) -> OutputFormatContent {
        let recursive_classes = compute_recursive_classes(&self.class_deps);

        let mut enums = self.enums.into_iter().collect::<Vec<_>>();
        enums.sort_by(|(a, _), (b, _)| a.cmp(b));
        let enums = enums.into_iter().map(|(_, v)| v).collect::<Vec<_>>();

        let mut classes = self.classes.into_iter().collect::<Vec<_>>();
        classes.sort_by(|(a, _), (b, _)| {
            let (a_name, a_mode) = a;
            let (b_name, b_mode) = b;
            match a_name.cmp(b_name) {
                std::cmp::Ordering::Equal => mode_rank(*a_mode).cmp(&mode_rank(*b_mode)),
                other => other,
            }
        });
        let classes = classes.into_iter().map(|(_, v)| v).collect::<Vec<_>>();

        OutputFormatContent::target(target)
            .enums(enums)
            .classes(classes)
            .recursive_classes(recursive_classes)
            .structural_recursive_aliases(self.structural_recursive_aliases)
            .build()
    }
}

/// Internal type metadata (baml-bridge compatible).
pub trait BamlTypeInternal {
    fn baml_internal_name() -> &'static str;
    fn baml_type_ir() -> TypeIR;
    fn register(_reg: &mut Registry) {}
}

/// Convert from BamlValue to Rust (baml-bridge compatible).
pub trait BamlValueConvert: Sized {
    fn try_from_baml_value(value: BamlValue, path: Vec<String>) -> Result<Self, BamlConvertError>;
}

/// Convert from Rust to BamlValue (baml-bridge compatible).
pub trait ToBamlValue {
    fn to_baml_value(&self) -> BamlValue;
}

/// Adapter for custom field-level conversion / schema representation.
pub trait BamlAdapter<T> {
    fn type_ir() -> TypeIR;
    fn register(_reg: &mut Registry) {}
    fn try_from_baml(value: BamlValue, path: Vec<String>) -> Result<T, BamlConvertError>;
}

/// Full BamlType trait (baml-bridge compatible).
///
/// Named `BamlTypeTrait` to avoid collision with the `#[BamlType]` attribute macro.
pub trait BamlTypeTrait: BamlTypeInternal + BamlValueConvert + Sized + 'static {
    fn baml_output_format() -> &'static OutputFormatContent;

    fn baml_internal_name() -> &'static str {
        <Self as BamlTypeInternal>::baml_internal_name()
    }

    fn baml_type_ir() -> TypeIR {
        <Self as BamlTypeInternal>::baml_type_ir()
    }
}

impl<T: Facet<'static>> BamlTypeInternal for T {
    fn baml_internal_name() -> &'static str {
        for attr in T::SHAPE.attributes {
            if attr.ns != Some("bamltype") || attr.key != "internal_name" {
                continue;
            }

            if let Some(name) = attr.get_as::<&'static str>() {
                return name;
            }
        }

        if let Some(name) = T::SHAPE.get_builtin_attr_value::<&'static str>("internal_name") {
            name
        } else {
            std::any::type_name::<T>()
        }
    }

    fn baml_type_ir() -> TypeIR {
        build_type_ir_from_shape(T::SHAPE)
    }
}

impl<T: Facet<'static>> BamlValueConvert for T {
    fn try_from_baml_value(value: BamlValue, _path: Vec<String>) -> Result<Self, BamlConvertError> {
        convert::from_baml_value(value).map_err(BamlConvertError::from)
    }
}

impl<T: Facet<'static>> ToBamlValue for T {
    fn to_baml_value(&self) -> BamlValue {
        convert::to_baml_value(self).unwrap_or(BamlValue::Null)
    }
}

impl<T: BamlSchema> BamlTypeTrait for T {
    fn baml_output_format() -> &'static OutputFormatContent {
        &T::baml_schema().output_format
    }
}

/// Add constraints to a TypeIR (baml-bridge compatible).
pub fn with_constraints(mut type_ir: TypeIR, constraints: Vec<Constraint>) -> TypeIR {
    type_ir.meta_mut().constraints.extend(constraints);
    type_ir
}

/// Default streaming behavior helper (baml-bridge compatible).
pub fn default_streaming_behavior() -> type_meta::base::StreamingBehavior {
    type_meta::base::StreamingBehavior::default()
}

/// Lookup helper matching baml-bridge semantics (`name` then optional alias).
pub fn get_field<'a>(
    map: &'a BamlMap<String, BamlValue>,
    name: &str,
    alias: Option<&str>,
) -> Option<&'a BamlValue> {
    map.get(name)
        .or_else(|| alias.and_then(|alias| map.get(alias)))
}

fn collect_class_refs(field_type: &TypeIR, deps: &mut IndexSet<String>) {
    match field_type {
        TypeIR::Class { name, .. } => {
            deps.insert(name.clone());
        }
        TypeIR::RecursiveTypeAlias { name, .. } => {
            deps.insert(name.clone());
        }
        TypeIR::List(inner, _) => collect_class_refs(inner, deps),
        TypeIR::Map(key, value, _) => {
            collect_class_refs(key, deps);
            collect_class_refs(value, deps);
        }
        TypeIR::Union(union, _) => {
            for item in union.iter_include_null() {
                collect_class_refs(item, deps);
            }
        }
        TypeIR::Tuple(items, _) => {
            for item in items {
                collect_class_refs(item, deps);
            }
        }
        TypeIR::Arrow(arrow, _) => {
            for param in &arrow.param_types {
                collect_class_refs(param, deps);
            }
            collect_class_refs(&arrow.return_type, deps);
        }
        TypeIR::Primitive(..) | TypeIR::Enum { .. } | TypeIR::Literal(..) | TypeIR::Top(..) => {}
    }
}

fn compute_recursive_classes(class_deps: &IndexMap<String, IndexSet<String>>) -> IndexSet<String> {
    struct Tarjan<'a> {
        next_index: usize,
        indices: HashMap<String, usize>,
        lowlink: HashMap<String, usize>,
        stack: Vec<String>,
        on_stack: HashSet<String>,
        deps: &'a IndexMap<String, IndexSet<String>>,
        recursive: HashSet<String>,
    }

    impl<'a> Tarjan<'a> {
        fn new(deps: &'a IndexMap<String, IndexSet<String>>) -> Self {
            Self {
                next_index: 0,
                indices: HashMap::new(),
                lowlink: HashMap::new(),
                stack: Vec::new(),
                on_stack: HashSet::new(),
                deps,
                recursive: HashSet::new(),
            }
        }

        fn strongconnect(&mut self, node: &str) {
            let node_key = node.to_string();
            self.indices.insert(node_key.clone(), self.next_index);
            self.lowlink.insert(node_key.clone(), self.next_index);
            self.next_index += 1;
            self.stack.push(node_key.clone());
            self.on_stack.insert(node_key.clone());

            let child_names = self
                .deps
                .get(node)
                .map(|children| children.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            for child in child_names {
                if !self.indices.contains_key(&child) {
                    self.strongconnect(&child);
                    let lowlink_child = self.lowlink.get(&child).copied().unwrap_or(0);
                    let lowlink_node = self.lowlink.get(&node_key).copied().unwrap_or(0);
                    self.lowlink
                        .insert(node_key.clone(), lowlink_node.min(lowlink_child));
                } else if self.on_stack.contains(&child) {
                    let index_child = self.indices.get(&child).copied().unwrap_or(0);
                    let lowlink_node = self.lowlink.get(&node_key).copied().unwrap_or(0);
                    self.lowlink
                        .insert(node_key.clone(), lowlink_node.min(index_child));
                }
            }

            let node_index = self.indices.get(&node_key).copied().unwrap_or(0);
            let node_lowlink = self.lowlink.get(&node_key).copied().unwrap_or(0);
            if node_lowlink == node_index {
                let mut scc = Vec::new();
                while let Some(w) = self.stack.pop() {
                    self.on_stack.remove(&w);
                    scc.push(w.clone());
                    if w == node_key {
                        break;
                    }
                }

                if scc.len() > 1 {
                    for name in scc {
                        self.recursive.insert(name);
                    }
                } else if let Some(name) = scc.first()
                    && self
                        .deps
                        .get(name)
                        .map(|edges| edges.contains(name))
                        .unwrap_or(false)
                {
                    self.recursive.insert(name.clone());
                }
            }
        }
    }

    let mut tarjan = Tarjan::new(class_deps);
    for node in class_deps.keys() {
        if !tarjan.indices.contains_key(node) {
            tarjan.strongconnect(node);
        }
    }

    let mut sorted = tarjan.recursive.into_iter().collect::<Vec<_>>();
    sorted.sort();
    IndexSet::from_iter(sorted)
}

fn mode_rank(mode: StreamingMode) -> u8 {
    match mode {
        StreamingMode::NonStreaming => 0,
        StreamingMode::Streaming => 1,
    }
}
