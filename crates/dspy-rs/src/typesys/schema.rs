//! In-house type model that replaces BAML's `TypeIR` / `OutputFormatContent`.
//!
//! The runtime value model is `serde_json::Value`; the *type* model is [`FieldType`]
//! plus a registry of class/enum definitions ([`OutputSchema`]). Both are derived from
//! facet `Shape` metadata at runtime — facet stays as the single source of truth for
//! type structure, exactly as before, but the translation target is now local types
//! instead of vendored BAML crates.

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use facet::{Def, Facet, Field, ScalarType, Shape, Type, UserType};
use indexmap::IndexMap;
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::constraint::{Constraint, ConstraintKind};

/// Structural type of a signature/nested field, mirroring the subset of BAML's `TypeIR`
/// that DSRs actually uses. Class/enum variants carry the *internal name* used as a key
/// into [`OutputSchema`].
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    String,
    Int,
    Float,
    Bool,
    /// A fixed string value, used for untagged unit-enum unions.
    Literal(String),
    List(Box<FieldType>),
    Optional(Box<FieldType>),
    Map(Box<FieldType>, Box<FieldType>),
    /// Named struct; look up the definition in [`OutputSchema::classes`].
    Class(String),
    /// Named unit enum; look up the definition in [`OutputSchema::enums`].
    Enum(String),
    /// Union of alternatives (e.g. untagged enums rendered as `A | B`).
    Union(Vec<FieldType>),
}

impl FieldType {
    pub fn is_optional(&self) -> bool {
        matches!(self, FieldType::Optional(_))
    }

    pub fn optional(inner: FieldType) -> FieldType {
        if inner.is_optional() {
            inner
        } else {
            FieldType::Optional(Box::new(inner))
        }
    }

    /// Peels `Optional`/`List` wrappers to reach the underlying leaf type.
    pub fn peel(&self) -> &FieldType {
        match self {
            FieldType::Optional(inner) | FieldType::List(inner) => inner.peel(),
            other => other,
        }
    }
}

/// A single field inside a [`ClassDef`].
#[derive(Debug, Clone)]
pub struct FieldDef {
    /// Rust field name.
    pub name: String,
    /// LM-facing name (after `#[alias]`/serde rename); equals `name` when unaliased.
    pub rendered_name: String,
    pub field_type: FieldType,
    pub docs: Option<String>,
    pub constraints: Vec<Constraint>,
}

/// A struct definition reachable from a signature's output type.
#[derive(Debug, Clone)]
pub struct ClassDef {
    pub internal_name: String,
    pub rendered_name: String,
    pub docs: Option<String>,
    pub fields: Vec<FieldDef>,
    pub constraints: Vec<Constraint>,
}

/// A single value of a unit enum.
#[derive(Debug, Clone)]
pub struct EnumValueDef {
    pub name: String,
    pub rendered_name: String,
    pub docs: Option<String>,
}

/// A unit-enum definition reachable from a signature's output type.
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub internal_name: String,
    pub rendered_name: String,
    pub docs: Option<String>,
    pub values: Vec<EnumValueDef>,
}

/// The full type description for a value: the root [`FieldType`] plus every class/enum
/// definition it references. Replaces BAML's `OutputFormatContent`.
#[derive(Debug, Clone, Default)]
pub struct OutputSchema {
    pub target: FieldType,
    pub classes: IndexMap<String, ClassDef>,
    pub enums: IndexMap<String, EnumDef>,
}

impl Default for FieldType {
    fn default() -> Self {
        FieldType::String
    }
}

impl OutputSchema {
    /// Builds the schema for a facet type `T` from its `Shape`.
    pub fn from_shape(shape: &'static Shape) -> Self {
        let mut builder = SchemaBuilder::new();
        let target = builder.build_field_type(shape);
        OutputSchema {
            target,
            classes: builder.classes,
            enums: builder.enums,
        }
    }

    /// Returns the rendered (LM-facing) name for a class/enum internal name, falling back
    /// to the last `::` segment of the token when it isn't a known class/enum.
    pub fn rendered_name(&self, token: &str) -> String {
        if let Some(class) = self.classes.get(token) {
            return class.rendered_name.clone();
        }
        if let Some(enm) = self.enums.get(token) {
            return enm.rendered_name.clone();
        }
        token.rsplit("::").next().unwrap_or(token).to_string()
    }
}

/// Runtime trait exposing the type model + serde-backed value conversion for a type.
///
/// Implemented for every `facet::Facet + Serialize + DeserializeOwned` type via a blanket
/// impl, so `#[derive(Signature)]` and `#[Schema]` don't need to emit any of this by hand.
pub trait Schema: Serialize + DeserializeOwned + 'static {
    /// The cached [`OutputSchema`] for this type.
    fn output_schema() -> &'static OutputSchema;

    /// The internal (fully-qualified) name of this type.
    fn internal_name() -> &'static str;

    /// The root [`FieldType`] for this type.
    fn field_type() -> FieldType {
        Self::output_schema().target.clone()
    }
}

impl<T> Schema for T
where
    T: for<'a> Facet<'a> + Serialize + DeserializeOwned + 'static,
{
    fn output_schema() -> &'static OutputSchema {
        static CACHE: OnceLock<RwLock<HashMap<TypeId, &'static OutputSchema>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
        {
            let guard = cache.read().expect("output schema cache poisoned");
            if let Some(schema) = guard.get(&TypeId::of::<T>()) {
                return schema;
            }
        }
        let built = OutputSchema::from_shape(<T as Facet<'_>>::SHAPE);
        let leaked: &'static OutputSchema = Box::leak(Box::new(built));
        let mut guard = cache.write().expect("output schema cache poisoned");
        guard.entry(TypeId::of::<T>()).or_insert(leaked)
    }

    fn internal_name() -> &'static str {
        internal_name_for_shape(<T as Facet<'_>>::SHAPE)
    }
}

/// Builds just the [`FieldType`] for a shape, discarding the class/enum registry.
///
/// Used by `SignatureSchema` to type individual signature fields; the shared registry is
/// obtained separately from the whole output type's [`Schema::output_schema`].
pub fn field_type_from_shape(shape: &'static Shape) -> FieldType {
    SchemaBuilder::new().build_field_type(shape)
}

/// Computes the internal name for a shape: module path + type identifier when available.
pub fn internal_name_for_shape(shape: &'static Shape) -> &'static str {
    // Interned into a process-global set so we can hand out `'static` names.
    static NAMES: OnceLock<RwLock<HashMap<String, &'static str>>> = OnceLock::new();
    let names = NAMES.get_or_init(|| RwLock::new(HashMap::new()));
    let computed = match shape.module_path {
        Some(module) if !module.is_empty() => format!("{module}::{}", shape.type_identifier),
        _ => shape.type_identifier.to_string(),
    };
    {
        let guard = names.read().expect("name intern poisoned");
        if let Some(existing) = guard.get(&computed) {
            return existing;
        }
    }
    let mut guard = names.write().expect("name intern poisoned");
    if let Some(existing) = guard.get(&computed) {
        return existing;
    }
    let leaked: &'static str = Box::leak(computed.clone().into_boxed_str());
    guard.insert(computed, leaked);
    leaked
}

struct SchemaBuilder {
    classes: IndexMap<String, ClassDef>,
    enums: IndexMap<String, EnumDef>,
    visited: HashMap<facet::ConstTypeId, FieldType>,
}

impl SchemaBuilder {
    fn new() -> Self {
        Self {
            classes: IndexMap::new(),
            enums: IndexMap::new(),
            visited: HashMap::new(),
        }
    }

    fn build_field_type(&mut self, shape: &'static Shape) -> FieldType {
        if let Some(existing) = self.visited.get(&shape.id) {
            return existing.clone();
        }

        match &shape.def {
            Def::Scalar => self.build_scalar(shape),
            Def::Option(option_def) => FieldType::optional(self.build_field_type(option_def.t)),
            Def::List(list_def) => FieldType::List(Box::new(self.build_field_type(list_def.t))),
            Def::Array(arr_def) => FieldType::List(Box::new(self.build_field_type(arr_def.t))),
            Def::Set(set_def) => FieldType::List(Box::new(self.build_field_type(set_def.t))),
            Def::Map(map_def) => FieldType::Map(
                Box::new(self.build_field_type(map_def.k)),
                Box::new(self.build_field_type(map_def.v)),
            ),
            Def::Pointer(ptr_def) => match ptr_def.pointee {
                Some(pointee) => self.build_field_type(pointee),
                None => panic!(
                    "typesys: pointer shape `{}` has no pointee",
                    shape.type_identifier
                ),
            },
            Def::Undefined => {
                if let Some(inner) = shape.inner {
                    self.build_field_type(inner)
                } else {
                    self.build_from_type(shape)
                }
            }
            _ => self.build_from_type(shape),
        }
    }

    fn build_from_type(&mut self, shape: &'static Shape) -> FieldType {
        match &shape.ty {
            Type::User(UserType::Struct(struct_type)) => self.build_struct(shape, struct_type),
            Type::User(UserType::Enum(enum_type)) => self.build_enum(shape, enum_type),
            Type::Primitive(primitive) => build_primitive(primitive),
            _ => panic!(
                "typesys: unsupported shape `{}` ({:?})",
                shape.type_identifier, shape.def
            ),
        }
    }

    fn build_scalar(&self, shape: &'static Shape) -> FieldType {
        if let Type::Primitive(primitive) = &shape.ty {
            return build_primitive(primitive);
        }
        match shape.scalar_type() {
            Some(ScalarType::Bool) => FieldType::Bool,
            Some(ScalarType::Char | ScalarType::Str) => FieldType::String,
            Some(ScalarType::F32 | ScalarType::F64) => FieldType::Float,
            Some(
                ScalarType::U8
                | ScalarType::U16
                | ScalarType::U32
                | ScalarType::U64
                | ScalarType::U128
                | ScalarType::USize
                | ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::I128
                | ScalarType::ISize,
            ) => FieldType::Int,
            _ => match shape.type_identifier {
                "String" | "str" | "Cow<str>" | "Cow<'_, str>" | "Cow<'static, str>" => {
                    FieldType::String
                }
                other => panic!("typesys: unsupported scalar shape `{other}`"),
            },
        }
    }

    fn build_struct(
        &mut self,
        shape: &'static Shape,
        struct_type: &facet::StructType,
    ) -> FieldType {
        let internal_name = internal_name_for_shape(shape).to_string();
        let rendered_name = rendered_name_for_shape(shape);
        let type_ir = FieldType::Class(internal_name.clone());
        self.visited.insert(shape.id, type_ir.clone());

        let mut fields = Vec::new();
        for field in struct_type.fields.iter() {
            if field.should_skip_deserializing() {
                continue;
            }
            let mut field_type = self.build_field_type(field.shape());
            if field.has_default() && !field_type.is_optional() {
                field_type = FieldType::optional(field_type);
            }
            fields.push(FieldDef {
                name: field.name.to_string(),
                rendered_name: field.effective_name().to_string(),
                field_type,
                docs: doc_to_description(field.doc),
                constraints: constraints_from_field(field),
            });
        }

        self.classes.insert(
            internal_name.clone(),
            ClassDef {
                internal_name: internal_name.clone(),
                rendered_name,
                docs: doc_to_description(shape.doc),
                fields,
                constraints: Vec::new(),
            },
        );

        type_ir
    }

    fn build_enum(&mut self, shape: &'static Shape, enum_type: &facet::EnumType) -> FieldType {
        let is_data_enum = enum_type
            .variants
            .iter()
            .any(|variant| !variant.data.fields.is_empty());
        if is_data_enum {
            panic!(
                "typesys: data-carrying enums are not supported (`{}`); use a struct",
                shape.type_identifier
            );
        }

        let internal_name = internal_name_for_shape(shape).to_string();
        let rendered_name = rendered_name_for_shape(shape);
        let type_ir = FieldType::Enum(internal_name.clone());
        self.visited.insert(shape.id, type_ir.clone());

        let values = enum_type
            .variants
            .iter()
            .map(|variant| EnumValueDef {
                name: variant.name.to_string(),
                rendered_name: variant.effective_name().to_string(),
                docs: doc_to_description(variant.doc),
            })
            .collect();

        self.enums.insert(
            internal_name.clone(),
            EnumDef {
                internal_name: internal_name.clone(),
                rendered_name,
                docs: doc_to_description(shape.doc),
                values,
            },
        );

        type_ir
    }
}

fn build_primitive(primitive: &facet::PrimitiveType) -> FieldType {
    use facet::{NumericType, PrimitiveType, TextualType};
    match primitive {
        PrimitiveType::Boolean => FieldType::Bool,
        PrimitiveType::Numeric(NumericType::Integer { .. }) => FieldType::Int,
        PrimitiveType::Numeric(NumericType::Float) => FieldType::Float,
        PrimitiveType::Textual(TextualType::Str | TextualType::Char) => FieldType::String,
        PrimitiveType::Never => panic!("typesys: `never` type cannot be represented"),
    }
}

fn rendered_name_for_shape(shape: &'static Shape) -> String {
    if shape.rename.is_some() {
        return shape.effective_name().to_string();
    }
    if let Some(name) = shape.get_builtin_attr_value::<&'static str>("rename") {
        return name.to_string();
    }
    shape.type_identifier.to_string()
}

fn doc_to_description(doc: &'static [&'static str]) -> Option<String> {
    if doc.is_empty() {
        return None;
    }
    let joined = doc
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n");
    if joined.is_empty() { None } else { Some(joined) }
}

/// Reads `#[check]`/`#[assert]` constraints declared directly on a facet field via the
/// `bamltype`-namespaced attrs the derive still emits.
fn constraints_from_field(field: &Field) -> Vec<Constraint> {
    // Constraints on signature fields are attached via `FieldMetadataSpec` in
    // `core/schema.rs`; nested-type field constraints are not currently surfaced.
    let _ = field;
    let _ = ConstraintKind::Check;
    Vec::new()
}
