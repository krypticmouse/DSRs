//! Schema builder - walks facet Shapes to build BAML schemas.
//!
//! This module implements the core reflection-to-BAML translation.

use std::collections::HashMap;

use baml_types::{Constraint, StreamingMode, TypeIR, type_meta};
use facet::{Attr, ConstTypeId, Def, Field, ScalarType, Shape, Type, UserType};
use internal_baml_jinja::types::{Class, Enum, Name, OutputFormatContent};

use crate::SchemaBundle;
use crate::adapters::FieldCodecRegisterContext;
use crate::facet_ext;
use crate::schema_registry::SchemaRegistry;

/// Build a SchemaBundle from a facet Shape.
///
/// TODO(dsrs-schema-result-api): expose non-panicking Result-returning schema build API publicly after downstream migration.
pub fn build_schema_bundle(shape: &'static Shape) -> SchemaBundle {
    let mut builder = SchemaBuilder::new();
    let target = builder.build_type_ir(shape);
    let output_format = builder.into_output_format(target.clone());
    SchemaBundle {
        target,
        output_format,
    }
}

/// Build a TypeIR from a facet Shape (without building full schema).
///
/// Used by runtime helpers to provide `baml_type_ir::<T>()` for any `Facet` type.
pub fn build_type_ir_from_shape(shape: &'static Shape) -> TypeIR {
    let mut builder = SchemaBuilder::new();
    builder.build_type_ir(shape)
}

/// Compute the BAML internal name for a shape.
///
/// This mirrors bridge behavior: explicit `#[baml(internal_name = ...)]` takes
/// precedence, otherwise module path + type identifier when available.
pub fn internal_name_for_shape(shape: &'static Shape) -> String {
    if let Some(name) = bamltype_internal_name(shape.attributes) {
        return name.to_string();
    }

    match shape.module_path {
        Some(module) if !module.is_empty() => format!("{module}::{}", shape.type_identifier),
        _ => shape.type_identifier.to_string(),
    }
}

/// Compute the rendered/display name for a shape.
///
/// Facet currently stores container rename in `Shape::rename` for some type kinds,
/// but for others it may only be present in builtin attrs. Prefer the explicit
/// shape field, then fall back to builtin attr lookup for parity with legacy bridge.
fn rendered_name_for_shape(shape: &'static Shape) -> String {
    if shape.rename.is_some() {
        return shape.effective_name().to_string();
    }

    if let Some(name) = shape.get_builtin_attr_value::<&'static str>("rename") {
        return name.to_string();
    }

    shape.type_identifier.to_string()
}

/// Internal builder state for schema construction.
struct SchemaBuilder {
    /// Memoization: shape id -> (internal_name, TypeIR)
    visited: HashMap<ConstTypeId, (String, TypeIR)>,

    /// Collected schema elements.
    registry: SchemaRegistry,

    /// Track which internal names are already used (for collision handling)
    used_internal_names: HashMap<String, ConstTypeId>,

    /// Collision suffix counter
    name_counter: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntReprMode {
    String,
    I64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapKeyReprMode {
    String,
    Pairs,
}

#[derive(Clone, Debug)]
struct MapEntryContext {
    owner_internal_name: String,
    field_name: String,
    rendered_field: String,
    variant_name: Option<String>,
    variant_rendered: Option<String>,
}

impl SchemaBuilder {
    fn new() -> Self {
        Self {
            visited: HashMap::new(),
            registry: SchemaRegistry::new(),
            used_internal_names: HashMap::new(),
            name_counter: 0,
        }
    }

    fn fail_unsupported_shape(context: &str, shape: &'static Shape) -> ! {
        panic!(
            "schema build failed: {context}; shape_id={:?}, type_identifier={}, def={:?}",
            shape.id, shape.type_identifier, shape.def
        );
    }

    /// Build TypeIR from a facet Shape.
    fn build_type_ir(&mut self, shape: &'static Shape) -> TypeIR {
        // Check if already visited (handles recursion)
        if let Some((_, type_ir)) = self.visited.get(&shape.id) {
            return type_ir.clone();
        }

        // Handle based on semantic Def first.
        match &shape.def {
            Def::Scalar => self.build_scalar_ir(shape),
            Def::Option(option_def) => {
                let inner_ir = self.build_type_ir(option_def.t);
                TypeIR::optional(inner_ir)
            }
            Def::List(list_def) => self.build_list_ir(list_def.t),
            Def::Array(arr_def) => self.build_list_ir(arr_def.t),
            Def::Map(map_def) => {
                let key_ir = self.build_type_ir(map_def.k);
                let value_ir = self.build_type_ir(map_def.v);
                TypeIR::map(key_ir, value_ir)
            }
            Def::Set(set_def) => self.build_list_ir(set_def.t),
            Def::Pointer(ptr_def) => {
                // Smart pointers - unwrap to inner type when available
                if let Some(pointee) = ptr_def.pointee {
                    self.build_type_ir(pointee)
                } else {
                    Self::fail_unsupported_shape(
                        "pointer shape missing pointee while building TypeIR",
                        shape,
                    )
                }
            }
            Def::Undefined => {
                if let Some(inner) = shape.inner {
                    return self.build_type_ir(inner);
                }
                self.build_from_type(shape)
            }
            _ => self.build_from_type(shape),
        }
    }

    fn build_list_ir(&mut self, item_shape: &'static Shape) -> TypeIR {
        TypeIR::list(self.build_type_ir(item_shape))
    }

    /// Build TypeIR from the shape's Type field (for user-defined types).
    fn build_from_type(&mut self, shape: &'static Shape) -> TypeIR {
        match &shape.ty {
            Type::User(UserType::Struct(struct_type)) => self.build_struct_ir(shape, struct_type),
            Type::User(UserType::Enum(enum_type)) => self.build_enum_ir(shape, enum_type),
            Type::Primitive(primitive) => self.build_primitive_ir(shape, primitive),
            _ => Self::fail_unsupported_shape("unsupported shape type in build_from_type", shape),
        }
    }

    /// Build TypeIR for scalar/primitive shapes.
    fn build_scalar_ir(&self, shape: &'static Shape) -> TypeIR {
        match &shape.ty {
            Type::Primitive(primitive) => self.build_primitive_ir(shape, primitive),
            _ => Self::build_known_scalar_ir(shape).unwrap_or_else(|| {
                Self::fail_unsupported_shape(
                    "Def::Scalar shape is not a supported primitive/scalar",
                    shape,
                )
            }),
        }
    }

    fn build_known_scalar_ir(shape: &'static Shape) -> Option<TypeIR> {
        match shape.scalar_type()? {
            ScalarType::Bool => Some(TypeIR::bool()),
            ScalarType::Char | ScalarType::Str => Some(TypeIR::string()),
            ScalarType::F32 | ScalarType::F64 => Some(TypeIR::float()),
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
            | ScalarType::ISize => Some(TypeIR::int()),
            ScalarType::ConstTypeId => Some(TypeIR::string()),
            ScalarType::Unit => None,
            _ => match shape.type_identifier {
                "String" | "Cow<str>" | "Cow<'_, str>" | "Cow<'static, str>" => {
                    Some(TypeIR::string())
                }
                "SocketAddr" | "IpAddr" | "Ipv4Addr" | "Ipv6Addr" => Some(TypeIR::string()),
                _ => None,
            },
        }
    }

    /// Build TypeIR for primitive types.
    fn build_primitive_ir(
        &self,
        shape: &'static Shape,
        primitive: &facet::PrimitiveType,
    ) -> TypeIR {
        use facet::{NumericType, PrimitiveType, TextualType};

        match primitive {
            PrimitiveType::Boolean => TypeIR::bool(),
            PrimitiveType::Numeric(NumericType::Integer { .. }) => TypeIR::int(),
            PrimitiveType::Numeric(NumericType::Float) => TypeIR::float(),
            PrimitiveType::Textual(TextualType::Str) => TypeIR::string(),
            PrimitiveType::Textual(TextualType::Char) => TypeIR::string(),
            PrimitiveType::Never => Self::fail_unsupported_shape(
                "PrimitiveType::Never cannot be represented in BAML schema",
                shape,
            ),
        }
    }

    /// Build TypeIR for struct types, registering the class.
    fn build_struct_ir(
        &mut self,
        shape: &'static Shape,
        struct_type: &facet::StructType,
    ) -> TypeIR {
        let internal_name = self.generate_internal_name(shape);
        let display_name = rendered_name_for_shape(shape);
        let type_constraints = constraints_from_attrs(shape.attributes);

        // Register as visited BEFORE recursing (handles cycles).
        let mut type_ir = TypeIR::class(&internal_name);
        type_ir
            .meta_mut()
            .constraints
            .extend(type_constraints.clone());
        self.visited
            .insert(shape.id, (internal_name.clone(), type_ir.clone()));

        // Build fields.
        let mut fields = Vec::new();
        for field in struct_type.fields.iter() {
            if field.should_skip_deserializing() {
                continue;
            }

            let mut field_ir = self.build_field_type_ir(field, &internal_name, None, None);
            field_ir
                .meta_mut()
                .constraints
                .extend(constraints_from_attrs(field.attributes));
            if field.has_default() && !field_ir.is_optional() {
                field_ir = TypeIR::optional(field_ir);
            }

            let field_name = field.name.to_string();
            let rendered_name = field.effective_name().to_string();
            let description = doc_to_description(field.doc);
            let name = name_with_optional_alias(field_name, rendered_name);
            fields.push((name, field_ir, description, false));
        }

        let description = doc_to_description(shape.doc);

        let class = Class {
            name: name_with_optional_alias(internal_name.clone(), display_name),
            description,
            namespace: StreamingMode::NonStreaming,
            fields,
            constraints: type_constraints,
            streaming_behavior: Default::default(),
        };

        self.register_class(class);

        type_ir
    }

    /// Build TypeIR for enum types.
    fn build_enum_ir(&mut self, shape: &'static Shape, enum_type: &facet::EnumType) -> TypeIR {
        let internal_name = self.generate_internal_name(shape);
        let display_name = rendered_name_for_shape(shape);
        let type_constraints = constraints_from_attrs(shape.attributes);

        let is_data_enum = enum_type
            .variants
            .iter()
            .any(|variant| !variant.data.fields.is_empty());

        if !is_data_enum {
            if shape.is_untagged() {
                let literals: Vec<TypeIR> = enum_type
                    .variants
                    .iter()
                    .map(|variant| TypeIR::literal_string(variant.effective_name().to_string()))
                    .collect();

                let mut type_ir = TypeIR::union_with_meta(literals, type_meta::IR::default());
                type_ir
                    .meta_mut()
                    .constraints
                    .extend(type_constraints.clone());

                self.visited
                    .insert(shape.id, (internal_name.clone(), type_ir.clone()));
                return type_ir;
            }

            let mut type_ir = TypeIR::r#enum(&internal_name);
            type_ir
                .meta_mut()
                .constraints
                .extend(type_constraints.clone());

            self.visited
                .insert(shape.id, (internal_name.clone(), type_ir.clone()));

            let mut values = Vec::new();
            for variant in enum_type.variants.iter() {
                let variant_name = variant.name.to_string();
                let rendered_name = variant.effective_name().to_string();
                let description = doc_to_description(variant.doc);
                let name = name_with_optional_alias(variant_name, rendered_name);
                values.push((name, description));
            }

            let description = doc_to_description(shape.doc);
            let enm = Enum {
                name: name_with_optional_alias(internal_name.clone(), display_name),
                description,
                values,
                constraints: type_constraints,
            };

            self.registry.register_enum(enm);
            return type_ir;
        }

        // Data enum: represented as union of generated variant classes.
        let variant_class_names: Vec<String> = enum_type
            .variants
            .iter()
            .map(|variant| format!("{internal_name}__{}", variant.name))
            .collect();

        let union_variants: Vec<TypeIR> = variant_class_names.iter().map(TypeIR::class).collect();

        let mut type_ir = TypeIR::union_with_meta(union_variants, type_meta::IR::default());
        type_ir
            .meta_mut()
            .constraints
            .extend(type_constraints.clone());

        // Register as visited before variant field recursion for cycle handling.
        self.visited
            .insert(shape.id, (internal_name.clone(), type_ir.clone()));

        let tag_name = shape.get_tag_attr().unwrap_or("type").to_string();

        for variant in enum_type.variants.iter() {
            let variant_internal_name = format!("{internal_name}__{}", variant.name);
            let rendered_variant = variant.effective_name().to_string();
            let variant_rendered_name = format!("{display_name}_{rendered_variant}");
            let variant_description = doc_to_description(variant.doc);

            let mut fields = Vec::new();
            fields.push((
                Name::new(tag_name.clone()),
                TypeIR::literal_string(rendered_variant.clone()),
                None,
                false,
            ));

            for field in variant.data.fields.iter() {
                if field.should_skip_deserializing() {
                    continue;
                }

                let mut field_ir = self.build_field_type_ir(
                    field,
                    &internal_name,
                    Some(variant.name),
                    Some(variant.effective_name()),
                );
                field_ir
                    .meta_mut()
                    .constraints
                    .extend(constraints_from_attrs(field.attributes));
                if field.has_default() && !field_ir.is_optional() {
                    field_ir = TypeIR::optional(field_ir);
                }

                let field_name = field.name.to_string();
                let rendered_field = field.effective_name().to_string();
                let field_description = doc_to_description(field.doc);
                let field_name = name_with_optional_alias(field_name, rendered_field);
                fields.push((field_name, field_ir, field_description, false));
            }

            let class = Class {
                name: name_with_optional_alias(variant_internal_name, variant_rendered_name),
                description: variant_description,
                namespace: StreamingMode::NonStreaming,
                fields,
                constraints: Vec::new(),
                streaming_behavior: Default::default(),
            };

            self.register_class(class);
        }

        type_ir
    }

    fn build_field_type_ir(
        &mut self,
        field: &Field,
        owner_internal_name: &str,
        variant_name: Option<&str>,
        variant_rendered: Option<&str>,
    ) -> TypeIR {
        if let Some(with) = facet_ext::with_adapter_fns(field.attributes) {
            (with.register)(FieldCodecRegisterContext {
                registry: &mut self.registry,
                owner_internal_name: Some(owner_internal_name),
                field_name: Some(field.name),
                rendered_field_name: Some(field.effective_name()),
                variant_name,
                rendered_variant_name: variant_rendered,
            });
            return (with.type_ir)();
        }

        if let Some(int_repr) = field_int_repr(field) {
            return Self::build_int_repr_ir(field.shape(), int_repr);
        }

        if let Some(map_repr) = field_map_key_repr(field) {
            let entry_ctx = MapEntryContext {
                owner_internal_name: owner_internal_name.to_string(),
                field_name: field.name.to_string(),
                rendered_field: field.effective_name().to_string(),
                variant_name: variant_name.map(std::string::ToString::to_string),
                variant_rendered: variant_rendered.map(std::string::ToString::to_string),
            };
            return self.build_map_key_repr_ir(field.shape(), map_repr, Some(entry_ctx));
        }

        self.build_type_ir(field.shape())
    }

    fn build_int_repr_ir(shape: &'static Shape, repr: IntReprMode) -> TypeIR {
        match &shape.def {
            Def::Option(option_def) => {
                TypeIR::optional(Self::build_int_repr_ir(option_def.t, repr))
            }
            Def::List(list_def) => TypeIR::list(Self::build_int_repr_ir(list_def.t, repr)),
            Def::Array(arr_def) => TypeIR::list(Self::build_int_repr_ir(arr_def.t, repr)),
            Def::Pointer(ptr_def) => {
                if let Some(pointee) = ptr_def.pointee {
                    Self::build_int_repr_ir(pointee, repr)
                } else {
                    Self::fail_unsupported_shape(
                        "int_repr override encountered pointer shape without pointee",
                        shape,
                    )
                }
            }
            _ => match repr {
                IntReprMode::String => TypeIR::string(),
                IntReprMode::I64 => TypeIR::int(),
            },
        }
    }

    fn build_map_key_repr_ir(
        &mut self,
        shape: &'static Shape,
        repr: MapKeyReprMode,
        entry_ctx: Option<MapEntryContext>,
    ) -> TypeIR {
        match &shape.def {
            Def::Option(option_def) => {
                TypeIR::optional(self.build_map_key_repr_ir(option_def.t, repr, entry_ctx))
            }
            Def::List(list_def) => {
                TypeIR::list(self.build_map_key_repr_ir(list_def.t, repr, entry_ctx))
            }
            Def::Array(arr_def) => {
                TypeIR::list(self.build_map_key_repr_ir(arr_def.t, repr, entry_ctx))
            }
            Def::Pointer(ptr_def) => {
                if let Some(pointee) = ptr_def.pointee {
                    self.build_map_key_repr_ir(pointee, repr, entry_ctx)
                } else {
                    Self::fail_unsupported_shape(
                        "map_key_repr override encountered pointer shape without pointee",
                        shape,
                    )
                }
            }
            Def::Map(map_def) => match repr {
                MapKeyReprMode::String => {
                    let value_ir = self.build_type_ir(map_def.v);
                    TypeIR::map(TypeIR::string(), value_ir)
                }
                MapKeyReprMode::Pairs => {
                    let ctx = entry_ctx.unwrap_or_else(|| MapEntryContext {
                        owner_internal_name: "MapEntry".to_string(),
                        field_name: "entries".to_string(),
                        rendered_field: "entries".to_string(),
                        variant_name: None,
                        variant_rendered: None,
                    });

                    let (entry_internal_name, rendered_entry_name) = map_entry_names(&ctx);
                    self.ensure_map_entry_class(
                        &entry_internal_name,
                        Some(rendered_entry_name),
                        map_def.k,
                        map_def.v,
                    );

                    TypeIR::list(TypeIR::class(entry_internal_name))
                }
            },
            _ => self.build_type_ir(shape),
        }
    }

    fn ensure_map_entry_class(
        &mut self,
        internal_name: &str,
        rendered_name: Option<String>,
        key_shape: &'static Shape,
        value_shape: &'static Shape,
    ) {
        let key_ir = self.build_type_ir(key_shape);
        let value_ir = self.build_type_ir(value_shape);

        let class = Class {
            name: match rendered_name {
                Some(rendered) => name_with_optional_alias(internal_name.to_string(), rendered),
                None => Name::new(internal_name.to_string()),
            },
            description: None,
            namespace: StreamingMode::NonStreaming,
            fields: vec![
                (Name::new("key".to_string()), key_ir, None, false),
                (Name::new("value".to_string()), value_ir, None, false),
            ],
            constraints: Vec::new(),
            streaming_behavior: Default::default(),
        };

        self.register_class(class);
    }

    fn register_class(&mut self, class: Class) {
        self.registry.register_class(class);
    }

    /// Generate a unique internal name for a type.
    fn generate_internal_name(&mut self, shape: &'static Shape) -> String {
        let base = internal_name_for_shape(shape);

        if let Some(existing_shape_id) = self.used_internal_names.get(&base) {
            if *existing_shape_id == shape.id {
                return base;
            }

            let mut candidate;
            loop {
                self.name_counter += 1;
                candidate = format!("{base}__{}", self.name_counter);
                if !self.used_internal_names.contains_key(&candidate) {
                    break;
                }
            }
            self.used_internal_names.insert(candidate.clone(), shape.id);
            return candidate;
        }

        self.used_internal_names.insert(base.clone(), shape.id);
        base
    }

    /// Finalize and produce the OutputFormatContent.
    fn into_output_format(self, target: TypeIR) -> OutputFormatContent {
        self.registry.build(target)
    }
}

fn field_int_repr(field: &Field) -> Option<IntReprMode> {
    let repr = bamltype_int_repr(field.attributes)?;

    match repr {
        "string" => Some(IntReprMode::String),
        "i64" => Some(IntReprMode::I64),
        _ => None,
    }
}

fn field_map_key_repr(field: &Field) -> Option<MapKeyReprMode> {
    let repr = bamltype_map_key_repr(field.attributes)?;

    match repr {
        "string" => Some(MapKeyReprMode::String),
        "pairs" => Some(MapKeyReprMode::Pairs),
        _ => None,
    }
}

fn bamltype_internal_name(attrs: &'static [Attr]) -> Option<&'static str> {
    bamltype_attr_static_str(attrs, "internal_name")
}

fn bamltype_int_repr(attrs: &'static [Attr]) -> Option<&'static str> {
    bamltype_attr_static_str(attrs, "int_repr")
}

fn bamltype_map_key_repr(attrs: &'static [Attr]) -> Option<&'static str> {
    bamltype_attr_static_str(attrs, "map_key_repr")
}

fn bamltype_attr_static_str(attrs: &'static [Attr], key: &str) -> Option<&'static str> {
    for attr in attrs {
        if attr.ns != Some("bamltype") || attr.key != key {
            continue;
        }

        if let Some(value) = attr.get_as::<&'static str>() {
            return Some(*value);
        }
    }

    None
}

fn map_entry_names(ctx: &MapEntryContext) -> (String, String) {
    let suffix = match &ctx.variant_name {
        Some(variant_name) => format!("{variant_name}__{}__Entry", ctx.field_name),
        None => format!("{}__Entry", ctx.field_name),
    };

    let internal_name = format!("{}::{suffix}", ctx.owner_internal_name);

    let rendered = match &ctx.variant_rendered {
        Some(variant_rendered) => {
            format!("{variant_rendered}{}Entry", ctx.rendered_field)
        }
        None => format!("{}Entry", ctx.rendered_field),
    };

    (internal_name, rendered)
}

fn constraints_from_attrs(attrs: &'static [Attr]) -> Vec<Constraint> {
    let mut out = Vec::new();

    for attr in attrs {
        if attr.ns != Some("bamltype") {
            continue;
        }

        if attr.key == "check" {
            let Some(ext_attr) = attr.get_as::<facet_ext::Attr>() else {
                continue;
            };
            if let facet_ext::Attr::Check(payload) = ext_attr {
                out.push(Constraint::new_check(payload.label, payload.expr));
            }
        } else if attr.key == "assert" {
            let Some(ext_attr) = attr.get_as::<facet_ext::Attr>() else {
                continue;
            };
            if let facet_ext::Attr::Assert(payload) = ext_attr {
                out.push(Constraint::new_assert(payload.label, payload.expr));
            }
        }
    }

    out
}

fn doc_to_description(doc: &'static [&'static str]) -> Option<String> {
    if doc.is_empty() {
        return None;
    }

    Some(
        doc.iter()
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn name_with_optional_alias(real_name: String, rendered_name: String) -> Name {
    if real_name == rendered_name {
        Name::new(real_name)
    } else {
        Name::new_with_alias(real_name, Some(rendered_name))
    }
}
