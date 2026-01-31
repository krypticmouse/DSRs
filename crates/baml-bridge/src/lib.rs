pub use baml_types;
pub use internal_baml_jinja;
pub use internal_baml_jinja::types::{HoistClasses, MapStyle, RenderOptions};
pub use jsonish;

#[cfg(feature = "derive")]
pub use baml_bridge_derive::BamlType;

#[cfg(feature = "pyo3")]
pub mod py;

pub mod prompt;

mod convert;
mod registry;
mod to_value;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
    sync::Arc,
};

use baml_types::{BamlValue, Constraint, ConstraintLevel, ResponseCheck, TypeIR};
use internal_baml_jinja::types::OutputFormatContent;
use jsonish::deserializer::{
    coercer::{run_user_checks, ParsingError},
    deserialize_flags::{DeserializerConditions, Flag},
    types::BamlValueWithFlags,
};
use sha2::{Digest, Sha256};

pub use convert::{get_field, BamlConvertError, BamlValueConvert};
pub use registry::{default_streaming_behavior, Registry};
pub use prompt::*;
pub use to_value::ToBamlValue;

pub trait BamlTypeInternal {
    fn baml_internal_name() -> &'static str;
    fn baml_type_ir() -> TypeIR;
    fn register(reg: &mut Registry);
}

pub trait BamlType: BamlTypeInternal + BamlValueConvert + Sized + 'static {
    fn baml_internal_name() -> &'static str {
        <Self as BamlTypeInternal>::baml_internal_name()
    }

    fn baml_type_ir() -> TypeIR {
        <Self as BamlTypeInternal>::baml_type_ir()
    }

    fn baml_output_format() -> &'static OutputFormatContent;

    fn try_from_baml_value(value: BamlValue) -> Result<Self, BamlConvertError> {
        <Self as BamlValueConvert>::try_from_baml_value(value, Vec::new())
    }
}

pub trait BamlAdapter<T> {
    fn type_ir() -> TypeIR;
    fn register(_reg: &mut Registry) {}
    fn try_from_baml(value: BamlValue, path: Vec<String>) -> Result<T, BamlConvertError>;
}

#[derive(Debug, Clone)]
pub struct Parsed<T> {
    pub value: T,
    pub baml_value: BamlValue,
    pub flags: Vec<Flag>,
    pub checks: Vec<ResponseCheck>,
    pub explanations: Vec<ParsingError>,
}

#[derive(Debug, thiserror::Error)]
pub enum BamlParseError {
    #[error("jsonish parse error: {0}")]
    Jsonish(#[from] anyhow::Error),
    #[error("constraint asserts failed")]
    ConstraintAssertsFailed { failed: Vec<ResponseCheck> },
    #[error("conversion error: {0}")]
    Convert(#[from] BamlConvertError),
}

pub fn render_schema<T: BamlType>(
    options: RenderOptions,
) -> Result<Option<String>, minijinja::Error> {
    T::baml_output_format().render(options)
}

pub fn parse_llm_output<T: BamlType>(
    raw: &str,
    is_done: bool,
) -> Result<Parsed<T>, BamlParseError> {
    let output_format = T::baml_output_format();
    let parsed = match jsonish::from_str(output_format, &output_format.target, raw, is_done) {
        Ok(parsed) => parsed,
        Err(err) => {
            if has_assert_failure(&err) {
                let failed = collect_assert_constraints(output_format);
                return Err(BamlParseError::ConstraintAssertsFailed { failed });
            }
            return Err(BamlParseError::Jsonish(err));
        }
    };

    let baml_value_with_meta: baml_types::BamlValueWithMeta<baml_types::TypeIR> =
        parsed.clone().into();
    let baml_value: BamlValue = baml_value_with_meta.into();

    let value = <T as BamlType>::try_from_baml_value(baml_value.clone())?;

    let mut flags = Vec::new();
    collect_flags(&parsed, &mut flags);

    let mut checks = Vec::new();
    collect_checks(&parsed, &mut checks)?;

    let mut explanations = Vec::new();
    parsed.explanation_impl(vec!["<root>".to_string()], &mut explanations);

    let mut failed_asserts = Vec::new();
    collect_assert_failures(&parsed, &mut failed_asserts)?;
    if !failed_asserts.is_empty() {
        return Err(BamlParseError::ConstraintAssertsFailed {
            failed: failed_asserts,
        });
    }

    Ok(Parsed {
        value,
        baml_value,
        flags,
        checks,
        explanations,
    })
}

pub fn with_constraints(mut r#type: TypeIR, constraints: Vec<Constraint>) -> TypeIR {
    r#type.meta_mut().constraints.extend(constraints);
    r#type
}

pub fn schema_fingerprint(
    output_format: &OutputFormatContent,
    options: RenderOptions,
) -> Result<String, minijinja::Error> {
    let rendered = output_format.render(options)?.unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(rendered.as_bytes());
    hasher.update(output_format.target.to_string().as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

fn collect_flags_recursive(value: &BamlValueWithFlags, flags: &mut Vec<Flag>) {
    match value {
        BamlValueWithFlags::String(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Int(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Float(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Bool(v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Enum(_, _, v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::Media(_, v) => {
            collect_from_conditions(&v.flags, flags);
        }
        BamlValueWithFlags::List(conds, _, items) => {
            collect_from_conditions(conds, flags);
            for item in items {
                collect_flags_recursive(item, flags);
            }
        }
        BamlValueWithFlags::Map(conds, _, items) => {
            collect_from_conditions(conds, flags);
            for (_, (entry_flags, entry_value)) in items {
                collect_from_conditions(entry_flags, flags);
                collect_flags_recursive(entry_value, flags);
            }
        }
        BamlValueWithFlags::Class(_, conds, _, fields) => {
            collect_from_conditions(conds, flags);
            for (_, field_value) in fields {
                collect_flags_recursive(field_value, flags);
            }
        }
        BamlValueWithFlags::Null(_, conds) => {
            collect_from_conditions(conds, flags);
        }
    }
}

fn collect_from_conditions(conditions: &DeserializerConditions, flags: &mut Vec<Flag>) {
    flags.extend(conditions.flags.iter().cloned());
}

fn collect_flags(value: &BamlValueWithFlags, flags: &mut Vec<Flag>) {
    collect_flags_recursive(value, flags);
}

fn collect_checks(
    value: &BamlValueWithFlags,
    checks: &mut Vec<ResponseCheck>,
) -> Result<(), BamlParseError> {
    let baml_value = baml_value_from_flags(value);
    let results = run_user_checks(&baml_value, value.field_type()).map_err(BamlParseError::from)?;

    for result in results {
        if let Some(check) = ResponseCheck::from_check_result(result) {
            checks.push(check);
        }
    }

    match value {
        BamlValueWithFlags::List(_, _, items) => {
            for item in items {
                collect_checks(item, checks)?;
            }
        }
        BamlValueWithFlags::Map(_, _, items) => {
            for (_, (_, entry_value)) in items {
                collect_checks(entry_value, checks)?;
            }
        }
        BamlValueWithFlags::Class(_, _, _, fields) => {
            for (_, field_value) in fields {
                collect_checks(field_value, checks)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn collect_assert_failures(
    value: &BamlValueWithFlags,
    failed: &mut Vec<ResponseCheck>,
) -> Result<(), BamlParseError> {
    let baml_value = baml_value_from_flags(value);
    let results = run_user_checks(&baml_value, value.field_type()).map_err(BamlParseError::from)?;

    for (constraint, ok) in results {
        if constraint.level == ConstraintLevel::Assert && !ok {
            failed.push(ResponseCheck {
                name: constraint.label.unwrap_or_else(|| "assert".to_string()),
                expression: constraint.expression.0,
                status: "failed".to_string(),
            });
        }
    }

    match value {
        BamlValueWithFlags::List(_, _, items) => {
            for item in items {
                collect_assert_failures(item, failed)?;
            }
        }
        BamlValueWithFlags::Map(_, _, items) => {
            for (_, (_, entry_value)) in items {
                collect_assert_failures(entry_value, failed)?;
            }
        }
        BamlValueWithFlags::Class(_, _, _, fields) => {
            for (_, field_value) in fields {
                collect_assert_failures(field_value, failed)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn collect_assert_constraints(of: &OutputFormatContent) -> Vec<ResponseCheck> {
    let mut failed = Vec::new();
    let mut seen = HashSet::new();

    collect_assert_constraints_in_type(&of.target, &mut failed, &mut seen);

    for class in of.classes.values() {
        for constraint in &class.constraints {
            push_assert_constraint(constraint, &mut failed, &mut seen);
        }
        for (_, field_type, _, _, _) in &class.fields {
            collect_assert_constraints_in_type(field_type, &mut failed, &mut seen);
        }
    }

    for r#enum in of.enums.values() {
        for constraint in &r#enum.constraints {
            push_assert_constraint(constraint, &mut failed, &mut seen);
        }
    }

    failed
}

fn collect_assert_constraints_in_type(
    r#type: &TypeIR,
    failed: &mut Vec<ResponseCheck>,
    seen: &mut HashSet<(String, String)>,
) {
    for constraint in &r#type.meta().constraints {
        push_assert_constraint(constraint, failed, seen);
    }

    match r#type {
        TypeIR::List(inner, _) => collect_assert_constraints_in_type(inner, failed, seen),
        TypeIR::Map(key, value, _) => {
            collect_assert_constraints_in_type(key, failed, seen);
            collect_assert_constraints_in_type(value, failed, seen);
        }
        TypeIR::Union(union, _) => {
            for item in union.iter_include_null() {
                collect_assert_constraints_in_type(item, failed, seen);
            }
        }
        TypeIR::Tuple(items, _) => {
            for item in items {
                collect_assert_constraints_in_type(item, failed, seen);
            }
        }
        TypeIR::Arrow(arrow, _) => {
            for param in &arrow.param_types {
                collect_assert_constraints_in_type(param, failed, seen);
            }
            collect_assert_constraints_in_type(&arrow.return_type, failed, seen);
        }
        TypeIR::Top(_)
        | TypeIR::Primitive(..)
        | TypeIR::Enum { .. }
        | TypeIR::Literal(..)
        | TypeIR::Class { .. }
        | TypeIR::RecursiveTypeAlias { .. } => {}
    }
}

fn push_assert_constraint(
    constraint: &Constraint,
    failed: &mut Vec<ResponseCheck>,
    seen: &mut HashSet<(String, String)>,
) {
    if constraint.level != ConstraintLevel::Assert {
        return;
    }

    let name = constraint
        .label
        .clone()
        .unwrap_or_else(|| "assert".to_string());
    let expr = constraint.expression.0.clone();
    if seen.insert((name.clone(), expr.clone())) {
        failed.push(ResponseCheck {
            name,
            expression: expr,
            status: "failed".to_string(),
        });
    }
}

fn has_assert_failure(err: &anyhow::Error) -> bool {
    err.to_string().contains("Assertions failed.")
}

fn baml_value_from_flags(value: &BamlValueWithFlags) -> BamlValue {
    match value {
        BamlValueWithFlags::String(v) => BamlValue::String(v.value.clone()),
        BamlValueWithFlags::Int(v) => BamlValue::Int(v.value),
        BamlValueWithFlags::Float(v) => BamlValue::Float(v.value),
        BamlValueWithFlags::Bool(v) => BamlValue::Bool(v.value),
        BamlValueWithFlags::Enum(name, _, v) => BamlValue::Enum(name.clone(), v.value.clone()),
        BamlValueWithFlags::Media(_, v) => BamlValue::Media(v.value.clone()),
        BamlValueWithFlags::List(_, _, items) => {
            BamlValue::List(items.iter().map(baml_value_from_flags).collect())
        }
        BamlValueWithFlags::Map(_, _, items) => BamlValue::Map(
            items
                .iter()
                .map(|(k, (_, v))| (k.clone(), baml_value_from_flags(v)))
                .collect(),
        ),
        BamlValueWithFlags::Class(name, _, _, fields) => BamlValue::Class(
            name.clone(),
            fields
                .iter()
                .map(|(k, v)| (k.clone(), baml_value_from_flags(v)))
                .collect(),
        ),
        BamlValueWithFlags::Null(_, _) => BamlValue::Null,
    }
}

impl BamlTypeInternal for String {
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::string()
    }

    fn register(_reg: &mut Registry) {}
}

impl BamlTypeInternal for bool {
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::bool()
    }

    fn register(_reg: &mut Registry) {}
}

impl BamlTypeInternal for f64 {
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::float()
    }

    fn register(_reg: &mut Registry) {}
}

impl BamlTypeInternal for f32 {
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::float()
    }

    fn register(_reg: &mut Registry) {}
}

impl BamlTypeInternal for i64 {
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::int()
    }

    fn register(_reg: &mut Registry) {}
}

macro_rules! impl_baml_int_internal {
    ($ty:ty) => {
        impl BamlTypeInternal for $ty {
            fn baml_internal_name() -> &'static str {
                std::any::type_name::<Self>()
            }

            fn baml_type_ir() -> TypeIR {
                TypeIR::int()
            }

            fn register(_reg: &mut Registry) {}
        }
    };
}

impl_baml_int_internal!(i8);
impl_baml_int_internal!(i16);
impl_baml_int_internal!(i32);
impl_baml_int_internal!(isize);
impl_baml_int_internal!(u8);
impl_baml_int_internal!(u16);
impl_baml_int_internal!(u32);

impl<T> BamlTypeInternal for Option<T>
where
    T: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::optional(T::baml_type_ir())
    }

    fn register(reg: &mut Registry) {
        T::register(reg);
    }
}

impl<T> BamlTypeInternal for Vec<T>
where
    T: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::list(T::baml_type_ir())
    }

    fn register(reg: &mut Registry) {
        T::register(reg);
    }
}

impl<T> BamlTypeInternal for Box<T>
where
    T: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        T::baml_type_ir()
    }

    fn register(reg: &mut Registry) {
        T::register(reg);
    }
}

impl<T> BamlTypeInternal for Arc<T>
where
    T: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        T::baml_type_ir()
    }

    fn register(reg: &mut Registry) {
        T::register(reg);
    }
}

impl<T> BamlTypeInternal for Rc<T>
where
    T: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        T::baml_type_ir()
    }

    fn register(reg: &mut Registry) {
        T::register(reg);
    }
}

impl<V> BamlTypeInternal for HashMap<String, V>
where
    V: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::map(TypeIR::string(), V::baml_type_ir())
    }

    fn register(reg: &mut Registry) {
        V::register(reg);
    }
}

impl<V> BamlTypeInternal for BTreeMap<String, V>
where
    V: BamlTypeInternal,
{
    fn baml_internal_name() -> &'static str {
        std::any::type_name::<Self>()
    }

    fn baml_type_ir() -> TypeIR {
        TypeIR::map(TypeIR::string(), V::baml_type_ir())
    }

    fn register(reg: &mut Registry) {
        V::register(reg);
    }
}
