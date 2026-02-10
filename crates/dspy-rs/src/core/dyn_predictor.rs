use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use bamltype::facet_reflect::Poke;
use facet::{ConstTypeId, Def, Facet, KnownPointer, Shape, Type, UserType};

use crate::{BamlValue, Example, PredictError, Predicted, SignatureSchema};

#[async_trait::async_trait]
pub trait DynPredictor: Send + Sync {
    fn schema(&self) -> &SignatureSchema;
    fn instruction(&self) -> String;
    fn set_instruction(&mut self, instruction: String);
    fn demos_as_examples(&self) -> Vec<Example>;
    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()>;
    fn dump_state(&self) -> PredictState;
    fn load_state(&mut self, state: PredictState) -> Result<()>;
    async fn forward_untyped(
        &self,
        input: BamlValue,
    ) -> std::result::Result<Predicted<BamlValue>, PredictError>;
}

#[derive(Clone, Debug, Default)]
pub struct PredictState {
    pub demos: Vec<Example>,
    pub instruction_override: Option<String>,
}

#[derive(Clone, Copy, Debug, facet::Facet)]
#[facet(opaque)]
pub struct PredictAccessorFns {
    pub accessor: fn(*mut ()) -> *mut dyn DynPredictor,
}

impl PartialEq for PredictAccessorFns {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.accessor, other.accessor)
    }
}

impl Eq for PredictAccessorFns {}

static ACCESSOR_REGISTRY: OnceLock<Mutex<HashMap<ConstTypeId, PredictAccessorFns>>> =
    OnceLock::new();

pub fn register_predict_accessor(
    shape: &'static Shape,
    accessor: fn(*mut ()) -> *mut dyn DynPredictor,
) {
    let registry = ACCESSOR_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = registry.lock().expect("predict accessor registry lock poisoned");
    guard
        .entry(shape.id)
        .or_insert(PredictAccessorFns { accessor });
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NamedParametersError {
    #[error("container `{ty}` at `{path}` contains a parameter leaf")]
    Container { path: String, ty: &'static str },
    #[error("parameter marker at `{path}` is missing a registered accessor")]
    MissingAttr { path: String },
}

#[tracing::instrument(
    level = "debug",
    name = "dsrs.named_parameters",
    skip(module),
)]
pub fn named_parameters<M>(
    module: &mut M,
) -> std::result::Result<Vec<(String, &mut dyn DynPredictor)>, NamedParametersError>
where
    M: for<'a> Facet<'a>,
{
    let mut raw_handles = Vec::<(String, *mut dyn DynPredictor)>::new();
    walk_value(Poke::new(module), "", &mut raw_handles)?;

    let mut handles = Vec::with_capacity(raw_handles.len());
    for (path, ptr) in raw_handles {
        // SAFETY: pointers are created from a single exclusive traversal over `module`.
        let handle = unsafe { &mut *ptr };
        handles.push((path, handle));
    }

    Ok(handles)
}

fn walk_value(
    mut value: Poke<'_, '_>,
    path: &str,
    out: &mut Vec<(String, *mut dyn DynPredictor)>,
) -> std::result::Result<(), NamedParametersError> {
    let shape = value.shape();
    if is_parameter_shape(shape) {
        let accessor = registered_accessor(shape).ok_or_else(|| NamedParametersError::MissingAttr {
            path: display_path(path),
        })?;
        let ptr = (accessor.accessor)(value.data_mut().as_mut_byte_ptr().cast::<()>());
        out.push((path.to_string(), ptr));
        return Ok(());
    }

    let mut struct_value = match value.into_struct() {
        Ok(struct_value) => struct_value,
        Err(_) => return Ok(()),
    };

    for idx in 0..struct_value.field_count() {
        let field = struct_value.ty().fields[idx];
        if field.should_skip_deserializing() {
            continue;
        }

        let field_path = push_field(path, field.name);
        if let Some(ty) = container_name(field.shape())
            && contains_parameter(field.shape(), &mut HashSet::new())
        {
            return Err(NamedParametersError::Container {
                path: field_path,
                ty,
            });
        }

        let child = struct_value.field(idx).map_err(|_| NamedParametersError::MissingAttr {
            path: display_path(&field_path),
        })?;
        walk_value(child, &field_path, out)?;
    }

    Ok(())
}

fn contains_parameter(shape: &'static Shape, visiting: &mut HashSet<ConstTypeId>) -> bool {
    if is_parameter_shape(shape) {
        return true;
    }

    if !visiting.insert(shape.id) {
        return false;
    }

    let found = match shape.ty {
        Type::User(UserType::Struct(struct_def)) => struct_def
            .fields
            .iter()
            .filter(|field| !field.should_skip_deserializing())
            .any(|field| contains_parameter(field.shape(), visiting)),
        _ => match shape.def {
            Def::List(def) => contains_parameter(def.t(), visiting),
            Def::Option(def) => contains_parameter(def.t(), visiting),
            Def::Map(def) => {
                contains_parameter(def.k(), visiting) || contains_parameter(def.v(), visiting)
            }
            Def::Array(def) => contains_parameter(def.t(), visiting),
            Def::Slice(def) => contains_parameter(def.t(), visiting),
            Def::Set(def) => contains_parameter(def.t(), visiting),
            Def::Result(def) => {
                contains_parameter(def.t(), visiting) || contains_parameter(def.e(), visiting)
            }
            Def::Pointer(def) => def.pointee().is_some_and(|inner| contains_parameter(inner, visiting)),
            _ => false,
        },
    };

    visiting.remove(&shape.id);
    found
}

fn container_name(shape: &'static Shape) -> Option<&'static str> {
    match shape.def {
        Def::List(_) => Some("Vec"),
        Def::Option(_) => Some("Option"),
        Def::Map(_) => Some("HashMap"),
        // Slice 5 guard: pointer-like wrappers cannot be safely traversed yet.
        Def::Pointer(def) => Some(match def.known {
            Some(KnownPointer::Box) => "Box",
            Some(KnownPointer::Rc) => "Rc",
            Some(KnownPointer::Arc) => "Arc",
            _ => "Pointer",
        }),
        _ => None,
    }
}

fn is_parameter_shape(shape: &'static Shape) -> bool {
    shape.type_identifier == "Predict"
}

fn registered_accessor(shape: &'static Shape) -> Option<PredictAccessorFns> {
    let registry = ACCESSOR_REGISTRY.get()?;
    let guard = registry.lock().ok()?;
    guard.get(&shape.id).copied()
}

fn push_field(path: &str, field: &str) -> String {
    if path.is_empty() {
        field.to_string()
    } else {
        format!("{path}.{field}")
    }
}

fn display_path(path: &str) -> String {
    if path.is_empty() {
        "<root>".to_string()
    } else {
        path.to_string()
    }
}
