use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use bamltype::facet_reflect::Peek;
use facet::{ConstTypeId, Def, Facet, KnownPointer, Shape, Type, UserType};

use crate::{Example, SignatureSchema};

/// Type-erased optimizer handle to a [`crate::Predict`] leaf.
///
/// Optimizers need to inspect and mutate Predict parameters (demos, instructions)
/// without knowing the concrete signature type. This trait bridges that gap. An
/// optimizer iterates over `(path, &mut dyn DynPredictor)` pairs from
/// [`named_parameters`] and works entirely through this interface:
///
/// ```
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// let mut predict = Predict::<QA>::new();
/// for (path, predictor) in named_parameters(&mut predict).unwrap() {
///     let demos = predictor.demos_as_examples();
///     predictor.set_instruction("Be concise.".into());
/// }
/// ```
///
/// Normal users never touch this — you pass your module to `optimizer.compile()`
/// and it uses `DynPredictor` internally.
///
pub trait DynPredictor: Send + Sync {
    /// Returns the [`SignatureSchema`] for this predictor's signature.
    fn schema(&self) -> &SignatureSchema;

    /// Returns the current instruction (override or default from the signature).
    fn instruction(&self) -> String;

    /// Overrides the instruction for this predictor.
    fn set_instruction(&mut self, instruction: String);

    /// Returns current demos as type-erased [`Example`]s.
    fn demos_as_examples(&self) -> Vec<Example>;

    /// Sets demos from type-erased [`Example`]s, converting to typed `Demo<S>` internally.
    ///
    /// # Errors
    ///
    /// Returns an error if any example can't be converted to the predictor's typed
    /// `Demo<S>` (schema mismatch).
    fn set_demos_from_examples(&mut self, demos: Vec<Example>) -> Result<()>;

    /// Snapshots the predictor's mutable state (demos + instruction override).
    fn dump_state(&self) -> PredictState;

    /// Restores predictor state from a snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the demos can't be converted to the predictor's typed format.
    fn load_state(&mut self, state: PredictState) -> Result<()>;
}

/// Serializable snapshot of a [`crate::Predict`]'s mutable state.
///
/// Contains demos (as type-erased [`Example`]s) and the instruction override.
/// Used by [`DynPredictor::dump_state`]/[`DynPredictor::load_state`] for
/// saving and restoring optimized parameters.
#[derive(Clone, Debug, Default)]
pub struct PredictState {
    /// The demos as type-erased examples.
    pub demos: Vec<Example>,
    /// The instruction override, if any.
    pub instruction_override: Option<String>,
}

#[derive(Clone, Copy, Debug, facet::Facet)]
#[facet(opaque)]
pub struct PredictAccessorFns {
    pub accessor_mut: fn(*mut ()) -> *mut dyn DynPredictor,
}

impl PartialEq for PredictAccessorFns {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.accessor_mut, other.accessor_mut)
    }
}

impl Eq for PredictAccessorFns {}

// FIXME(dsrs-s2): Temporary bridge for S2 until Facet supports shape-local typed attr payloads
// on generic containers (e.g. Predict<S>) without E0401 in macro-generated statics.
// Intended solution:
// 1. Read `PredictAccessorFns` directly from shape-local attrs on the discovered leaf shape.
// 2. Delete this global registry and stop requiring explicit runtime registration.
// Upstream tracking:
// - Issue: https://github.com/facet-rs/facet/issues/2039
// - PR: https://github.com/facet-rs/facet/pull/2040
// - PR: https://github.com/facet-rs/facet/pull/2041
// TODO(post-v6): Remove registry fallback once upstream lands and DSRs upgrades facet.
static ACCESSOR_REGISTRY: OnceLock<Mutex<HashMap<ConstTypeId, PredictAccessorFns>>> =
    OnceLock::new();

fn accessor_registry() -> &'static Mutex<HashMap<ConstTypeId, PredictAccessorFns>> {
    ACCESSOR_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn register_predict_accessor(
    shape: &'static Shape,
    accessor_mut: fn(*mut ()) -> *mut dyn DynPredictor,
) {
    let registration = PredictAccessorFns { accessor_mut };
    let mut guard = accessor_registry()
        .lock()
        .expect("predict accessor registry lock poisoned");
    if let Some(existing) = guard.get(&shape.id) {
        assert_eq!(
            *existing, registration,
            "conflicting predict accessor registration for shape id={:?} type_identifier={}",
            shape.id,
            shape.type_identifier
        );
        return;
    }
    guard.insert(shape.id, registration);
}

/// Error from [`named_parameters`] when the Facet walker encounters an unsupported structure.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NamedParametersError {
    /// A `Predict` leaf was found inside an unsupported container (`Rc`, `Arc`, etc.).
    #[error("container `{ty}` at `{path}` contains a parameter leaf")]
    Container { path: String, ty: &'static str },

    /// A `Predict`-like leaf was found but hasn't registered its accessor functions.
    /// This means `Predict::new()` or `Predict::builder().build()` was never called for
    /// this concrete `Predict<S>`.
    // NOTE(dsrs-s2): Error message will simplify once Facet supports shape-local
    // accessor payloads and the global registry workaround is removed.
    #[error(
        "parameter-like leaf at `{path}` has no registered accessor — was `Predict::new()` or `.build()` called for this concrete type?"
    )]
    MissingAttr { path: String },
}

/// Discovers all [`crate::Predict`] leaves in a module by walking its struct fields.
///
/// Returns `(dotted_path, &mut dyn DynPredictor)` pairs. Paths reflect the field
/// hierarchy: a `ChainOfThought` inside field `answer` yields `"answer.predictor"`.
///
/// Takes exclusive `&mut` — you can't `call()` the module during discovery. This is
/// intentional: optimization needs to mutate state without races.
///
/// The walker follows struct fields and common containers (`Option`, `Vec`,
/// `HashMap<String, _>`, `Box`). It does not follow `Rc`, `Arc`, or other smart
/// pointers — those error explicitly. If a `Predict` leaf exists but wasn't
/// constructed via `new()`/`build()`, you get [`NamedParametersError::MissingAttr`]
/// (the accessor wasn't registered — see [`crate::Predict`] doc on construction).
///
/// # Errors
///
/// - [`Container`](NamedParametersError::Container): `Predict` inside unsupported container
/// - [`MissingAttr`](NamedParametersError::MissingAttr): `Predict` without registered accessor
///
/// ```
/// use dspy_rs::*;
/// use dspy_rs::doctest::*;
///
/// let mut predict = Predict::<QA>::new();
/// for (path, predictor) in named_parameters(&mut predict).unwrap() {
///     println!("{}: {} demos", path, predictor.demos_as_examples().len());
/// }
/// ```
#[tracing::instrument(level = "debug", name = "dsrs.named_parameters", skip(module))]
pub fn named_parameters<M>(
    module: &mut M,
) -> std::result::Result<Vec<(String, &mut dyn DynPredictor)>, NamedParametersError>
where
    M: for<'a> Facet<'a>,
{
    let mut raw_handles = Vec::<(String, *mut dyn DynPredictor)>::new();
    walk_value::<MutableAccess>(Peek::new(&*module), "", &mut raw_handles)?;

    let mut handles = Vec::with_capacity(raw_handles.len());
    for (path, ptr) in raw_handles {
        // SAFETY: pointers are created from a single exclusive traversal over `module`.
        let handle = unsafe { &mut *ptr };
        handles.push((path, handle));
    }

    Ok(handles)
}

trait WalkAccess {
    type RawPtr;

    fn pointer(accessor: PredictAccessorFns, value: Peek<'_, '_>) -> Self::RawPtr;
}

struct MutableAccess;

impl WalkAccess for MutableAccess {
    type RawPtr = *mut dyn DynPredictor;

    fn pointer(accessor: PredictAccessorFns, value: Peek<'_, '_>) -> Self::RawPtr {
        // SAFETY: `named_parameters` has exclusive access to `module` for the full traversal.
        // We only cast to a mutable pointer after the read-only walk has located the leaf.
        (accessor.accessor_mut)((value.data().as_byte_ptr() as *mut u8).cast::<()>())
    }
}

fn walk_value<Access: WalkAccess>(
    value: Peek<'_, '_>,
    path: &str,
    out: &mut Vec<(String, Access::RawPtr)>,
) -> std::result::Result<(), NamedParametersError> {
    let shape = value.shape();
    if let Some(accessor) = lookup_registered_predict_accessor(shape) {
        out.push((path.to_string(), Access::pointer(accessor, value)));
        return Ok(());
    }
    if is_predict_type_name(shape) {
        return Err(NamedParametersError::MissingAttr {
            path: display_path(path),
        });
    }

    if matches!(shape.ty, Type::User(UserType::Struct(_))) {
        let struct_value = value.into_struct().expect("shape says struct");
        for idx in 0..struct_value.field_count() {
            let field = struct_value.ty().fields[idx];
            if field.should_skip_deserializing() {
                continue;
            }

            let field_path = push_field(path, field.name);
            let child = struct_value
                .field(idx)
                .map_err(|_| NamedParametersError::MissingAttr {
                    path: display_path(&field_path),
                })?;
            walk_value::<Access>(child, &field_path, out)?;
        }
        return Ok(());
    }

    match shape.def {
        Def::Option(_) => {
            if let Some(inner) = value.into_option().expect("shape says option").value() {
                walk_value::<Access>(inner, path, out)?;
            }
            Ok(())
        }
        Def::List(_) | Def::Array(_) | Def::Slice(_) => {
            for (idx, child) in value
                .into_list_like()
                .expect("shape says list-like")
                .iter()
                .enumerate()
            {
                let child_path = push_index(path, idx);
                walk_value::<Access>(child, &child_path, out)?;
            }
            Ok(())
        }
        Def::Map(_) => {
            let mut entries = value
                .into_map()
                .expect("shape says map")
                .iter()
                .map(|(key, value)| {
                    key.as_str().map(|name| (name.to_string(), value)).ok_or(
                        NamedParametersError::Container {
                            path: display_path(path),
                            ty: "HashMap",
                        },
                    )
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;

            entries.sort_by(|(left, _), (right, _)| left.as_bytes().cmp(right.as_bytes()));
            for (key, child) in entries {
                let child_path = push_map_key(path, &key);
                walk_value::<Access>(child, &child_path, out)?;
            }
            Ok(())
        }
        Def::Pointer(pointer_def) => match pointer_def.known {
            Some(KnownPointer::Box) => {
                if let Some(inner) = value
                    .into_pointer()
                    .expect("shape says pointer")
                    .borrow_inner()
                {
                    walk_value::<Access>(inner, path, out)?;
                }
                Ok(())
            }
            _ => {
                if contains_parameter(shape, &mut HashSet::new()) {
                    return Err(NamedParametersError::Container {
                        path: display_path(path),
                        ty: pointer_name(pointer_def.known),
                    });
                }
                Ok(())
            }
        },
        _ => Ok(()),
    }
}

fn contains_parameter(shape: &'static Shape, visiting: &mut HashSet<ConstTypeId>) -> bool {
    if lookup_registered_predict_accessor(shape).is_some() || is_predict_type_name(shape) {
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
            Def::Pointer(def) => def
                .pointee()
                .is_some_and(|inner| contains_parameter(inner, visiting)),
            _ => false,
        },
    };

    visiting.remove(&shape.id);
    found
}

fn is_predict_type_name(shape: &'static Shape) -> bool {
    // Temporary diagnostic-only guard: we never use this for successful dispatch.
    // Success requires a registered accessor; this path exists to fail loudly when
    // a Predict-like leaf appears without registration.
    shape.type_identifier == "Predict"
}

fn lookup_registered_predict_accessor(shape: &'static Shape) -> Option<PredictAccessorFns> {
    let registry = ACCESSOR_REGISTRY.get()?;
    let guard = registry
        .lock()
        .expect("predict accessor registry lock poisoned");
    guard.get(&shape.id).copied()
}

fn push_field(path: &str, field: &str) -> String {
    if path.is_empty() {
        field.to_string()
    } else {
        format!("{path}.{field}")
    }
}

fn push_index(path: &str, index: usize) -> String {
    if path.is_empty() {
        format!("[{index}]")
    } else {
        format!("{path}[{index}]")
    }
}

fn push_map_key(path: &str, key: &str) -> String {
    let escaped = escape_map_key(key);
    if path.is_empty() {
        format!("['{escaped}']")
    } else {
        format!("{path}['{escaped}']")
    }
}

fn escape_map_key(key: &str) -> String {
    let mut escaped = String::with_capacity(key.len());
    for ch in key.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '\'' => escaped.push_str("\\'"),
            c if c.is_control() => escaped.push_str(&format!("\\u{{{:X}}}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

fn display_path(path: &str) -> String {
    if path.is_empty() {
        "<root>".to_string()
    } else {
        path.to_string()
    }
}

fn pointer_name(pointer: Option<KnownPointer>) -> &'static str {
    match pointer {
        Some(KnownPointer::Box) => "Box",
        Some(KnownPointer::Rc) => "Rc",
        Some(KnownPointer::Arc) => "Arc",
        _ => "Pointer",
    }
}
