use std::collections::HashSet;
use std::ops::ControlFlow;

use anyhow::Result;
use bamltype::facet_reflect::Peek;
use facet::{ConstTypeId, Def, Facet, KnownPointer, Shape, Type, UserType};

use crate::SignatureSchema;
use crate::data::example::Example as RawExample;

/// Type-erased optimizer handle to a [`crate::Predict`] leaf.
///
/// Optimizers need to inspect and mutate Predict parameters (demos, instructions)
/// without knowing the concrete signature type. Discovery uses
/// [`visit_named_predictors_mut`], which walks the module tree and passes each
/// discovered `(path, &mut dyn DynPredictor)` leaf to a selector callback.
///
/// Normal users never touch this — you pass your module to `optimizer.compile()`
/// and it uses `DynPredictor` internally.
pub(crate) trait DynPredictor: Send + Sync {
    /// Returns the [`SignatureSchema`] for this predictor's signature.
    fn schema(&self) -> &SignatureSchema;

    /// Returns the current instruction (override or default from the signature).
    fn instruction(&self) -> String;

    /// Overrides the instruction for this predictor.
    fn set_instruction(&mut self, instruction: String);

    /// Returns current demos as type-erased [`Example`]s.
    fn demos_as_examples(&self) -> Vec<RawExample>;

    /// Sets demos from type-erased [`Example`]s, converting to typed `Example<S>` internally.
    ///
    /// # Errors
    ///
    /// Returns an error if any example can't be converted to the predictor's typed
    /// `Example<S>` (schema mismatch).
    fn set_demos_from_examples(&mut self, demos: Vec<RawExample>) -> Result<()>;

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
pub(crate) struct PredictState {
    /// The demos as type-erased examples.
    pub demos: Vec<RawExample>,
    /// The instruction override, if any.
    pub instruction_override: Option<String>,
}

type VisitMutFn =
    fn(*mut (), &mut dyn FnMut(&mut dyn DynPredictor) -> ControlFlow<()>) -> ControlFlow<()>;

#[derive(Clone, Copy, Debug, facet::Facet)]
#[facet(opaque)]
pub(crate) struct PredictAccessorFns {
    pub visit_mut: VisitMutFn,
}

impl PartialEq for PredictAccessorFns {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.visit_mut, other.visit_mut)
    }
}

impl Eq for PredictAccessorFns {}

facet::define_attr_grammar! {
    ns "dsrs";
    crate_path $crate::core::dyn_predictor;

    pub enum Attr {
        PredictAccessor(Option<&'static PredictAccessorFns>),
    }
}

/// Error from [`visit_named_predictors_mut`] when the Facet walker encounters an unsupported structure.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum NamedParametersError {
    /// A `Predict` leaf was found inside an unsupported container (`Rc`, `Arc`, etc.).
    #[error("container `{ty}` at `{path}` contains a parameter leaf")]
    Container { path: String, ty: &'static str },

    /// A `Predict`-like leaf was found with missing or malformed shape-local accessor payload.
    #[error(
        "parameter-like leaf at `{path}` is missing a valid shape-local accessor payload (`#[facet(dsrs::predict_accessor = ...)]`)"
    )]
    MissingAttr { path: String },
}

/// Visits all [`crate::Predict`] leaves in a module by walking struct fields and
/// supported containers.
///
/// The callback acts as a selector: it receives each `(dotted_path, predictor)` pair
/// and may return `ControlFlow::Break(())` to stop traversal early.
///
/// Safety model:
/// - discovery has exclusive `&mut` access to `module` for the full traversal;
/// - leaf access requires a valid shape-local accessor payload attached to the leaf;
/// - unsupported shared-pointer containers (`Rc`, `Arc`) are rejected explicitly.
#[tracing::instrument(
    level = "debug",
    name = "dsrs.visit_named_predictors_mut",
    skip(module, visitor)
)]
pub(crate) fn visit_named_predictors_mut<M, F>(
    module: &mut M,
    mut visitor: F,
) -> std::result::Result<(), NamedParametersError>
where
    M: for<'a> Facet<'a>,
    F: FnMut(&str, &mut dyn DynPredictor) -> ControlFlow<()>,
{
    let _ = walk_value(Peek::new(&*module), "", &mut visitor)?;
    Ok(())
}

fn walk_value<F>(
    value: Peek<'_, '_>,
    path: &str,
    visitor: &mut F,
) -> std::result::Result<ControlFlow<()>, NamedParametersError>
where
    F: FnMut(&str, &mut dyn DynPredictor) -> ControlFlow<()>,
{
    let shape = value.shape();
    match resolve_predict_leaf(shape) {
        PredictLeafResolution::Accessor(accessor) => {
            let raw_ptr = (value.data().as_byte_ptr() as *mut u8).cast::<()>();
            let mut forward = |predictor: &mut dyn DynPredictor| visitor(path, predictor);
            return Ok((accessor.visit_mut)(raw_ptr, &mut forward));
        }
        PredictLeafResolution::Missing => {
            return Err(NamedParametersError::MissingAttr {
                path: display_path(path),
            });
        }
        PredictLeafResolution::NotLeaf => {}
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
            if let ControlFlow::Break(()) = walk_value(child, &field_path, visitor)? {
                return Ok(ControlFlow::Break(()));
            }
        }
        return Ok(ControlFlow::Continue(()));
    }

    match shape.def {
        Def::Option(_) => {
            if let Some(inner) = value.into_option().expect("shape says option").value()
                && let ControlFlow::Break(()) = walk_value(inner, path, visitor)?
            {
                return Ok(ControlFlow::Break(()));
            }
            Ok(ControlFlow::Continue(()))
        }
        Def::List(_) | Def::Array(_) | Def::Slice(_) => {
            for (idx, child) in value
                .into_list_like()
                .expect("shape says list-like")
                .iter()
                .enumerate()
            {
                let child_path = push_index(path, idx);
                if let ControlFlow::Break(()) = walk_value(child, &child_path, visitor)? {
                    return Ok(ControlFlow::Break(()));
                }
            }
            Ok(ControlFlow::Continue(()))
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
                if let ControlFlow::Break(()) = walk_value(child, &child_path, visitor)? {
                    return Ok(ControlFlow::Break(()));
                }
            }
            Ok(ControlFlow::Continue(()))
        }
        Def::Pointer(pointer_def) => match pointer_def.known {
            Some(KnownPointer::Box) => {
                if let Some(inner) = value
                    .into_pointer()
                    .expect("shape says pointer")
                    .borrow_inner()
                    && let ControlFlow::Break(()) = walk_value(inner, path, visitor)?
                {
                    return Ok(ControlFlow::Break(()));
                }
                Ok(ControlFlow::Continue(()))
            }
            _ => {
                // TODO(dsrs-shared-ptr-policy): define safe mutable-handle policy for Arc/Rc traversal.
                if contains_parameter(shape, &mut HashSet::new()) {
                    return Err(NamedParametersError::Container {
                        path: display_path(path),
                        ty: pointer_name(pointer_def.known),
                    });
                }
                Ok(ControlFlow::Continue(()))
            }
        },
        _ => Ok(ControlFlow::Continue(())),
    }
}

fn contains_parameter(shape: &'static Shape, visiting: &mut HashSet<ConstTypeId>) -> bool {
    if !matches!(resolve_predict_leaf(shape), PredictLeafResolution::NotLeaf) {
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

enum PredictLeafResolution {
    NotLeaf,
    Accessor(PredictAccessorFns),
    Missing,
}

fn resolve_predict_leaf(shape: &'static Shape) -> PredictLeafResolution {
    let has_leaf_marker = is_predict_shape_identity(shape);
    let mut accessor_count = 0usize;
    let mut accessor = None;
    let mut invalid = false;

    for attr in shape.attributes {
        if attr.ns != Some("dsrs") {
            continue;
        }

        if attr.key == "predict_accessor" {
            accessor_count += 1;
            match attr.get_as::<Attr>() {
                Some(Attr::PredictAccessor(Some(value))) => {
                    if accessor.is_some() {
                        invalid = true;
                    } else {
                        accessor = Some(**value);
                    }
                }
                _ => invalid = true,
            }
        }
    }

    if !has_leaf_marker {
        if accessor_count > 0 {
            return PredictLeafResolution::Missing;
        }
        return PredictLeafResolution::NotLeaf;
    }

    if invalid || accessor_count != 1 {
        return PredictLeafResolution::Missing;
    }

    match accessor {
        Some(accessor) => PredictLeafResolution::Accessor(accessor),
        None => PredictLeafResolution::Missing,
    }
}

fn is_predict_shape_identity(shape: &'static Shape) -> bool {
    shape.type_identifier == "Predict" && shape.module_path == Some("dspy_rs::predictors::predict")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate as dsrs;
    use crate::Signature;
    use crate::predictors::Predict as RealPredict;
    use std::ops::ControlFlow;
    use std::rc::Rc;
    use std::sync::Arc;

    #[derive(Signature, Clone, Debug)]
    struct DummySig {
        #[input]
        value: String,

        #[output]
        done: bool,
    }

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct SharedPointerModule {
        rc_predictor: Rc<RealPredict<DummySig>>,
        arc_predictor: Arc<RealPredict<DummySig>>,
    }

    #[test]
    fn named_parameters_rejects_shared_pointers() {
        let mut module = SharedPointerModule {
            rc_predictor: Rc::new(RealPredict::<DummySig>::new()),
            arc_predictor: Arc::new(RealPredict::<DummySig>::new()),
        };

        match visit_named_predictors_mut(&mut module, |_path, _predictor| ControlFlow::Continue(()))
        {
            Err(NamedParametersError::Container { path, ty }) => {
                assert_eq!(path, "rc_predictor");
                assert_eq!(ty, "Rc");
            }
            Ok(_) => panic!("walk unexpectedly succeeded"),
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[derive(facet::Facet)]
    #[facet(crate = facet, dsrs::predict_accessor)]
    struct MalformedAccessorLeaf;

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct MalformedAccessorModule {
        malformed: MalformedAccessorLeaf,
    }

    #[test]
    fn named_parameters_rejects_malformed_predict_accessor_payload() {
        let mut module = MalformedAccessorModule {
            malformed: MalformedAccessorLeaf,
        };

        match visit_named_predictors_mut(&mut module, |_path, _predictor| ControlFlow::Continue(()))
        {
            Err(NamedParametersError::MissingAttr { path }) => {
                assert_eq!(path, "malformed");
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("walk unexpectedly succeeded"),
        }
    }

    #[derive(facet::Facet)]
    #[facet(
        crate = facet,
        dsrs::predict_accessor,
        dsrs::predict_accessor
    )]
    struct DuplicateAccessorLeaf;

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct DuplicateAccessorModule {
        duplicate: DuplicateAccessorLeaf,
    }

    #[test]
    fn named_parameters_rejects_duplicate_predict_accessor_attrs() {
        let mut module = DuplicateAccessorModule {
            duplicate: DuplicateAccessorLeaf,
        };

        match visit_named_predictors_mut(&mut module, |_path, _predictor| ControlFlow::Continue(()))
        {
            Err(NamedParametersError::MissingAttr { path }) => {
                assert_eq!(path, "duplicate");
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("walk unexpectedly succeeded"),
        }
    }

    #[derive(facet::Facet)]
    #[facet(crate = facet, dsrs::predict_accessor)]
    struct AccessorOnlyLeaf;

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct AccessorOnlyModule {
        leaf: AccessorOnlyLeaf,
    }

    #[test]
    fn named_parameters_rejects_accessor_without_leaf_marker() {
        let mut module = AccessorOnlyModule {
            leaf: AccessorOnlyLeaf,
        };

        match visit_named_predictors_mut(&mut module, |_path, _predictor| ControlFlow::Continue(()))
        {
            Err(NamedParametersError::MissingAttr { path }) => {
                assert_eq!(path, "leaf");
            }
            Err(other) => panic!("unexpected error: {other:?}"),
            Ok(_) => panic!("walk unexpectedly succeeded"),
        }
    }

    #[test]
    fn real_predict_shape_has_strict_identity_marker() {
        assert!(is_predict_shape_identity(RealPredict::<DummySig>::SHAPE));
    }

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct Predict;

    #[derive(facet::Facet)]
    #[facet(crate = facet)]
    struct SameNameModule {
        predictor: Predict,
    }

    #[test]
    fn type_name_alone_is_not_treated_as_predict_leaf() {
        let mut module = SameNameModule { predictor: Predict };
        let mut paths = Vec::new();

        visit_named_predictors_mut(&mut module, |path, _predictor| {
            paths.push(path.to_string());
            ControlFlow::Continue(())
        })
        .expect("walk should succeed");

        assert!(paths.is_empty());
    }
}
