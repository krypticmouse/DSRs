//! The fx/ModuleState ↔ [`Overlay`] bridge (RFC 0002 §2.4 migration contract).
//!
//! One candidate currency, three surfaces:
//!
//! - [`fx::Params`](crate::fx::Params) is the **unbound string form** of an
//!   overlay: `{name → (instruction?, demos)}` with no program attached.
//!   [`Params::bind`](crate::fx::Params::bind) resolves the names against a
//!   [`Program`] into an [`Overlay`];
//!   [`Params::from_overlay`](crate::fx::Params::from_overlay) unbinds an
//!   overlay back to named params.
//! - [`ModuleState`] (dotted-path persistence) is a **serde projection** of an
//!   overlay: [`ModuleState::to_overlay`] / [`Overlay::to_module_state`]
//!   round-trip through a program whose leaf names match the dotted paths.
//!   The existing `ModuleState` JSON format is unchanged.
//! - [`with_overlay`] is the fx-lane scope symmetric to
//!   [`fx::with_params`](crate::fx::with_params): it unbinds the overlay
//!   against the program and injects the resulting params ambiently, so a
//!   candidate minted for the interpreter drives a hand-written fx harness
//!   unchanged.
//!
//! # Value mapping
//!
//! `PredictState.instruction_override: None` means "use the incumbent" in
//! both lanes, so it maps to *no overlay entry* (the slot default reads
//! through). Empty `PredictState.demos` likewise maps to no entry. Flat demo
//! rows (input and output fields merged into one JSON object) split into
//! [`DemoRow`]'s input/output maps using the owning leaf's [`SignatureDef`]
//! field names — a field in neither side is an
//! [`OverlayError::DemoField`] error.
//!
//! In the overlay → params/state direction the projection is **restricted to
//! `Instruction` and `Demos` kinds** (RFC 0002 §2.4): `ToolDesc`, `ModelRef`,
//! `ContextPolicy`, and `Code` entries have no fx/ModuleState representation
//! and are skipped, never errors — the static lane simply has no slot for
//! them.

use std::collections::BTreeMap;

use crate::core::{ModuleState, PredictState};
use crate::ir::graph::{Node, Program};
use crate::ir::params::{DemoRow, Overlay, OverlayError, ParamOwner, ParamValue};
use crate::ir::sig::SignatureDef;
use crate::trace::JsonMap;

impl crate::fx::Params {
    /// Resolves these string-named params against `program` into an
    /// [`Overlay`] — the bound candidate form the interpreter reads through
    /// at render time.
    ///
    /// Every named entry must resolve to a leaf (`"<name>.instruction"` /
    /// `"<name>.demos"` param paths); an unknown name is
    /// [`OverlayError::UnknownPath`]. Entries with no instruction override
    /// and no demos still verify the leaf exists but set nothing (the slot
    /// defaults read through).
    pub fn bind(&self, program: &Program) -> Result<Overlay, OverlayError> {
        states_to_overlay(program, self.iter_states())
    }

    /// Unbinds `overlay` against `program` back to string-named params — the
    /// inverse of [`bind`](Self::bind), restricted to `Instruction`/`Demos`
    /// kinds (other kinds have no fx representation and are skipped).
    ///
    /// Fails with [`OverlayError::BaseMismatch`] when the overlay was minted
    /// against a different program.
    pub fn from_overlay(program: &Program, overlay: &Overlay) -> Result<Self, OverlayError> {
        let mut params = Self::new();
        for (name, state) in overlay_to_states(program, overlay)? {
            params.set(name, state);
        }
        Ok(params)
    }
}

impl ModuleState {
    /// Projects this saved state into an [`Overlay`] against a program whose
    /// leaf names match the dotted predictor paths.
    ///
    /// The serde format of `ModuleState` is unchanged — this is a view, not a
    /// migration. Unknown paths are [`OverlayError::UnknownPath`].
    pub fn to_overlay(&self, program: &Program) -> Result<Overlay, OverlayError> {
        states_to_overlay(
            program,
            self.predictors
                .iter()
                .map(|(name, state)| (name.as_str(), state)),
        )
    }
}

impl Overlay {
    /// Projects this overlay to the [`ModuleState`] persistence form,
    /// restricted to `Instruction`/`Demos` kinds (RFC 0002 §2.4) — entries of
    /// other kinds are skipped.
    ///
    /// Fails with [`OverlayError::BaseMismatch`] when the overlay was minted
    /// against a different program.
    pub fn to_module_state(&self, program: &Program) -> Result<ModuleState, OverlayError> {
        Ok(ModuleState {
            predictors: overlay_to_states(program, self)?,
        })
    }
}

/// Runs `fut` with `overlay` (unbound against `program`) as the ambient
/// [`fx::Params`](crate::fx::Params) scope — the fx-lane equivalent of
/// passing the overlay to [`Interpreter::run`](crate::ir::Interpreter::run).
///
/// IR-loaded programs and fx harnesses share one candidate currency: the same
/// `Overlay` an optimizer evaluates through the interpreter drives an fx
/// harness whose [`fx::predict`](crate::fx::predict) names match the
/// program's leaf names. Scoping semantics are exactly
/// [`fx::with_params`](crate::fx::with_params) (task-local; spawned subtasks
/// do not inherit; nesting replaces).
///
/// Fails before running `fut` when the overlay does not unbind against
/// `program` (base mismatch).
pub async fn with_overlay<Fut: Future>(
    program: &Program,
    overlay: &Overlay,
    fut: Fut,
) -> Result<Fut::Output, OverlayError> {
    let params = crate::fx::Params::from_overlay(program, overlay)?;
    Ok(crate::fx::with_params(params, fut).await)
}

tokio::task_local! {
    /// The ambient overlay `#[module]` executable fns read (RFC 0003 §5).
    static CURRENT_OVERLAY: std::sync::Arc<Overlay>;
}

/// Runs `fut` with `overlay` as the ambient candidate for every `#[module]`
/// fn called on this task — the interpreter-lane sibling of
/// [`fx::with_params`](crate::fx::with_params). Scoping matches it exactly:
/// task-local, spawned subtasks do not inherit, nesting replaces.
///
/// The overlay's [`base`](Overlay::base) is checked by `Interpreter::run`
/// against each module's program, not here — one scope can span calls into
/// several modules, and only the matching one accepts it.
pub async fn with_ambient_overlay<Fut: Future>(
    overlay: std::sync::Arc<Overlay>,
    fut: Fut,
) -> Fut::Output {
    CURRENT_OVERLAY.scope(overlay, fut).await
}

/// The ambient overlay, if a [`with_ambient_overlay`] scope is active on
/// this task. Read by `#[module]`-generated executable fns immediately
/// before `Interpreter::run`.
pub fn current_overlay() -> Option<std::sync::Arc<Overlay>> {
    CURRENT_OVERLAY.try_with(std::sync::Arc::clone).ok()
}

// ---------------------------------------------------------------------------
// Core conversions
// ---------------------------------------------------------------------------

/// Builds an overlay from `(leaf name, PredictState)` pairs — the shared core
/// of [`fx::Params::bind`](crate::fx::Params::bind) and
/// [`ModuleState::to_overlay`].
pub(crate) fn states_to_overlay<'a, I>(
    program: &Program,
    states: I,
) -> Result<Overlay, OverlayError>
where
    I: IntoIterator<Item = (&'a str, &'a PredictState)>,
{
    let mut overlay = Overlay::new(program);
    for (name, state) in states {
        let instruction_path = format!("{name}.instruction");
        let instruction_id =
            program
                .param_id(&instruction_path)
                .ok_or_else(|| OverlayError::UnknownPath {
                    path: instruction_path,
                })?;
        if let Some(text) = &state.instruction_override {
            overlay.set(
                program,
                instruction_id,
                ParamValue::Instruction { text: text.clone() },
            )?;
        }

        if !state.demos.is_empty() {
            let demos_path = format!("{name}.demos");
            let demos_id =
                program
                    .param_id(&demos_path)
                    .ok_or_else(|| OverlayError::UnknownPath {
                        path: demos_path.clone(),
                    })?;
            let def = leaf_sig_of(program, demos_id);
            let rows = state
                .demos
                .iter()
                .map(|flat| split_demo_row(def, &demos_path, flat))
                .collect::<Result<Vec<_>, _>>()?;
            overlay.set(program, demos_id, ParamValue::Demos { rows })?;
        }
    }
    Ok(overlay)
}

/// Projects an overlay down to `(leaf name → PredictState)`, restricted to
/// `Instruction`/`Demos` kinds — the shared core of
/// [`fx::Params::from_overlay`](crate::fx::Params::from_overlay) and
/// [`Overlay::to_module_state`].
pub(crate) fn overlay_to_states(
    program: &Program,
    overlay: &Overlay,
) -> Result<BTreeMap<String, PredictState>, OverlayError> {
    if overlay.base != program.meta.program_hash {
        return Err(OverlayError::BaseMismatch {
            expected: overlay.base,
            got: program.meta.program_hash,
        });
    }
    let mut states: BTreeMap<String, PredictState> = BTreeMap::new();
    for (id, value) in overlay.entries() {
        let ParamOwner::Node(node) = program.params[id].owner else {
            continue; // tool-owned slots have no fx/ModuleState surface
        };
        let Some(leaf) = program.leaf_name(node) else {
            continue;
        };
        match value {
            ParamValue::Instruction { text } => {
                states
                    .entry(leaf.to_string())
                    .or_default()
                    .instruction_override = Some(text.clone());
            }
            ParamValue::Demos { rows } => {
                states.entry(leaf.to_string()).or_default().demos =
                    rows.iter().map(flatten_demo_row).collect();
            }
            // Restricted projection (RFC 0002 §2.4): ToolDesc / ModelRef /
            // ContextPolicy / Code have no representation in the static lane.
            _ => {}
        }
    }
    Ok(states)
}

/// The signature of the leaf that owns a demos slot. Demos slots only exist
/// on `Predict`/`AgentLoop` nodes — validated program invariant.
fn leaf_sig_of(program: &Program, demos_id: crate::ir::params::ParamId) -> &SignatureDef {
    let ParamOwner::Node(node) = program.params[demos_id].owner else {
        unreachable!("validated: demos slots are node-owned");
    };
    let sig = match &program.nodes[node] {
        Node::Predict(n) => n.sig,
        Node::AgentLoop(n) => n.sig,
        _ => unreachable!("validated: demos slots live on Predict/AgentLoop leaves"),
    };
    &program.sigs[sig]
}

/// Splits a flat demo row (input and output fields merged — the
/// [`PredictState::demos`] shape) into a [`DemoRow`] using the leaf
/// signature's field names.
fn split_demo_row(def: &SignatureDef, path: &str, flat: &JsonMap) -> Result<DemoRow, OverlayError> {
    let mut input = JsonMap::new();
    let mut output = JsonMap::new();
    for (field, value) in flat {
        if def.inputs.iter().any(|f| &*f.name == field.as_str()) {
            input.insert(field.clone(), value.clone());
        } else if def.outputs.iter().any(|f| &*f.name == field.as_str()) {
            output.insert(field.clone(), value.clone());
        } else {
            return Err(OverlayError::DemoField {
                path: path.to_string(),
                field: field.clone(),
            });
        }
    }
    Ok(DemoRow { input, output })
}

/// Merges a [`DemoRow`] back into the flat `PredictState` shape.
fn flatten_demo_row(row: &DemoRow) -> JsonMap {
    let mut flat = row.input.clone();
    flat.extend(row.output.iter().map(|(k, v)| (k.clone(), v.clone())));
    flat
}
