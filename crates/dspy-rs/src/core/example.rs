//! How trainset rows connect to modules.
//!
//! A trainset row is any struct you like — it can carry fields the module never
//! sees (gold labels, metadata, metric-only context like supporting facts). The
//! evaluation loop and optimizers only need one thing from a row: the module's
//! input. [`ToInput`] is that projection. [`ToOutput`] is its counterpart for
//! seeding labeled demos from gold data.
//!
//! Three ways to get an impl:
//!
//! 1. **`#[derive(Example)]`** — the row projects into *any* input/output type
//!    by field name through serde. No signature is named on the row; extra row
//!    fields are ignored by the projection. A missing or mismatched field is a
//!    runtime error on first use.
//! 2. **A `(input, output)` tuple** — `(S::Input, S::Output)` pairs implement
//!    both traits with no conversion at all (compile-time checked).
//! 3. **A hand-written impl** — for rows whose field names don't line up.

use anyhow::{Context, Result};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Projects a trainset row into a module's input type.
///
/// Implemented by `#[derive(Example)]` (field-name projection via serde), by
/// `(I, O)` tuples, or by hand.
pub trait ToInput<I> {
    fn to_input(&self) -> Result<I>;
}

/// Projects a trainset row into a signature's output type — the gold label.
///
/// Used for seeding labeled few-shot demos from a trainset
/// (`Demo::new(row.to_input()?, row.to_output()?)`). Optimizers that harvest
/// demos from traces don't need it.
pub trait ToOutput<O> {
    fn to_output(&self) -> Result<O>;
}

/// `(input, output)` pairs are rows with no extra fields.
impl<I: Clone, O> ToInput<I> for (I, O) {
    fn to_input(&self) -> Result<I> {
        Ok(self.0.clone())
    }
}

/// `(input, output)` pairs are rows with no extra fields.
impl<I, O: Clone> ToOutput<O> for (I, O) {
    fn to_output(&self) -> Result<O> {
        Ok(self.1.clone())
    }
}

/// Field-name projection: serializes `row` and deserializes `U` out of it.
///
/// Fields of `row` that `U` doesn't declare are ignored; fields `U` requires
/// that `row` lacks (or that don't type-match) produce an error naming both
/// types. This is the engine behind `#[derive(Example)]`.
pub fn project<T, U>(row: &T) -> Result<U>
where
    T: Serialize,
    U: DeserializeOwned,
{
    let value = serde_json::to_value(row).with_context(|| {
        format!(
            "cannot serialize row `{}` for projection",
            std::any::type_name::<T>()
        )
    })?;
    serde_json::from_value(value).with_context(|| {
        format!(
            "cannot project row `{}` into `{}` (fields are matched by name)",
            std::any::type_name::<T>(),
            std::any::type_name::<U>()
        )
    })
}
