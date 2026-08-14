//! In-house constraint model + evaluation, replacing BAML's `Constraint` / `run_user_checks`.
//!
//! `#[check]` / `#[assert]` expressions are evaluated with `minijinja` (already a
//! dependency). The value under test is bound as `this`, matching BAML's jinja semantics
//! (e.g. `this >= 0.0 and this <= 1.0`, `this|length > 0`).

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use minijinja::{Environment, Expression};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Shared environment for constraint expressions — building an `Environment` per
/// evaluation is measurable waste on the parse hot path.
static CONSTRAINT_ENV: LazyLock<Environment<'static>> = LazyLock::new(Environment::new);

/// Compiled constraint expressions, keyed by the `&'static str` the signature macro
/// emitted. `None` marks an expression that failed to compile (cached so a bad
/// expression doesn't recompile on every parse either).
static COMPILED_EXPRESSIONS: LazyLock<
    RwLock<HashMap<&'static str, Option<Expression<'static, 'static>>>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));

/// Whether a constraint is a soft `check` (reported) or a hard `assert` (fails the call).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    Check,
    Assert,
}

/// Back-compat alias for the old public name.
pub type ConstraintLevel = ConstraintKind;

/// A single `#[check]`/`#[assert]` constraint attached to a field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Constraint {
    pub level: ConstraintKind,
    pub label: Option<String>,
    pub expression: String,
}

impl Constraint {
    pub fn new_check(label: impl Into<String>, expression: impl Into<String>) -> Self {
        Self {
            level: ConstraintKind::Check,
            label: Some(label.into()),
            expression: expression.into(),
        }
    }

    pub fn new_assert(label: impl Into<String>, expression: impl Into<String>) -> Self {
        Self {
            level: ConstraintKind::Assert,
            label: Some(label.into()),
            expression: expression.into(),
        }
    }
}

/// The outcome of evaluating a constraint against a value.
#[derive(Debug, Clone)]
pub struct ConstraintOutcome {
    pub level: ConstraintKind,
    pub label: String,
    pub expression: String,
    pub passed: bool,
}

/// A reported check result, mirroring the old `ResponseCheck` shape used by GEPA/optimizers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseCheck {
    pub name: String,
    pub expression: String,
    pub status: String,
}

/// Evaluates every constraint against `value`, binding it as `this` in a jinja expression.
///
/// A constraint that fails to evaluate (bad expression, wrong type) is treated as not
/// passing rather than erroring, matching the tolerant spirit of the old pipeline.
pub fn evaluate_constraints(value: &Value, constraints: &[Constraint]) -> Vec<ConstraintOutcome> {
    constraints
        .iter()
        .map(|constraint| {
            let passed = eval_expression(&constraint.expression, value).unwrap_or(false);
            let label = constraint.label.clone().unwrap_or_else(|| match constraint.level {
                ConstraintKind::Assert => "assert".to_string(),
                ConstraintKind::Check => "check".to_string(),
            });
            ConstraintOutcome {
                level: constraint.level,
                label,
                expression: constraint.expression.clone(),
                passed,
            }
        })
        .collect()
}

/// Evaluates a runtime (non-`'static`) constraint expression against `value`,
/// binding it as `this`. Compiles per call — dynamic-lane constraints are owned
/// strings, and caching them process-wide would reintroduce the leak-per-load
/// that RFC 0002 IR-1 removed. Failed evaluations return `false`, matching
/// [`evaluate_constraints`].
pub fn evaluate_expression(expression: &str, value: &Value) -> bool {
    eval_expression(expression, value).unwrap_or(false)
}

fn eval_expression(expression: &str, value: &Value) -> Result<bool, minijinja::Error> {
    let ctx = minijinja::context! { this => value };
    let expr = CONSTRAINT_ENV.compile_expression(expression)?;
    let result = expr.eval(ctx)?;
    Ok(result.is_true())
}

/// Evaluates a `'static` constraint expression against `value`, compiling it at
/// most once per process.
///
/// This is the parse hot path: signature constraints arrive as `&'static str`
/// from the derive macro, so the compiled expression is cached by pointer-stable
/// key. Failed evaluations (bad expression, wrong type) return `false`, matching
/// [`evaluate_constraints`].
pub fn evaluate_constraint_expression(expression: &'static str, value: &Value) -> bool {
    {
        let cache = COMPILED_EXPRESSIONS.read().expect("constraint cache poisoned");
        if let Some(entry) = cache.get(expression) {
            return entry
                .as_ref()
                .map(|expr| eval_compiled(expr, value))
                .unwrap_or(false);
        }
    }

    let compiled = CONSTRAINT_ENV.compile_expression(expression).ok();
    let passed = compiled
        .as_ref()
        .map(|expr| eval_compiled(expr, value))
        .unwrap_or(false);
    COMPILED_EXPRESSIONS
        .write()
        .expect("constraint cache poisoned")
        .entry(expression)
        .or_insert(compiled);
    passed
}

fn eval_compiled(expr: &Expression<'static, 'static>, value: &Value) -> bool {
    let ctx = minijinja::context! { this => value };
    expr.eval(ctx).map(|result| result.is_true()).unwrap_or(false)
}
