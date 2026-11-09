//! In-house constraint model + evaluation, replacing BAML's `Constraint` / `run_user_checks`.
//!
//! `#[check]` / `#[assert]` expressions are evaluated with `minijinja` (already a
//! dependency). The value under test is bound as `this`, matching BAML's jinja semantics
//! (e.g. `this >= 0.0 and this <= 1.0`, `this|length > 0`).

use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use minijinja::machinery::ast::{BinOpKind, Expr, UnaryOpKind};
use minijinja::machinery::parse_expr;
use minijinja::{Environment, Expression};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Shared environment for constraint expressions — building an `Environment` per
/// evaluation is measurable waste on the parse hot path.
static CONSTRAINT_ENV: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    env.set_fuel(Some(10_000));
    env
});

/// Constraint expressions are predicates, not a general template language.
/// Inspect the parsed tree before MiniJinja can fold a costly constant.
pub(crate) fn validate_constraint_expression(expression: &str) -> Result<(), &'static str> {
    if expression.len() > 1024 {
        return Err("constraint expression exceeds 1024 bytes");
    }
    let parsed = parse_expr(expression).map_err(|_| "invalid constraint expression")?;
    let mut pending = vec![&parsed];
    while let Some(expr) = pending.pop() {
        match expr {
            Expr::Var(var) if var.id == "this" => {}
            Expr::Const(_) => {}
            Expr::UnaryOp(op) if matches!(&op.op, UnaryOpKind::Not | UnaryOpKind::Neg) => {
                pending.push(&op.expr);
            }
            Expr::BinOp(op)
                if matches!(
                    &op.op,
                    BinOpKind::Eq
                        | BinOpKind::Ne
                        | BinOpKind::Lt
                        | BinOpKind::Lte
                        | BinOpKind::Gt
                        | BinOpKind::Gte
                        | BinOpKind::ScAnd
                        | BinOpKind::ScOr
                        | BinOpKind::In
                ) =>
            {
                pending.push(&op.left);
                pending.push(&op.right);
            }
            Expr::Filter(filter)
                if matches!(filter.name, "length" | "int" | "float")
                    && filter.args.is_empty()
                    && filter.expr.is_some() =>
            {
                pending.push(filter.expr.as_ref().unwrap());
            }
            Expr::GetAttr(attr) => pending.push(&attr.expr),
            Expr::GetItem(item) => {
                pending.push(&item.expr);
                pending.push(&item.subscript_expr);
            }
            _ => return Err("constraint expression uses an unsupported operation"),
        }
    }
    Ok(())
}

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

/// Evaluates a runtime (non-`'static`) constraint expression against `value`,
/// binding it as `this`. Compiles per call — dynamic-lane constraints are owned
/// strings, and caching them process-wide would reintroduce the leak-per-load
/// that RFC 0002 IR-1 removed. Failed evaluations (bad expression, wrong type)
/// return `false`.
pub fn evaluate_expression(expression: &str, value: &Value) -> bool {
    if validate_constraint_expression(expression).is_err() {
        return false;
    }
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
/// [`evaluate_expression`].
pub fn evaluate_constraint_expression(expression: &'static str, value: &Value) -> bool {
    {
        let cache = COMPILED_EXPRESSIONS
            .read()
            .expect("constraint cache poisoned");
        if let Some(entry) = cache.get(expression) {
            return entry
                .as_ref()
                .map(|expr| eval_compiled(expr, value))
                .unwrap_or(false);
        }
    }

    let compiled = validate_constraint_expression(expression)
        .ok()
        .and_then(|_| CONSTRAINT_ENV.compile_expression(expression).ok());
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
    expr.eval(ctx)
        .map(|result| result.is_true())
        .unwrap_or(false)
}
