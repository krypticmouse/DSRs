use std::collections::HashMap;

use baml_types::{BamlValue, JinjaExpression};
use minijinja::value::Value;
use regex::Regex;

pub fn get_env<'a>() -> minijinja::Environment<'a> {
    let mut env = minijinja::Environment::new();

    env.set_formatter(|output, state, value| {
        let value = if value.is_none() {
            &Value::from("null")
        } else {
            value
        };

        minijinja::escape_formatter(output, state, value)
    });

    env.set_debug(true);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_filter("regex_match", regex_match);
    env.add_filter("sum", sum_filter);
    env
}

fn regex_match(value: String, regex: String) -> bool {
    match Regex::new(&regex) {
        Err(_) => false,
        Ok(re) => re.is_match(&value),
    }
}

fn sum_filter(value: Vec<Value>) -> Value {
    let int_sum: Option<i64> = value
        .iter()
        .map(|v| <i64>::try_from(v.clone()).ok())
        .collect::<Option<Vec<_>>>()
        .map(|ints| ints.into_iter().sum());
    let float_sum: Option<f64> = value
        .into_iter()
        .map(|v| <f64>::try_from(v).ok())
        .collect::<Option<Vec<_>>>()
        .map(|floats| floats.into_iter().sum());
    if int_sum.is_none() && float_sum.is_none() {
        log::warn!("The `sum` jinja filter was run against non-numeric arguments")
    }
    int_sum.map_or(float_sum.map_or(Value::from(0), Value::from), Value::from)
}

pub fn render_expression(
    expression: &JinjaExpression,
    ctx: &HashMap<String, minijinja::Value>,
) -> anyhow::Result<String> {
    let env = get_env();
    let template = format!(r#"{{{{ {} }}}}"#, expression.0);
    let args_dict = minijinja::Value::from_serialize(ctx);
    Ok(env.render_str(&template, &args_dict)?)
}

pub fn evaluate_predicate(
    this: &BamlValue,
    predicate_expression: &JinjaExpression,
) -> Result<bool, anyhow::Error> {
    let ctx: HashMap<String, minijinja::Value> =
        HashMap::from([("this".to_string(), minijinja::Value::from_serialize(this))]);
    match render_expression(predicate_expression, &ctx)?.as_ref() {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(anyhow::anyhow!("Predicate did not evaluate to a boolean")),
    }
}
