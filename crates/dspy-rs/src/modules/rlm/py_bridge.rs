use anyhow::anyhow;
use bamltype::baml_types::ir_type::UnionTypeViewGeneric;
use bamltype::baml_types::{BamlMap, BamlValue, LiteralValue, StreamingMode, TypeIR, TypeValue};
use bamltype::internal_baml_jinja::types::{Class, OutputFormatContent};
use bamltype::jsonish;
use bamltype::jsonish::deserializer::coercer::run_user_checks;
use bamltype::{BamlParseError, BamlType};
use pyo3::IntoPyObjectExt;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyList, PyListMethods, PyModule, PyString,
    PyTuple, PyTupleMethods, PyTypeMethods,
};
use pyo3::{Bound, Py, PyAny, PyResult, Python};
use serde_json::Value as JsonValue;

use super::submit::SubmitHandler;
use super::tools::LlmTools;
use crate::{BamlConvertError, ConstraintLevel, ResponseCheck, Signature};

pub fn setup_interpreter_globals<S: Signature>(
    py: Python<'_>,
    input: &S::Input,
    submit_handler: &SubmitHandler,
    llm_tools: Option<&LlmTools>,
) -> PyResult<Py<PyDict>> {
    let globals = PyDict::new(py);

    let input_value = input.to_baml_value();
    match input_value {
        BamlValue::Class(_, ref fields) | BamlValue::Map(ref fields) => {
            for (name, value) in fields {
                globals.set_item(name, baml_value_to_py(py, value)?)?;
            }
        }
        other => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "RLM input must serialize to object-like BamlValue, got {}",
                other.r#type()
            )));
        }
    }

    if let Some(llm_tools) = llm_tools {
        let tools_py = Py::new(py, llm_tools.clone())?;
        let tools_bound = tools_py.bind(py);
        globals.set_item("llm_query", tools_bound.getattr("llm_query")?)?;
        globals.set_item(
            "llm_query_batched",
            tools_bound.getattr("llm_query_batched")?,
        )?;
    }
    globals.set_item("SUBMIT", Py::new(py, submit_handler.clone())?)?;

    Ok(globals.unbind())
}

/// Convert BamlValue tree to Python objects recursively.
pub fn baml_value_to_py(py: Python<'_>, value: &BamlValue) -> PyResult<Py<PyAny>> {
    match value {
        BamlValue::String(value) => Ok(value.clone().into_py_any(py)?),
        BamlValue::Int(value) => Ok(value.into_py_any(py)?),
        BamlValue::Float(value) => Ok(value.into_py_any(py)?),
        BamlValue::Bool(value) => Ok(value.into_py_any(py)?),
        BamlValue::Null => Ok(py.None()),
        BamlValue::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(baml_value_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        BamlValue::Map(map) => {
            let dict = PyDict::new(py);
            for (key, value) in map.iter() {
                dict.set_item(key, baml_value_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        BamlValue::Enum(_, variant) => Ok(variant.clone().into_py_any(py)?),
        BamlValue::Class(_, fields) => {
            let dict = PyDict::new(py);
            for (key, value) in fields.iter() {
                dict.set_item(key, baml_value_to_py(py, value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
        BamlValue::Media(_) => Err(pyo3::exceptions::PyTypeError::new_err(
            "Media values are not supported in RLM V1",
        )),
    }
}

pub fn kwargs_to_baml_value<S: Signature>(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
) -> Result<BamlValue, BamlParseError> {
    let schema = S::schema();
    let output_format = schema.output_format();
    let mut fields = BamlMap::new();

    for field in schema.output_fields() {
        let value = kwargs
            .get_item(field.lm_name)
            .map_err(py_err_to_parse)?
            .ok_or_else(|| missing_field_error(&[], field.lm_name))?;
        let baml_value = py_to_baml_value(py, &value, &field.type_ir, output_format)
            .map_err(|err| add_field_context(err, field.lm_name))?;
        fields.insert(field.rust_name.to_string(), baml_value);
    }

    if let Some(class_name) = output_class_name(output_format) {
        Ok(BamlValue::Class(class_name, fields))
    } else {
        Ok(BamlValue::Map(fields))
    }
}

pub fn collect_checks_for_output<S: Signature>(
    value: &BamlValue,
) -> Result<Vec<ResponseCheck>, BamlParseError> {
    let schema = S::schema();

    let fields = match value {
        BamlValue::Class(_, fields) | BamlValue::Map(fields) => fields,
        other => {
            return Err(BamlParseError::Convert(BamlConvertError::new(
                Vec::new(),
                "object",
                format!("{other:?}"),
                "expected an object",
            )));
        }
    };

    let mut checks = Vec::new();
    let mut failed = Vec::new();

    for field in schema.output_fields() {
        let Some(value) = fields.get(field.rust_name.as_str()) else {
            return Err(missing_field_error(&[], field.rust_name.as_str()));
        };

        let results = run_user_checks(value, &field.type_ir).map_err(BamlParseError::from)?;
        for (constraint, ok) in results {
            if constraint.level == ConstraintLevel::Assert && !ok {
                failed.push(ResponseCheck {
                    name: constraint
                        .label
                        .clone()
                        .unwrap_or_else(|| "assert".to_string()),
                    expression: constraint.expression.0.clone(),
                    status: "failed".to_string(),
                });
            }

            if let Some(check) = ResponseCheck::from_check_result((constraint, ok)) {
                checks.push(check);
            }
        }
    }

    if !failed.is_empty() {
        return Err(BamlParseError::ConstraintAssertsFailed { failed });
    }

    Ok(checks)
}

fn output_class_name(output_format: &OutputFormatContent) -> Option<String> {
    let mut current = output_format.target.clone();
    loop {
        match current {
            TypeIR::Class { name, .. } => return Some(name),
            TypeIR::RecursiveTypeAlias { name, .. } => {
                if let Some(next) = output_format.structural_recursive_aliases.get(&name) {
                    current = next.clone();
                    continue;
                }
                return None;
            }
            _ => return None,
        }
    }
}

fn add_field_context(err: BamlParseError, field: &str) -> BamlParseError {
    match err {
        BamlParseError::Convert(err) => {
            let mut path = Vec::with_capacity(err.path.len() + 1);
            path.push(field.to_string());
            path.extend(err.path);
            BamlParseError::Convert(BamlConvertError::new(
                path,
                err.expected,
                err.got,
                err.message,
            ))
        }
        BamlParseError::Jsonish(inner) => BamlParseError::Convert(BamlConvertError::new(
            vec![field.to_string()],
            "schema",
            "python",
            inner.to_string(),
        )),
        other => other,
    }
}

pub fn py_to_baml_value(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    r#type: &TypeIR,
    output_format: &OutputFormatContent,
) -> Result<BamlValue, BamlParseError> {
    let obj = if obj.hasattr("__baml__").map_err(py_err_to_parse)? {
        obj.call_method0("__baml__").map_err(py_err_to_parse)?
    } else {
        obj.clone()
    };
    let obj = normalize_python_object(py, &obj).map_err(py_err_to_parse)?;
    let mut path = Vec::new();
    py_to_baml_value_inner(py, &obj, r#type, output_format, &mut path)
}

pub fn normalize_python_object<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if obj.is_instance_of::<PyDict>() || obj.is_instance_of::<PyList>() {
        return Ok(obj.clone());
    }

    if let Ok(value) = obj.call_method0("model_dump") {
        return Ok(value);
    }

    if let Ok(value) = obj.call_method0("dict") {
        return Ok(value);
    }

    if let Ok(value) = obj.call_method0("_asdict") {
        return Ok(value);
    }

    if let Ok(dataclasses) = PyModule::import(py, "dataclasses")
        && let Ok(is_dataclass) = dataclasses.getattr("is_dataclass")
        && is_dataclass.call1((obj,))?.is_truthy()?
        && let Ok(asdict) = dataclasses.getattr("asdict")
    {
        return asdict.call1((obj,));
    }

    if let Ok(attrs) = PyModule::import(py, "attr")
        && let Ok(has) = attrs.getattr("has")
        && has.call1((obj,))?.is_truthy()?
        && let Ok(asdict) = attrs.getattr("asdict")
    {
        return asdict.call1((obj,));
    }

    if let Ok(dict) = obj.getattr("__dict__") {
        return Ok(dict);
    }

    Ok(obj.clone())
}

fn py_to_baml_value_inner(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    r#type: &TypeIR,
    output_format: &OutputFormatContent,
    path: &mut Vec<String>,
) -> Result<BamlValue, BamlParseError> {
    let resolved = resolve_recursive_type(r#type, output_format);

    if !is_string_target(&resolved) && obj.is_instance_of::<PyString>() {
        let raw = obj.extract::<String>().map_err(py_err_to_parse)?;
        if let Ok(parsed_json) = serde_json::from_str::<JsonValue>(&raw) {
            let py_obj = json_value_to_py(py, &parsed_json).into_bound(py);
            return py_to_baml_value_inner(py, &py_obj, &resolved, output_format, path);
        }
    }

    match &resolved {
        TypeIR::Primitive(TypeValue::String, _) => obj
            .extract::<String>()
            .map(BamlValue::String)
            .map_err(py_err_to_parse),
        TypeIR::Primitive(TypeValue::Int, _) => {
            if obj.is_instance_of::<PyBool>() {
                return Err(conversion_error(path, &resolved, obj));
            }
            obj.extract::<i64>()
                .map(BamlValue::Int)
                .map_err(py_err_to_parse)
        }
        TypeIR::Primitive(TypeValue::Float, _) => {
            if obj.is_instance_of::<PyBool>() {
                return Err(conversion_error(path, &resolved, obj));
            }
            obj.extract::<f64>()
                .map(BamlValue::Float)
                .map_err(py_err_to_parse)
        }
        TypeIR::Primitive(TypeValue::Bool, _) => obj
            .extract::<bool>()
            .map(BamlValue::Bool)
            .map_err(py_err_to_parse),
        TypeIR::Primitive(TypeValue::Null, _) => {
            if obj.is_none() {
                Ok(BamlValue::Null)
            } else {
                Err(conversion_error(path, &resolved, obj))
            }
        }
        TypeIR::Primitive(TypeValue::Media(_), _) => Err(conversion_error(path, &resolved, obj)),
        TypeIR::Enum { name, .. } => {
            let raw = obj.extract::<String>().map_err(py_err_to_parse)?;
            let enum_type = output_format.enums.get(name).ok_or_else(|| {
                BamlParseError::Jsonish(anyhow!("missing enum definition for {name}"))
            })?;
            let mut matches_variant = false;
            for (value, _) in &enum_type.values {
                if value.real_name() == raw || value.rendered_name() == raw {
                    matches_variant = true;
                    break;
                }
            }
            if !matches_variant {
                return Err(conversion_error(path, &resolved, obj));
            }
            Ok(BamlValue::Enum(name.to_string(), raw))
        }
        TypeIR::Literal(LiteralValue::String(literal), _) => {
            let raw = obj.extract::<String>().map_err(py_err_to_parse)?;
            if raw == *literal {
                Ok(BamlValue::String(raw))
            } else {
                Err(conversion_error(path, &resolved, obj))
            }
        }
        TypeIR::Literal(LiteralValue::Int(literal), _) => {
            if obj.is_instance_of::<PyBool>() {
                return Err(conversion_error(path, &resolved, obj));
            }
            let raw = obj.extract::<i64>().map_err(py_err_to_parse)?;
            if raw == *literal {
                Ok(BamlValue::Int(raw))
            } else {
                Err(conversion_error(path, &resolved, obj))
            }
        }
        TypeIR::Literal(LiteralValue::Bool(literal), _) => {
            let raw = obj.extract::<bool>().map_err(py_err_to_parse)?;
            if raw == *literal {
                Ok(BamlValue::Bool(raw))
            } else {
                Err(conversion_error(path, &resolved, obj))
            }
        }
        TypeIR::Class { name, .. } => {
            py_to_class_value(py, obj, name.as_str(), output_format, path)
        }
        TypeIR::List(item_type, _) => {
            py_to_list_value(py, obj, item_type.as_ref(), output_format, path)
        }
        TypeIR::Map(key_type, value_type, _) => py_to_map_value(
            py,
            obj,
            key_type.as_ref(),
            value_type.as_ref(),
            output_format,
            path,
        ),
        TypeIR::Tuple(items, _) => py_to_tuple_value(py, obj, items, output_format, path),
        TypeIR::RecursiveTypeAlias { name, .. } => Err(BamlParseError::Jsonish(anyhow!(
            "missing recursive alias {name}"
        ))),
        TypeIR::Top(_) => py_any_to_baml_value_untyped(py, obj),
        TypeIR::Arrow(_, _) => Err(conversion_error(path, &resolved, obj)),
        TypeIR::Union(inner, _) => match inner.view() {
            UnionTypeViewGeneric::Null => {
                if obj.is_none() {
                    Ok(BamlValue::Null)
                } else {
                    Err(conversion_error(path, &resolved, obj))
                }
            }
            UnionTypeViewGeneric::Optional(t) => {
                if obj.is_none() {
                    Ok(BamlValue::Null)
                } else {
                    py_to_baml_value_inner(py, obj, t, output_format, path)
                }
            }
            UnionTypeViewGeneric::OneOf(types) | UnionTypeViewGeneric::OneOfOptional(types) => {
                let mut last_err: Option<BamlParseError> = None;
                for t in types {
                    match py_to_baml_value_inner(py, obj, t, output_format, path) {
                        Ok(value) => return Ok(value),
                        Err(err) => last_err = Some(err),
                    }
                }
                Err(last_err.unwrap_or_else(|| conversion_error(path, &resolved, obj)))
            }
        },
    }
}

fn py_to_class_value(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    class_name: &str,
    output_format: &OutputFormatContent,
    path: &mut Vec<String>,
) -> Result<BamlValue, BamlParseError> {
    let dict = match obj.cast::<PyDict>() {
        Ok(dict) => dict,
        Err(_) => {
            if let Some(value) =
                orjson_fallback_to_baml(py, obj, &TypeIR::class(class_name), output_format)
            {
                return Ok(value);
            }
            return Err(conversion_error(path, &TypeIR::class(class_name), obj));
        }
    };

    let class = find_class(output_format, class_name).ok_or_else(|| {
        BamlParseError::Jsonish(anyhow!("missing class definition for {class_name}"))
    })?;

    let mut fields = BamlMap::new();
    for field in &class.fields {
        let (name, field_type, _, _) = field;
        let rendered: &str = name.rendered_name();
        let real: &str = name.real_name();

        let value = dict
            .get_item(rendered)
            .map_err(py_err_to_parse)?
            .or_else(|| dict.get_item(real).ok().flatten());

        let value = match value {
            Some(value) => value,
            None => {
                if field_type.is_optional() {
                    fields.insert(real.to_string(), BamlValue::Null);
                    continue;
                }
                return Err(missing_field_error(path, real));
            }
        };

        path.push(real.to_string());
        let field_value = py_to_baml_value_inner(py, &value, field_type, output_format, path)?;
        path.pop();
        fields.insert(real.to_string(), field_value);
    }

    Ok(BamlValue::Class(class_name.to_string(), fields))
}

fn py_to_map_value(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    key_type: &TypeIR,
    value_type: &TypeIR,
    output_format: &OutputFormatContent,
    path: &mut Vec<String>,
) -> Result<BamlValue, BamlParseError> {
    if !matches!(
        key_type,
        TypeIR::Primitive(TypeValue::String, _) | TypeIR::Literal(LiteralValue::String(_), _)
    ) {
        return Err(BamlParseError::Convert(BamlConvertError::new(
            path.clone(),
            "string",
            schema_type_name(key_type),
            "map keys must be strings",
        )));
    }

    let dict = match obj.cast::<PyDict>() {
        Ok(dict) => dict,
        Err(_) => {
            if let Some(value) = orjson_fallback_to_baml(
                py,
                obj,
                &TypeIR::map(key_type.clone(), value_type.clone()),
                output_format,
            ) {
                return Ok(value);
            }
            return Err(conversion_error(
                path,
                &TypeIR::map(key_type.clone(), value_type.clone()),
                obj,
            ));
        }
    };

    let mut map = BamlMap::new();
    for (key, value) in dict.iter() {
        let key = key
            .extract::<String>()
            .map_err(|_| conversion_error(path, key_type, &key))?;
        path.push(key.clone());
        let value = py_to_baml_value_inner(py, &value, value_type, output_format, path)?;
        path.pop();
        map.insert(key, value);
    }

    Ok(BamlValue::Map(map))
}

fn py_to_list_value(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    item_type: &TypeIR,
    output_format: &OutputFormatContent,
    path: &mut Vec<String>,
) -> Result<BamlValue, BamlParseError> {
    let list = if let Ok(list) = obj.cast::<PyList>() {
        list
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for (idx, item) in tuple.iter().enumerate() {
            path.push(idx.to_string());
            let value = py_to_baml_value_inner(py, &item, item_type, output_format, path)?;
            path.pop();
            items.push(value);
        }
        return Ok(BamlValue::List(items));
    } else {
        if let Some(value) =
            orjson_fallback_to_baml(py, obj, &TypeIR::list(item_type.clone()), output_format)
        {
            return Ok(value);
        }
        return Err(conversion_error(
            path,
            &TypeIR::list(item_type.clone()),
            obj,
        ));
    };

    let mut items = Vec::with_capacity(list.len());
    for (idx, item) in list.iter().enumerate() {
        path.push(idx.to_string());
        let value = py_to_baml_value_inner(py, &item, item_type, output_format, path)?;
        path.pop();
        items.push(value);
    }

    Ok(BamlValue::List(items))
}

fn py_to_tuple_value(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    items: &[TypeIR],
    output_format: &OutputFormatContent,
    path: &mut Vec<String>,
) -> Result<BamlValue, BamlParseError> {
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        if tuple.len() != items.len() {
            return Err(conversion_error(path, &TypeIR::tuple(items.to_vec()), obj));
        }
        let mut values = Vec::with_capacity(items.len());
        for (idx, (item, item_type)) in tuple.iter().zip(items.iter()).enumerate() {
            path.push(idx.to_string());
            let value = py_to_baml_value_inner(py, &item, item_type, output_format, path)?;
            path.pop();
            values.push(value);
        }
        return Ok(BamlValue::List(values));
    }

    if let Ok(list) = obj.cast::<PyList>() {
        if list.len() != items.len() {
            return Err(conversion_error(path, &TypeIR::tuple(items.to_vec()), obj));
        }
        let mut values = Vec::with_capacity(items.len());
        for (idx, (item, item_type)) in list.iter().zip(items.iter()).enumerate() {
            path.push(idx.to_string());
            let value = py_to_baml_value_inner(py, &item, item_type, output_format, path)?;
            path.pop();
            values.push(value);
        }
        return Ok(BamlValue::List(values));
    }

    Err(conversion_error(path, &TypeIR::tuple(items.to_vec()), obj))
}

fn py_any_to_baml_value_untyped(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> Result<BamlValue, BamlParseError> {
    if obj.is_none() {
        return Ok(BamlValue::Null);
    }

    if obj.is_instance_of::<PyBool>() {
        return obj
            .extract::<bool>()
            .map(BamlValue::Bool)
            .map_err(py_err_to_parse);
    }

    if let Ok(value) = obj.extract::<i64>() {
        return Ok(BamlValue::Int(value));
    }

    if let Ok(value) = obj.extract::<f64>() {
        return Ok(BamlValue::Float(value));
    }

    if let Ok(value) = obj.extract::<String>() {
        return Ok(BamlValue::String(value));
    }

    if let Ok(dict) = obj.cast::<PyDict>() {
        let mut map = BamlMap::new();
        for (key, value) in dict.iter() {
            let key = key.extract::<String>().map_err(py_err_to_parse)?;
            let value = py_any_to_baml_value_untyped(py, &value)?;
            map.insert(key, value);
        }
        return Ok(BamlValue::Map(map));
    }

    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_any_to_baml_value_untyped(py, &item)?);
        }
        return Ok(BamlValue::List(items));
    }

    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_any_to_baml_value_untyped(py, &item)?);
        }
        return Ok(BamlValue::List(items));
    }

    let raw = python_object_to_json_string(py, obj)?;
    let parsed: JsonValue =
        serde_json::from_str(&raw).map_err(|err| BamlParseError::Jsonish(anyhow!(err)))?;
    Ok(json_value_to_baml_value(&parsed))
}

fn python_object_to_json_string(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> Result<String, BamlParseError> {
    if let Ok(orjson) = PyModule::import(py, "orjson")
        && let Ok(dumps) = orjson.getattr("dumps")
        && let Ok(raw) = dumps.call1((obj,))
        && let Ok(bytes) = raw.extract::<Vec<u8>>()
    {
        return String::from_utf8(bytes).map_err(|err| BamlParseError::Jsonish(anyhow!(err)));
    }

    let json = PyModule::import(py, "json").map_err(py_err_to_parse)?;
    let dumps = json.getattr("dumps").map_err(py_err_to_parse)?;
    dumps
        .call1((obj,))
        .map_err(py_err_to_parse)?
        .extract::<String>()
        .map_err(py_err_to_parse)
}

fn json_value_to_py(py: Python<'_>, value: &JsonValue) -> Py<PyAny> {
    match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(value) => value.into_py_any(py).unwrap_or_else(|_| py.None()),
        JsonValue::Number(value) => value
            .as_i64()
            .map(|value| value.into_py_any(py).unwrap_or_else(|_| py.None()))
            .or_else(|| {
                value
                    .as_f64()
                    .map(|value| value.into_py_any(py).unwrap_or_else(|_| py.None()))
            })
            .unwrap_or_else(|| py.None()),
        JsonValue::String(value) => value.clone().into_py_any(py).unwrap_or_else(|_| py.None()),
        JsonValue::Array(values) => {
            let list = PyList::empty(py);
            for item in values {
                let _ = list.append(json_value_to_py(py, item));
            }
            list.into_any().unbind()
        }
        JsonValue::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                let _ = dict.set_item(key, json_value_to_py(py, value));
            }
            dict.into_any().unbind()
        }
    }
}

fn json_value_to_baml_value(value: &JsonValue) -> BamlValue {
    match value {
        JsonValue::Null => BamlValue::Null,
        JsonValue::Bool(value) => BamlValue::Bool(*value),
        JsonValue::Number(value) => {
            if let Some(value) = value.as_i64() {
                BamlValue::Int(value)
            } else if let Some(value) = value.as_f64() {
                BamlValue::Float(value)
            } else {
                BamlValue::Null
            }
        }
        JsonValue::String(value) => BamlValue::String(value.clone()),
        JsonValue::Array(values) => {
            BamlValue::List(values.iter().map(json_value_to_baml_value).collect())
        }
        JsonValue::Object(values) => BamlValue::Map(
            values
                .iter()
                .map(|(key, value)| (key.clone(), json_value_to_baml_value(value)))
                .collect(),
        ),
    }
}

fn resolve_recursive_type(r#type: &TypeIR, output_format: &OutputFormatContent) -> TypeIR {
    let mut current = r#type.clone();
    loop {
        let next = match &current {
            TypeIR::RecursiveTypeAlias { name, .. } => output_format
                .structural_recursive_aliases
                .get(name)
                .cloned(),
            _ => None,
        };

        match next {
            Some(next) => current = next,
            None => return current,
        }
    }
}

fn find_class<'a>(output_format: &'a OutputFormatContent, class_name: &str) -> Option<&'a Class> {
    let key = (class_name.to_string(), StreamingMode::NonStreaming);
    if let Some(class) = output_format.classes.get(&key) {
        return Some(class);
    }

    output_format
        .classes
        .iter()
        .find(|((name, _), _)| name == class_name)
        .map(|(_, class)| class)
}

fn is_string_target(r#type: &TypeIR) -> bool {
    matches!(
        r#type,
        TypeIR::Primitive(TypeValue::String, _) | TypeIR::Literal(LiteralValue::String(_), _)
    )
}

fn conversion_error(path: &[String], expected: &TypeIR, got: &Bound<'_, PyAny>) -> BamlParseError {
    let got_type = py_type_name(got);
    BamlParseError::Convert(BamlConvertError::new(
        path.to_vec(),
        "schema",
        got_type,
        format!("expected {}", schema_type_name(expected)),
    ))
}

fn schema_type_name(type_ir: &TypeIR) -> String {
    crate::core::render_type_name_for_prompt_with(type_ir, crate::core::simplify_type_token)
}

fn missing_field_error(path: &[String], field: &str) -> BamlParseError {
    let mut full_path = path.to_vec();
    full_path.push(field.to_string());

    BamlParseError::Convert(BamlConvertError::new(
        full_path,
        "field",
        "missing",
        format!("missing required field {field}"),
    ))
}

fn py_type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .ok()
        .and_then(|name| name.extract::<String>().ok())
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn py_err_to_parse(err: pyo3::PyErr) -> BamlParseError {
    BamlParseError::Jsonish(anyhow!(err.to_string()))
}

fn orjson_fallback_to_baml(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    r#type: &TypeIR,
    output_format: &OutputFormatContent,
) -> Option<BamlValue> {
    let raw = python_object_to_json_string(py, obj).ok()?;
    let parsed = jsonish::from_str(output_format, r#type, &raw, true).ok()?;
    Some(BamlValue::from(parsed))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyo3::types::{PyDict, PyDictMethods};
    use tokio::runtime::Handle;

    use super::*;
    use crate::Signature;
    use crate::modules::rlm::{LlmQuery, SubmitSlot};

    #[derive(Signature, Clone, Debug)]
    struct BridgeSig {
        #[input]
        question: String,

        #[input]
        count: i64,

        #[output]
        answer: String,

        #[output]
        #[check("this >= 0.0", label = "non_negative")]
        score: f64,
    }

    #[derive(Signature, Clone, Debug)]
    struct AssertSig {
        #[input]
        prompt: String,

        #[output]
        #[assert("this > 0", label = "positive")]
        score: i64,
    }

    struct MockLm;

    #[async_trait::async_trait]
    impl LlmQuery for MockLm {
        async fn query(&self, prompt: &str) -> anyhow::Result<String> {
            Ok(format!("mock:{prompt}"))
        }
    }

    #[test]
    fn baml_value_to_py_supports_common_types() {
        Python::attach(|py| {
            let value = BamlValue::Map(BamlMap::from_iter([
                ("name".to_string(), BamlValue::String("alice".to_string())),
                (
                    "nums".to_string(),
                    BamlValue::List(vec![BamlValue::Int(1), BamlValue::Int(2)]),
                ),
                ("ok".to_string(), BamlValue::Bool(true)),
                (
                    "nested".to_string(),
                    BamlValue::Class(
                        "Nested".to_string(),
                        BamlMap::from_iter([("x".to_string(), BamlValue::Float(1.25))]),
                    ),
                ),
            ]));

            let py_obj = baml_value_to_py(py, &value).expect("convert to py");
            let dict = py_obj.bind(py).cast::<PyDict>().expect("dict");
            assert_eq!(
                dict.get_item("name")
                    .expect("getitem")
                    .expect("name")
                    .extract::<String>()
                    .expect("name str"),
                "alice"
            );
            assert!(
                dict.get_item("ok")
                    .expect("getitem")
                    .expect("ok")
                    .extract::<bool>()
                    .expect("ok bool")
            );
        });
    }

    #[test]
    fn kwargs_to_baml_value_validates_typed_fields() {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("answer", "done").expect("set answer");
            kwargs.set_item("score", 0.85).expect("set score");

            let converted = kwargs_to_baml_value::<BridgeSig>(py, &kwargs).expect("convert kwargs");
            let BamlValue::Class(_, fields) = converted else {
                panic!("expected class output");
            };
            assert_eq!(
                fields.get("answer"),
                Some(&BamlValue::String("done".to_string()))
            );
            assert_eq!(fields.get("score"), Some(&BamlValue::Float(0.85)));
        });
    }

    #[test]
    fn kwargs_to_baml_value_reports_type_error_context() {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("answer", "done").expect("set answer");
            kwargs.set_item("score", "oops").expect("set score");

            let err = kwargs_to_baml_value::<BridgeSig>(py, &kwargs).expect_err("should fail");
            match err {
                BamlParseError::Convert(err) => {
                    assert_eq!(err.path.first().map(|s| s.as_str()), Some("score"));
                }
                other => panic!("unexpected error: {other}"),
            }
        });
    }

    #[test]
    fn collect_checks_for_output_reports_assert_failures() {
        let value = BamlValue::Map(BamlMap::from_iter([(
            "score".to_string(),
            BamlValue::Int(-1),
        )]));

        let err = collect_checks_for_output::<AssertSig>(&value).expect_err("assert should fail");
        match err {
            BamlParseError::ConstraintAssertsFailed { failed } => {
                assert_eq!(failed.len(), 1);
                assert_eq!(failed[0].name, "positive");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn setup_interpreter_globals_injects_inputs_and_tools() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");

        runtime.block_on(async {
            Python::attach(|py| {
                let slot: SubmitSlot = Arc::new(std::sync::Mutex::new(None));
                let submit = SubmitHandler::new::<BridgeSig>(Arc::clone(&slot));
                let tools = LlmTools::with_budget(Arc::new(MockLm), 2, Handle::current());

                let input = BridgeSigInput {
                    question: "what?".to_string(),
                    count: 3,
                };

                let globals =
                    setup_interpreter_globals::<BridgeSig>(py, &input, &submit, Some(&tools))
                        .expect("setup globals")
                        .bind(py)
                        .clone();

                assert!(globals.get_item("question").expect("getitem").is_some());
                assert!(globals.get_item("count").expect("getitem").is_some());
                assert!(globals.get_item("llm_query").expect("getitem").is_some());
                assert!(
                    globals
                        .get_item("llm_query_batched")
                        .expect("getitem")
                        .is_some()
                );
                assert!(globals.get_item("SUBMIT").expect("getitem").is_some());
            });
        });
    }

    #[test]
    fn setup_interpreter_globals_without_sub_lm_tools_still_injects_submit_and_inputs() {
        Python::attach(|py| {
            let slot: SubmitSlot = Arc::new(std::sync::Mutex::new(None));
            let submit = SubmitHandler::new::<BridgeSig>(Arc::clone(&slot));
            let input = BridgeSigInput {
                question: "what?".to_string(),
                count: 3,
            };

            let globals = setup_interpreter_globals::<BridgeSig>(py, &input, &submit, None)
                .expect("setup globals")
                .bind(py)
                .clone();

            assert!(globals.get_item("question").expect("getitem").is_some());
            assert!(globals.get_item("count").expect("getitem").is_some());
            assert!(globals.get_item("SUBMIT").expect("getitem").is_some());
            assert!(globals.get_item("llm_query").expect("getitem").is_none());
            assert!(
                globals
                    .get_item("llm_query_batched")
                    .expect("getitem")
                    .is_none()
            );
        });
    }
}
