use anyhow::anyhow;
use baml_types::ir_type::UnionTypeViewGeneric;
use baml_types::{
    BamlMap, BamlMedia, BamlValue, LiteralValue, StreamingMode, TypeIR, TypeValue,
};
use internal_baml_jinja::types::{Class, OutputFormatContent};
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyList, PyListMethods, PyModule, PyString,
    PyStringMethods, PyTuple, PyTupleMethods, PyTypeMethods,
};
use pyo3::{Bound, Py, PyAny, PyResult, Python};
use pyo3::IntoPyObjectExt;
use serde_json::Value as JsonValue;

use crate::{BamlConvertError, BamlParseError};

pub fn baml_value_to_py(py: Python<'_>, value: &BamlValue) -> Py<PyAny> {
    match value {
        BamlValue::String(value) => into_py_object(py, value.clone()),
        BamlValue::Int(value) => into_py_object(py, *value),
        BamlValue::Float(value) => into_py_object(py, *value),
        BamlValue::Bool(value) => into_py_object(py, *value),
        BamlValue::Null => py.None(),
        BamlValue::Media(media) => json_value_to_py(py, &media_to_json_value(media)),
        BamlValue::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                let _ = list.append(baml_value_to_py(py, item));
            }
            list.into_any().unbind()
        }
        BamlValue::Map(map) => {
            let dict = PyDict::new(py);
            for (key, value) in map.iter() {
                let _ = dict.set_item(key, baml_value_to_py(py, value));
            }
            dict.into_any().unbind()
        }
        BamlValue::Enum(_, variant) => into_py_object(py, variant.clone()),
        BamlValue::Class(_, fields) => {
            let dict = PyDict::new(py);
            for (key, value) in fields.iter() {
                let _ = dict.set_item(key, baml_value_to_py(py, value));
            }
            dict.into_any().unbind()
        }
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

    if let Ok(dataclasses) = PyModule::import(py, "dataclasses") {
        if let Ok(is_dataclass) = dataclasses.getattr("is_dataclass") {
            if is_dataclass.call1((obj,))?.is_truthy()? {
                if let Ok(asdict) = dataclasses.getattr("asdict") {
                    return asdict.call1((obj,));
                }
            }
        }
    }

    if let Ok(attrs) = PyModule::import(py, "attr") {
        if let Ok(has) = attrs.getattr("has") {
            if has.call1((obj,))?.is_truthy()? {
                if let Ok(asdict) = attrs.getattr("asdict") {
                    return asdict.call1((obj,));
                }
            }
        }
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
        let parsed = jsonish::from_str(output_format, &resolved, &raw, true)
            .map_err(BamlParseError::from)?;
        return Ok(BamlValue::from(parsed));
    }

    match &resolved {
        TypeIR::Union(union, _) => match union.view() {
            UnionTypeViewGeneric::Null => {
                if obj.is_none() {
                    Ok(BamlValue::Null)
                } else {
                    Err(conversion_error(path, &resolved, obj))
                }
            }
            UnionTypeViewGeneric::Optional(inner) => {
                if obj.is_none() {
                    Ok(BamlValue::Null)
                } else {
                    py_to_baml_value_inner(py, obj, inner, output_format, path)
                }
            }
            UnionTypeViewGeneric::OneOf(options)
            | UnionTypeViewGeneric::OneOfOptional(options) => {
                if obj.is_none() && union.is_optional() {
                    return Ok(BamlValue::Null);
                }
                let mut last_err = None;
                for option in options {
                    match py_to_baml_value_inner(py, obj, option, output_format, path) {
                        Ok(value) => return Ok(value),
                        Err(err) => last_err = Some(err),
                    }
                }
                Err(last_err.unwrap_or_else(|| conversion_error(path, &resolved, obj)))
            }
        },
        TypeIR::Primitive(TypeValue::String, _) => {
            let value = obj.extract::<String>().map_err(py_err_to_parse)?;
            Ok(BamlValue::String(value))
        }
        TypeIR::Primitive(TypeValue::Int, _) => {
            if obj.is_instance_of::<PyBool>() {
                return Err(conversion_error(path, &resolved, obj));
            }
            let value = obj.extract::<i64>().map_err(py_err_to_parse)?;
            Ok(BamlValue::Int(value))
        }
        TypeIR::Primitive(TypeValue::Float, _) => {
            let value = obj.extract::<f64>().map_err(py_err_to_parse)?;
            Ok(BamlValue::Float(value))
        }
        TypeIR::Primitive(TypeValue::Bool, _) => {
            let value = obj.extract::<bool>().map_err(py_err_to_parse)?;
            Ok(BamlValue::Bool(value))
        }
        TypeIR::Primitive(TypeValue::Null, _) => {
            if obj.is_none() {
                Ok(BamlValue::Null)
            } else {
                Err(conversion_error(path, &resolved, obj))
            }
        }
        TypeIR::Primitive(TypeValue::Media(_), _) => {
            let json_value = python_object_to_json_value(py, obj)?;
            let media = serde_json::from_value::<BamlMedia>(json_value)
                .map_err(|err| BamlParseError::Jsonish(anyhow!(err)))?;
            Ok(BamlValue::Media(media))
        }
        TypeIR::Enum { name, .. } => {
            let raw = obj.extract::<String>().map_err(py_err_to_parse)?;
            let enum_name = name.clone();
            let enum_def = output_format.enums.get(&enum_name);
            if let Some(enum_def) = enum_def {
                if let Some((variant, _)) = enum_def
                    .values
                    .iter()
                    .find(|(name, _)| name.rendered_name() == raw || name.real_name() == raw)
                {
                    return Ok(BamlValue::Enum(
                        enum_name,
                        variant.real_name().to_string(),
                    ));
                }
                return Err(conversion_error(path, &resolved, obj));
            }
            Ok(BamlValue::Enum(enum_name, raw))
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
            if let Some(value) = orjson_fallback_to_baml(py, obj, &TypeIR::class(class_name), output_format)
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
    for (name, field_type, _, _) in &class.fields {
        let rendered = name.rendered_name();
        let real = name.real_name();
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
            format!("{}", key_type.diagnostic_repr()),
            "map keys must be strings",
        )));
    }

    let dict = match obj.cast::<PyDict>() {
        Ok(dict) => dict,
        Err(_) => {
            if let Some(value) = orjson_fallback_to_baml(py, obj, &TypeIR::map(key_type.clone(), value_type.clone()), output_format)
            {
                return Ok(value);
            }
            return Err(conversion_error(path, &TypeIR::map(key_type.clone(), value_type.clone()), obj));
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
        if let Some(value) = orjson_fallback_to_baml(py, obj, &TypeIR::list(item_type.clone()), output_format)
        {
            return Ok(value);
        }
        return Err(conversion_error(path, &TypeIR::list(item_type.clone()), obj));
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
        let value = obj.extract::<bool>().map_err(py_err_to_parse)?;
        return Ok(BamlValue::Bool(value));
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
            let key = key
                .str()
                .map_err(py_err_to_parse)?
                .to_str()
                .map_err(py_err_to_parse)?
                .to_string();
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

    let json_value = python_object_to_json_value(py, obj)?;
    Ok(json_value_to_baml_value(&json_value))
}

fn python_object_to_json_value(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> Result<JsonValue, BamlParseError> {
    let raw = python_object_to_json_string(py, obj)?;
    serde_json::from_str(&raw).map_err(|err| BamlParseError::Jsonish(anyhow!(err)))
}

fn python_object_to_json_string(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> Result<String, BamlParseError> {
    if let Ok(orjson) = PyModule::import(py, "orjson") {
        if let Ok(dumps) = orjson.getattr("dumps") {
            if let Ok(bytes) = dumps.call1((obj,)) {
                if let Ok(bytes) = bytes.extract::<Vec<u8>>() {
                    return String::from_utf8(bytes)
                        .map_err(|err| BamlParseError::Jsonish(anyhow!(err)));
                }
            }
        }
    }

    let json = PyModule::import(py, "json").map_err(py_err_to_parse)?;
    let dumps = json.getattr("dumps").map_err(py_err_to_parse)?;
    dumps
        .call1((obj,))
        .map_err(py_err_to_parse)?
        .extract::<String>()
        .map_err(py_err_to_parse)
}

fn into_py_object<T>(py: Python<'_>, value: T) -> Py<PyAny>
where
    for<'a> T: IntoPyObjectExt<'a>,
{
    value.into_py_any(py).unwrap_or_else(|_| py.None())
}

fn json_value_to_py(py: Python<'_>, value: &JsonValue) -> Py<PyAny> {
    match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(value) => into_py_object(py, *value),
        JsonValue::Number(value) => value
            .as_i64()
            .map(|value| into_py_object(py, value))
            .or_else(|| value.as_f64().map(|value| into_py_object(py, value)))
            .unwrap_or_else(|| py.None()),
        JsonValue::String(value) => into_py_object(py, value.clone()),
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

fn media_to_json_value(media: &BamlMedia) -> JsonValue {
    serde_json::to_value(media).unwrap_or(JsonValue::Null)
}

fn resolve_recursive_type(r#type: &TypeIR, output_format: &OutputFormatContent) -> TypeIR {
    let mut current = r#type.clone();
    loop {
        let next = match &current {
            TypeIR::RecursiveTypeAlias { name, .. } => {
                output_format.structural_recursive_aliases.get(name).cloned()
            }
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
        format!("expected {}", expected.diagnostic_repr()),
    ))
}

fn missing_field_error(path: &[String], field: &str) -> BamlParseError {
    BamlParseError::Convert(BamlConvertError::new(
        path.to_vec(),
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
