use anyhow::anyhow;
use bamltype::BamlParseError;
use bamltype::baml_types::ir_type::UnionTypeViewGeneric;
use bamltype::baml_types::{BamlMap, BamlValue, LiteralValue, StreamingMode, TypeIR, TypeValue};
use bamltype::internal_baml_jinja::types::{Class, OutputFormatContent};
use bamltype::jsonish;
use bamltype::jsonish::deserializer::coercer::run_user_checks;
use pyo3::IntoPyObjectExt;
use pyo3::types::{
    PyAnyMethods, PyBool, PyDict, PyDictMethods, PyFloat, PyInt, PyList, PyListMethods, PyModule,
    PyString, PyTuple, PyTupleMethods, PyTypeMethods,
};
use pyo3::{Bound, Py, PyAny, PyResult, Python};
use serde_json::Value as JsonValue;

use super::runtime::{InterpreterSetup, MethodSignature, MethodSource, RlmInputFields};
use super::submit::SubmitHandler;
use super::tools::LlmTools;
use crate::{BamlConvertError, ConstraintLevel, ResponseCheck, Signature};

const RESERVED_GLOBAL_NAMES: [&str; 4] = ["llm_query", "llm_query_batched", "SUBMIT", "cleanup"];

pub fn setup_interpreter_globals<S: Signature>(
    py: Python<'_>,
    input: &S::Input,
    submit_handler: &SubmitHandler,
    llm_tools: Option<&LlmTools>,
) -> PyResult<InterpreterSetup>
where
    S::Input: RlmInputFields,
{
    let globals = PyDict::new(py);

    if let Some(name) = input
        .rlm_field_names()
        .iter()
        .copied()
        .find(|name| RESERVED_GLOBAL_NAMES.contains(name))
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "RLM input field '{name}' conflicts with reserved runtime binding. Rename this field (reserved names: {}).",
            RESERVED_GLOBAL_NAMES.join(", ")
        )));
    }
    input.inject_into_python(py, &globals)?;
    let methods_by_var = collect_methods_by_var(py, &globals, input.rlm_field_names())?;

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
    inject_cleanup_helper(py, &globals, input.rlm_field_names())?;

    Ok(InterpreterSetup {
        globals: globals.unbind(),
        methods_by_var,
    })
}

fn inject_cleanup_helper(
    py: Python<'_>,
    globals: &Bound<'_, PyDict>,
    injected_roots: &[&str],
) -> PyResult<()> {
    let mut protected_names = injected_roots
        .iter()
        .map(|name| (*name).to_string())
        .collect::<Vec<_>>();
    protected_names.extend(RESERVED_GLOBAL_NAMES.iter().map(|name| (*name).to_string()));
    protected_names.sort();
    protected_names.dedup();

    let protected_names_literal = protected_names
        .iter()
        .map(|name| format!("{name:?}"))
        .collect::<Vec<_>>()
        .join(", ");

    let source = format!(
        r#"def cleanup(*keep, _protected=frozenset([{protected_names_literal}])):
    """Clear user-created globals while preserving injected/runtime bindings.

    Usage:
        cleanup()
        cleanup("name_to_keep", "other_name")
    """
    namespace = globals()
    keep_names = [str(name) for name in keep]
    protected = set(_protected)
    protected.update(keep_names)

    cleared = []
    for name in list(namespace.keys()):
        if name in protected or name.startswith("__"):
            continue
        del namespace[name]
        cleared.append(name)

    cleared.sort()
    kept_present = sorted(name for name in protected if name in namespace)
    missing_requested = sorted(name for name in keep_names if name not in namespace)

    def _fmt(names):
        return ", ".join(f"`{{item}}`" for item in names) if names else "(none)"

    message = (
        f"cleanup(): cleared {{len(cleared)}} var(s): {{_fmt(cleared)}}\n"
        f"kept: {{_fmt(kept_present)}}"
    )
    if missing_requested:
        message += f"\nrequested but not found: {{_fmt(missing_requested)}}"
    return message
"#
    );

    let builtins = PyModule::import(py, "builtins")?;
    builtins
        .getattr("exec")?
        .call1((source.as_str(), globals, globals))?;

    if let Some(cleanup_fn) = globals.get_item("cleanup")? {
        cleanup_fn.setattr("__source__", source.as_str())?;
    }
    Ok(())
}

fn collect_methods_by_var(
    py: Python<'_>,
    globals: &Bound<'_, PyDict>,
    field_names: &[&str],
) -> PyResult<std::collections::BTreeMap<String, Vec<MethodSignature>>> {
    let inspect = PyModule::import(py, "inspect")?;
    let mut methods_by_var = std::collections::BTreeMap::new();

    for field_name in field_names {
        let Some(value) = globals.get_item(field_name)? else {
            continue;
        };
        let methods = collect_visible_methods_for_object(&inspect, &value)?;
        methods_by_var.insert((*field_name).to_string(), methods);
    }

    Ok(methods_by_var)
}

fn collect_visible_methods_for_object(
    inspect: &Bound<'_, PyModule>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<MethodSignature>> {
    if value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyList>()
        || value.is_instance_of::<PyDict>()
        || value.is_instance_of::<PyTuple>()
    {
        return Ok(Vec::new());
    }

    let class = value.get_type();
    let members = inspect.call_method1("getmembers", (&class, inspect.getattr("isroutine")?))?;
    let members = members.cast::<PyList>()?;
    let mut methods = Vec::new();

    for member in members.iter() {
        let tuple = member.cast::<PyTuple>()?;
        if tuple.len() != 2 {
            continue;
        }
        let name = tuple.get_item(0)?.extract::<String>()?;
        let is_dunder = name.starts_with("__") && name.ends_with("__");
        if name == "__baml__"
            || (is_dunder && !matches!(name.as_str(), "__len__" | "__iter__" | "__getitem__"))
        {
            continue;
        }

        let callable = tuple.get_item(1)?;
        let doc = extract_trimmed_docstring(&callable)?;
        if doc.is_empty() {
            continue;
        }

        methods.push(MethodSignature {
            signature: sanitize_signature(
                &extract_signature(inspect, &callable).unwrap_or_else(|| "()".to_string()),
            ),
            source: classify_method_source(&name),
            name,
            doc,
            is_dunder,
        });
    }

    methods.sort_by(|a, b| {
        a.name
            .cmp(&b.name)
            .then(a.signature.cmp(&b.signature))
            .then(a.doc.cmp(&b.doc))
    });
    methods.dedup_by(|a, b| a.name == b.name && a.signature == b.signature);
    Ok(methods)
}

fn extract_trimmed_docstring(callable: &Bound<'_, PyAny>) -> PyResult<String> {
    let Some(raw_doc) = callable.getattr("__doc__")?.extract::<Option<String>>()? else {
        return Ok(String::new());
    };
    Ok(raw_doc.trim().to_string())
}

fn extract_signature(inspect: &Bound<'_, PyModule>, callable: &Bound<'_, PyAny>) -> Option<String> {
    if let Ok(text_sig) = callable.getattr("__text_signature__")
        && let Ok(Some(text_sig)) = text_sig.extract::<Option<String>>()
    {
        let trimmed = text_sig.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }

    inspect
        .call_method1("signature", (callable,))
        .ok()
        .and_then(|sig| sig.str().ok())
        .and_then(|sig| sig.extract::<String>().ok())
        .map(|sig| sig.trim().to_string())
        .filter(|sig| !sig.is_empty())
        .or_else(|| {
            callable
                .call_method0("__signature__")
                .ok()
                .and_then(|sig| sig.str().ok())
                .and_then(|sig| sig.extract::<String>().ok())
                .map(|sig| sig.trim().to_string())
                .filter(|sig| !sig.is_empty())
        })
        .or_else(|| None)
}

fn sanitize_signature(raw_signature: &str) -> String {
    let mut signature = raw_signature.trim().to_string();

    if signature == "($self)" || signature == "($self, /)" {
        signature = "()".to_string();
    } else if signature.starts_with("($self, /, ") {
        signature = signature.replacen("($self, /, ", "(", 1);
    } else if signature.starts_with("($self, ") {
        signature = signature.replacen("($self, ", "(", 1);
    }

    if signature == "(self)" || signature == "(self, /)" {
        signature = "()".to_string();
    } else if signature.starts_with("(self, /, ") {
        signature = signature.replacen("(self, /, ", "(", 1);
    } else if signature.starts_with("(self, ") {
        signature = signature.replacen("(self, ", "(", 1);
    }
    signature = signature.replace("($self, /)", "()");
    signature = signature.replace("($self,)", "()");
    signature = signature.replace(", /)", ")");
    signature = signature.replace(", /, ", ", ");

    if !signature.starts_with('(') {
        signature = format!("({signature})");
    }

    simplify_qualified_type_paths(&signature)
}

fn simplify_qualified_type_paths(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut token = String::new();

    let flush = |out: &mut String, token: &mut String| {
        if token.is_empty() {
            return;
        }
        if token.contains('.') {
            if let Some(last) = token.rsplit('.').next() {
                out.push_str(last);
            }
        } else {
            out.push_str(token);
        }
        token.clear();
    };

    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' {
            token.push(ch);
        } else {
            flush(&mut out, &mut token);
            out.push(ch);
        }
    }
    flush(&mut out, &mut token);
    out
}

fn classify_method_source(name: &str) -> MethodSource {
    match name {
        "__len__" | "__iter__" | "__getitem__" | "__repr__" | "__baml__" => MethodSource::Generated,
        _ => MethodSource::Custom,
    }
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

        let field_value = with_path_segment(path, real.to_string(), |path| {
            py_to_baml_value_inner(py, &value, field_type, output_format, path)
        })?;
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
        let value = with_path_segment(path, key.clone(), |path| {
            py_to_baml_value_inner(py, &value, value_type, output_format, path)
        })?;
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
            let value = with_path_segment(path, idx.to_string(), |path| {
                py_to_baml_value_inner(py, &item, item_type, output_format, path)
            })?;
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
        let value = with_path_segment(path, idx.to_string(), |path| {
            py_to_baml_value_inner(py, &item, item_type, output_format, path)
        })?;
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
            let value = with_path_segment(path, idx.to_string(), |path| {
                py_to_baml_value_inner(py, &item, item_type, output_format, path)
            })?;
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
            let value = with_path_segment(path, idx.to_string(), |path| {
                py_to_baml_value_inner(py, &item, item_type, output_format, path)
            })?;
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

fn with_path_segment<T>(
    path: &mut Vec<String>,
    segment: String,
    convert: impl FnOnce(&mut Vec<String>) -> Result<T, BamlParseError>,
) -> Result<T, BamlParseError> {
    path.push(segment);
    let result = convert(path);
    path.pop();
    result
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

    use bamltype::baml_types::ir_type::UnionConstructor;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods};
    use tokio::runtime::Handle;

    use super::*;
    use crate::BamlType;
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

    #[derive(Signature, Clone, Debug)]
    struct ReservedNameSig {
        #[input]
        llm_query: String,

        #[output]
        answer: String,
    }

    #[pyclass]
    #[BamlType]
    #[derive(Clone, Debug)]
    struct MethodFixture {
        label: String,
    }

    #[pymethods]
    impl MethodFixture {
        #[new]
        fn new(label: String) -> Self {
            Self { label }
        }

        #[pyo3(text_signature = "(query)")]
        /// Search entries by query text.
        fn search(&self, query: String) -> String {
            format!("{}:{query}", self.label)
        }

        /// Return the character count for this fixture label.
        fn __len__(&self) -> usize {
            self.label.chars().count()
        }

        fn undocumented(&self) -> String {
            self.label.clone()
        }
    }

    #[derive(Signature, Clone, Debug)]
    struct MethodFixtureSig {
        #[input]
        trajectory: MethodFixture,

        #[output]
        answer: String,
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

                let setup =
                    setup_interpreter_globals::<BridgeSig>(py, &input, &submit, Some(&tools))
                        .expect("setup globals");
                let globals = setup.globals.bind(py).clone();

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
                assert!(globals.get_item("cleanup").expect("getitem").is_some());
                assert!(setup.methods_by_var.contains_key("question"));
                assert!(setup.methods_by_var.contains_key("count"));
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

            let setup = setup_interpreter_globals::<BridgeSig>(py, &input, &submit, None)
                .expect("setup globals");
            let globals = setup.globals.bind(py).clone();

            assert!(globals.get_item("question").expect("getitem").is_some());
            assert!(globals.get_item("count").expect("getitem").is_some());
            assert!(globals.get_item("SUBMIT").expect("getitem").is_some());
            assert!(globals.get_item("cleanup").expect("getitem").is_some());
            assert!(globals.get_item("llm_query").expect("getitem").is_none());
            assert!(
                globals
                    .get_item("llm_query_batched")
                    .expect("getitem")
                    .is_none()
            );
            assert!(setup.methods_by_var.contains_key("question"));
            assert!(setup.methods_by_var.contains_key("count"));
        });
    }

    #[test]
    fn cleanup_helper_clears_scratch_variables_and_exposes_source() {
        Python::attach(|py| {
            let slot: SubmitSlot = Arc::new(std::sync::Mutex::new(None));
            let submit = SubmitHandler::new::<BridgeSig>(Arc::clone(&slot));
            let input = BridgeSigInput {
                question: "what?".to_string(),
                count: 3,
            };

            let setup = setup_interpreter_globals::<BridgeSig>(py, &input, &submit, None)
                .expect("setup globals");
            let globals = setup.globals.bind(py).clone();
            globals
                .set_item("retro_corrections", vec!["a", "b"])
                .expect("set scratch");
            globals
                .set_item("themes", vec!["x"])
                .expect("set keep target");

            let cleanup_fn = globals
                .get_item("cleanup")
                .expect("getitem")
                .expect("cleanup fn");
            let message = cleanup_fn
                .call1(("themes",))
                .expect("call cleanup")
                .extract::<String>()
                .expect("cleanup message");

            assert!(message.contains("cleared 1 var(s)"));
            assert!(message.contains("`retro_corrections`"));
            assert!(message.contains("`themes`"));
            assert!(
                globals
                    .get_item("retro_corrections")
                    .expect("getitem")
                    .is_none()
            );
            assert!(globals.get_item("themes").expect("getitem").is_some());
            assert!(globals.get_item("question").expect("getitem").is_some());
            assert!(globals.get_item("SUBMIT").expect("getitem").is_some());

            let source = cleanup_fn
                .getattr("__source__")
                .expect("cleanup source attr")
                .extract::<String>()
                .expect("cleanup source string");
            assert!(source.contains("def cleanup("));
            assert!(source.contains("Clear user-created globals"));
        });
    }

    #[test]
    fn setup_interpreter_globals_rejects_reserved_input_names() {
        Python::attach(|py| {
            let slot: SubmitSlot = Arc::new(std::sync::Mutex::new(None));
            let submit = SubmitHandler::new::<ReservedNameSig>(Arc::clone(&slot));
            let input = ReservedNameSigInput {
                llm_query: "collision".to_string(),
            };

            let err = setup_interpreter_globals::<ReservedNameSig>(py, &input, &submit, None)
                .expect_err("reserved input names should fail setup");
            let message = err.to_string();
            assert!(message.contains("llm_query"));
            assert!(message.contains("reserved runtime binding"));
        });
    }

    #[test]
    fn setup_interpreter_globals_collects_filtered_method_metadata() {
        Python::attach(|py| {
            let slot: SubmitSlot = Arc::new(std::sync::Mutex::new(None));
            let submit = SubmitHandler::new::<MethodFixtureSig>(Arc::clone(&slot));
            let input = MethodFixtureSigInput {
                trajectory: MethodFixture {
                    label: "root".to_string(),
                },
            };

            let setup = setup_interpreter_globals::<MethodFixtureSig>(py, &input, &submit, None)
                .expect("setup globals");
            let methods = setup
                .methods_by_var
                .get("trajectory")
                .expect("trajectory methods");

            assert_eq!(
                setup.methods_by_var.keys().collect::<Vec<_>>(),
                vec![&"trajectory".to_string()],
                "keys must match injected variable names"
            );
            assert!(
                methods.windows(2).all(|w| w[0].name <= w[1].name),
                "method list should be deterministic and sorted by name"
            );
            assert!(methods.iter().any(|m| m.name == "search"));
            assert!(methods.iter().any(|m| m.name == "__len__"));
            assert!(!methods.iter().any(|m| m.name == "undocumented"));
            assert!(!methods.iter().any(|m| m.name == "__baml__"));

            let search = methods
                .iter()
                .find(|m| m.name == "search")
                .expect("search method metadata");
            assert!(search.signature.contains("query"));
            assert!(!search.signature.contains("self"));
            assert!(search.doc.contains("Search entries"));
            assert!(matches!(search.source, MethodSource::Custom));
            assert!(!search.is_dunder);

            let dunder_len = methods
                .iter()
                .find(|m| m.name == "__len__")
                .expect("__len__ metadata");
            assert!(dunder_len.is_dunder);
            assert!(matches!(dunder_len.source, MethodSource::Generated));
            assert!(!dunder_len.doc.trim().is_empty());
        });
    }

    #[test]
    fn sanitize_signature_removes_python_self_variants() {
        assert_eq!(
            sanitize_signature("($self, path_fragment)"),
            "(path_fragment)"
        );
        assert_eq!(
            sanitize_signature("($self, /, path_fragment)"),
            "(path_fragment)"
        );
        assert_eq!(
            sanitize_signature("(self, /, path_fragment)"),
            "(path_fragment)"
        );
        assert_eq!(sanitize_signature("($self, /)"), "()");
    }

    #[test]
    fn sanitize_signature_simplifies_qualified_type_paths() {
        let raw = "(query: builtins.str, other: tanha.types.Sessions) -> tanha.types.Sessions";
        let sanitized = sanitize_signature(raw);
        assert!(!sanitized.contains("builtins."));
        assert!(!sanitized.contains("tanha.types."));
        assert!(sanitized.contains("str"));
        assert!(sanitized.contains("Sessions"));
    }

    #[test]
    fn union_attempts_do_not_leak_path_segments_between_branches() {
        Python::attach(|py| {
            let list = PyList::empty(py);
            list.append(3).expect("append");

            let union = TypeIR::union(vec![
                TypeIR::list(TypeIR::literal_int(1)),
                TypeIR::list(TypeIR::literal_int(2)),
            ]);
            let output_format = BridgeSig::schema().output_format();

            let err = py_to_baml_value(py, list.as_any(), &union, output_format)
                .expect_err("union should fail to parse mismatched literal");
            match err {
                BamlParseError::Convert(err) => {
                    assert_eq!(
                        err.path,
                        vec!["0".to_string()],
                        "path should represent one nesting level, not accumulate from prior union attempts"
                    );
                }
                other => panic!("unexpected error: {other}"),
            }
        });
    }
}
