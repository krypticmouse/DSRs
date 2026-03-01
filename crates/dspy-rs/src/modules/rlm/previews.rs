use std::collections::{BTreeMap, BTreeSet};

use bamltype::baml_types::ir_type::{TypeGeneric, UnionTypeViewGeneric};
use bamltype::baml_types::type_meta::base::TypeMeta;
use bamltype::baml_types::{StreamingMode, TypeIR, TypeValue};
use bamltype::internal_baml_jinja::types::{Class, OutputFormatContent};
use tracing::{debug, info_span};

use super::runtime::MethodSignature;
use crate::{BamlType, Facet, FieldSchema, Signature, SignatureSchema};

#[derive(Clone, Copy)]
struct RenderBudget {
    max_methods: usize,
    max_depth: usize,
}

impl RenderBudget {
    const fn relaxed() -> Self {
        Self {
            max_methods: usize::MAX,
            max_depth: 12,
        }
    }
}

pub(super) fn render_previews<S: Signature>(
    _input: &S::Input,
    methods_by_var: &BTreeMap<String, Vec<MethodSignature>>,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
) -> String
where
    S::Input: BamlType + for<'a> Facet<'a>,
{
    let schema = SignatureSchema::of::<S>();
    let input_format = <S::Input as BamlType>::baml_output_format();

    let render_span = info_span!(
        "rlm.preview.render",
        input_fields = schema.input_fields().len(),
        method_vars = methods_by_var.len(),
        output_len = tracing::field::Empty
    );
    let _render_guard = render_span.enter();

    let budget = RenderBudget::relaxed();
    let rendered = render_with_budget(
        schema,
        input_format,
        methods_by_var,
        methods_by_type,
        budget,
    );
    let output_len = rendered.chars().count();
    debug!(
        output_len,
        max_methods = budget.max_methods,
        max_depth = budget.max_depth,
        "preview rendered"
    );
    render_span.record("output_len", output_len);
    rendered
}

pub(super) fn is_primitive_input_type(type_ir: &TypeIR) -> bool {
    let Some(inner) = strip_optional(type_ir) else {
        return false;
    };

    matches!(
        inner,
        TypeGeneric::Primitive(TypeValue::String, _)
            | TypeGeneric::Primitive(TypeValue::Int, _)
            | TypeGeneric::Primitive(TypeValue::Float, _)
            | TypeGeneric::Primitive(TypeValue::Bool, _)
    )
}

pub(super) fn type_label(type_ir: &TypeIR, output_format: &OutputFormatContent) -> String {
    clean_type_expr(type_ir, output_format)
}

pub(super) fn render_type_shape(
    type_ir: &TypeIR,
    output_format: &OutputFormatContent,
    indent: usize,
) -> Vec<String> {
    let mut visited = BTreeSet::new();
    let methods_by_type = BTreeMap::new();
    render_type_node(
        type_ir,
        output_format,
        &methods_by_type,
        indent,
        0,
        RenderBudget::relaxed().max_depth,
        &mut visited,
    )
}

fn render_with_budget(
    schema: &SignatureSchema,
    input_format: &OutputFormatContent,
    methods_by_var: &BTreeMap<String, Vec<MethodSignature>>,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    budget: RenderBudget,
) -> String {
    let mut lines = Vec::new();
    let mut rendered_any = false;

    for field in schema.input_fields() {
        if is_primitive_input_type(&field.type_ir) {
            continue;
        }

        rendered_any = true;
        lines.extend(render_variable_block(
            field,
            input_format,
            methods_by_var
                .get(field.rust_name.as_str())
                .map(Vec::as_slice),
            methods_by_type,
            budget,
        ));
        lines.push(String::new());
    }

    if !rendered_any {
        lines.push("(No complex input variables.)".to_string());
    }

    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }

    lines.join("\n")
}

fn render_variable_block(
    field: &FieldSchema,
    output_format: &OutputFormatContent,
    methods: Option<&[MethodSignature]>,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    budget: RenderBudget,
) -> Vec<String> {
    let mut lines = Vec::new();

    lines.push(format!(
        "Variable: `{}` (access it in your code)",
        field.rust_name
    ));
    lines.push(format!(
        "Type: {}",
        type_label(&field.type_ir, output_format)
    ));

    if !field.docs.trim().is_empty() {
        lines.push(format!("Description: {}", normalize_doc_text(&field.docs)));
    }

    lines.push("Schema:".to_string());

    let mut visited = BTreeSet::new();
    lines.extend(render_root_schema(
        &field.type_ir,
        output_format,
        methods,
        methods_by_type,
        2,
        0,
        budget,
        &mut visited,
    ));

    lines
}

fn render_root_schema(
    type_ir: &TypeIR,
    output_format: &OutputFormatContent,
    methods: Option<&[MethodSignature]>,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    indent: usize,
    depth: usize,
    budget: RenderBudget,
    visited: &mut BTreeSet<String>,
) -> Vec<String> {
    if let Some((class_name, mode)) = class_type_ref(type_ir)
        && let Some(class) = output_format.classes.get(&(class_name.to_string(), mode))
    {
        return render_class_block(
            class,
            output_format,
            methods,
            methods_by_type,
            indent,
            depth,
            budget,
            visited,
        );
    }

    render_type_node(
        type_ir,
        output_format,
        methods_by_type,
        indent,
        depth,
        budget.max_depth,
        visited,
    )
}

fn render_class_block(
    class: &Class,
    output_format: &OutputFormatContent,
    methods: Option<&[MethodSignature]>,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    indent: usize,
    depth: usize,
    budget: RenderBudget,
    visited: &mut BTreeSet<String>,
) -> Vec<String> {
    let class_name = class.name.rendered_name().to_string();
    if depth >= budget.max_depth || !visited.insert(class_name.clone()) {
        return vec![format!("{}{}", spaces(indent), class_name)];
    }

    let mut lines = Vec::new();
    lines.push(format!("{}{} {{", spaces(indent), class_name));

    let methods = methods.or_else(|| methods_by_type.get(&class_name).map(Vec::as_slice));
    if let Some(methods) = methods {
        let methods = methods
            .iter()
            .filter(|method| !method.is_dunder)
            .take(budget.max_methods)
            .collect::<Vec<_>>();

        if !methods.is_empty() {
            lines.push(format!("{}// methods", spaces(indent + 2)));
            for method in methods {
                lines.push(format!(
                    "{}{}",
                    spaces(indent + 2),
                    render_method_line(method)
                ));
            }
        }
    }

    lines.push(format!("{}// shape", spaces(indent + 2)));
    for (field_name, field_type, description, _) in &class.fields {
        lines.extend(render_field_line(
            field_name.real_name(),
            field_type,
            description.as_deref(),
            output_format,
            methods_by_type,
            indent + 2,
            depth + 1,
            budget,
            visited,
        ));
    }

    lines.push(format!("{}}}", spaces(indent)));
    lines
}

fn render_field_line(
    field_name: &str,
    field_type: &TypeIR,
    description: Option<&str>,
    output_format: &OutputFormatContent,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    indent: usize,
    depth: usize,
    budget: RenderBudget,
    visited: &mut BTreeSet<String>,
) -> Vec<String> {
    let mut lines = Vec::new();
    let rendered = render_type_node(
        field_type,
        output_format,
        methods_by_type,
        indent + 2,
        depth,
        budget.max_depth,
        visited,
    );

    if rendered.len() == 1 {
        let mut line = format!(
            "{}{}: {}",
            spaces(indent),
            field_name,
            rendered[0].trim_start()
        );
        if let Some(description) = description
            && !description.trim().is_empty()
        {
            line.push_str(" // ");
            line.push_str(&normalize_doc_text(description));
        }
        lines.push(line);
        return lines;
    }

    let mut first_line = format!(
        "{}{}: {}",
        spaces(indent),
        field_name,
        rendered[0].trim_start()
    );
    if let Some(description) = description
        && !description.trim().is_empty()
    {
        first_line.push_str(" // ");
        first_line.push_str(&normalize_doc_text(description));
    }
    lines.push(first_line);
    lines.extend(rendered.into_iter().skip(1));

    lines
}

fn render_type_node(
    type_ir: &TypeIR,
    output_format: &OutputFormatContent,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    indent: usize,
    depth: usize,
    max_depth: usize,
    visited: &mut BTreeSet<String>,
) -> Vec<String> {
    if depth >= max_depth {
        return vec![format!(
            "{}{}",
            spaces(indent),
            type_label(type_ir, output_format)
        )];
    }

    if let Some(optional_inner) = optional_inner(type_ir)
        && is_simple_type(optional_inner)
    {
        return vec![format!(
            "{}{} | null",
            spaces(indent),
            type_label(optional_inner, output_format)
        )];
    }

    match type_ir {
        TypeGeneric::List(inner, _) => {
            render_list_node(
                inner,
                output_format,
                methods_by_type,
                indent,
                depth + 1,
                max_depth,
                visited,
            )
        }
        TypeGeneric::Map(key, value, _) => {
            let key_name = type_label(key, output_format);
            if is_simple_type(value) {
                return vec![format!(
                    "{}map<{}, {}>",
                    spaces(indent),
                    key_name,
                    type_label(value, output_format)
                )];
            }

            let mut lines = vec![format!("{}map<{},", spaces(indent), key_name)];
            lines.extend(render_type_node(
                value,
                output_format,
                methods_by_type,
                indent + 2,
                depth + 1,
                max_depth,
                visited,
            ));
            lines.push(format!("{}>", spaces(indent)));
            lines
        }
        TypeGeneric::Class { name, mode, .. } => {
            if let Some(class) = output_format.classes.get(&(name.to_string(), *mode)) {
                let class_methods = methods_by_type
                    .get(class.name.rendered_name())
                    .map(Vec::as_slice);
                render_class_block(
                    class,
                    output_format,
                    class_methods,
                    methods_by_type,
                    indent,
                    depth,
                    RenderBudget::relaxed(),
                    visited,
                )
            } else {
                vec![format!("{}{}", spaces(indent), short_name(name))]
            }
        }
        TypeGeneric::Enum { name, .. } => vec![format!(
            "{}{}",
            spaces(indent),
            enum_name(name, output_format)
        )],
        TypeGeneric::Union(union, _) => {
            render_union_node(
                union,
                output_format,
                methods_by_type,
                indent,
                depth,
                max_depth,
                visited,
            )
        }
        TypeGeneric::RecursiveTypeAlias { name, .. } => {
            if let Some(alias) = output_format.structural_recursive_aliases.get(name) {
                render_type_node(
                    alias,
                    output_format,
                    methods_by_type,
                    indent,
                    depth + 1,
                    max_depth,
                    visited,
                )
            } else {
                vec![format!("{}{}", spaces(indent), short_name(name))]
            }
        }
        TypeGeneric::Primitive(value, _) => {
            vec![format!("{}{}", spaces(indent), primitive_name(*value))]
        }
        TypeGeneric::Literal(literal, _) => {
            vec![format!("{}{:?}", spaces(indent), literal)]
        }
        _ => vec![format!(
            "{}{}",
            spaces(indent),
            clean_diagnostic_repr(type_ir)
        )],
    }
}

fn render_list_node(
    inner: &TypeIR,
    output_format: &OutputFormatContent,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    indent: usize,
    depth: usize,
    max_depth: usize,
    visited: &mut BTreeSet<String>,
) -> Vec<String> {
    if is_simple_type(inner) {
        return vec![format!(
            "{}list[{}]",
            spaces(indent),
            type_label(inner, output_format)
        )];
    }

    let mut lines = vec![format!("{}list[", spaces(indent))];
    lines.extend(render_type_node(
        inner,
        output_format,
        methods_by_type,
        indent + 2,
        depth,
        max_depth,
        visited,
    ));
    lines.push(format!("{}]", spaces(indent)));
    lines
}

fn render_union_node(
    union: &bamltype::baml_types::ir_type::UnionTypeGeneric<TypeMeta>,
    output_format: &OutputFormatContent,
    methods_by_type: &BTreeMap<String, Vec<MethodSignature>>,
    indent: usize,
    depth: usize,
    max_depth: usize,
    visited: &mut BTreeSet<String>,
) -> Vec<String> {
    if let UnionTypeViewGeneric::Optional(inner) = union.view() {
        if is_simple_type(inner) {
            return vec![format!(
                "{}{} | null",
                spaces(indent),
                type_label(inner, output_format)
            )];
        }
    }

    let mut lines = vec![format!("{}one of:", spaces(indent))];
    for option in union.iter_include_null() {
        let option = unwrap_single_payload_variant_class(option, output_format).unwrap_or(option);
        let rendered = render_type_node(
            option,
            output_format,
            methods_by_type,
            indent + 4,
            depth + 1,
            max_depth,
            visited,
        );
        if rendered.is_empty() {
            continue;
        }

        lines.push(format!(
            "{}- {}",
            spaces(indent + 2),
            rendered[0].trim_start()
        ));
        for extra in rendered.iter().skip(1) {
            lines.push(extra.to_string());
        }
    }

    lines
}

fn unwrap_single_payload_variant_class<'a>(
    type_ir: &'a TypeIR,
    output_format: &'a OutputFormatContent,
) -> Option<&'a TypeIR> {
    let TypeGeneric::Class { name, mode, .. } = type_ir else {
        return None;
    };
    let class = output_format.classes.get(&(name.to_string(), *mode))?;
    if class.fields.len() != 2 {
        return None;
    }

    let mut literal_count = 0usize;
    let mut payload: Option<&TypeIR> = None;
    for (_, field_type, _, _) in &class.fields {
        if matches!(field_type, TypeGeneric::Literal(..)) {
            literal_count += 1;
            continue;
        }
        if payload.is_some() {
            return None;
        }
        payload = Some(field_type);
    }

    if literal_count == 1 { payload } else { None }
}

fn render_method_line(method: &MethodSignature) -> String {
    let mut line = format!(".{}{}", method.name, method.signature);
    let doc = normalize_doc_text(&method.doc);
    if !doc.is_empty() {
        line.push_str(" // ");
        line.push_str(&doc);
    }
    line
}

fn is_simple_type(type_ir: &TypeIR) -> bool {
    if let Some(inner) = strip_optional(type_ir) {
        return matches!(
            inner,
            TypeGeneric::Primitive(..)
                | TypeGeneric::Enum { .. }
                | TypeGeneric::Literal(..)
                | TypeGeneric::Top(..)
        );
    }

    matches!(
        type_ir,
        TypeGeneric::Primitive(..)
            | TypeGeneric::Enum { .. }
            | TypeGeneric::Literal(..)
            | TypeGeneric::Top(..)
    )
}

fn strip_optional(type_ir: &TypeIR) -> Option<&TypeIR> {
    match type_ir {
        TypeGeneric::Union(union, _) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => Some(inner),
            _ => None,
        },
        _ => Some(type_ir),
    }
}

fn optional_inner(type_ir: &TypeIR) -> Option<&TypeIR> {
    match type_ir {
        TypeGeneric::Union(union, _) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => Some(inner),
            _ => None,
        },
        _ => None,
    }
}

fn class_type_ref(type_ir: &TypeIR) -> Option<(&str, StreamingMode)> {
    match type_ir {
        TypeGeneric::Class { name, mode, .. } => Some((name.as_str(), *mode)),
        TypeGeneric::Union(union, _) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => class_type_ref(inner),
            _ => None,
        },
        _ => None,
    }
}

fn clean_type_expr(type_ir: &TypeIR, output_format: &OutputFormatContent) -> String {
    match type_ir {
        TypeGeneric::Primitive(value, _) => primitive_name(*value).to_string(),
        TypeGeneric::Class { name, mode, .. } => output_format
            .classes
            .get(&(name.to_string(), *mode))
            .map(|class| class.name.rendered_name().to_string())
            .unwrap_or_else(|| short_name(name)),
        TypeGeneric::Enum { name, .. } => enum_name(name, output_format),
        TypeGeneric::List(inner, _) => {
            format!("list[{}]", clean_type_expr(inner, output_format))
        }
        TypeGeneric::Map(key, value, _) => format!(
            "map<{}, {}>",
            clean_type_expr(key, output_format),
            clean_type_expr(value, output_format)
        ),
        TypeGeneric::Union(union, _) => {
            if let UnionTypeViewGeneric::Optional(inner) = union.view() {
                return format!("{} | null", clean_type_expr(inner, output_format));
            }

            let variants = union
                .iter_include_null()
                .into_iter()
                .map(|variant| clean_type_expr(variant, output_format))
                .collect::<Vec<_>>();
            variants.join(" | ")
        }
        TypeGeneric::RecursiveTypeAlias { name, .. } => short_name(name),
        _ => clean_diagnostic_repr(type_ir),
    }
}

fn clean_diagnostic_repr(type_ir: &TypeIR) -> String {
    let mut out = type_ir.diagnostic_repr().to_string();
    out = out.replace("class `", "");
    out = out.replace("enum `", "");
    out = out.replace('`', "");
    for token in ["class ", "enum "] {
        out = out.replace(token, "");
    }
    short_path_tokens(&out)
}

fn short_path_tokens(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut token = String::new();

    let flush = |out: &mut String, token: &mut String| {
        if token.is_empty() {
            return;
        }
        if token.contains("::") {
            if let Some(last) = token.rsplit("::").next() {
                out.push_str(last);
            }
        } else {
            out.push_str(token);
        }
        token.clear();
    };

    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == ':' {
            token.push(ch);
        } else {
            flush(&mut out, &mut token);
            out.push(ch);
        }
    }
    flush(&mut out, &mut token);
    out
}

fn enum_name(internal: &str, output_format: &OutputFormatContent) -> String {
    output_format
        .enums
        .get(internal)
        .map(|enm| enm.name.rendered_name().to_string())
        .unwrap_or_else(|| short_name(internal))
}

fn primitive_name(value: TypeValue) -> &'static str {
    match value {
        TypeValue::String => "string",
        TypeValue::Int => "int",
        TypeValue::Float => "float",
        TypeValue::Bool => "bool",
        TypeValue::Null => "null",
        _ => "value",
    }
}

fn short_name(path: &str) -> String {
    path.rsplit("::").next().unwrap_or(path).to_string()
}

fn normalize_doc_text(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn spaces(count: usize) -> String {
    " ".repeat(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BamlType;
    use crate::Signature;

    #[derive(Clone, Debug)]
    #[BamlType]
    struct PreviewAction {
        /// Tool name.
        name: String,
        /// JSON arguments.
        arguments: String,
        /// Tool output.
        result: Option<String>,
        /// True if the tool errored.
        is_error: bool,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct PreviewTurn {
        /// User message that started this turn.
        trigger: Option<String>,
        /// Tool actions in this turn.
        actions: Vec<PreviewAction>,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct PreviewSession {
        /// First user message, truncated.
        brief: Option<String>,
        /// Turn sequence.
        turns: Vec<PreviewTurn>,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct PreviewSessions {
        /// Stored sessions.
        items: Vec<PreviewSession>,
    }

    #[derive(Signature, Clone, Debug)]
    struct PreviewSig {
        #[input]
        title: String,

        #[input]
        count: i64,

        #[output]
        answer: String,
    }

    #[test]
    fn primitive_inputs_are_skipped() {
        let input = PreviewSigInput {
            title: "x".to_string(),
            count: 3,
        };
        let rendered = render_previews::<PreviewSig>(&input, &BTreeMap::new(), &BTreeMap::new());
        assert!(rendered.contains("(No complex input variables.)"));
    }

    #[derive(Signature, Clone, Debug)]
    struct RichPreviewSig {
        #[input]
        /// Turn-level trajectories for each development session.
        sessions: PreviewSessions,

        #[output]
        answer: String,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct DuplicateActions {
        /// Primary action list.
        primary: Vec<PreviewAction>,
        /// Backup action list.
        backup: Vec<PreviewAction>,
    }

    #[derive(Signature, Clone, Debug)]
    struct DedupPreviewSig {
        #[input]
        /// Two fields that reference the same nested class.
        actions: DuplicateActions,

        #[output]
        answer: String,
    }

    #[test]
    fn schema_rendering_has_methods_shape_comments_and_nested_lists() {
        let input = RichPreviewSigInput {
            sessions: PreviewSessions {
                items: vec![PreviewSession {
                    brief: Some("Investigate signal drop".to_string()),
                    turns: vec![PreviewTurn {
                        trigger: Some("start".to_string()),
                        actions: vec![PreviewAction {
                            name: "search".to_string(),
                            arguments: "{\"q\":\"start\"}".to_string(),
                            result: Some("ok".to_string()),
                            is_error: false,
                        }],
                    }],
                }],
            },
        };
        let methods = BTreeMap::from([(
            "sessions".to_string(),
            vec![
                MethodSignature {
                    name: "search".to_string(),
                    signature: "(query)".to_string(),
                    doc: "Find matching sessions.".to_string(),
                    source: super::super::runtime::MethodSource::Custom,
                    is_dunder: false,
                },
                MethodSignature {
                    name: "hidden".to_string(),
                    signature: "()".to_string(),
                    doc: "".to_string(),
                    source: super::super::runtime::MethodSource::Custom,
                    is_dunder: false,
                },
            ],
        )]);

        let rendered = render_previews::<RichPreviewSig>(&input, &methods, &BTreeMap::new());
        assert!(rendered.contains("Variable: `sessions` (access it in your code)"));
        assert!(rendered.contains("Type: PreviewSessions"));
        assert!(
            rendered.contains("Description: Turn-level trajectories for each development session.")
        );
        assert!(rendered.contains("// methods"));
        assert!(rendered.contains(".search(query) // Find matching sessions."));
        assert!(rendered.contains(".hidden()"));
        assert!(!rendered.contains(".hidden() //"));
        assert!(rendered.contains("// shape"));
        assert!(rendered.contains("items: list[ // Stored sessions."));
        assert!(rendered.contains("brief: string | null // First user message, truncated."));
        assert!(rendered.contains("turns: list[ // Turn sequence."));
        assert!(rendered.contains("PreviewTurn {"));
        assert!(rendered.contains("actions: list[ // Tool actions in this turn."));
        assert!(rendered.contains("PreviewAction {"));
        assert!(rendered.contains("name: string // Tool name."));
        assert!(rendered.contains("arguments: string // JSON arguments."));
        assert!(rendered.contains("result: string | null // Tool output."));
        assert!(rendered.contains("is_error: bool // True if the tool errored."));
        assert!(
            rendered.contains("trigger: string | null // User message that started this turn.")
        );
        assert!(!rendered.contains("Vec<"));
        assert!(!rendered.contains("String"));
        assert!(!rendered.contains("i64"));
        assert!(!rendered.contains("$self"));
    }

    #[test]
    fn shared_nested_type_is_rendered_once_then_referenced() {
        let input = DedupPreviewSigInput {
            actions: DuplicateActions {
                primary: vec![PreviewAction {
                    name: "search".to_string(),
                    arguments: "{}".to_string(),
                    result: None,
                    is_error: false,
                }],
                backup: vec![PreviewAction {
                    name: "grep".to_string(),
                    arguments: "{}".to_string(),
                    result: None,
                    is_error: false,
                }],
            },
        };

        let rendered =
            render_previews::<DedupPreviewSig>(&input, &BTreeMap::new(), &BTreeMap::new());
        assert_eq!(rendered.matches("PreviewAction {").count(), 1);
        assert_eq!(rendered.matches("name: string // Tool name.").count(), 1);
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    enum UnionIndentChoice {
        First { data: PreviewSession },
        Second { data: PreviewAction },
    }

    #[derive(Signature, Clone, Debug)]
    struct UnionIndentSig {
        #[input]
        choice: UnionIndentChoice,

        #[output]
        answer: String,
    }

    #[test]
    fn union_option_continuations_preserve_nested_indentation() {
        let input = UnionIndentSigInput {
            choice: UnionIndentChoice::First {
                data: PreviewSession {
                    brief: Some("Investigate signal drop".to_string()),
                    turns: vec![PreviewTurn {
                        trigger: Some("start".to_string()),
                        actions: vec![PreviewAction {
                            name: "search".to_string(),
                            arguments: "{\"q\":\"start\"}".to_string(),
                            result: Some("ok".to_string()),
                            is_error: false,
                        }],
                    }],
                },
            },
        };

        let rendered =
            render_previews::<UnionIndentSig>(&input, &BTreeMap::new(), &BTreeMap::new());
        let option_indent = rendered
            .lines()
            .find(|line| line.contains("- PreviewSession {"))
            .map(|line| line.chars().take_while(|ch| *ch == ' ').count())
            .expect("union option line present");
        let brief_indent = rendered
            .lines()
            .find(|line| line.contains("brief: string | null"))
            .map(|line| line.chars().take_while(|ch| *ch == ' ').count())
            .expect("nested brief line present");
        assert!(
            brief_indent > option_indent,
            "nested field indent should be deeper than its union option line"
        );
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct WrapA {
        value: String,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct WrapB {
        value: String,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    enum WrapperUnion {
        First { data: WrapA },
        Second { data: WrapB },
    }

    #[derive(Signature, Clone, Debug)]
    struct WrapperUnionSig {
        #[input]
        item: WrapperUnion,

        #[output]
        answer: String,
    }

    #[test]
    fn single_payload_data_enum_variants_render_as_payload_union_arms() {
        let input = WrapperUnionSigInput {
            item: WrapperUnion::First {
                data: WrapA {
                    value: "x".to_string(),
                },
            },
        };
        let rendered =
            render_previews::<WrapperUnionSig>(&input, &BTreeMap::new(), &BTreeMap::new());

        assert!(rendered.contains("one of:"));
        assert!(rendered.contains("WrapA {"));
        assert!(rendered.contains("WrapB {"));
        assert!(!rendered.contains("WrapperUnion_First {"));
        assert!(!rendered.contains("WrapperUnion_Second {"));
        assert!(!rendered.contains("type: String(\"First\")"));
        assert!(!rendered.contains("type: String(\"Second\")"));
        assert!(!rendered.contains("data: WrapA {"));
        assert!(!rendered.contains("data: WrapB {"));
    }

    #[test]
    fn render_method_line_collapses_multiline_docs_to_single_line() {
        let method = MethodSignature {
            name: "after".to_string(),
            signature: "(date)".to_string(),
            doc: "Returns `Sessions`: sessions on or after an ISO date prefix like `2026-02-25`.\n\nReturns a `Sessions` sub-collection so calls can be chained.".to_string(),
            source: super::super::runtime::MethodSource::Custom,
            is_dunder: false,
        };

        let rendered = render_method_line(&method);
        assert_eq!(
            rendered,
            ".after(date) // Returns `Sessions`: sessions on or after an ISO date prefix like `2026-02-25`. Returns a `Sessions` sub-collection so calls can be chained."
        );
    }

    #[test]
    fn nested_class_methods_render_when_provided_by_type_name() {
        let input = RichPreviewSigInput {
            sessions: PreviewSessions {
                items: vec![PreviewSession {
                    brief: Some("Investigate signal drop".to_string()),
                    turns: vec![PreviewTurn {
                        trigger: Some("start".to_string()),
                        actions: vec![PreviewAction {
                            name: "search".to_string(),
                            arguments: "{\"q\":\"start\"}".to_string(),
                            result: Some("ok".to_string()),
                            is_error: false,
                        }],
                    }],
                }],
            },
        };
        let methods_by_type = BTreeMap::from([(
            "PreviewSession".to_string(),
            vec![MethodSignature {
                name: "thread".to_string(),
                signature: "(participants)".to_string(),
                doc: "Conversation view for selected participants.".to_string(),
                source: super::super::runtime::MethodSource::Custom,
                is_dunder: false,
            }],
        )]);

        let rendered =
            render_previews::<RichPreviewSig>(&input, &BTreeMap::new(), &methods_by_type);
        assert!(rendered.contains("PreviewSession {"));
        assert!(rendered.contains(".thread(participants) // Conversation view for selected participants."));
    }

    #[test]
    fn methods_without_docstrings_are_rendered_without_comment_suffix() {
        let input = RichPreviewSigInput {
            sessions: PreviewSessions {
                items: vec![PreviewSession {
                    brief: Some("Investigate signal drop".to_string()),
                    turns: vec![PreviewTurn {
                        trigger: Some("start".to_string()),
                        actions: vec![PreviewAction {
                            name: "search".to_string(),
                            arguments: "{\"q\":\"start\"}".to_string(),
                            result: Some("ok".to_string()),
                            is_error: false,
                        }],
                    }],
                }],
            },
        };
        let methods = BTreeMap::from([(
            "sessions".to_string(),
            vec![MethodSignature {
                name: "undocumented".to_string(),
                signature: "()".to_string(),
                doc: "".to_string(),
                source: super::super::runtime::MethodSource::Custom,
                is_dunder: false,
            }],
        )]);

        let rendered = render_previews::<RichPreviewSig>(&input, &methods, &BTreeMap::new());
        assert!(rendered.contains(".undocumented()"));
        assert!(!rendered.contains(".undocumented() //"));
    }
}
