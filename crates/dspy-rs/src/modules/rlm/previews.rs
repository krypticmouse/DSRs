use std::collections::{BTreeMap, BTreeSet};

use bamltype::baml_types::ir_type::{TypeGeneric, UnionTypeViewGeneric};
use bamltype::baml_types::{BamlMap, TypeIR};

use crate::{
    BamlType, BamlValue, ConstraintKind, Facet, FieldSchema, OutputFormatContent, Signature,
    SignatureSchema,
};

const TOP_LEVEL_STRING_LIMIT: usize = 500;
const NESTED_STRING_LIMIT: usize = 100;
const STRUCT_PREVIEW_DEPTH_CAP: usize = 2;
const STRUCT_PREVIEW_BREADTH_CAP: usize = 8;
const SOFT_PREVIEW_BUDGET: usize = 4 * 1024;
const FIELD_STATS_FULL_SCAN: usize = 2_000;
const FIELD_STATS_SAMPLE: usize = 512;

#[derive(Clone, Copy)]
struct RenderBudget {
    top_level_limit: usize,
    nested_limit: usize,
    include_middle_samples: bool,
}

impl RenderBudget {
    const fn default() -> Self {
        Self {
            top_level_limit: TOP_LEVEL_STRING_LIMIT,
            nested_limit: NESTED_STRING_LIMIT,
            include_middle_samples: true,
        }
    }

    const fn shorter_strings() -> Self {
        Self {
            top_level_limit: 320,
            nested_limit: 64,
            include_middle_samples: true,
        }
    }

    const fn no_middle_samples() -> Self {
        Self {
            top_level_limit: 200,
            nested_limit: 48,
            include_middle_samples: false,
        }
    }
}

pub(super) fn render_previews<S: Signature>(input: &S::Input) -> String
where
    S::Input: BamlType + for<'a> Facet<'a>,
{
    let schema = SignatureSchema::of::<S>();
    let root = input.to_baml_value();
    let input_format = <S::Input as BamlType>::baml_output_format();

    let budgets = [
        RenderBudget::default(),
        RenderBudget::shorter_strings(),
        RenderBudget::no_middle_samples(),
    ];

    for budget in budgets {
        let rendered = render_with_budget(schema, &root, input_format, budget);
        if rendered.chars().count() <= SOFT_PREVIEW_BUDGET || !budget.include_middle_samples {
            return rendered;
        }
    }

    String::new()
}

fn render_with_budget(
    schema: &SignatureSchema,
    root: &BamlValue,
    input_format: &OutputFormatContent,
    budget: RenderBudget,
) -> String {
    let mut lines = vec!["## Variables".to_string(), String::new()];

    for field in schema.input_fields() {
        lines.push(format!(
            "{}: {}",
            field.lm_name,
            field.type_ir.diagnostic_repr()
        ));

        if !field.docs.trim().is_empty() {
            lines.push(format!("  {}", field.docs.trim()));
        }

        for constraint in field.constraints {
            let marker = match constraint.kind {
                ConstraintKind::Check => "soft",
                ConstraintKind::Assert => "hard",
            };
            lines.push(format!(
                "  Constraint: {marker} {} ({})",
                constraint.label, constraint.expression
            ));
        }

        if let Some(value) = schema.navigate_field(field.path(), root) {
            for line in
                render_value_block(value, Some(&field.type_ir), field, input_format, 0, budget)
            {
                lines.push(format!("  {line}"));
            }
        } else {
            lines.push("  <missing>".to_string());
        }

        lines.push(String::new());
    }

    lines.push("## Expected Output".to_string());
    for field in schema.output_fields() {
        lines.push(format!(
            "{}: {}",
            field.lm_name,
            field.type_ir.diagnostic_repr()
        ));
    }

    while lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }

    lines.join("\n")
}

fn render_value_block(
    value: &BamlValue,
    type_ir: Option<&TypeIR>,
    field: &FieldSchema,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    if let Some(inner) = optional_inner(type_ir) {
        if matches!(value, BamlValue::Null) {
            return vec!["None".to_string()];
        }
        let mut lines = vec!["(Present)".to_string()];
        lines.extend(render_value_block(
            value,
            Some(inner),
            field,
            input_format,
            depth,
            budget,
        ));
        return lines;
    }

    match value {
        BamlValue::String(text) => render_string_block(text, depth > 0, budget),
        BamlValue::Int(v) => vec![format!("Value: {v}")],
        BamlValue::Float(v) => vec![format!("Value: {v}")],
        BamlValue::Bool(v) => vec![format!("Value: {v}")],
        BamlValue::Null => vec!["None".to_string()],
        BamlValue::Enum(_, variant) => vec![format!("Variant: {variant}")],
        BamlValue::Media(_) => vec!["Media (preview omitted)".to_string()],
        BamlValue::List(items) => {
            render_list_block(items, item_type(type_ir), input_format, depth, budget)
        }
        BamlValue::Map(map) => {
            render_map_block(map, map_value_type(type_ir), input_format, depth, budget)
        }
        BamlValue::Class(class_name, fields) => {
            render_struct_block(class_name, fields, type_ir, input_format, depth, budget)
        }
    }
}

fn render_string_block(text: &str, nested: bool, budget: RenderBudget) -> Vec<String> {
    if nested {
        return vec![truncate_string(text, budget.nested_limit)];
    }

    let len = text.chars().count();
    let lines = text.lines().count().max(1);
    let mut out = vec![format!("Length: {len} chars, Lines: {lines}")];

    if len > 50 {
        if let Some(summary) = summarize_json_string(text) {
            out.push(format!("(JSON String) {summary}"));
        }
    }

    out.push(format!(
        "Value: {}",
        truncate_string(text, budget.top_level_limit)
    ));
    out
}

fn render_list_block(
    items: &[BamlValue],
    item_type: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    let mut lines = vec![format!("Count: {} items", items.len())];

    if let Some(schema_line) = class_schema_line(item_type, input_format) {
        lines.push(format!("Schema: {schema_line}"));
    }

    if let Some(distribution) = scalar_distribution(items) {
        lines.push(format!("Distribution: {distribution}"));
    }

    if let Some(stats) = compute_field_stats(items) {
        lines.push(format!("Field stats: {}", stats.summary));
        if let Some(note) = stats.sampling_note {
            lines.push(note);
        }
    }

    if depth >= STRUCT_PREVIEW_DEPTH_CAP {
        return lines;
    }

    for idx in sample_indices(items.len(), depth, budget.include_middle_samples) {
        let rendered = render_inline_value(&items[idx], item_type, input_format, depth + 1, budget);
        lines.push(format!("Sample [{idx}]: {rendered}"));
    }

    lines
}

fn render_map_block(
    map: &BamlMap<String, BamlValue>,
    value_type: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    let mut lines = vec![format!("Keys: {} items", map.len())];
    if map.is_empty() || depth >= STRUCT_PREVIEW_DEPTH_CAP {
        return lines;
    }

    let mut keys = map.keys().collect::<Vec<_>>();
    keys.sort_unstable();

    for idx in sample_indices(keys.len(), depth, budget.include_middle_samples) {
        let key = keys[idx];
        if let Some(value) = map.get(key) {
            let rendered = render_inline_value(value, value_type, input_format, depth + 1, budget);
            lines.push(format!("Sample [{key:?}]: {rendered}"));
        }
    }

    lines
}

fn render_struct_block(
    class_name: &str,
    fields: &BamlMap<String, BamlValue>,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    let mut lines = vec![];

    if let Some(schema_line) = class_schema_line(type_ir, input_format) {
        lines.push(format!("Schema: {schema_line}"));
    } else {
        lines.push(format!("Schema: {}", fallback_schema(fields)));
    }

    if depth >= STRUCT_PREVIEW_DEPTH_CAP {
        lines.push(format!("Preview: {class_name} ({} fields)", fields.len()));
        return lines;
    }

    let preview = render_inline_struct(class_name, fields, type_ir, input_format, depth, budget);
    lines.push(format!("Preview: {preview}"));
    lines
}

fn render_inline_value(
    value: &BamlValue,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> String {
    if let Some(inner) = optional_inner(type_ir) {
        return if matches!(value, BamlValue::Null) {
            "None".to_string()
        } else {
            format!(
                "(Present) {}",
                render_inline_value(value, Some(inner), input_format, depth, budget)
            )
        };
    }

    match value {
        BamlValue::String(text) => truncate_string(text, budget.nested_limit),
        BamlValue::Int(v) => v.to_string(),
        BamlValue::Float(v) => v.to_string(),
        BamlValue::Bool(v) => v.to_string(),
        BamlValue::Null => "None".to_string(),
        BamlValue::Enum(_, variant) => variant.to_string(),
        BamlValue::Media(_) => "Media".to_string(),
        BamlValue::List(items) => {
            if depth >= STRUCT_PREVIEW_DEPTH_CAP {
                return format!("Count: {} items", items.len());
            }
            let idxs = sample_indices(items.len(), depth, budget.include_middle_samples);
            let inner = item_type(type_ir);
            if idxs.is_empty() {
                return "Count: 0 items".to_string();
            }
            let samples = idxs
                .iter()
                .map(|idx| {
                    format!(
                        "sample[{idx}]={}",
                        render_inline_value(&items[*idx], inner, input_format, depth + 1, budget,)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("Count: {} items; {samples}", items.len())
        }
        BamlValue::Map(map) => {
            if depth >= STRUCT_PREVIEW_DEPTH_CAP {
                return format!("Keys: {} items", map.len());
            }
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let value_type = map_value_type(type_ir);
            let pairs = sample_indices(keys.len(), depth, budget.include_middle_samples)
                .into_iter()
                .filter_map(|idx| {
                    let key = keys[idx];
                    map.get(key).map(|entry| {
                        format!(
                            "{key:?}: {}",
                            render_inline_value(entry, value_type, input_format, depth + 1, budget,)
                        )
                    })
                })
                .collect::<Vec<_>>();
            if pairs.is_empty() {
                format!("Keys: {} items", map.len())
            } else {
                format!("Keys: {} items; {}", map.len(), pairs.join(", "))
            }
        }
        BamlValue::Class(class_name, fields) => {
            render_inline_struct(class_name, fields, type_ir, input_format, depth, budget)
        }
    }
}

fn render_inline_struct(
    class_name: &str,
    fields: &BamlMap<String, BamlValue>,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> String {
    if depth >= STRUCT_PREVIEW_DEPTH_CAP {
        return format!("{class_name} ({} fields)", fields.len());
    }

    let ordered = ordered_struct_fields(fields, type_ir, input_format);
    let mut parts = Vec::new();

    for (idx, (name, value, child_ty)) in ordered.into_iter().enumerate() {
        if idx >= STRUCT_PREVIEW_BREADTH_CAP {
            break;
        }
        let rendered = match value {
            BamlValue::String(text) => truncate_string(text, budget.nested_limit),
            BamlValue::Int(v) => v.to_string(),
            BamlValue::Float(v) => v.to_string(),
            BamlValue::Bool(v) => v.to_string(),
            BamlValue::Null => "None".to_string(),
            BamlValue::Enum(_, variant) => variant.to_string(),
            BamlValue::Media(_) => "Media".to_string(),
            BamlValue::List(items) => format!("Count: {} items", items.len()),
            BamlValue::Map(map) => format!("Keys: {} items", map.len()),
            BamlValue::Class(inner_name, inner_fields) => {
                let _ = child_ty;
                let _ = input_format;
                let _ = budget;
                format!("{inner_name} ({} fields)", inner_fields.len())
            }
        };
        parts.push(format!("{name}: {rendered}"));
    }

    if fields.len() > STRUCT_PREVIEW_BREADTH_CAP {
        parts.push(format!(
            "... (+{} fields)",
            fields.len() - STRUCT_PREVIEW_BREADTH_CAP
        ));
    }

    format!("{class_name} {{ {} }}", parts.join(", "))
}

fn ordered_struct_fields<'a>(
    fields: &'a BamlMap<String, BamlValue>,
    type_ir: Option<&'a TypeIR>,
    input_format: &'a OutputFormatContent,
) -> Vec<(&'a str, &'a BamlValue, Option<&'a TypeIR>)> {
    if let Some((class_name, mode)) = class_type_ref(type_ir) {
        if let Some(class) = input_format.classes.get(&(class_name.to_string(), mode)) {
            let mut ordered = Vec::new();
            for (field_name, field_ty, _, _) in &class.fields {
                let key = field_name.real_name();
                if let Some(value) = fields.get(key) {
                    ordered.push((key, value, Some(field_ty)));
                }
            }

            for (key, value) in fields {
                if !ordered
                    .iter()
                    .any(|(existing, _, _)| *existing == key.as_str())
                {
                    ordered.push((key.as_str(), value, None));
                }
            }
            return ordered;
        }
    }

    let mut fallback = fields
        .iter()
        .map(|(key, value)| (key.as_str(), value, None))
        .collect::<Vec<_>>();
    fallback.sort_by(|a, b| a.0.cmp(b.0));
    fallback
}

fn class_schema_line(
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
) -> Option<String> {
    let (class_name, mode) = class_type_ref(type_ir)?;
    let class = input_format.classes.get(&(class_name.to_string(), mode))?;

    let mut fields = class
        .fields
        .iter()
        .take(STRUCT_PREVIEW_BREADTH_CAP)
        .map(|(field_name, field_type, _, _)| {
            format!(
                "{}: {}",
                field_name.real_name(),
                field_type.diagnostic_repr()
            )
        })
        .collect::<Vec<_>>();

    if class.fields.len() > STRUCT_PREVIEW_BREADTH_CAP {
        fields.push(format!(
            "... (+{} fields)",
            class.fields.len() - STRUCT_PREVIEW_BREADTH_CAP
        ));
    }

    Some(format!("{{ {} }}", fields.join(", ")))
}

fn fallback_schema(fields: &BamlMap<String, BamlValue>) -> String {
    let mut keys = fields.keys().collect::<Vec<_>>();
    keys.sort_unstable();
    let mut parts = keys
        .iter()
        .take(STRUCT_PREVIEW_BREADTH_CAP)
        .filter_map(|key| {
            fields
                .get(*key)
                .map(|value| format!("{key}: {}", primitive_type_name(value)))
        })
        .collect::<Vec<_>>();

    if fields.len() > STRUCT_PREVIEW_BREADTH_CAP {
        parts.push(format!(
            "... (+{} fields)",
            fields.len() - STRUCT_PREVIEW_BREADTH_CAP
        ));
    }

    format!("{{ {} }}", parts.join(", "))
}

fn primitive_type_name(value: &BamlValue) -> &'static str {
    match value {
        BamlValue::String(_) => "string",
        BamlValue::Int(_) => "int",
        BamlValue::Float(_) => "float",
        BamlValue::Bool(_) => "bool",
        BamlValue::Map(_) => "map",
        BamlValue::List(_) => "list",
        BamlValue::Media(_) => "media",
        BamlValue::Enum(_, _) => "enum",
        BamlValue::Class(_, _) => "class",
        BamlValue::Null => "null",
    }
}

fn sample_indices(len: usize, depth: usize, include_middle: bool) -> Vec<usize> {
    if len == 0 || depth >= STRUCT_PREVIEW_DEPTH_CAP {
        return Vec::new();
    }

    if depth == 1 {
        return vec![0];
    }

    if len <= 3 {
        return (0..len).collect();
    }

    let mut indices = vec![0, len - 1];
    if include_middle {
        indices.push(len / 2);
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn optional_inner(type_ir: Option<&TypeIR>) -> Option<&TypeIR> {
    match type_ir {
        Some(TypeGeneric::Union(union, _)) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => Some(inner),
            _ => None,
        },
        _ => None,
    }
}

fn item_type(type_ir: Option<&TypeIR>) -> Option<&TypeIR> {
    match type_ir {
        Some(TypeGeneric::List(inner, _)) => Some(inner),
        Some(TypeGeneric::Union(union, _)) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => item_type(Some(inner)),
            _ => None,
        },
        _ => None,
    }
}

fn map_value_type(type_ir: Option<&TypeIR>) -> Option<&TypeIR> {
    match type_ir {
        Some(TypeGeneric::Map(_, value, _)) => Some(value),
        Some(TypeGeneric::Union(union, _)) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => map_value_type(Some(inner)),
            _ => None,
        },
        _ => None,
    }
}

fn class_type_ref(type_ir: Option<&TypeIR>) -> Option<(&str, bamltype::baml_types::StreamingMode)> {
    match type_ir {
        Some(TypeGeneric::Class { name, mode, .. }) => Some((name.as_str(), *mode)),
        Some(TypeGeneric::Union(union, _)) => match union.view() {
            UnionTypeViewGeneric::Optional(inner) => class_type_ref(Some(inner)),
            _ => None,
        },
        _ => None,
    }
}

fn truncate_string(text: &str, limit: usize) -> String {
    let total = text.chars().count();
    if total <= limit {
        return format!("{:?}", text);
    }

    let head_len = limit / 2;
    let tail_len = limit.saturating_sub(head_len);
    let head = text.chars().take(head_len).collect::<String>();
    let tail = text
        .chars()
        .rev()
        .take(tail_len)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();

    format!(
        "{:?} ... ({} chars omitted) ... {:?}",
        head,
        total.saturating_sub(head_len + tail_len),
        tail
    )
}

fn summarize_json_string(text: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(text).ok()?;

    match value {
        serde_json::Value::Object(map) => {
            let mut keys = map.keys().cloned().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut summary = keys.iter().take(8).cloned().collect::<Vec<_>>().join(", ");
            if keys.len() > 8 {
                summary.push_str(&format!(", ... (+{} keys)", keys.len() - 8));
            }
            Some(format!("object keys: {summary}"))
        }
        serde_json::Value::Array(items) => {
            let first = items.first().map(json_value_name).unwrap_or("empty");
            Some(format!(
                "array with {} items (first item type: {first})",
                items.len()
            ))
        }
        _ => None,
    }
}

fn json_value_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn scalar_distribution(items: &[BamlValue]) -> Option<String> {
    if items.is_empty() {
        return None;
    }

    let mut numeric = Vec::new();
    let mut t = 0usize;
    let mut f = 0usize;
    let mut variants: BTreeMap<String, usize> = BTreeMap::new();

    for item in items {
        match item {
            BamlValue::Int(v) => numeric.push(*v as f64),
            BamlValue::Float(v) => numeric.push(*v),
            BamlValue::Bool(true) => t += 1,
            BamlValue::Bool(false) => f += 1,
            BamlValue::Enum(_, variant) => *variants.entry(variant.clone()).or_insert(0) += 1,
            _ => return None,
        }
    }

    if !numeric.is_empty() {
        let min = numeric.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
        let max = numeric.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
        let mean = numeric.iter().sum::<f64>() / numeric.len() as f64;
        return Some(format!(
            "min={}; max={}; mean={}",
            number(min),
            number(max),
            number(mean)
        ));
    }

    if t + f == items.len() {
        return Some(format!("true={t}, false={f}"));
    }

    if !variants.is_empty() {
        return Some(
            variants
                .into_iter()
                .map(|(variant, count)| format!("{variant}: {count}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    None
}

struct FieldStats {
    summary: String,
    sampling_note: Option<String>,
}

#[derive(Default)]
struct FieldAgg {
    present: usize,
    missing: usize,
    strings: Vec<usize>,
    unique_values: BTreeSet<String>,
    numbers: Vec<f64>,
    bool_true: usize,
    bool_false: usize,
}

fn compute_field_stats(items: &[BamlValue]) -> Option<FieldStats> {
    if items.is_empty() {
        return None;
    }

    let rows = items
        .iter()
        .map(|item| match item {
            BamlValue::Class(_, fields) | BamlValue::Map(fields) => Some(fields),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let sample_indices = if rows.len() <= FIELD_STATS_FULL_SCAN {
        (0..rows.len()).collect::<Vec<_>>()
    } else {
        stride_sample(rows.len(), FIELD_STATS_SAMPLE)
    };

    let sampled_rows = sample_indices
        .iter()
        .map(|idx| rows[*idx])
        .collect::<Vec<_>>();

    let mut field_names = BTreeSet::new();
    for row in &sampled_rows {
        for key in row.keys() {
            field_names.insert(key.clone());
        }
    }

    let mut parts = Vec::new();

    for field_name in field_names {
        let mut agg = FieldAgg::default();
        for row in &sampled_rows {
            match row.get(&field_name) {
                None | Some(BamlValue::Null) => agg.missing += 1,
                Some(BamlValue::String(v)) => {
                    agg.present += 1;
                    agg.strings.push(v.chars().count());
                    if agg.unique_values.len() <= 4096 {
                        agg.unique_values.insert(v.clone());
                    }
                }
                Some(BamlValue::Int(v)) => {
                    agg.present += 1;
                    agg.numbers.push(*v as f64);
                }
                Some(BamlValue::Float(v)) => {
                    agg.present += 1;
                    agg.numbers.push(*v);
                }
                Some(BamlValue::Bool(true)) => {
                    agg.present += 1;
                    agg.bool_true += 1;
                }
                Some(BamlValue::Bool(false)) => {
                    agg.present += 1;
                    agg.bool_false += 1;
                }
                Some(v) => {
                    agg.present += 1;
                    if let BamlValue::Enum(_, variant) = v {
                        if agg.unique_values.len() <= 4096 {
                            agg.unique_values.insert(variant.clone());
                        }
                    }
                }
            }
        }

        let mut rendered = if !agg.numbers.is_empty() && agg.numbers.len() == agg.present {
            let min = agg.numbers.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
            let max = agg
                .numbers
                .iter()
                .fold(f64::NEG_INFINITY, |acc, v| acc.max(*v));
            let mean = agg.numbers.iter().sum::<f64>() / agg.numbers.len() as f64;
            format!(
                "min={}; max={}; mean={}",
                number(min),
                number(max),
                number(mean)
            )
        } else if !agg.strings.is_empty() && agg.strings.len() == agg.present {
            let min = agg.strings.iter().min().copied().unwrap_or(0);
            let max = agg.strings.iter().max().copied().unwrap_or(0);
            if is_categorical_field(&field_name, agg.unique_values.len(), agg.present) {
                format!("{} unique values", agg.unique_values.len())
            } else {
                format!("{min}-{max} chars")
            }
        } else if agg.bool_true + agg.bool_false == agg.present {
            format!("true={}, false={}", agg.bool_true, agg.bool_false)
        } else {
            continue;
        };

        if agg.missing > 0 {
            let pct = ((agg.missing as f64 / sampled_rows.len() as f64) * 100.0).round() as usize;
            rendered.push_str(&format!(", {pct}% null"));
        }

        parts.push(format!("{field_name}: {rendered}"));
    }

    if parts.is_empty() {
        return None;
    }

    Some(FieldStats {
        summary: parts.join("; "),
        sampling_note: (rows.len() > FIELD_STATS_FULL_SCAN)
            .then(|| format!("(sampled {} of {})", sampled_rows.len(), rows.len())),
    })
}

fn stride_sample(total: usize, target: usize) -> Vec<usize> {
    if total <= target {
        return (0..total).collect();
    }

    let step = total as f64 / target as f64;
    let mut out = (0..target)
        .map(|i| ((i as f64) * step).floor() as usize)
        .collect::<Vec<_>>();

    if let Some(last) = out.last_mut() {
        *last = total - 1;
    }

    out.sort_unstable();
    out.dedup();
    out
}

fn is_categorical_field(name: &str, unique: usize, present: usize) -> bool {
    let lowered = name.to_ascii_lowercase();
    if ["id", "type", "category", "status", "label"]
        .iter()
        .any(|token| lowered.contains(token))
    {
        return true;
    }

    unique <= 32 || (unique as f64 / present.max(1) as f64) <= 0.2
}

fn number(value: f64) -> String {
    if (value.fract()).abs() < f64::EPSILON {
        format!("{value:.0}")
    } else {
        format!("{value:.3}")
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{BamlType, Signature};

    use super::render_previews;

    #[derive(Clone, Debug)]
    #[BamlType]
    struct Paper {
        title: String,
        abstract_text: String,
        year: i32,
        category: String,
        email: Option<String>,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    enum State {
        Ready,
        Failed,
    }

    #[derive(Signature, Clone, Debug)]
    /// Scalar preview test.
    struct ScalarSig {
        #[input]
        #[check("this != ''", label = "non_empty")]
        text: String,

        #[input]
        payload: String,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    /// Collection preview test.
    struct CollectionSig {
        #[input]
        papers: Vec<Paper>,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    /// Mixed preview test.
    struct MixedSig {
        #[input]
        maybe_note: Option<String>,

        #[input]
        scores: HashMap<String, i32>,

        #[input]
        states: Vec<State>,

        #[input]
        nested: Vec<Vec<String>>,

        #[output]
        ok: bool,
    }

    #[test]
    fn scalar_preview_shows_truncation_json_and_constraints() {
        let rendered = render_previews::<ScalarSig>(&ScalarSigInput {
            text: "x".repeat(620),
            payload: serde_json::json!({
                "a": 1,
                "b": [1, 2, 3],
                "longer_payload_to_force_json_detection": "abcdefghijklmnopqrstuvwxyz"
            })
            .to_string(),
        });

        assert!(rendered.contains("Constraint: soft non_empty"));
        assert!(rendered.contains("Length: 620 chars"));
        assert!(rendered.contains("chars omitted"));
        assert!(
            rendered.contains("(JSON String) object keys")
                || rendered.contains("(JSON String)")
                || rendered.contains("object keys")
        );
        assert!(rendered.contains("## Expected Output"));
    }

    #[test]
    fn collection_preview_shows_samples_and_field_stats() {
        let papers = (0..2_101)
            .map(|idx| Paper {
                title: format!("paper-{idx}"),
                abstract_text: "z".repeat(180 + (idx % 20) as usize),
                year: 2020 + (idx % 5),
                category: format!("cat-{}", idx % 7),
                email: (idx % 3 == 0).then(|| format!("author{idx}@example.com")),
            })
            .collect::<Vec<_>>();

        let rendered = render_previews::<CollectionSig>(&CollectionSigInput { papers });

        assert!(rendered.contains("Count: 2101 items"));
        assert!(rendered.contains("Field stats:"));
        assert!(rendered.contains("(sampled"));
        assert!(rendered.contains("Sample [0]:"));
        assert!(rendered.contains("Sample [1050]:"));
        assert!(rendered.contains("Sample [2100]:"));
    }

    #[test]
    fn mixed_preview_shows_optional_map_enum_and_nested_depth_rules() {
        let mut scores = HashMap::new();
        scores.insert("zeta".to_string(), 9);
        scores.insert("alpha".to_string(), 1);
        scores.insert("middle".to_string(), 5);
        scores.insert("omega".to_string(), 7);

        let rendered = render_previews::<MixedSig>(&MixedSigInput {
            maybe_note: None,
            scores,
            states: vec![State::Ready, State::Failed, State::Ready],
            nested: vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string(), "d".to_string()],
                vec!["e".to_string(), "f".to_string()],
            ],
        });

        assert!(rendered.contains("maybe_note"));
        assert!(rendered.contains("None"));
        assert!(rendered.contains("Keys: 4 items"));
        assert!(rendered.contains("Sample [\"alpha\"]"));
        assert!(rendered.contains("Sample [\"omega\"]"));
        assert!(rendered.contains("Distribution:"));
        assert!(rendered.contains("Ready: 2"));
        assert!(rendered.contains("Count: 3 items"));
    }
}
