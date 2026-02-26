use std::collections::{BTreeMap, BTreeSet};

use bamltype::baml_types::TypeIR;
use bamltype::baml_types::ir_type::{TypeGeneric, UnionTypeViewGeneric};
use bamltype::facet::{Type, UserType};
use bamltype::facet_reflect::{HasFields, Peek};

use crate::{
    BamlType, ConstraintKind, Facet, FieldPath, OutputFormatContent, Signature, SignatureSchema,
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
    let root = Peek::new(input);
    let input_format = <S::Input as BamlType>::baml_output_format();

    let budgets = [
        RenderBudget::default(),
        RenderBudget::shorter_strings(),
        RenderBudget::no_middle_samples(),
    ];

    for budget in budgets {
        let rendered = render_with_budget(schema, root, input_format, budget);
        if rendered.chars().count() <= SOFT_PREVIEW_BUDGET || !budget.include_middle_samples {
            return rendered;
        }
    }

    String::new()
}

fn render_with_budget(
    schema: &SignatureSchema,
    root: Peek<'_, '_>,
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

        if let Some(value) = peek_at_field_path(root, field.path()) {
            for line in render_value_block(value, Some(&field.type_ir), input_format, 0, budget) {
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
    value: Peek<'_, '_>,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    if let Some(inner) = optional_inner(type_ir) {
        return match summarize_peek(value) {
            PeekSummary::None => vec!["None".to_string()],
            summary => {
                let mut lines = vec!["(Present)".to_string()];
                lines.extend(render_value_block_from_summary(
                    summary,
                    Some(inner),
                    input_format,
                    depth,
                    budget,
                ));
                lines
            }
        };
    }
    render_value_block_from_summary(summarize_peek(value), type_ir, input_format, depth, budget)
}

#[derive(Clone)]
enum StructFieldValue<'mem, 'facet> {
    Peek(Peek<'mem, 'facet>),
    String(String),
}

enum PeekSummary<'mem, 'facet> {
    None,
    Media,
    String(&'mem str),
    Bool(bool),
    SignedInt(i128),
    UnsignedInt(u128),
    Float(f64),
    UnitEnum(String),
    List(bamltype::facet_reflect::PeekListLike<'mem, 'facet>),
    Map(Vec<(String, Peek<'mem, 'facet>)>),
    StructLike {
        class_name: String,
        fields: Vec<(String, StructFieldValue<'mem, 'facet>)>,
    },
    Unknown(Peek<'mem, 'facet>),
}

fn summarize_peek<'mem, 'facet>(value: Peek<'mem, 'facet>) -> PeekSummary<'mem, 'facet> {
    let Some(value) = collapse_option_chain(value) else {
        return PeekSummary::None;
    };

    if is_media_shape(value.shape()) {
        return PeekSummary::Media;
    }

    if let Some(text) = value.as_str() {
        return PeekSummary::String(text);
    }
    if let Ok(v) = value.get::<bool>() {
        return PeekSummary::Bool(*v);
    }
    if let Some(v) = peek_signed_i128(value) {
        return PeekSummary::SignedInt(v);
    }
    if let Some(v) = peek_unsigned_u128(value) {
        return PeekSummary::UnsignedInt(v);
    }
    if let Ok(v) = value.get::<f64>() {
        return PeekSummary::Float(*v);
    }
    if let Ok(v) = value.get::<f32>() {
        return PeekSummary::Float(*v as f64);
    }
    if let Some(v) = unit_enum_variant(value) {
        return PeekSummary::UnitEnum(v);
    }
    if let Ok(list) = value.into_list_like() {
        return PeekSummary::List(list);
    }
    if let Ok(map) = value.into_map() {
        return PeekSummary::Map(map_entries_for_preview(map));
    }
    if let Some((class_name, fields)) = struct_like_fields(value) {
        return PeekSummary::StructLike { class_name, fields };
    }
    // Keep unknown types generic; known media goes through `PeekSummary::Media`.
    PeekSummary::Unknown(value)
}

fn render_value_block_from_summary(
    summary: PeekSummary<'_, '_>,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    match summary {
        PeekSummary::None => vec!["None".to_string()],
        PeekSummary::Media => vec!["Media (preview omitted)".to_string()],
        PeekSummary::String(text) => render_string_block(text, depth > 0, budget),
        PeekSummary::Bool(v) => vec![format!("Value: {v}")],
        PeekSummary::SignedInt(v) => vec![format!("Value: {v}")],
        PeekSummary::UnsignedInt(v) => vec![format!("Value: {v}")],
        PeekSummary::Float(v) => vec![format!("Value: {v}")],
        PeekSummary::UnitEnum(variant) => vec![format!("Variant: {variant}")],
        PeekSummary::List(list) => {
            render_list_block(&list, item_type(type_ir), input_format, depth, budget)
        }
        PeekSummary::Map(entries) => render_map_block(
            &entries,
            map_value_type(type_ir),
            input_format,
            depth,
            budget,
        ),
        PeekSummary::StructLike { class_name, fields } => {
            render_struct_block(&class_name, &fields, type_ir, input_format, depth, budget)
        }
        PeekSummary::Unknown(value) => vec![format!("Value: {}", value)],
    }
}

fn render_string_block(text: &str, nested: bool, budget: RenderBudget) -> Vec<String> {
    if nested {
        return vec![truncate_string(text, budget.nested_limit)];
    }

    let len = text.chars().count();
    let lines = text.lines().count().max(1);
    let mut out = vec![format!("Length: {len} chars, Lines: {lines}")];

    if len > 50
        && let Some(summary) = summarize_json_string(text)
    {
        out.push(format!("(JSON String) {summary}"));
    }

    out.push(format!(
        "Value: {}",
        truncate_string(text, budget.top_level_limit)
    ));
    out
}

fn render_list_block(
    items: &bamltype::facet_reflect::PeekListLike<'_, '_>,
    item_type: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    let mut lines = vec![format!("Count: {} items", items.len())];

    if let Some(schema_line) = class_schema_line(item_type, input_format) {
        lines.push(format!("Schema: {schema_line}"));
    }

    let sample = items.iter().collect::<Vec<_>>();

    if let Some(distribution) = scalar_distribution(&sample) {
        lines.push(format!("Distribution: {distribution}"));
    }

    if let Some(stats) = compute_field_stats(&sample) {
        lines.push(format!("Field stats: {}", stats.summary));
        if let Some(note) = stats.sampling_note {
            lines.push(note);
        }
    }

    if depth >= STRUCT_PREVIEW_DEPTH_CAP {
        return lines;
    }

    for idx in sample_indices(items.len(), depth, budget.include_middle_samples) {
        if let Some(item) = items.get(idx) {
            let rendered = render_inline_value(item, item_type, input_format, depth + 1, budget);
            lines.push(format!("Sample [{idx}]: {rendered}"));
        }
    }

    lines
}

fn render_map_block(
    entries: &[(String, Peek<'_, '_>)],
    value_type: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> Vec<String> {
    let mut lines = vec![format!("Keys: {} items", entries.len())];
    if entries.is_empty() || depth >= STRUCT_PREVIEW_DEPTH_CAP {
        return lines;
    }

    for idx in sample_indices(entries.len(), depth, budget.include_middle_samples) {
        let (key, value) = &entries[idx];
        let rendered = render_inline_value(*value, value_type, input_format, depth + 1, budget);
        lines.push(format!("Sample [{key:?}]: {rendered}"));
    }

    lines
}

fn render_struct_block(
    class_name: &str,
    fields: &[(String, StructFieldValue<'_, '_>)],
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
    value: Peek<'_, '_>,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> String {
    if let Some(inner) = optional_inner(type_ir) {
        return match summarize_peek(value) {
            PeekSummary::None => "None".to_string(),
            summary => format!(
                "(Present) {}",
                render_inline_value_from_summary(summary, Some(inner), input_format, depth, budget)
            ),
        };
    }
    render_inline_value_from_summary(summarize_peek(value), type_ir, input_format, depth, budget)
}

fn render_inline_value_from_summary(
    summary: PeekSummary<'_, '_>,
    type_ir: Option<&TypeIR>,
    input_format: &OutputFormatContent,
    depth: usize,
    budget: RenderBudget,
) -> String {
    match summary {
        PeekSummary::None => "None".to_string(),
        PeekSummary::Media => "Media".to_string(),
        PeekSummary::String(text) => truncate_string(text, budget.nested_limit),
        PeekSummary::Bool(v) => v.to_string(),
        PeekSummary::SignedInt(v) => v.to_string(),
        PeekSummary::UnsignedInt(v) => v.to_string(),
        PeekSummary::Float(v) => v.to_string(),
        PeekSummary::UnitEnum(variant) => variant,
        PeekSummary::List(list) => {
            if depth >= STRUCT_PREVIEW_DEPTH_CAP {
                return format!("Count: {} items", list.len());
            }
            let idxs = sample_indices(list.len(), depth, budget.include_middle_samples);
            let inner = item_type(type_ir);
            if idxs.is_empty() {
                return "Count: 0 items".to_string();
            }
            let samples = idxs
                .iter()
                .filter_map(|idx| {
                    list.get(*idx).map(|item| {
                        format!(
                            "sample[{idx}]={}",
                            render_inline_value(item, inner, input_format, depth + 1, budget)
                        )
                    })
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("Count: {} items; {samples}", list.len())
        }
        PeekSummary::Map(entries) => {
            if depth >= STRUCT_PREVIEW_DEPTH_CAP {
                return format!("Keys: {} items", entries.len());
            }
            let value_type = map_value_type(type_ir);
            let pairs = sample_indices(entries.len(), depth, budget.include_middle_samples)
                .into_iter()
                .map(|idx| {
                    let (key, entry) = &entries[idx];
                    format!(
                        "{key:?}: {}",
                        render_inline_value(*entry, value_type, input_format, depth + 1, budget)
                    )
                })
                .collect::<Vec<_>>();
            if pairs.is_empty() {
                format!("Keys: {} items", entries.len())
            } else {
                format!("Keys: {} items; {}", entries.len(), pairs.join(", "))
            }
        }
        PeekSummary::StructLike { class_name, fields } => {
            render_inline_struct(&class_name, &fields, type_ir, input_format, depth, budget)
        }
        PeekSummary::Unknown(value) => format!("{}", value),
    }
}

fn render_inline_struct(
    class_name: &str,
    fields: &[(String, StructFieldValue<'_, '_>)],
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

    for (idx, (name, value, _child_ty)) in ordered.into_iter().enumerate() {
        if idx >= STRUCT_PREVIEW_BREADTH_CAP {
            break;
        }
        let rendered = render_inline_field_value(value, budget);
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

fn ordered_struct_fields<'a, 'mem, 'facet>(
    fields: &'a [(String, StructFieldValue<'mem, 'facet>)],
    type_ir: Option<&'a TypeIR>,
    input_format: &'a OutputFormatContent,
) -> Vec<(
    &'a str,
    &'a StructFieldValue<'mem, 'facet>,
    Option<&'a TypeIR>,
)> {
    if let Some((class_name, mode)) = class_type_ref(type_ir)
        && let Some(class) = input_format.classes.get(&(class_name.to_string(), mode))
    {
        let mut ordered = Vec::new();
        for (field_name, field_ty, _, _) in &class.fields {
            let key = field_name.real_name();
            if let Some((_, value)) = fields.iter().find(|(name, _)| name == key) {
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

fn fallback_schema(fields: &[(String, StructFieldValue<'_, '_>)]) -> String {
    let mut keys = fields
        .iter()
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    keys.sort_unstable();
    let mut parts = keys
        .iter()
        .take(STRUCT_PREVIEW_BREADTH_CAP)
        .filter_map(|key| {
            fields
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value)
                .map(|value| format!("{key}: {}", primitive_type_name(value)))
        })
        .collect::<Vec<_>>();

    if keys.len() > STRUCT_PREVIEW_BREADTH_CAP {
        parts.push(format!(
            "... (+{} fields)",
            keys.len() - STRUCT_PREVIEW_BREADTH_CAP
        ));
    }

    format!("{{ {} }}", parts.join(", "))
}

fn primitive_type_name(value: &StructFieldValue<'_, '_>) -> &'static str {
    match value {
        StructFieldValue::String(_) => "string",
        StructFieldValue::Peek(value) => primitive_type_name_peek(*value),
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

fn scalar_distribution(items: &[Peek<'_, '_>]) -> Option<String> {
    if items.is_empty() {
        return None;
    }

    let mut numeric = Vec::new();
    let mut t = 0usize;
    let mut f = 0usize;
    let mut variants: BTreeMap<String, usize> = BTreeMap::new();

    for item in items {
        match summarize_peek(*item) {
            PeekSummary::SignedInt(v) => numeric.push(v as f64),
            PeekSummary::UnsignedInt(v) => numeric.push(v as f64),
            PeekSummary::Float(v) => numeric.push(v),
            PeekSummary::Bool(true) => t += 1,
            PeekSummary::Bool(false) => f += 1,
            PeekSummary::UnitEnum(variant) => *variants.entry(variant).or_insert(0) += 1,
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

fn compute_field_stats(items: &[Peek<'_, '_>]) -> Option<FieldStats> {
    if items.is_empty() {
        return None;
    }

    let rows = items
        .iter()
        .map(|item| row_for_field_stats(*item))
        .collect::<Option<Vec<_>>>()?;

    let sample_indices = if rows.len() <= FIELD_STATS_FULL_SCAN {
        (0..rows.len()).collect::<Vec<_>>()
    } else {
        stride_sample(rows.len(), FIELD_STATS_SAMPLE)
    };

    let sampled_rows = sample_indices
        .iter()
        .map(|idx| &rows[*idx])
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
                None => agg.missing += 1,
                Some(StructFieldValue::String(v)) => {
                    agg.present += 1;
                    agg.strings.push(v.chars().count());
                    if agg.unique_values.len() <= 4096 {
                        agg.unique_values.insert(v.clone());
                    }
                }
                Some(StructFieldValue::Peek(value)) => match summarize_peek(*value) {
                    PeekSummary::None => agg.missing += 1,
                    PeekSummary::String(v) => {
                        agg.present += 1;
                        agg.strings.push(v.chars().count());
                        if agg.unique_values.len() <= 4096 {
                            agg.unique_values.insert(v.to_string());
                        }
                    }
                    PeekSummary::SignedInt(v) => {
                        agg.present += 1;
                        agg.numbers.push(v as f64);
                    }
                    PeekSummary::UnsignedInt(v) => {
                        agg.present += 1;
                        agg.numbers.push(v as f64);
                    }
                    PeekSummary::Float(v) => {
                        agg.present += 1;
                        agg.numbers.push(v);
                    }
                    PeekSummary::Bool(true) => {
                        agg.present += 1;
                        agg.bool_true += 1;
                    }
                    PeekSummary::Bool(false) => {
                        agg.present += 1;
                        agg.bool_false += 1;
                    }
                    PeekSummary::UnitEnum(variant) => {
                        agg.present += 1;
                        if agg.unique_values.len() <= 4096 {
                            agg.unique_values.insert(variant);
                        }
                    }
                    _ => agg.present += 1,
                },
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

fn render_inline_field_value(value: &StructFieldValue<'_, '_>, budget: RenderBudget) -> String {
    match value {
        StructFieldValue::String(text) => truncate_string(text, budget.nested_limit),
        StructFieldValue::Peek(value) => match summarize_peek(*value) {
            PeekSummary::None => "None".to_string(),
            PeekSummary::Media => "Media".to_string(),
            PeekSummary::String(text) => truncate_string(text, budget.nested_limit),
            PeekSummary::Bool(v) => v.to_string(),
            PeekSummary::SignedInt(v) => v.to_string(),
            PeekSummary::UnsignedInt(v) => v.to_string(),
            PeekSummary::Float(v) => v.to_string(),
            PeekSummary::UnitEnum(variant) => variant,
            PeekSummary::List(list) => format!("Count: {} items", list.len()),
            PeekSummary::Map(entries) => format!("Keys: {} items", entries.len()),
            PeekSummary::StructLike { class_name, fields } => {
                format!("{class_name} ({} fields)", fields.len())
            }
            PeekSummary::Unknown(value) => format!("{}", value),
        },
    }
}

fn primitive_type_name_peek(value: Peek<'_, '_>) -> &'static str {
    match summarize_peek(value) {
        PeekSummary::None => "null",
        PeekSummary::Media => "media",
        PeekSummary::String(_) => "string",
        PeekSummary::Bool(_) => "bool",
        PeekSummary::SignedInt(_) | PeekSummary::UnsignedInt(_) => "int",
        PeekSummary::Float(_) => "float",
        PeekSummary::UnitEnum(_) => "enum",
        PeekSummary::List(_) => "list",
        PeekSummary::Map(_) => "map",
        PeekSummary::StructLike { .. } => "class",
        PeekSummary::Unknown(_) => "object",
    }
}

fn peek_at_field_path<'mem, 'facet>(
    root: Peek<'mem, 'facet>,
    path: &FieldPath,
) -> Option<Peek<'mem, 'facet>> {
    let parts = path.iter().collect::<Vec<_>>();
    let mut current = root.innermost_peek();
    for (idx, part) in parts.iter().enumerate() {
        let struct_peek = current.into_struct().ok()?;
        let mut next = struct_peek.field_by_name(part).ok()?.innermost_peek();
        if idx + 1 < parts.len()
            && let Ok(opt) = next.into_option()
        {
            next = opt.value()?.innermost_peek();
        }
        current = next;
    }
    Some(current)
}

fn map_entries_for_preview<'mem, 'facet>(
    map: bamltype::facet_reflect::PeekMap<'mem, 'facet>,
) -> Vec<(String, Peek<'mem, 'facet>)> {
    let mut entries = BTreeMap::new();
    for (key, value) in map.iter() {
        entries.insert(key_to_string(key), value);
    }
    entries.into_iter().collect()
}

fn struct_like_fields<'mem, 'facet>(
    value: Peek<'mem, 'facet>,
) -> Option<(String, Vec<(String, StructFieldValue<'mem, 'facet>)>)> {
    let value = value.innermost_peek();

    if let Ok(struct_peek) = value.into_struct() {
        let class_name = bamltype::internal_name_for_shape(value.shape());
        let fields = struct_peek
            .fields_for_serialize()
            .map(|(field_item, field_value)| {
                (
                    field_item.effective_name().to_string(),
                    StructFieldValue::Peek(field_value),
                )
            })
            .collect::<Vec<_>>();
        return Some((class_name, fields));
    }

    if let Ok(enum_peek) = value.into_enum() {
        if !enum_has_data_variants(value.shape()) {
            return None;
        }

        let class_name = bamltype::internal_name_for_shape(value.shape());
        let variant = enum_peek.active_variant().ok()?;
        let mut fields = Vec::new();
        let tag_name = value.shape().get_tag_attr().unwrap_or("type");
        fields.push((
            tag_name.to_string(),
            StructFieldValue::String(variant.effective_name().to_string()),
        ));
        fields.extend(
            enum_peek
                .fields_for_serialize()
                .map(|(field_item, field_value)| {
                    (
                        field_item.effective_name().to_string(),
                        StructFieldValue::Peek(field_value),
                    )
                }),
        );
        return Some((class_name, fields));
    }

    None
}

fn row_for_field_stats<'mem, 'facet>(
    value: Peek<'mem, 'facet>,
) -> Option<BTreeMap<String, StructFieldValue<'mem, 'facet>>> {
    if let Some((_, fields)) = struct_like_fields(value) {
        let mut row = BTreeMap::new();
        for (name, field_value) in fields {
            row.insert(name, field_value);
        }
        return Some(row);
    }

    let map = value.innermost_peek().into_map().ok()?;
    let mut row = BTreeMap::new();
    for (key, value) in map.iter() {
        row.insert(key_to_string(key), StructFieldValue::Peek(value));
    }
    Some(row)
}

fn unit_enum_variant(value: Peek<'_, '_>) -> Option<String> {
    let value = value.innermost_peek();
    let enum_peek = value.into_enum().ok()?;
    if enum_has_data_variants(value.shape()) {
        return None;
    }
    Some(
        enum_peek
            .active_variant()
            .ok()?
            .effective_name()
            .to_string(),
    )
}

fn is_media_shape(shape: &'static bamltype::facet::Shape) -> bool {
    shape.type_identifier.ends_with("BamlMedia")
        || shape.type_identifier.contains("::media::BamlMedia")
}

fn enum_has_data_variants(shape: &'static bamltype::facet::Shape) -> bool {
    let Type::User(UserType::Enum(enum_type)) = &shape.ty else {
        return false;
    };
    enum_type
        .variants
        .iter()
        .any(|variant| !variant.data.fields.is_empty())
}

fn key_to_string(key: Peek<'_, '_>) -> String {
    if let Some(s) = key.as_str() {
        return s.to_string();
    }
    if let Some(value) = peek_signed_i128(key) {
        return value.to_string();
    }
    if let Some(value) = peek_unsigned_u128(key) {
        return value.to_string();
    }
    if let Ok(value) = key.get::<bool>() {
        return value.to_string();
    }
    format!("{key}")
}

fn peek_signed_i128(value: Peek<'_, '_>) -> Option<i128> {
    if let Ok(v) = value.get::<i128>() {
        return Some(*v);
    }
    if let Ok(v) = value.get::<i64>() {
        return Some(*v as i128);
    }
    if let Ok(v) = value.get::<i32>() {
        return Some(*v as i128);
    }
    if let Ok(v) = value.get::<i16>() {
        return Some(*v as i128);
    }
    if let Ok(v) = value.get::<i8>() {
        return Some(*v as i128);
    }
    if let Ok(v) = value.get::<isize>() {
        return Some(*v as i128);
    }
    None
}

fn peek_unsigned_u128(value: Peek<'_, '_>) -> Option<u128> {
    if let Ok(v) = value.get::<u128>() {
        return Some(*v);
    }
    if let Ok(v) = value.get::<u64>() {
        return Some(*v as u128);
    }
    if let Ok(v) = value.get::<u32>() {
        return Some(*v as u128);
    }
    if let Ok(v) = value.get::<u16>() {
        return Some(*v as u128);
    }
    if let Ok(v) = value.get::<u8>() {
        return Some(*v as u128);
    }
    if let Ok(v) = value.get::<usize>() {
        return Some(*v as u128);
    }
    None
}

fn collapse_option_chain<'mem, 'facet>(value: Peek<'mem, 'facet>) -> Option<Peek<'mem, 'facet>> {
    let mut current = value.innermost_peek();
    loop {
        match current.into_option() {
            Ok(option) => match option.value() {
                Some(inner) => current = inner.innermost_peek(),
                None => return None,
            },
            Err(_) => return Some(current),
        }
    }
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

    #[derive(Clone, Debug)]
    #[BamlType]
    #[allow(
        dead_code,
        reason = "Payload fields are exercised via reflection in preview tests."
    )]
    enum ActionState {
        Final { answer: String, confidence: i32 },
        Retry { reason: String },
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct RowWithNestedOptionals {
        id: String,
        attrs: Option<HashMap<String, Option<i32>>>,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct RowWithDoubleOptional {
        id: String,
        maybe_note: Option<Option<String>>,
    }

    #[derive(Clone, Debug)]
    #[BamlType]
    struct FlattenedNote {
        reason: String,
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

    #[derive(Signature, Clone, Debug)]
    struct EnumPayloadSig {
        #[input]
        action: ActionState,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    struct NestedOptionalStatsSig {
        #[input]
        rows: Vec<RowWithNestedOptionals>,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    struct EmptyCollectionsSig {
        #[input]
        empty_list: Vec<String>,

        #[input]
        empty_map: HashMap<String, i32>,

        #[input]
        nested_empty: Vec<Vec<String>>,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    struct FlattenAliasSig {
        #[input]
        #[flatten]
        inner: FlattenedNote,

        #[input]
        #[alias("topic_alias")]
        topic: String,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    struct NestedOptionalSig {
        #[input]
        maybe_note: Option<Option<String>>,

        #[output]
        ok: bool,
    }

    #[derive(Signature, Clone, Debug)]
    struct NestedOptionalStatsParitySig {
        #[input]
        rows: Vec<RowWithDoubleOptional>,

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

    #[test]
    fn enum_payload_variant_renders_as_struct_like_preview() {
        let rendered = render_previews::<EnumPayloadSig>(&EnumPayloadSigInput {
            action: ActionState::Final {
                answer: "ship it".to_string(),
                confidence: 9,
            },
        });

        assert!(rendered.contains("action:"));
        assert!(rendered.contains("Preview:"));
        assert!(rendered.contains("type: \"Final\""));
        assert!(rendered.contains("answer: \"ship it\""));
        assert!(rendered.contains("confidence: 9"));
    }

    #[test]
    fn vec_struct_field_stats_handles_nested_optional_map_values() {
        let rows = vec![
            RowWithNestedOptionals {
                id: "row-a".to_string(),
                attrs: Some(HashMap::from([
                    ("x".to_string(), Some(1)),
                    ("y".to_string(), None),
                ])),
            },
            RowWithNestedOptionals {
                id: "row-b".to_string(),
                attrs: None,
            },
            RowWithNestedOptionals {
                id: "row-c".to_string(),
                attrs: Some(HashMap::from([("z".to_string(), Some(3))])),
            },
        ];

        let rendered =
            render_previews::<NestedOptionalStatsSig>(&NestedOptionalStatsSigInput { rows });

        assert!(rendered.contains("Count: 3 items"));
        assert!(rendered.contains("Field stats:"));
        assert!(rendered.contains("id:"));
        assert!(rendered.contains("Sample [0]:"));
    }

    #[test]
    fn empty_collections_keep_counts_without_samples() {
        let rendered = render_previews::<EmptyCollectionsSig>(&EmptyCollectionsSigInput {
            empty_list: Vec::new(),
            empty_map: HashMap::new(),
            nested_empty: vec![Vec::new()],
        });

        assert!(rendered.contains("empty_list"));
        assert!(rendered.contains("Count: 0 items"));
        assert!(rendered.contains("empty_map"));
        assert!(rendered.contains("Keys: 0 items"));
        assert!(rendered.contains("nested_empty"));
        assert!(rendered.contains("Sample [0]: Count: 0 items"));
        assert!(!rendered.contains("Sample [\""));
    }

    #[test]
    fn map_samples_are_deterministically_sorted_under_peek() {
        let mut scores = HashMap::new();
        scores.insert("zeta".to_string(), 9);
        scores.insert("alpha".to_string(), 1);
        scores.insert("middle".to_string(), 5);
        scores.insert("omega".to_string(), 7);

        let rendered = render_previews::<MixedSig>(&MixedSigInput {
            maybe_note: Some("note".to_string()),
            scores,
            states: vec![State::Ready],
            nested: vec![vec!["a".to_string()]],
        });

        let alpha = rendered
            .find("Sample [\"alpha\"]")
            .expect("alpha sample must be present");
        let omega = rendered
            .find("Sample [\"omega\"]")
            .expect("omega sample must be present");
        let zeta = rendered
            .find("Sample [\"zeta\"]")
            .expect("zeta sample must be present");

        assert!(
            alpha < omega && omega < zeta,
            "map key order is not deterministic:\n{rendered}"
        );
    }

    #[test]
    fn flattened_alias_field_path_resolves_value_under_peek() {
        let rendered = render_previews::<FlattenAliasSig>(&FlattenAliasSigInput {
            inner: FlattenedNote {
                reason: "because evidence".to_string(),
            },
            topic: "biology".to_string(),
        });

        assert!(rendered.contains("reason: string"));
        assert!(rendered.contains("Value: \"because evidence\""));
        assert!(rendered.contains("topic_alias: string"));
        assert!(rendered.contains("Value: \"biology\""));
        assert!(!rendered.contains("<missing>"));
    }

    #[test]
    fn nested_optional_some_none_renders_none_without_present_prefix() {
        let rendered = render_previews::<NestedOptionalSig>(&NestedOptionalSigInput {
            maybe_note: Some(None),
        });

        assert!(rendered.contains("maybe_note"));
        assert!(rendered.contains("None"));
        assert!(!rendered.contains("(Present) None"));
    }

    #[test]
    fn nested_optional_some_none_counts_as_null_in_field_stats() {
        let rows = vec![
            RowWithDoubleOptional {
                id: "r1".to_string(),
                maybe_note: Some(None),
            },
            RowWithDoubleOptional {
                id: "r2".to_string(),
                maybe_note: None,
            },
            RowWithDoubleOptional {
                id: "r3".to_string(),
                maybe_note: Some(Some("x".to_string())),
            },
        ];

        let rendered =
            render_previews::<NestedOptionalStatsParitySig>(&NestedOptionalStatsParitySigInput {
                rows,
            });

        assert!(rendered.contains("Field stats:"));
        assert!(rendered.contains("maybe_note:"));
        assert!(rendered.contains("67% null"));
    }
}
