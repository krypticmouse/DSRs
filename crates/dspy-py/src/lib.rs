use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use bamltype::baml_types::ir_type::UnionConstructor;
use bamltype::baml_types::{BamlValue, StreamingMode, TypeIR};
use bamltype::internal_baml_jinja::types::{Class, Enum, Name, OutputFormatContent};
use bamltype::{RenderOptions, SchemaRegistry, default_streaming_behavior};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use regex::Regex;
use serde::Deserialize;
use serde_json::{Map, Value};

// Whitespace-tolerant: models routinely emit `[[ ## name ##]]` (missing a space) — observed at
// ~6% of deepseek-v4-flash calls in production, where a strict match made the whole completion
// unparseable ("missing output field") and forced an expensive adapter-fallback re-call. The
// canonical render (below) always emits `[[ ## name ## ]]`; only PARSING is lenient.
static FIELD_HEADER_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\[\[\s*##\s*(\w+)\s*##\s*\]\]").expect("valid marker regex"));

#[derive(Debug, Clone, Deserialize)]
struct AdapterField {
    name: String,
    #[serde(default)]
    description: String,
    #[allow(dead_code)]
    #[serde(default)]
    format: Option<String>,
    schema: Value,
}

#[derive(Debug, Deserialize)]
struct AdapterSpec {
    #[serde(default)]
    input_fields: Vec<AdapterField>,
    #[serde(default)]
    output_fields: Vec<AdapterField>,
    #[serde(default)]
    instruction: String,
}

#[derive(Debug, Clone)]
struct CompiledOutputField {
    name: String,
    description: String,
    type_ir: TypeIR,
}

#[derive(Debug)]
struct CompiledSpec {
    input_fields: Vec<AdapterField>,
    output_fields: Vec<CompiledOutputField>,
    output_format: OutputFormatContent,
}

#[derive(Debug, Default)]
struct SchemaCompiler {
    registry: SchemaRegistry,
    defs_by_scope: HashMap<String, Value>,
    def_aliases: HashMap<String, String>,
    compiled_defs: HashMap<String, TypeIR>,
    in_progress_defs: HashSet<String>,
    used_names: HashSet<String>,
}

impl SchemaCompiler {
    fn scope_id(name: &str) -> String {
        sanitize_identifier(name)
    }

    fn add_scope_defs(&mut self, scope: &str, schema: &Value) {
        for defs_key in ["$defs", "definitions"] {
            let Some(defs_map) = schema.get(defs_key).and_then(Value::as_object) else {
                continue;
            };
            for (def_name, def_schema) in defs_map {
                let scoped_name = format!("{scope}::{def_name}");
                self.defs_by_scope.insert(scoped_name, def_schema.clone());
            }
        }
    }

    fn unique_name(&mut self, raw: &str) -> String {
        let base = sanitize_identifier(raw);
        if self.used_names.insert(base.clone()) {
            return base;
        }

        let mut idx = 2usize;
        loop {
            let candidate = format!("{base}_{idx}");
            if self.used_names.insert(candidate.clone()) {
                return candidate;
            }
            idx += 1;
        }
    }

    fn compile_schema(
        &mut self,
        schema: &Value,
        hint_name: Option<&str>,
        scope: &str,
    ) -> Result<TypeIR, String> {
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
            return self.compile_ref(reference, scope);
        }

        if let Some(enum_values) = schema.get("enum").and_then(Value::as_array) {
            return self.compile_enum(schema, enum_values, hint_name);
        }

        if let Some(constant) = schema.get("const")
            && let Some(literal) = literal_type_from_value(constant)
        {
            return Ok(literal);
        }

        if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
            return self.compile_union(any_of, hint_name, scope);
        }
        if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
            return self.compile_union(one_of, hint_name, scope);
        }
        if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
            return self.compile_union(all_of, hint_name, scope);
        }

        if let Some(type_name) = schema.get("type").and_then(Value::as_str) {
            return self.type_from_keyword(type_name, schema, hint_name, scope);
        }

        if let Some(type_array) = schema.get("type").and_then(Value::as_array) {
            let mut choices = Vec::new();
            for keyword in type_array {
                let Some(keyword) = keyword.as_str() else {
                    continue;
                };
                choices.push(self.type_from_keyword(keyword, schema, hint_name, scope)?);
            }
            return Ok(match choices.len() {
                0 => TypeIR::top(),
                1 => choices.remove(0),
                _ => TypeIR::union(choices),
            });
        }

        if schema.get("properties").is_some() || schema.get("additionalProperties").is_some() {
            return self.compile_object(schema, hint_name, scope);
        }

        if schema.get("items").is_some() {
            return self.compile_array(schema, hint_name, scope);
        }

        Ok(TypeIR::top())
    }

    fn compile_ref(&mut self, reference: &str, scope: &str) -> Result<TypeIR, String> {
        let Some(ref_name) = parse_local_ref_name(reference) else {
            return Err(format!(
                "unsupported non-local JSON schema reference: {reference}"
            ));
        };

        let scoped_ref = format!("{scope}::{ref_name}");

        let alias_name = if let Some(existing) = self.def_aliases.get(&scoped_ref) {
            existing.clone()
        } else {
            let fresh = self.unique_name(&ref_name);
            self.def_aliases.insert(scoped_ref.clone(), fresh.clone());
            fresh
        };

        if let Some(compiled) = self.compiled_defs.get(&scoped_ref) {
            return Ok(compiled.clone());
        }

        if self.in_progress_defs.contains(&scoped_ref) {
            return Ok(TypeIR::class(alias_name));
        }

        let def_schema = self
            .defs_by_scope
            .get(&scoped_ref)
            .ok_or_else(|| format!("reference `{reference}` not found in this field scope"))?
            .clone();

        self.in_progress_defs.insert(scoped_ref.clone());
        let compiled = self.compile_schema(&def_schema, Some(&alias_name), scope)?;
        self.in_progress_defs.remove(&scoped_ref);
        self.compiled_defs.insert(scoped_ref, compiled.clone());

        Ok(compiled)
    }

    fn compile_union(
        &mut self,
        variants: &[Value],
        hint_name: Option<&str>,
        scope: &str,
    ) -> Result<TypeIR, String> {
        let mut choices = Vec::new();
        for (idx, variant_schema) in variants.iter().enumerate() {
            let child_hint = hint_name.map(|hint| format!("{hint}Variant{}", idx + 1));
            let variant_type = self.compile_schema(variant_schema, child_hint.as_deref(), scope)?;
            choices.push(variant_type);
        }

        Ok(match choices.len() {
            0 => TypeIR::top(),
            1 => choices.remove(0),
            _ => TypeIR::union(choices),
        })
    }

    fn compile_enum(
        &mut self,
        schema: &Value,
        values: &[Value],
        hint_name: Option<&str>,
    ) -> Result<TypeIR, String> {
        let all_strings = values.iter().all(|value| value.is_string());
        if all_strings {
            let enum_base_name = schema
                .get("title")
                .and_then(Value::as_str)
                .or(hint_name)
                .unwrap_or("Enum");
            let enum_name = self.unique_name(enum_base_name);
            let enum_description = schema
                .get("description")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);

            let mut enum_values = Vec::new();
            for value in values {
                let variant_name = value
                    .as_str()
                    .ok_or_else(|| "string enum value unexpectedly missing".to_string())?
                    .to_string();
                enum_values.push((Name::new(variant_name), None));
            }

            self.registry.register_enum(Enum {
                name: Name::new(enum_name.clone()),
                description: enum_description,
                values: enum_values,
                constraints: Vec::new(),
            });

            return Ok(TypeIR::r#enum(&enum_name));
        }

        let mut literals = Vec::new();
        for value in values {
            if let Some(literal) = literal_type_from_value(value) {
                literals.push(literal);
            }
        }

        Ok(match literals.len() {
            0 => TypeIR::top(),
            1 => literals.remove(0),
            _ => TypeIR::union(literals),
        })
    }

    fn compile_array(
        &mut self,
        schema: &Value,
        hint_name: Option<&str>,
        scope: &str,
    ) -> Result<TypeIR, String> {
        if let Some(prefix_items) = schema.get("prefixItems").and_then(Value::as_array) {
            let mut members = Vec::new();
            for (idx, item_schema) in prefix_items.iter().enumerate() {
                let child_hint = hint_name.map(|hint| format!("{hint}TupleItem{}", idx + 1));
                members.push(self.compile_schema(item_schema, child_hint.as_deref(), scope)?);
            }

            if let Some(items_schema) = schema.get("items") {
                match items_schema {
                    Value::Bool(false) => {}
                    Value::Bool(true) => members.push(TypeIR::string()),
                    other => {
                        let child_hint = hint_name.map(|hint| format!("{hint}TupleRest"));
                        members.push(self.compile_schema(other, child_hint.as_deref(), scope)?);
                    }
                }
            }

            return Ok(TypeIR::list(match members.len() {
                0 => TypeIR::string(),
                1 => members.remove(0),
                _ => TypeIR::union(members),
            }));
        }

        if let Some(items) = schema.get("items").and_then(Value::as_array) {
            let mut tuple_members = Vec::new();
            for (idx, item_schema) in items.iter().enumerate() {
                let child_hint = hint_name.map(|hint| format!("{hint}Item{}", idx + 1));
                tuple_members.push(self.compile_schema(
                    item_schema,
                    child_hint.as_deref(),
                    scope,
                )?);
            }
            return Ok(TypeIR::list(match tuple_members.len() {
                0 => TypeIR::top(),
                1 => tuple_members.remove(0),
                _ => TypeIR::union(tuple_members),
            }));
        }

        if let Some(item_schema) = schema.get("items") {
            let child_hint = hint_name.map(|hint| format!("{hint}Item"));
            let inner = self.compile_schema(item_schema, child_hint.as_deref(), scope)?;
            return Ok(TypeIR::list(inner));
        }

        Ok(TypeIR::list(TypeIR::string()))
    }

    fn compile_object(
        &mut self,
        schema: &Value,
        hint_name: Option<&str>,
        scope: &str,
    ) -> Result<TypeIR, String> {
        if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
            let class_base_name = schema
                .get("title")
                .and_then(Value::as_str)
                .or(hint_name)
                .unwrap_or("Object");
            let class_name = self.unique_name(class_base_name);
            let class_description = schema
                .get("description")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);

            let required = schema
                .get("required")
                .and_then(Value::as_array)
                .map(|required_fields| {
                    required_fields
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect::<HashSet<String>>()
                })
                .unwrap_or_default();

            let mut class_fields = Vec::new();
            for (prop_name, prop_schema) in properties {
                let child_hint = format!("{}{}", class_name, sanitize_identifier(prop_name));
                let mut prop_type = self.compile_schema(prop_schema, Some(&child_hint), scope)?;
                if !required.contains(prop_name) {
                    prop_type = TypeIR::optional(prop_type);
                }

                let description = prop_schema
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);

                class_fields.push((Name::new(prop_name.clone()), prop_type, description, false));
            }

            self.registry.register_class(Class {
                name: Name::new(class_name.clone()),
                description: class_description,
                namespace: StreamingMode::NonStreaming,
                fields: class_fields,
                constraints: Vec::new(),
                streaming_behavior: default_streaming_behavior(),
            });

            return Ok(TypeIR::class(class_name));
        }

        if let Some(additional_props) = schema.get("additionalProperties") {
            if matches!(additional_props, Value::Bool(false)) {
                let class_name = self.unique_name(
                    schema
                        .get("title")
                        .and_then(Value::as_str)
                        .or(hint_name)
                        .unwrap_or("Object"),
                );
                self.registry.register_class(Class {
                    name: Name::new(class_name.clone()),
                    description: schema
                        .get("description")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned),
                    namespace: StreamingMode::NonStreaming,
                    fields: Vec::new(),
                    constraints: Vec::new(),
                    streaming_behavior: default_streaming_behavior(),
                });
                return Ok(TypeIR::class(class_name));
            }

            if matches!(additional_props, Value::Bool(true)) {
                return Ok(TypeIR::map(TypeIR::string(), TypeIR::top()));
            }

            let value_hint = hint_name.map(|hint| format!("{hint}Value"));
            let value_type = self.compile_schema(additional_props, value_hint.as_deref(), scope)?;
            return Ok(TypeIR::map(TypeIR::string(), value_type));
        }

        Ok(TypeIR::map(TypeIR::string(), TypeIR::top()))
    }

    fn type_from_keyword(
        &mut self,
        keyword: &str,
        schema: &Value,
        hint_name: Option<&str>,
        scope: &str,
    ) -> Result<TypeIR, String> {
        match keyword {
            "string" => Ok(TypeIR::string()),
            "integer" => Ok(TypeIR::int()),
            "number" => Ok(TypeIR::float()),
            "boolean" => Ok(TypeIR::bool()),
            "null" => Ok(TypeIR::null()),
            "array" => self.compile_array(schema, hint_name, scope),
            "object" => self.compile_object(schema, hint_name, scope),
            _ => Ok(TypeIR::top()),
        }
    }
}

fn parse_local_ref_name(reference: &str) -> Option<String> {
    let body = reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))?;
    body.split('/').next().map(ToOwned::to_owned)
}

fn sanitize_identifier(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }

    if out.is_empty() {
        return "Type".to_string();
    }

    if out
        .chars()
        .next()
        .map(|ch| ch.is_ascii_digit())
        .unwrap_or(false)
    {
        out.insert(0, 'T');
    }

    out
}

fn literal_type_from_value(value: &Value) -> Option<TypeIR> {
    if let Some(str_value) = value.as_str() {
        return Some(TypeIR::literal_string(str_value.to_string()));
    }
    if let Some(int_value) = value.as_i64() {
        return Some(TypeIR::literal_int(int_value));
    }
    if let Some(bool_value) = value.as_bool() {
        return Some(TypeIR::literal_bool(bool_value));
    }
    None
}

fn compile_spec(spec: &AdapterSpec) -> Result<CompiledSpec, String> {
    let mut compiler = SchemaCompiler::default();

    for field in &spec.output_fields {
        let scope = SchemaCompiler::scope_id(&field.name);
        compiler.add_scope_defs(&scope, &field.schema);
    }

    let mut output_fields = Vec::new();
    for field in &spec.output_fields {
        let scope = SchemaCompiler::scope_id(&field.name);
        let type_ir = compiler.compile_schema(&field.schema, Some(&field.name), &scope)?;
        output_fields.push(CompiledOutputField {
            name: field.name.clone(),
            description: field.description.clone(),
            type_ir,
        });
    }

    let output_class_name = compiler.unique_name("DSPyOutput");
    let output_class_fields = output_fields
        .iter()
        .map(|field| {
            (
                Name::new(field.name.clone()),
                field.type_ir.clone(),
                if field.description.is_empty() {
                    None
                } else {
                    Some(field.description.clone())
                },
                false,
            )
        })
        .collect::<Vec<_>>();

    compiler.registry.register_class(Class {
        name: Name::new(output_class_name.clone()),
        description: if spec.instruction.is_empty() {
            None
        } else {
            Some(spec.instruction.clone())
        },
        namespace: StreamingMode::NonStreaming,
        fields: output_class_fields,
        constraints: Vec::new(),
        streaming_behavior: default_streaming_behavior(),
    });

    let output_format = compiler.registry.build(TypeIR::class(output_class_name));

    Ok(CompiledSpec {
        input_fields: spec.input_fields.clone(),
        output_fields,
        output_format,
    })
}

fn simplify_type_name(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '`' {
                    break;
                }
                token.push(next);
            }
            let simplified = token.rsplit("::").next().unwrap_or(&token);
            result.push_str(simplified);
        } else {
            result.push(ch);
        }
    }
    result
}

fn render_type_name_for_prompt(type_ir: &TypeIR) -> String {
    let raw = type_ir.diagnostic_repr().to_string();
    let simplified = simplify_type_name(&raw);
    simplified
        .replace("class ", "")
        .replace("enum ", "")
        .replace(" | ", " or ")
        .trim()
        .to_string()
}

fn split_schema_definitions(schema: &str) -> Option<(String, String)> {
    let lines: Vec<&str> = schema.lines().collect();
    let mut index = 0;
    let mut definitions = Vec::new();
    let mut parsed_any = false;

    while index < lines.len() {
        let start_index = index;

        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }

        while index < lines.len() && lines[index].trim_start().starts_with("//") {
            index += 1;
        }

        while index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
        }

        if index >= lines.len() {
            break;
        }

        let name_line = lines[index].trim();
        if name_line.is_empty() {
            break;
        }
        index += 1;

        if index >= lines.len() || lines[index].trim() != "----" {
            index = start_index;
            break;
        }
        index += 1;

        let mut values_found = 0;
        while index < lines.len() {
            let trimmed = lines[index].trim_start();
            if trimmed.is_empty() {
                break;
            }
            if trimmed.starts_with('-') {
                values_found += 1;
                index += 1;
                continue;
            }
            break;
        }

        if values_found == 0 {
            index = start_index;
            break;
        }

        let mut block_end = index;
        if index < lines.len() && lines[index].trim().is_empty() {
            index += 1;
            block_end = index;
        }

        definitions.extend_from_slice(&lines[start_index..block_end]);
        parsed_any = true;
    }

    if !parsed_any {
        return None;
    }

    let mut main_lines = Vec::new();
    if index < lines.len() {
        main_lines.extend_from_slice(&lines[index..]);
    }

    let defs = definitions.join("\n").trim_end().to_string();
    let main = main_lines.join("\n").trim_start().to_string();
    if defs.is_empty() || main.is_empty() {
        None
    } else {
        Some((defs, main))
    }
}

fn format_schema_for_prompt(schema: &str) -> String {
    let Some((definitions, main)) = split_schema_definitions(schema) else {
        return schema.to_string();
    };

    format!("Definitions (used below):\n\n{definitions}\n\n{main}")
}

/// One leading `Name` / `----` / `- value…` definition block in a rendered
/// field schema, with the source line indices it spans (including any leading
/// `//` comment describing it and one trailing blank line).
struct DefinitionBlock {
    name: String,
    line_indices: Vec<usize>,
}

/// Parse the contiguous run of enum/class definition blocks at the top of a
/// rendered field schema. Returns the blocks and the index where the main body
/// (the part that actually references them) begins.
fn parse_leading_definition_blocks(lines: &[&str]) -> (Vec<DefinitionBlock>, usize) {
    let mut blocks = Vec::new();
    let mut index = 0;

    loop {
        let block_start = index;
        let mut cursor = index;
        while cursor < lines.len() && lines[cursor].trim().is_empty() {
            cursor += 1;
        }
        while cursor < lines.len() && lines[cursor].trim_start().starts_with("//") {
            cursor += 1;
        }
        while cursor < lines.len() && lines[cursor].trim().is_empty() {
            cursor += 1;
        }
        if cursor >= lines.len() {
            break;
        }
        let name = lines[cursor].trim().to_string();
        if name.is_empty() || cursor + 1 >= lines.len() || lines[cursor + 1].trim() != "----" {
            break;
        }
        cursor += 2;
        let mut values = 0;
        while cursor < lines.len() {
            let trimmed = lines[cursor].trim_start();
            if trimmed.is_empty() {
                break;
            }
            if trimmed.starts_with('-') {
                values += 1;
                cursor += 1;
                continue;
            }
            break;
        }
        if values == 0 {
            break;
        }
        let mut block_end = cursor;
        if block_end < lines.len() && lines[block_end].trim().is_empty() {
            block_end += 1;
        }
        blocks.push(DefinitionBlock {
            name,
            line_indices: (block_start..block_end).collect(),
        });
        index = block_end;
    }

    (blocks, index)
}

/// BAML renders a field against the whole signature's type registry, so it
/// prepends EVERY named enum/class definition to each field's schema — even
/// fields (including primitive `string` fields) that reference none of them.
/// Drop the definition blocks this field's body does not actually use, so each
/// field carries only its own `Definitions (used below)`.
fn strip_unreferenced_definitions(schema: &str) -> String {
    let lines: Vec<&str> = schema.lines().collect();
    let (blocks, main_start) = parse_leading_definition_blocks(&lines);
    if blocks.is_empty() {
        return schema.to_string();
    }

    // A definition is referenced when its name appears (as a whole word) in an
    // actual type position — a non-comment body line such as `role: Role,` —
    // not merely inside a `//` description. Grow the referencing text with kept
    // blocks so transitive references between definitions survive.
    let mut referencing = lines[main_start..]
        .iter()
        .filter(|line| !line.trim_start().starts_with("//"))
        .copied()
        .collect::<Vec<_>>()
        .join("\n");

    let mut kept = vec![false; blocks.len()];
    loop {
        let mut changed = false;
        for (idx, block) in blocks.iter().enumerate() {
            if kept[idx] {
                continue;
            }
            let pattern = format!(r"\b{}\b", regex::escape(&block.name));
            let referenced = Regex::new(&pattern)
                .map(|re| re.is_match(&referencing))
                .unwrap_or(true); // on the impossible regex error, keep the block
            if referenced {
                kept[idx] = true;
                changed = true;
                for &line_index in &block.line_indices {
                    referencing.push('\n');
                    referencing.push_str(lines[line_index]);
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut out: Vec<&str> = Vec::new();
    for (idx, block) in blocks.iter().enumerate() {
        if kept[idx] {
            for &line_index in &block.line_indices {
                out.push(lines[line_index]);
            }
        }
    }
    out.extend_from_slice(&lines[main_start..]);
    out.join("\n").trim().to_string()
}

fn render_field_type_schema(
    parent_format: &OutputFormatContent,
    type_ir: &TypeIR,
) -> Result<String, String> {
    let field_format = OutputFormatContent {
        enums: parent_format.enums.clone(),
        classes: parent_format.classes.clone(),
        recursive_classes: parent_format.recursive_classes.clone(),
        structural_recursive_aliases: parent_format.structural_recursive_aliases.clone(),
        target: type_ir.clone(),
    };

    let schema = field_format
        .render(RenderOptions::default().with_prefix(None))
        .map_err(|err| err.to_string())?
        .unwrap_or_else(|| type_ir.diagnostic_repr().to_string());

    Ok(strip_unreferenced_definitions(&schema))
}

fn render_field_structure_core(spec: &AdapterSpec) -> Result<String, String> {
    let compiled = compile_spec(spec)?;
    let mut lines = vec![
        "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
        String::new(),
    ];

    for field in &compiled.input_fields {
        lines.push(format!("[[ ## {} ## ]]", field.name));
        lines.push(field.name.clone());
        lines.push(String::new());
    }

    for field in &compiled.output_fields {
        let type_name = render_type_name_for_prompt(&field.type_ir);
        let schema = render_field_type_schema(&compiled.output_format, &field.type_ir)?;

        lines.push(format!("[[ ## {} ## ]]", field.name));
        lines.push(format!(
            "Output field `{}` should be of type: {type_name}",
            field.name
        ));

        if !schema.is_empty() && schema != type_name {
            lines.push(String::new());
            lines.push(format_schema_for_prompt(&schema));
        }

        lines.push(String::new());
    }

    lines.push("[[ ## completed ## ]]".to_string());
    Ok(lines.join("\n"))
}

fn parse_sections(content: &str) -> HashMap<String, String> {
    let mut sections: Vec<(Option<String>, Vec<String>)> = vec![(None, Vec::new())];

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(captures) = FIELD_HEADER_PATTERN.captures(trimmed) {
            let header = captures
                .get(1)
                .map(|value| value.as_str().to_string())
                .unwrap_or_default();
            let marker = captures
                .get(0)
                .expect("header capture should include full marker");
            let remaining = trimmed[marker.end()..].trim();

            let mut field_lines = Vec::new();
            if !remaining.is_empty() {
                field_lines.push(remaining.to_string());
            }
            sections.push((Some(header), field_lines));
        } else if let Some((_, lines)) = sections.last_mut() {
            lines.push(line.to_string());
        }
    }

    let mut parsed = HashMap::new();
    for (header, lines) in sections {
        let Some(name) = header else {
            continue;
        };
        parsed
            .entry(name)
            .or_insert_with(|| lines.join("\n").trim().to_string());
    }

    parsed
}

fn parse_response_core(
    spec: &AdapterSpec,
    completion: &str,
    is_done: bool,
) -> Result<Map<String, Value>, String> {
    let compiled = compile_spec(spec)?;
    let sections = parse_sections(completion);

    let mut parsed_output = Map::new();
    let mut errors = Vec::new();

    for field in &compiled.output_fields {
        let Some(raw_text) = sections.get(&field.name) else {
            errors.push(format!(
                "missing output field `{}` in LM response",
                field.name
            ));
            continue;
        };

        let parsed = match bamltype::jsonish::from_str(
            &compiled.output_format,
            &field.type_ir,
            raw_text,
            is_done,
        ) {
            Ok(value) => value,
            Err(err) => {
                errors.push(format!(
                    "failed to parse output field `{}` with JSONish: {err}",
                    field.name
                ));
                continue;
            }
        };

        let baml_value: BamlValue = parsed.into();
        let json_value = serde_json::to_value(baml_value).map_err(|err| err.to_string())?;
        parsed_output.insert(field.name.clone(), json_value);
    }

    if errors.is_empty() {
        Ok(parsed_output)
    } else {
        Err(errors.join("\n"))
    }
}

#[pyfunction]
fn render_field_structure(spec_json: &str) -> PyResult<String> {
    let spec: AdapterSpec = serde_json::from_str(spec_json)
        .map_err(|err| PyValueError::new_err(format!("invalid adapter spec JSON: {err}")))?;
    render_field_structure_core(&spec)
        .map_err(|err| PyValueError::new_err(format!("failed to render field structure: {err}")))
}

#[pyfunction(signature = (spec_json, completion, is_done = true))]
fn parse_response(spec_json: &str, completion: &str, is_done: bool) -> PyResult<String> {
    let spec: AdapterSpec = serde_json::from_str(spec_json)
        .map_err(|err| PyValueError::new_err(format!("invalid adapter spec JSON: {err}")))?;
    let parsed = parse_response_core(&spec, completion, is_done)
        .map_err(|err| PyValueError::new_err(format!("failed to parse response: {err}")))?;

    serde_json::to_string(&Value::Object(parsed))
        .map_err(|err| PyValueError::new_err(format!("failed to serialize parsed response: {err}")))
}

#[pymodule]
fn _dsrs_dspy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_field_structure, m)?)?;
    m.add_function(wrap_pyfunction!(parse_response, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn nested_spec() -> AdapterSpec {
        serde_json::from_value(json!({
            "input_fields": [
                {
                    "name": "note",
                    "description": "Clinical note",
                    "schema": {"type": "string"}
                }
            ],
            "output_fields": [
                {
                    "name": "patient",
                    "description": "Extracted patient object",
                    "schema": {
                        "$defs": {
                            "Address": {
                                "type": "object",
                                "title": "Address",
                                "properties": {
                                    "street": {"type": "string"}
                                },
                                "required": ["street"]
                            }
                        },
                        "type": "object",
                        "title": "Patient",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {
                                "anyOf": [
                                    {"type": "integer"},
                                    {"type": "null"}
                                ]
                            },
                            "address": {"$ref": "#/$defs/Address"}
                        },
                        "required": ["name", "age", "address"]
                    }
                }
            ],
            "instruction": "Extract patient information"
        }))
        .expect("valid test spec")
    }

    #[test]
    fn render_field_structure_uses_baml_schema() {
        let spec = nested_spec();
        let rendered = render_field_structure_core(&spec).expect("render should succeed");

        assert!(rendered.contains("[[ ## patient ## ]]"));
        assert!(rendered.contains("Output field `patient` should be of type"));
        assert!(rendered.contains("patient"));
    }

    #[test]
    fn parse_response_uses_jsonish_for_nested_object() {
        let spec = nested_spec();
        let completion = r#"
[[ ## patient ## ]]
{
  "name": "Ada",
  "age": 30,
  "address": {
    "street": "Main",
  },
}
[[ ## completed ## ]]
"#;

        let parsed = parse_response_core(&spec, completion, true).expect("parse should succeed");

        assert_eq!(parsed["patient"]["name"], "Ada");
        assert_eq!(parsed["patient"]["age"], 30);
        assert_eq!(parsed["patient"]["address"]["street"], "Main");
    }

    #[test]
    fn parse_response_tolerates_marker_whitespace_variants() {
        // deepseek-v4-flash emits `[[ ## name ##]]` (missing space) on a real fraction of calls;
        // strict marker matching turned those into missing-field hard failures.
        let spec = nested_spec();
        let completion = "[[ ## patient ##]]\n{\"name\": \"Ada\", \"age\": 30, \"address\": {\"street\": \"Main\"}}\n[[##completed##]]\n";

        let parsed = parse_response_core(&spec, completion, true).expect("parse should succeed");

        assert_eq!(parsed["patient"]["name"], "Ada");
        assert_eq!(parsed["patient"]["age"], 30);
    }

    #[test]
    fn parse_response_preserves_python_none_as_null_in_nested_output() {
        let spec = nested_spec();
        let completion = r#"
[[ ## patient ## ]]
{'name': 'Ada', 'age': None, 'address': {'street': 'Main'}}
[[ ## completed ## ]]
"#;

        let parsed = parse_response_core(&spec, completion, true).expect("parse should succeed");

        assert_eq!(parsed["patient"]["age"], Value::Null);
    }

    #[test]
    fn parse_response_reports_missing_field() {
        let spec: AdapterSpec = serde_json::from_value(json!({
            "output_fields": [
                {
                    "name": "answer",
                    "description": "",
                    "schema": {"type": "string"}
                }
            ]
        }))
        .expect("valid simple spec");

        let err = parse_response_core(&spec, "[[ ## completed ## ]]", true)
            .expect_err("should fail due to missing marker");

        assert!(err.contains("missing output field `answer`"));
    }
}

#[cfg(test)]
mod property_tests;
