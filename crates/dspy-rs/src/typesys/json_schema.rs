//! Provider-native JSON Schema projection from the owned signature IR.

use serde_json::{Map, Value, json};

use crate::ir::SignatureDef;

use super::{FieldType, TypeTable};

/// Projects output fields and their reachable types into a strict JSON Schema.
/// Object properties use LM-facing aliases; parsers map these back to canonical names.
pub fn json_schema_for_outputs(def: &SignatureDef, types: &TypeTable) -> Value {
    let mut defs = Map::new();
    let mut visiting = Vec::new();
    for field in def.outputs.iter() {
        collect_defs(&field.ty, types, &mut defs, &mut visiting);
    }
    let mut properties = Map::new();
    let mut required = Vec::new();
    for field in def.outputs.iter() {
        let mut schema = type_schema(&field.ty, types);
        if let (Some(description), Some(obj)) = (field.docs.as_deref(), schema.as_object_mut()) {
            obj.insert("description".into(), json!(description));
        }
        properties.insert(field.lm_name.to_string(), schema);
        required.push(field.lm_name.to_string());
    }
    let mut root = Map::new();
    root.insert("type".into(), json!("object"));
    root.insert("properties".into(), Value::Object(properties));
    root.insert("required".into(), json!(required));
    root.insert("additionalProperties".into(), json!(false));
    if !defs.is_empty() {
        root.insert("$defs".into(), Value::Object(defs));
    }
    Value::Object(root)
}

/// Whether this signature fits the JSON Schema subset accepted by providers. JSON object
/// keys are strings, so a map with a non-string key must stay on delimiter output.
pub fn json_schema_compatible(def: &SignatureDef, types: &TypeTable) -> bool {
    let mut visited = Vec::new();
    def.outputs
        .iter()
        .all(|field| compatible_type(&field.ty, types, &mut visited))
}

fn compatible_type(ty: &FieldType, types: &TypeTable, visited: &mut Vec<String>) -> bool {
    match ty {
        // Open maps are not valid in providers' strict schema subset. Keep them
        // open by selecting JSON mode or delimiter output instead of closing them.
        FieldType::Map(_, _) => false,
        FieldType::Optional(inner) | FieldType::List(inner) => {
            compatible_type(inner, types, visited)
        }
        FieldType::Union(items) => items
            .iter()
            .all(|item| compatible_type(item, types, visited)),
        FieldType::Class(name) if !visited.contains(name) => {
            visited.push(name.clone());
            let ok = types.classes.get(name).is_none_or(|class| {
                class
                    .fields
                    .iter()
                    .all(|field| compatible_type(&field.field_type, types, visited))
            });
            visited.pop();
            ok
        }
        _ => true,
    }
}

fn collect_defs(
    ty: &FieldType,
    types: &TypeTable,
    defs: &mut Map<String, Value>,
    visiting: &mut Vec<String>,
) {
    match ty {
        FieldType::Class(name) if !defs.contains_key(name) => {
            let Some(class) = types.classes.get(name) else {
                return;
            };
            if visiting.contains(name) {
                return;
            }
            visiting.push(name.clone());
            let mut properties = Map::new();
            let mut required = Vec::new();
            for field in &class.fields {
                collect_defs(&field.field_type, types, defs, visiting);
                let mut schema = type_schema(&field.field_type, types);
                if let (Some(description), Some(obj)) =
                    (field.docs.as_deref(), schema.as_object_mut())
                {
                    obj.insert("description".into(), json!(description));
                }
                properties.insert(field.rendered_name.clone(), schema);
                required.push(field.rendered_name.clone());
            }
            let mut schema = Map::new();
            schema.insert("type".into(), json!("object"));
            schema.insert("properties".into(), Value::Object(properties));
            schema.insert("required".into(), json!(required));
            schema.insert("additionalProperties".into(), json!(false));
            if let Some(description) = &class.docs {
                schema.insert("description".into(), json!(description));
            }
            defs.insert(name.clone(), Value::Object(schema));
            visiting.pop();
        }
        FieldType::Enum(name) if !defs.contains_key(name) => {
            if let Some(enm) = types.enums.get(name) {
                let mut schema = Map::new();
                schema.insert("type".into(), json!("string"));
                schema.insert(
                    "enum".into(),
                    json!(
                        enm.values
                            .iter()
                            .map(|v| &v.rendered_name)
                            .collect::<Vec<_>>()
                    ),
                );
                if let Some(description) = &enm.docs {
                    schema.insert("description".into(), json!(description));
                }
                defs.insert(name.clone(), Value::Object(schema));
            }
        }
        FieldType::List(inner) | FieldType::Optional(inner) => {
            collect_defs(inner, types, defs, visiting)
        }
        FieldType::Map(key, value) => {
            collect_defs(key, types, defs, visiting);
            collect_defs(value, types, defs, visiting);
        }
        FieldType::Union(items) => {
            for item in items {
                collect_defs(item, types, defs, visiting);
            }
        }
        _ => {}
    }
}

fn type_schema(ty: &FieldType, types: &TypeTable) -> Value {
    match ty {
        FieldType::String => json!({"type":"string"}),
        FieldType::Int => json!({"type":"integer"}),
        FieldType::Float => json!({"type":"number"}),
        FieldType::Bool => json!({"type":"boolean"}),
        FieldType::Literal(value) => json!({"type":"string", "const":value}),
        FieldType::List(inner) => json!({"type":"array", "items":type_schema(inner, types)}),
        // Strict providers require every property. Optionality is nullable here.
        FieldType::Optional(inner) => json!({"anyOf":[type_schema(inner, types), {"type":"null"}]}),
        FieldType::Map(key, value) => {
            if !matches!(key.as_ref(), FieldType::String) {
                return json!({"type":"object", "additionalProperties":type_schema(value, types)});
            }
            json!({"type":"object", "additionalProperties":type_schema(value, types)})
        }
        FieldType::Class(name) | FieldType::Enum(name) => json!({"$ref":format!("#/$defs/{name}")}),
        FieldType::Union(items) => {
            json!({"anyOf":items.iter().map(|item| type_schema(item, types)).collect::<Vec<_>>()})
        }
    }
}

/// Maps LM-facing field aliases to canonical output keys recursively.
pub fn canonicalize_output_aliases(value: Value, def: &SignatureDef, types: &TypeTable) -> Value {
    let fields = def
        .outputs
        .iter()
        .map(|f| (f.lm_name.as_ref(), f.name.as_ref(), &f.ty))
        .collect::<Vec<_>>();
    canonicalize_object(value, &fields, types)
}

/// Rewrites canonical output keys to LM-facing names in few-shot JSON examples.
pub fn alias_output_fields(value: Value, def: &SignatureDef, types: &TypeTable) -> Value {
    let fields = def
        .outputs
        .iter()
        .map(|f| (f.name.as_ref(), f.lm_name.as_ref(), &f.ty))
        .collect::<Vec<_>>();
    alias_object(value, &fields, types)
}

fn alias_object(value: Value, fields: &[(&str, &str, &FieldType)], types: &TypeTable) -> Value {
    let Value::Object(input) = value else {
        return value;
    };
    let mut out = Map::new();
    for (canonical, lm, ty) in fields {
        let Some(mut item) = input.get(*canonical).or_else(|| input.get(*lm)).cloned() else {
            continue;
        };
        alias_nested(&mut item, ty, types);
        out.insert((*lm).into(), item);
    }
    Value::Object(out)
}

fn alias_nested(value: &mut Value, ty: &FieldType, types: &TypeTable) {
    match ty {
        FieldType::Optional(inner) => alias_nested(value, inner, types),
        FieldType::List(inner) => {
            if let Value::Array(values) = value {
                for item in values {
                    alias_nested(item, inner, types);
                }
            }
        }
        FieldType::Class(name) => {
            if let Some(class) = types.classes.get(name) {
                if let Some(input) = value.as_object() {
                    let input = input.clone();
                    let fields = class
                        .fields
                        .iter()
                        .map(|f| (f.name.as_str(), f.rendered_name.as_str(), &f.field_type))
                        .collect::<Vec<_>>();
                    *value = alias_object(Value::Object(input), &fields, types);
                }
            }
        }
        FieldType::Map(_, item) => {
            if let Value::Object(values) = value {
                for item_value in values.values_mut() {
                    alias_nested(item_value, item, types);
                }
            }
        }
        FieldType::Union(items) => {
            for item in items {
                if let FieldType::Class(name) = item
                    && let Some(class) = types.classes.get(name)
                    && class.fields.iter().any(|field| {
                        value.get(&field.name).is_some()
                            || value.get(&field.rendered_name).is_some()
                    })
                {
                    alias_nested(value, item, types);
                    break;
                }
            }
        }
        _ => {}
    }
}

fn canonicalize_object(
    value: Value,
    fields: &[(&str, &str, &FieldType)],
    types: &TypeTable,
) -> Value {
    let Value::Object(input) = value else {
        return value;
    };
    let mut out = Map::new();
    for (lm, canonical, ty) in fields {
        let Some(mut value) = input.get(*lm).or_else(|| input.get(*canonical)).cloned() else {
            continue;
        };
        canonicalize_nested(&mut value, ty, types);
        out.insert((*canonical).into(), value);
    }
    Value::Object(out)
}

fn canonicalize_nested(value: &mut Value, ty: &FieldType, types: &TypeTable) {
    match ty {
        FieldType::Optional(inner) => canonicalize_nested(value, inner, types),
        FieldType::List(inner) => if let Value::Array(values) = value { for value in values { canonicalize_nested(value, inner, types); } },
        FieldType::Class(name) => {
            if let Some(class) = types.classes.get(name) {
                if let Some(input) = value.as_object() {
                    let input = input.clone();
                    let mut fields = Vec::new();
                    for field in &class.fields {
                        fields.push((field.rendered_name.as_str(), field.name.as_str(), &field.field_type));
                    }
                    *value = canonicalize_object(Value::Object(input), &fields, types);
                }
            }
        }
        FieldType::Map(_, item) => if let Value::Object(values) = value { for value in values.values_mut() { canonicalize_nested(value, item, types); } },
        FieldType::Union(items) => if let Some(item) = items.iter().find(|item| matches!(item, FieldType::Class(name) if value.get("$type").and_then(Value::as_str) == Some(name))) { canonicalize_nested(value, item, types); },
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ConstraintDef, FieldDef as IrFieldDef};
    use crate::typesys::{ClassDef, EnumDef, EnumValueDef};

    fn base_types() -> TypeTable {
        let mut types = TypeTable::default();
        types.classes.insert(
            "Node".into(),
            ClassDef {
                internal_name: "Node".into(),
                rendered_name: "Node".into(),
                docs: Some("linked node".into()),
                fields: vec![
                    crate::typesys::FieldDef {
                        name: "label".into(),
                        rendered_name: "label_alias".into(),
                        field_type: FieldType::String,
                        docs: Some("node label".into()),
                        constraints: vec![],
                    },
                    crate::typesys::FieldDef {
                        name: "next".into(),
                        rendered_name: "next".into(),
                        field_type: FieldType::Optional(Box::new(FieldType::Class("Node".into()))),
                        docs: None,
                        constraints: vec![],
                    },
                ],
                constraints: vec![],
            },
        );
        types.enums.insert(
            "Mood".into(),
            EnumDef {
                internal_name: "Mood".into(),
                rendered_name: "Mood".into(),
                docs: None,
                values: vec![EnumValueDef {
                    name: "Happy".into(),
                    rendered_name: "joyful".into(),
                    docs: None,
                }],
            },
        );
        types
    }

    #[test]
    fn schema_has_aliases_nullable_optional_enum_and_recursive_defs() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output_full(IrFieldDef::new("root", FieldType::Class("Node".into())).aliased("tree"))
            .output(
                "maybe",
                FieldType::Optional(Box::new(FieldType::Enum("Mood".into()))),
            )
            .finish()
            .unwrap();
        let schema = json_schema_for_outputs(&def, &base_types());
        assert_eq!(schema["properties"]["tree"]["$ref"], "#/$defs/Node");
        assert_eq!(schema["properties"]["maybe"]["anyOf"][1]["type"], "null");
        assert_eq!(schema["required"], json!(["tree", "maybe"]));
        assert_eq!(
            schema["$defs"]["Node"]["properties"]["label_alias"]["description"],
            "node label"
        );
        assert_eq!(
            schema["$defs"]["Node"]["properties"]["next"]["anyOf"][1]["type"],
            "null"
        );
        assert_eq!(schema["$defs"]["Mood"]["enum"], json!(["joyful"]));
    }

    #[test]
    fn maps_remain_open_and_nested_non_string_maps_fall_back() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output(
                "m",
                FieldType::Map(Box::new(FieldType::String), Box::new(FieldType::Int)),
            )
            .finish()
            .unwrap();
        let schema = json_schema_for_outputs(&def, &TypeTable::default());
        assert_eq!(
            schema["properties"]["m"]["additionalProperties"]["type"],
            "integer"
        );
        assert!(!json_schema_compatible(&def, &TypeTable::default()));
        let mut types = base_types();
        types
            .classes
            .get_mut("Node")
            .unwrap()
            .fields
            .push(crate::typesys::FieldDef {
                name: "map".into(),
                rendered_name: "map".into(),
                field_type: FieldType::Map(Box::new(FieldType::Int), Box::new(FieldType::String)),
                docs: None,
                constraints: vec![],
            });
        let nested = SignatureDef::build("Nested")
            .input("seed", FieldType::String)
            .output("n", FieldType::Class("Node".into()))
            .finish()
            .unwrap();
        assert!(!json_schema_compatible(&nested, &types));
    }

    #[test]
    fn empty_output_signature_projects_empty_object() {
        let def = SignatureDef {
            name: "Empty".into(),
            instruction: "".into(),
            inputs: Box::new([]),
            outputs: Box::new([]),
        };
        let schema = json_schema_for_outputs(&def, &TypeTable::default());
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["properties"], json!({}));
        assert_eq!(schema["required"], json!([]));
        assert_eq!(schema["additionalProperties"], false);
    }

    #[test]
    fn aliases_canonicalize_recursively_without_string_round_trip() {
        let def = SignatureDef::build("Out")
            .input("seed", FieldType::String)
            .output_full(IrFieldDef::new("root", FieldType::Class("Node".into())).aliased("tree"))
            .output_full(
                IrFieldDef::new("verified", FieldType::Bool)
                    .with_constraint(ConstraintDef::check("affirmative", "this == true")),
            )
            .output_full(
                IrFieldDef::new("asserted", FieldType::Int)
                    .with_constraint(ConstraintDef::assert("this > 0")),
            )
            .finish()
            .unwrap();
        let types = base_types();
        let canonical = canonicalize_output_aliases(
            json!({"tree":{"label_alias":"a \"quoted\" value", "next":null},"verified":true,"asserted":1}),
            &def,
            &types,
        );
        let output = canonical.as_object().unwrap();
        let value = crate::typesys::coerce_value(
            output["root"].clone(),
            &FieldType::Class("Node".into()),
            &types,
        )
        .unwrap()
        .value;
        assert_eq!(value["label"], "a \"quoted\" value");
        assert_eq!(
            alias_output_fields(json!({"root":{"label":"x","next":null}}), &def, &types)["tree"]["label_alias"],
            "x"
        );
    }

    #[test]
    fn native_json_coercion_preserves_delimited_coercion_behavior() {
        let types = base_types();
        let node = crate::typesys::coerce_value(
            json!({"label_alias":"a \"quoted\" label"}),
            &FieldType::Class("Node".into()),
            &types,
        )
        .unwrap()
        .value;
        assert_eq!(node["label"], "a \"quoted\" label");
        assert_eq!(node["next"], Value::Null);

        let mood =
            crate::typesys::coerce_value(json!("JOYFUL"), &FieldType::Enum("Mood".into()), &types)
                .unwrap()
                .value;
        assert_eq!(mood, "Happy");

        let count = crate::typesys::coerce_value(json!("3.7"), &FieldType::Int, &types)
            .unwrap()
            .value;
        assert_eq!(count, 4);
    }
}
