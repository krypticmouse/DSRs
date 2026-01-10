use std::collections::{BTreeMap, HashMap};

use baml_bridge::baml_types::BamlValue;
use baml_bridge::{BamlType, BamlValueConvert, ToBamlValue};

#[derive(Debug, Clone, PartialEq, BamlType)]
struct RoundtripStruct {
    name: String,
    count: i32,
    tags: Vec<String>,
    meta: Option<String>,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
enum UnitEnum {
    Alpha,
    #[baml(alias = "beta")]
    Beta,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
#[baml(tag = "kind")]
enum DataEnum {
    Message { body: String, count: i64 },
    Empty,
}

#[derive(Debug, Clone, PartialEq, BamlType)]
struct NestedStruct {
    title: String,
    items: Option<Vec<RoundtripStruct>>,
    metadata: HashMap<String, Option<i32>>,
}

#[test]
fn roundtrip_primitives() {
    let samples = ["", "hello", "with spaces"];
    for value in samples {
        let baml = value.to_baml_value();
        let recovered = String::try_from_baml_value(baml, Vec::new()).expect("string roundtrip");
        assert_eq!(value, recovered);
    }

    for value in [0i32, 1, -1, i32::MAX, i32::MIN] {
        let baml = value.to_baml_value();
        let recovered = i32::try_from_baml_value(baml, Vec::new()).expect("i32 roundtrip");
        assert_eq!(value, recovered);
    }

    for value in [0.0f64, 1.25, -2.5] {
        let baml = value.to_baml_value();
        let recovered = f64::try_from_baml_value(baml, Vec::new()).expect("f64 roundtrip");
        assert_eq!(value, recovered);
    }

    for value in [0.5f32, 2.25, -3.0] {
        let baml = value.to_baml_value();
        let recovered = f32::try_from_baml_value(baml, Vec::new()).expect("f32 roundtrip");
        assert_eq!(value, recovered);
    }

    let value = true;
    let baml = value.to_baml_value();
    let recovered = bool::try_from_baml_value(baml, Vec::new()).expect("bool roundtrip");
    assert_eq!(value, recovered);
}

#[test]
fn roundtrip_containers() {
    let some: Option<i32> = Some(42);
    let baml = some.to_baml_value();
    let recovered = Option::<i32>::try_from_baml_value(baml, Vec::new()).expect("option some");
    assert_eq!(some, recovered);

    let none: Option<i32> = None;
    let baml = none.to_baml_value();
    let recovered = Option::<i32>::try_from_baml_value(baml, Vec::new()).expect("option none");
    assert_eq!(none, recovered);

    let list = vec!["a".to_string(), "b".to_string()];
    let baml = list.to_baml_value();
    let recovered = Vec::<String>::try_from_baml_value(baml, Vec::new()).expect("vec roundtrip");
    assert_eq!(list, recovered);

    let mut map = HashMap::new();
    map.insert("answer".to_string(), 42i32);
    let baml = map.to_baml_value();
    let recovered =
        HashMap::<String, i32>::try_from_baml_value(baml, Vec::new()).expect("hashmap roundtrip");
    assert_eq!(map, recovered);

    let mut tree = BTreeMap::new();
    tree.insert("left".to_string(), 1i64);
    tree.insert("right".to_string(), 2i64);
    let baml = tree.to_baml_value();
    let recovered =
        BTreeMap::<String, i64>::try_from_baml_value(baml, Vec::new()).expect("btreemap roundtrip");
    assert_eq!(tree, recovered);
}

#[test]
fn roundtrip_struct() {
    let value = RoundtripStruct {
        name: "example".to_string(),
        count: 7,
        tags: vec!["tag".to_string()],
        meta: Some("meta".to_string()),
    };
    let baml = value.to_baml_value();
    let recovered = <RoundtripStruct as BamlValueConvert>::try_from_baml_value(baml, Vec::new())
        .expect("struct roundtrip");
    assert_eq!(value, recovered);
}

#[test]
fn roundtrip_unit_enum() {
    let value = UnitEnum::Beta;
    let baml = value.to_baml_value();
    match &baml {
        BamlValue::Enum(_, variant) => assert_eq!(variant, "beta"),
        other => panic!("expected enum value, got {other:?}"),
    }
    let recovered = <UnitEnum as BamlValueConvert>::try_from_baml_value(baml, Vec::new())
        .expect("enum roundtrip");
    assert_eq!(value, recovered);
}

#[test]
fn roundtrip_data_enum() {
    let value = DataEnum::Message {
        body: "hello".to_string(),
        count: 3,
    };
    let baml = value.to_baml_value();
    match &baml {
        BamlValue::Class(_, map) | BamlValue::Map(map) => match map.get("kind") {
            Some(BamlValue::String(tag)) => assert_eq!(tag, "Message"),
            other => panic!("expected tag field, got {other:?}"),
        },
        other => panic!("expected class value, got {other:?}"),
    }
    let recovered = <DataEnum as BamlValueConvert>::try_from_baml_value(baml, Vec::new())
        .expect("data enum roundtrip");
    assert_eq!(value, recovered);
}

#[test]
fn roundtrip_nested() {
    let mut metadata = HashMap::new();
    metadata.insert("alpha".to_string(), Some(1));
    metadata.insert("beta".to_string(), None);

    let value = NestedStruct {
        title: "nested".to_string(),
        items: Some(vec![RoundtripStruct {
            name: "child".to_string(),
            count: 2,
            tags: vec!["x".to_string(), "y".to_string()],
            meta: None,
        }]),
        metadata,
    };

    let baml = value.to_baml_value();
    let recovered = <NestedStruct as BamlValueConvert>::try_from_baml_value(baml, Vec::new())
        .expect("nested roundtrip");
    assert_eq!(value, recovered);
}
