use std::collections::HashMap;

use baml_bridge::{parse_llm_output, BamlType};
use proptest::collection::{hash_map, vec};
use proptest::prelude::*;
use proptest::string::string_regex;

#[derive(Debug, Clone, PartialEq, BamlType, serde::Serialize, serde::Deserialize)]
struct RoundTripUser {
    name: String,
    age: u32,
    active: bool,
    tags: Vec<String>,
    meta: HashMap<String, i64>,
    nickname: Option<String>,
}

fn arb_string() -> impl Strategy<Value = String> {
    string_regex("[a-zA-Z0-9 _-]{0,12}").expect("valid regex")
}

fn arb_user() -> impl Strategy<Value = RoundTripUser> {
    (
        arb_string(),
        0_u32..1000_u32,
        any::<bool>(),
        vec(arb_string(), 0..5),
        hash_map(arb_string(), -1000_i64..1000_i64, 0..5),
        proptest::option::of(arb_string()),
    )
        .prop_map(|(name, age, active, tags, meta, nickname)| RoundTripUser {
            name,
            age,
            active,
            tags,
            meta,
            nickname,
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

    #[test]
    fn round_trip_user(user in arb_user()) {
        let json = serde_json::to_string(&user).expect("serialize");
        let parsed = parse_llm_output::<RoundTripUser>(&json, true).expect("parse");
        prop_assert_eq!(parsed.value, user);
    }
}
