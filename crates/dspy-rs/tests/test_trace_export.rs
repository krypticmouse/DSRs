//! Golden-file coverage for the trace exports (RFC 0001 §4f/§4g): a fully
//! deterministic canned trace (fixed ids, hashes, and timestamps — nothing to
//! normalize) must project to byte-stable JSON.
//!
//! To regenerate the goldens after an intentional format change:
//! `DSRS_BLESS=1 cargo test -p dspy-rs --test test_trace_export`

use std::collections::BTreeMap;

use dspy_rs::{
    CompId, Eval, LMConfig, LmUsage, Message, ModelEntry, ModelId, PrefixEntry, PrefixId, Span,
    SpanError, SpanErrorKind, SpanEvent, SpanId, Trace, TraceMeta, TraceOutcome,
};
use serde_json::{Value, json};

fn tool_call(id: &str, name: &str, args: Value) -> rig::message::ToolCall {
    match rig::completion::AssistantContent::tool_call(id, name, args) {
        rig::completion::AssistantContent::ToolCall(tc) => tc,
        _ => unreachable!("tool_call constructor returns a ToolCall"),
    }
}

fn usage(prompt: u64, completion: u64) -> LmUsage {
    LmUsage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
    }
}

fn json_map(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => unreachable!("json_map takes an object literal"),
    }
}

/// A deterministic 3-span rollout: a prefix-carrying drafter call, a
/// tool-looping call, and a failed provider call (no events).
fn canned_trace() -> Trace {
    let base_span = Span {
        id: SpanId(0),
        component: CompId(0),
        seq: 0,
        parent: None,
        links: Vec::new(),
        prefix: None,
        suffix: Vec::new(),
        input: None,
        model: ModelId(0),
        request_hash: 0,
        events: Vec::new(),
        raw_output: None,
        output: None,
        usage: LmUsage::default(),
        error: None,
        started_at_us: 0,
        duration_us: 0,
        complete: true,
    };

    let drafter = Span {
        id: SpanId(0),
        prefix: Some(PrefixId(0)),
        suffix: vec![Message::user("[[ ## question ## ]]\nq")],
        input: Some(json_map(json!({"question": "q"}))),
        request_hash: 0x1111,
        events: vec![SpanEvent::Exchange {
            message: Message::assistant("[[ ## answer ## ]]\ndraft"),
            usage: usage(10, 5),
        }],
        raw_output: Some("[[ ## answer ## ]]\ndraft".to_string()),
        output: Some(json_map(json!({"answer": "draft"}))),
        usage: usage(10, 5),
        started_at_us: 1_000_100,
        duration_us: 200,
        ..base_span.clone()
    };

    let tooler = Span {
        id: SpanId(1),
        component: CompId(1),
        links: vec![SpanId(0)],
        suffix: vec![Message::user("use the tool on: draft")],
        input: Some(json_map(json!({"question": "use the tool on: draft"}))),
        request_hash: 0x2222,
        events: vec![
            SpanEvent::Exchange {
                message: Message::tool_call(tool_call("call_1", "search", json!({"q": "draft"}))),
                usage: usage(13, 5),
            },
            SpanEvent::ToolRun {
                id: "call_1".to_string(),
                name: "search".to_string(),
                args: json!({"q": "draft"}),
                result: "found: relevant doc".to_string(),
                duration_us: 300,
                error: None,
            },
            SpanEvent::Exchange {
                message: Message::assistant("[[ ## answer ## ]]\nfinal"),
                usage: usage(7, 3),
            },
        ],
        raw_output: Some("[[ ## answer ## ]]\nfinal".to_string()),
        output: Some(json_map(json!({"answer": "final"}))),
        usage: usage(20, 8),
        started_at_us: 1_000_400,
        duration_us: 700,
        ..base_span.clone()
    };

    // A failed provider call: no events, no output — excluded from RL
    // transitions, exported to OTel with ERROR status.
    let failed = Span {
        id: SpanId(2),
        seq: 1,
        links: vec![SpanId(1)],
        suffix: vec![Message::user("[[ ## question ## ]]\nretry q")],
        input: Some(json_map(json!({"question": "retry q"}))),
        request_hash: 0x3333,
        error: Some(SpanError {
            kind: SpanErrorKind::Lm,
            message: "provider unreachable".to_string(),
        }),
        started_at_us: 1_001_200,
        duration_us: 90,
        ..base_span
    };

    let config = LMConfig {
        base_url: None,
        api_key: None,
        model: "openai:gpt-4o-mini".to_string(),
        temperature: 0.0,
        max_tokens: 128,
        max_tool_iterations: 4,
        max_retries: 0,
        retry_base_delay_ms: 1,
        cache: false,
    };

    Trace {
        meta: TraceMeta {
            v: 1,
            trace_id: "0123456789abcdef0123456789abcdef".to_string(),
            started_at_us: 1_000_000,
            candidate_hash: Some(42),
            input: Some(json_map(json!({"question": "q"}))),
            tags: BTreeMap::from([("optimizer".to_string(), "gepa".to_string())]),
        },
        components: vec!["drafter".to_string(), "tooler".to_string()],
        models: vec![ModelEntry::from_config(&config)],
        prefixes: vec![PrefixEntry {
            messages: vec![
                Message::system("You draft answers."),
                Message::user("[[ ## question ## ]]\ndemo-q"),
                Message::assistant("[[ ## answer ## ]]\ndemo-a"),
            ],
        }],
        spans: vec![drafter, tooler, failed],
        outcome: Some(TraceOutcome {
            output: Some(json_map(json!({"answer": "final"}))),
            error: None,
            eval: Some(Eval::with_feedback(0.75, "ok")),
            duration_us: 5_000,
        }),
        ..Trace::default()
    }
}

/// Compares produced JSON against a golden file; `DSRS_BLESS=1` rewrites it.
fn assert_matches_golden(produced: &Value, file_name: &str) {
    let path = format!(
        "{}/tests/fixtures/{file_name}",
        env!("CARGO_MANIFEST_DIR")
    );
    if std::env::var("DSRS_BLESS").is_ok() {
        let mut pretty = serde_json::to_string_pretty(produced).expect("serialize golden");
        pretty.push('\n');
        std::fs::write(&path, pretty).expect("write golden");
    }
    let golden = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("missing golden {path} (run with DSRS_BLESS=1): {err}"));
    let golden: Value = serde_json::from_str(&golden).expect("parse golden");
    assert_eq!(
        produced, &golden,
        "export drifted from {file_name}; if intentional, re-bless with DSRS_BLESS=1"
    );
}

#[test]
fn rl_rollout_matches_golden() {
    let trace = canned_trace();
    let rollout = trace.to_rl_rollout().expect("eval recorded");
    let produced = serde_json::to_value(&rollout).expect("serialize rollout");

    // Shape sanity before the byte-level golden: reward from the outcome,
    // failed span dropped, tool loop flattened into completion messages.
    assert_eq!(produced["reward"], json!(0.75));
    assert_eq!(produced["trace_id"], json!("0123456789abcdef0123456789abcdef"));
    let transitions = produced["transitions"].as_array().expect("transitions");
    assert_eq!(transitions.len(), 2, "the failed span emits no transition");
    assert_eq!(transitions[0]["component"], json!("drafter"));
    assert_eq!(
        transitions[0]["messages"].as_array().unwrap().len(),
        4,
        "prefix (system + demo pair) ++ suffix"
    );
    assert_eq!(transitions[1]["component"], json!("tooler"));
    assert_eq!(
        transitions[1]["completion"].as_array().unwrap().len(),
        3,
        "tool-call turn, tool-result turn, final answer"
    );
    assert_eq!(transitions[1]["usage"]["total_tokens"], json!(28));
    assert_eq!(transitions[1]["model"], json!("openai:gpt-4o-mini"));

    assert_matches_golden(&produced, "rl_rollout.golden.json");

    // The JSONL line is the same object.
    let line = rollout.to_json_line().expect("jsonl line");
    assert_eq!(
        serde_json::from_str::<Value>(&line).expect("parse line"),
        produced
    );
}

#[test]
fn rl_rollout_requires_a_recorded_eval() {
    let mut trace = canned_trace();
    trace.outcome.as_mut().unwrap().eval = None;
    assert!(trace.to_rl_rollout().is_none());
    trace.outcome = None;
    assert!(trace.to_rl_rollout().is_none());
}

#[test]
fn otel_export_matches_golden() {
    let trace = canned_trace();
    let produced = trace.to_otlp_json("dsrs-test", true);

    // Shape sanity before the byte-level golden.
    let spans = &produced["resourceSpans"][0]["scopeSpans"][0]["spans"];
    let spans = spans.as_array().expect("spans array");
    // Root + 3 predict spans + 1 tool child.
    assert_eq!(spans.len(), 5);

    let root = &spans[0];
    assert_eq!(root["name"], json!("dsrs.rollout"));
    assert_eq!(root["traceId"], json!("0123456789abcdef0123456789abcdef"));
    assert!(root.get("parentSpanId").is_none());
    assert_eq!(root["startTimeUnixNano"], json!("1000000000"));
    assert_eq!(root["endTimeUnixNano"], json!("1005000000"));

    let drafter = &spans[1];
    assert_eq!(drafter["name"], json!("drafter"));
    assert_eq!(drafter["kind"], json!(3), "predict spans are CLIENT");
    assert_eq!(drafter["parentSpanId"], root["spanId"]);
    let attrs: Vec<(&str, &Value)> = drafter["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|kv| (kv["key"].as_str().unwrap(), &kv["value"]))
        .collect();
    assert!(attrs.contains(&(
        "gen_ai.request.model",
        &json!({"stringValue": "openai:gpt-4o-mini"})
    )));
    assert!(attrs.contains(&("gen_ai.usage.input_tokens", &json!({"intValue": "10"}))));
    assert!(attrs.contains(&(
        "dsrs.request_hash",
        &json!({"stringValue": "0000000000001111"})
    )));

    let tool = &spans[3];
    assert_eq!(tool["name"], json!("tool:search"));
    assert_eq!(tool["parentSpanId"], spans[2]["spanId"]);
    assert_eq!(tool["kind"], json!(1), "tool spans are INTERNAL");

    let failed = &spans[4];
    assert_eq!(failed["status"]["code"], json!(2));
    assert!(
        failed["status"]["message"]
            .as_str()
            .unwrap()
            .contains("provider unreachable")
    );

    assert_matches_golden(&produced, "otel_export.golden.json");
}

#[test]
fn otel_content_capture_is_opt_in() {
    let trace = canned_trace();
    let produced = trace.to_otlp_json("dsrs-test", false);
    let spans = produced["resourceSpans"][0]["scopeSpans"][0]["spans"]
        .as_array()
        .expect("spans array")
        .clone();

    let all_attr_keys: Vec<String> = spans
        .iter()
        .flat_map(|span| span["attributes"].as_array().unwrap().iter())
        .map(|kv| kv["key"].as_str().unwrap().to_string())
        .collect();
    assert!(
        !all_attr_keys.iter().any(|key| key.contains("arguments") || key.contains("result")),
        "tool payloads are content"
    );
    for span in &spans {
        assert!(
            span.get("events").is_none(),
            "prompt/completion events are content: {span}"
        );
    }
    // Identity, usage, and timing stay.
    assert!(all_attr_keys.iter().any(|key| key == "gen_ai.usage.input_tokens"));
    assert!(all_attr_keys.iter().any(|key| key == "dsrs.request_hash"));
    assert!(all_attr_keys.iter().any(|key| key == "gen_ai.tool.name"));
}
