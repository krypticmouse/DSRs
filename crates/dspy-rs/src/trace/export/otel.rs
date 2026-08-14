//! OTel export (RFC 0001 §4g): one-way batch mapping of a finished [`Trace`]
//! onto OpenTelemetry GenAI semantic conventions — as plain serializable
//! structs in the OTLP/JSON wire shape, with **no OpenTelemetry dependency**.
//!
//! Mapping (RFC §4g):
//!
//! | trace format                        | OTel                                            |
//! |-------------------------------------|-------------------------------------------------|
//! | `meta.trace_id`                     | trace id (used verbatim if 32-hex, else hashed) |
//! | the rollout                         | root span `dsrs.rollout`, kind `INTERNAL`       |
//! | `Span`                              | span, name = component name, kind = `CLIENT`    |
//! | `span.parent` (else the root span)  | parent span id                                  |
//! | `started_at_us` / `duration_us`     | start/end timestamps (ns)                       |
//! | `models[span.model].config.model`   | `gen_ai.request.model`                          |
//! | `temperature` / `max_tokens`        | `gen_ai.request.temperature` / `.max_tokens`    |
//! | `usage.prompt/completion_tokens`    | `gen_ai.usage.input_tokens` / `.output_tokens`  |
//! | `prefix ++ suffix` (content opt-in) | `gen_ai.prompt` events                          |
//! | `raw_output` (content opt-in)       | `gen_ai.completion` event                       |
//! | `SpanEvent::ToolRun`                | child span `tool:{name}`, `gen_ai.tool.*`       |
//! | `span.error`                        | status `ERROR` + message                        |
//! | `seq` / `component` / hashes        | `dsrs.*` attributes                             |
//!
//! Prompt/completion/tool content is **opt-in** (`include_content`), mirroring
//! OTel's GenAI content-capture switch — spans stay exportable to shared
//! collectors without leaking prompt text.
//!
//! # Shipping to a collector
//!
//! [`Trace::to_otlp_json`] wraps the spans in a complete
//! `resourceSpans` envelope, directly acceptable to any OTLP/HTTP collector
//! (Jaeger, Tempo, otel-collector) — no SDK required:
//!
//! ```ignore
//! let payload = trace.to_otlp_json("my-service", /* include_content */ false);
//! reqwest::Client::new()
//!     .post("http://localhost:4318/v1/traces")
//!     .json(&payload)
//!     .send()
//!     .await?;
//! ```
//!
//! Timing note: `ToolRun` events record only their duration, so tool child
//! spans start at their parent span's start time — durations are exact,
//! offsets within the parent are not.

use serde::Serialize;

use crate::trace::span::{Span, SpanEvent, Trace};
use crate::utils::hash::stable_hash_debug;

/// OTLP `SPAN_KIND_INTERNAL`.
pub const SPAN_KIND_INTERNAL: u32 = 1;
/// OTLP `SPAN_KIND_CLIENT`.
pub const SPAN_KIND_CLIENT: u32 = 3;
/// OTLP `STATUS_CODE_ERROR`.
pub const STATUS_CODE_ERROR: u32 = 2;

/// One span in the OTLP/JSON wire shape (proto3 JSON mapping: camelCase keys,
/// 64-bit integers as decimal strings, ids as lowercase hex).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OtelSpan {
    /// 128-bit trace id, 32 lowercase hex chars.
    pub trace_id: String,
    /// 64-bit span id, 16 lowercase hex chars.
    pub span_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_span_id: Option<String>,
    pub name: String,
    pub kind: u32,
    pub start_time_unix_nano: String,
    pub end_time_unix_nano: String,
    pub attributes: Vec<OtelKeyValue>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<OtelEvent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<OtelStatus>,
}

/// OTLP `KeyValue`.
#[derive(Clone, Debug, Serialize)]
pub struct OtelKeyValue {
    pub key: String,
    pub value: OtelValue,
}

impl OtelKeyValue {
    fn str(key: &str, value: impl Into<String>) -> Self {
        Self {
            key: key.to_string(),
            value: OtelValue::StringValue(value.into()),
        }
    }

    fn int(key: &str, value: i64) -> Self {
        Self {
            key: key.to_string(),
            // proto3 JSON maps int64 to a decimal string.
            value: OtelValue::IntValue(value.to_string()),
        }
    }

    fn double(key: &str, value: f64) -> Self {
        Self {
            key: key.to_string(),
            value: OtelValue::DoubleValue(value),
        }
    }
}

/// OTLP `AnyValue` (the oneof arms this export emits).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum OtelValue {
    StringValue(String),
    IntValue(String),
    DoubleValue(f64),
}

/// OTLP span `Event`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OtelEvent {
    pub time_unix_nano: String,
    pub name: String,
    pub attributes: Vec<OtelKeyValue>,
}

/// OTLP span `Status`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OtelStatus {
    pub code: u32,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub message: String,
}

/// Deterministic span-id namespaces: predict spans, the root span, and tool
/// child spans never collide and stay greppable in collector UIs.
const PREDICT_SPAN_NS: u64 = 0x0100_0000_0000_0000;
const ROOT_SPAN_NS: u64 = 0x0200_0000_0000_0000;
const TOOL_SPAN_NS: u64 = 0x0300_0000_0000_0000;

fn hex_span_id(id: u64) -> String {
    format!("{id:016x}")
}

fn ns(us: u64) -> String {
    (us.saturating_mul(1_000)).to_string()
}

/// 32-hex trace id: the recorded id verbatim when it already is one (the
/// capture scope mints exactly this shape), otherwise a stable hash of it.
fn otel_trace_id(trace_id: &str) -> String {
    let is_32_hex =
        trace_id.len() == 32 && trace_id.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase());
    if is_32_hex {
        trace_id.to_string()
    } else {
        format!(
            "{:016x}{:016x}",
            stable_hash_debug(&("dsrs.otel.hi", trace_id)),
            stable_hash_debug(&("dsrs.otel.lo", trace_id)),
        )
    }
}

impl Trace {
    /// Maps this trace onto OTel-shaped spans: one root span for the rollout,
    /// one `CLIENT` span per `Predict` invocation (parented to its recorded
    /// parent span, else the root), and one child span per `ToolRun`.
    ///
    /// `include_content` gates prompt/completion/tool-payload capture (the
    /// GenAI content-capture switch); identity, usage, and timing attributes
    /// are always emitted.
    pub fn to_otel_spans(&self, include_content: bool) -> Vec<OtelSpan> {
        let trace_id = otel_trace_id(&self.meta.trace_id);
        let root_span_id = hex_span_id(ROOT_SPAN_NS);

        let root_end_us = match &self.outcome {
            Some(outcome) => self.meta.started_at_us + outcome.duration_us,
            None => self
                .spans
                .iter()
                .map(|span| span.started_at_us + span.duration_us)
                .max()
                .unwrap_or(self.meta.started_at_us),
        };
        let mut root_attributes = vec![OtelKeyValue::str("dsrs.trace_id", &self.meta.trace_id)];
        if let Some(candidate_hash) = self.meta.candidate_hash {
            root_attributes.push(OtelKeyValue::str(
                "dsrs.candidate_hash",
                format!("{candidate_hash:016x}"),
            ));
        }
        for (key, value) in &self.meta.tags {
            root_attributes.push(OtelKeyValue::str(&format!("dsrs.tag.{key}"), value));
        }
        let root_status = self
            .outcome
            .as_ref()
            .and_then(|outcome| outcome.error.as_ref())
            .map(|error| OtelStatus {
                code: STATUS_CODE_ERROR,
                message: error.clone(),
            });

        let mut spans = vec![OtelSpan {
            trace_id: trace_id.clone(),
            span_id: root_span_id.clone(),
            parent_span_id: None,
            name: "dsrs.rollout".to_string(),
            kind: SPAN_KIND_INTERNAL,
            start_time_unix_nano: ns(self.meta.started_at_us),
            end_time_unix_nano: ns(root_end_us),
            attributes: root_attributes,
            events: Vec::new(),
            status: root_status,
        }];

        for span in &self.spans {
            spans.push(self.otel_predict_span(span, &trace_id, &root_span_id, include_content));
            spans.extend(self.otel_tool_spans(span, &trace_id, include_content));
        }
        spans
    }

    fn otel_predict_span(
        &self,
        span: &Span,
        trace_id: &str,
        root_span_id: &str,
        include_content: bool,
    ) -> OtelSpan {
        let config = self.model(span);
        let mut attributes = vec![
            OtelKeyValue::str("gen_ai.request.model", &config.model),
            OtelKeyValue::double("gen_ai.request.temperature", config.temperature as f64),
            OtelKeyValue::int("gen_ai.request.max_tokens", config.max_tokens as i64),
            OtelKeyValue::int("gen_ai.usage.input_tokens", span.usage.prompt_tokens as i64),
            OtelKeyValue::int(
                "gen_ai.usage.output_tokens",
                span.usage.completion_tokens as i64,
            ),
            OtelKeyValue::str("dsrs.component", self.component_name(span.component)),
            OtelKeyValue::int("dsrs.seq", span.seq as i64),
            OtelKeyValue::str("dsrs.request_hash", format!("{:016x}", span.request_hash)),
        ];
        if let Some(candidate_hash) = self.meta.candidate_hash {
            attributes.push(OtelKeyValue::str(
                "dsrs.candidate_hash",
                format!("{candidate_hash:016x}"),
            ));
        }

        let mut events = Vec::new();
        if include_content {
            for message in self.prompt(span) {
                events.push(OtelEvent {
                    time_unix_nano: ns(span.started_at_us),
                    name: "gen_ai.prompt".to_string(),
                    attributes: vec![
                        OtelKeyValue::str("gen_ai.prompt.role", message.role.as_str()),
                        OtelKeyValue::str("gen_ai.prompt.content", message.content()),
                    ],
                });
            }
            if let Some(raw_output) = &span.raw_output {
                events.push(OtelEvent {
                    time_unix_nano: ns(span.started_at_us + span.duration_us),
                    name: "gen_ai.completion".to_string(),
                    attributes: vec![OtelKeyValue::str("gen_ai.completion.content", raw_output)],
                });
            }
        }

        OtelSpan {
            trace_id: trace_id.to_string(),
            span_id: hex_span_id(PREDICT_SPAN_NS | span.id.0 as u64),
            parent_span_id: Some(match span.parent {
                Some(parent) => hex_span_id(PREDICT_SPAN_NS | parent.0 as u64),
                None => root_span_id.to_string(),
            }),
            name: self.component_name(span.component).to_string(),
            kind: SPAN_KIND_CLIENT,
            start_time_unix_nano: ns(span.started_at_us),
            end_time_unix_nano: ns(span.started_at_us + span.duration_us),
            attributes,
            events,
            status: span.error.as_ref().map(|error| OtelStatus {
                code: STATUS_CODE_ERROR,
                message: format!("{}: {}", error.kind.as_str(), error.message),
            }),
        }
    }

    fn otel_tool_spans(
        &self,
        span: &Span,
        trace_id: &str,
        include_content: bool,
    ) -> Vec<OtelSpan> {
        span.events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| match event {
                SpanEvent::ToolRun {
                    id,
                    name,
                    args,
                    result,
                    duration_us,
                    error,
                } => {
                    let mut attributes = vec![
                        OtelKeyValue::str("gen_ai.tool.name", name),
                        OtelKeyValue::str("gen_ai.tool.call.id", id),
                    ];
                    if include_content {
                        attributes.push(OtelKeyValue::str(
                            "gen_ai.tool.call.arguments",
                            serde_json::to_string(args).unwrap_or_default(),
                        ));
                        attributes.push(OtelKeyValue::str("gen_ai.tool.call.result", result));
                    }
                    Some(OtelSpan {
                        trace_id: trace_id.to_string(),
                        span_id: hex_span_id(
                            TOOL_SPAN_NS | ((span.id.0 as u64) << 16) | index as u64,
                        ),
                        parent_span_id: Some(hex_span_id(PREDICT_SPAN_NS | span.id.0 as u64)),
                        name: format!("tool:{name}"),
                        kind: SPAN_KIND_INTERNAL,
                        // ToolRuns record duration only; anchor at parent start.
                        start_time_unix_nano: ns(span.started_at_us),
                        end_time_unix_nano: ns(span.started_at_us + duration_us),
                        attributes,
                        events: Vec::new(),
                        status: error.as_ref().map(|message| OtelStatus {
                            code: STATUS_CODE_ERROR,
                            message: message.clone(),
                        }),
                    })
                }
                _ => None,
            })
            .collect()
    }

    /// Complete OTLP/HTTP JSON payload (`resourceSpans` envelope) for this
    /// trace — POST it to a collector's `/v1/traces` endpoint as-is. See the
    /// module docs for an example.
    pub fn to_otlp_json(&self, service_name: &str, include_content: bool) -> serde_json::Value {
        serde_json::json!({
            "resourceSpans": [{
                "resource": {
                    "attributes": [
                        { "key": "service.name", "value": { "stringValue": service_name } }
                    ]
                },
                "scopeSpans": [{
                    "scope": { "name": "dsrs.trace", "version": env!("CARGO_PKG_VERSION") },
                    "spans": self.to_otel_spans(include_content)
                }]
            }]
        })
    }
}
