//! JSONL wire format for [`Trace`]: header line, span lines, optional footer.
//!
//! ```text
//! {"h":{...meta...},"components":[...],"models":[...],"prefixes":[...]}
//! {"id":0,"component":0,"seq":0,...}
//! {"f":{...outcome...}}
//! ```
//!
//! The footer is a separate line so evals can be attached after spans are
//! written without rewriting the file. Oversized text fields are truncated at
//! serialization time (never during capture) with a content-hash marker;
//! truncated spans are marked `complete = false` and refuse strict replay.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::trace::span::{ModelEntry, PrefixEntry, Span, SpanEvent, Trace, TraceMeta, TraceOutcome};
use crate::utils::hash::stable_hash_debug;
use crate::{ContentBlock, Message};

/// Highest format version this reader understands.
pub const TRACE_FORMAT_VERSION: u32 = 1;

/// Per-message-text and raw-output budget (bytes).
const TEXT_BUDGET: usize = 64 * 1024;
/// Tool args/result budget (bytes).
const TOOL_BUDGET: usize = 16 * 1024;

#[derive(Serialize)]
struct HeaderRef<'a> {
    h: &'a TraceMeta,
    components: &'a [String],
    models: &'a [ModelEntry],
    prefixes: &'a [PrefixEntry],
}

#[derive(Deserialize)]
struct HeaderOwned {
    h: TraceMeta,
    #[serde(default)]
    components: Vec<String>,
    #[serde(default)]
    models: Vec<ModelEntry>,
    #[serde(default)]
    prefixes: Vec<PrefixEntry>,
}

#[derive(Serialize, Deserialize)]
struct Footer {
    f: TraceOutcome,
}

impl Trace {
    /// Serializes to canonical JSONL: one header line, one line per span, one
    /// footer line when an outcome is recorded.
    pub fn to_jsonl(&self) -> Result<String> {
        let mut out = String::new();
        out.push_str(&serde_json::to_string(&HeaderRef {
            h: &self.meta,
            components: &self.components,
            models: &self.models,
            prefixes: &self.prefixes,
        })?);
        out.push('\n');
        for span in &self.spans {
            let line = match truncated_span(span) {
                Some(truncated) => serde_json::to_string(&truncated)?,
                None => serde_json::to_string(span)?,
            };
            out.push_str(&line);
            out.push('\n');
        }
        if let Some(outcome) = &self.outcome {
            out.push_str(&serde_json::to_string(&Footer { f: outcome.clone() })?);
            out.push('\n');
        }
        Ok(out)
    }

    /// Parses JSONL produced by [`to_jsonl`](Trace::to_jsonl). Unknown fields
    /// and unknown span-event tags are skipped; a version newer than this
    /// reader understands is rejected.
    pub fn from_jsonl(jsonl: &str) -> Result<Trace> {
        let mut lines = jsonl.lines().filter(|line| !line.trim().is_empty());
        let header_line = lines.next().ok_or_else(|| anyhow!("empty trace file"))?;
        let header: HeaderOwned =
            serde_json::from_str(header_line).context("failed to parse trace header line")?;
        if header.h.v > TRACE_FORMAT_VERSION {
            return Err(anyhow!(
                "trace format version {} is newer than supported version {}",
                header.h.v,
                TRACE_FORMAT_VERSION
            ));
        }

        let mut trace = Trace {
            meta: header.h,
            components: header.components,
            models: header.models,
            prefixes: header.prefixes,
            spans: Vec::new(),
            outcome: None,
        };

        for line in lines {
            if let Ok(footer) = serde_json::from_str::<Footer>(line) {
                trace.outcome = Some(footer.f);
                continue;
            }
            let mut span: Span =
                serde_json::from_str(line).context("failed to parse trace span line")?;
            // Unknown-tag placeholders don't survive re-serialization; drop
            // them here so a read-write cycle is stable.
            span.events
                .retain(|event| !matches!(event, SpanEvent::Unknown));
            trace.spans.push(span);
        }
        Ok(trace)
    }
}

fn truncate_marker(text: &str, budget: usize) -> String {
    let hash = stable_hash_debug(&text);
    let mut cut = budget;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}…[truncated fnv1a={hash:016x} len={}]",
        &text[..cut],
        text.len()
    )
}

fn truncate_text(text: &mut String, budget: usize) -> bool {
    if text.len() <= budget {
        return false;
    }
    *text = truncate_marker(text, budget);
    true
}

fn truncate_messages(messages: &mut [Message]) -> bool {
    let mut truncated = false;
    for message in messages {
        for block in &mut message.content {
            if let ContentBlock::Text { text } = block {
                truncated |= truncate_text(text, TEXT_BUDGET);
            }
        }
    }
    truncated
}

/// Returns a budget-respecting copy of the span when any field overflows,
/// `None` when the span fits as-is. Truncation clears `complete`.
fn truncated_span(span: &Span) -> Option<Span> {
    let over_budget = span
        .suffix
        .iter()
        .flat_map(|message| &message.content)
        .any(|block| matches!(block, ContentBlock::Text { text } if text.len() > TEXT_BUDGET))
        || span
            .raw_output
            .as_ref()
            .is_some_and(|text| text.len() > TEXT_BUDGET)
        || span.events.iter().any(|event| match event {
            SpanEvent::ToolRun { args, result, .. } => {
                result.len() > TOOL_BUDGET
                    || serde_json::to_string(args).map(|s| s.len()).unwrap_or(0) > TOOL_BUDGET
            }
            _ => false,
        });
    if !over_budget {
        return None;
    }

    let mut span = span.clone();
    truncate_messages(&mut span.suffix);
    if let Some(raw_output) = &mut span.raw_output {
        truncate_text(raw_output, TEXT_BUDGET);
    }
    for event in &mut span.events {
        if let SpanEvent::ToolRun { args, result, .. } = event {
            truncate_text(result, TOOL_BUDGET);
            let serialized = serde_json::to_string(args).unwrap_or_default();
            if serialized.len() > TOOL_BUDGET {
                *args = serde_json::Value::String(truncate_marker(&serialized, TOOL_BUDGET));
            }
        }
    }
    span.complete = false;
    Some(span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::span::{CompId, ModelId, SpanId};
    use crate::{LMConfig, LmUsage};

    fn test_span(suffix_text: &str) -> Span {
        Span {
            id: SpanId(0),
            component: CompId(0),
            seq: 0,
            parent: None,
            links: Vec::new(),
            prefix: None,
            suffix: vec![Message::user(suffix_text)],
            input: None,
            model: ModelId(0),
            request_hash: 7,
            events: vec![SpanEvent::Exchange {
                message: Message::assistant("out"),
                usage: LmUsage::default(),
            }],
            raw_output: Some("out".to_string()),
            output: None,
            usage: LmUsage::default(),
            error: None,
            started_at_us: 1,
            duration_us: 2,
            complete: true,
        }
    }

    fn test_trace(span: Span) -> Trace {
        Trace {
            meta: TraceMeta {
                v: 1,
                trace_id: "t".to_string(),
                started_at_us: 1,
                ..TraceMeta::default()
            },
            components: vec!["comp".to_string()],
            models: vec![ModelEntry::from_config(&LMConfig::default())],
            prefixes: Vec::new(),
            spans: vec![span],
            outcome: None,
        }
    }

    #[test]
    fn oversized_suffix_is_truncated_and_marked_incomplete() {
        let big = "x".repeat(TEXT_BUDGET * 2);
        let trace = test_trace(test_span(&big));
        let jsonl = trace.to_jsonl().expect("serialize");
        let parsed = Trace::from_jsonl(&jsonl).expect("deserialize");
        let span = &parsed.spans[0];
        assert!(!span.complete);
        let text = span.suffix[0].text_content();
        assert!(text.len() < big.len());
        assert!(text.contains("truncated fnv1a="));
        assert!(text.contains(&format!("len={}", big.len())));
    }

    #[test]
    fn newer_version_is_rejected() {
        let mut trace = test_trace(test_span("hi"));
        trace.meta.v = 99;
        let jsonl = trace.to_jsonl().expect("serialize");
        let err = Trace::from_jsonl(&jsonl).expect_err("should reject v99");
        assert!(err.to_string().contains("newer than supported"));
    }

    #[test]
    fn unknown_event_tags_are_skipped_on_read() {
        let trace = test_trace(test_span("hi"));
        let mut jsonl = trace.to_jsonl().expect("serialize");
        // Simulate a newer writer emitting an unknown event tag.
        jsonl = jsonl.replace(
            "\"events\":[",
            "\"events\":[{\"t\":\"future_event\",\"payload\":1},",
        );
        let parsed = Trace::from_jsonl(&jsonl).expect("unknown tags should be skipped");
        assert_eq!(parsed.spans[0].events.len(), 1);
        assert!(matches!(
            parsed.spans[0].events[0],
            SpanEvent::Exchange { .. }
        ));
    }
}
