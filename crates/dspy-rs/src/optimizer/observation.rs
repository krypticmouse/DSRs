//! Configurable trace evidence for instruction proposal prompts.

use serde_json::{Value, json};

use crate::trace::{Eval, Trace};

/// Projects a rollout into a minimal summary, with parsed training evidence
/// included only when the caller explicitly enables sharing. Complete raw
/// traces, including prompts, model configuration, tool events, and unparsed
/// output, are available only through the separate full-trace opt-in.
pub(super) fn proposal_observation(
    trace: &Trace,
    eval: &Eval,
    share_training_examples: bool,
    share_full_training_traces: bool,
) -> anyhow::Result<Value> {
    if share_full_training_traces {
        return Ok(json!({
            "score": eval.score,
            "feedback": eval.feedback,
            "trace": trace.to_jsonl()?,
        }));
    }

    let spans = trace
        .spans
        .iter()
        .map(|span| {
            let span_eval = span.eval.as_ref();
            let mut observation = json!({
                "component": trace.components.get(span.component.0 as usize),
                "seq": span.seq,
                "score": span_eval.map(|eval| eval.score),
            });
            if share_training_examples {
                observation["input"] = span
                    .input
                    .as_ref()
                    .map_or(Value::Null, |input| Value::Object(input.clone()));
                observation["output"] = span
                    .output
                    .as_ref()
                    .map_or(Value::Null, |output| Value::Object(output.clone()));
                observation["feedback"] = span_eval
                    .and_then(|eval| eval.feedback.as_ref())
                    .map_or(Value::Null, |feedback| json!(feedback));
            }
            observation
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "score": eval.score,
        "feedback": if share_training_examples {
            eval.feedback
                .as_deref()
                .map_or(Value::Null, |feedback| json!(feedback))
        } else {
            Value::Null
        },
        "spans": spans,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{ModelEntry, PrefixEntry, Span, SpanId, TraceMeta};

    #[test]
    fn proposal_observation_keeps_parsed_evidence_and_omits_raw_trace_data() {
        let span: Span = serde_json::from_value(json!({
            "id": 0,
            "component": 0,
            "seq": 0,
            "parent": null,
            "prefix": 0,
            "suffix": [{
                "role": "user",
                "content": [{"type": "text", "text": "SUFFIX_SECRET"}]
            }],
            "input": {"question": "TRAIN_QUESTION"},
            "model": 0,
            "request_hash": 17,
            "events": [{
                "t": "tool_run",
                "id": "call-1",
                "name": "lookup",
                "args": {"api_key": "TOOL_ARGS_SECRET"},
                "result": "TOOL_RESULT_SECRET",
                "duration_us": 1,
                "error": null
            }],
            "raw_output": "RAW_OUTPUT_SECRET",
            "output": {"answer": "TRAIN_ANSWER"},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": null,
            "started_at_us": 1,
            "duration_us": 1,
            "complete": true,
            "eval": {"score": 0.75, "feedback": "SPAN_FEEDBACK"}
        }))
        .expect("valid trace span");
        assert_eq!(span.id, SpanId(0));
        let trace = Trace {
            meta: TraceMeta {
                trace_id: "TRACE_ID_SECRET".into(),
                candidate_hash: Some(987654321),
                ..TraceMeta::default()
            },
            components: vec!["answerer".into()],
            models: vec![ModelEntry::from_config(&crate::LMConfig {
                base_url: Some("https://MODEL_CONFIG_SECRET".into()),
                ..crate::LMConfig::default()
            })],
            prefixes: vec![PrefixEntry {
                messages: vec![crate::Message::user("PREFIX_SECRET")],
            }],
            spans: vec![span],
            ..Trace::default()
        };

        let eval = Eval::with_feedback(0.8, "ROLLOUT_FEEDBACK");
        let default_observation = proposal_observation(&trace, &eval, false, false).unwrap();
        let default_text = default_observation.to_string();
        for forbidden in [
            "SUFFIX_SECRET",
            "PREFIX_SECRET",
            "TRACE_ID_SECRET",
            "987654321",
            "MODEL_CONFIG_SECRET",
            "gpt-4o-mini",
            "config_hash",
            "TRAIN_QUESTION",
            "TRAIN_ANSWER",
            "TOOL_ARGS_SECRET",
            "TOOL_RESULT_SECRET",
            "RAW_OUTPUT_SECRET",
            "ROLLOUT_FEEDBACK",
            "SPAN_FEEDBACK",
        ] {
            assert!(
                !default_text.contains(forbidden),
                "default observation unexpectedly included {forbidden}"
            );
        }
        assert!(default_text.contains("answerer"));
        assert_eq!(default_observation["score"], 0.8);
        assert_eq!(default_observation["spans"][0]["score"], 0.75);

        let observation = proposal_observation(&trace, &eval, true, false).unwrap();
        let text = observation.to_string();

        for forbidden in [
            "SUFFIX_SECRET",
            "TOOL_ARGS_SECRET",
            "TOOL_RESULT_SECRET",
            "RAW_OUTPUT_SECRET",
        ] {
            assert!(
                !text.contains(forbidden),
                "unexpectedly included {forbidden}"
            );
        }
        for useful in [
            "TRAIN_QUESTION",
            "TRAIN_ANSWER",
            "answerer",
            "ROLLOUT_FEEDBACK",
            "SPAN_FEEDBACK",
        ] {
            assert!(text.contains(useful), "missing useful evidence {useful}");
        }
        assert_eq!(observation["score"], 0.8);
        assert_eq!(observation["spans"][0]["score"], 0.75);

        let full_trace = proposal_observation(&trace, &eval, false, true).unwrap();
        let full_text = full_trace.to_string();
        for expected in [
            "TOOL_RESULT_SECRET",
            "TOOL_ARGS_SECRET",
            "RAW_OUTPUT_SECRET",
            "TRAIN_QUESTION",
            "TRAIN_ANSWER",
        ] {
            assert!(
                full_text.contains(expected),
                "full trace did not include {expected}"
            );
        }
    }
}
