/*
Example showing scoped trace capture for a composed module.

Wrapping a call in `trace::capture` records one span per `Predict` invocation:
the rendered prompt (interned prefix + live suffix), typed input/output as
JSON, usage, timing, and — for tool loops — every provider round-trip and tool
execution as ordered events. Spans are addressed by the same names the params
system uses, so `trace.for_component("answerer")` is the sub-trace of that one
predictor.

Run with:
```
cargo run --example 12-tracing
```
*/

use anyhow::Result;
use bon::Builder;
use dspy_rs::{
    CallMetadata, LM, Module, Predict, PredictError, Predicted, Signature, configure,
    init_tracing, trace,
};

#[derive(Signature, Clone, Debug)]
struct QASignature {
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Signature, Clone, Debug)]
struct RateSignature {
    #[input]
    question: String,

    #[input]
    answer: String,

    #[output]
    rating: i8,
}

#[derive(Builder)]
struct QARater {
    #[builder(default = Predict::<QASignature>::builder().named("answerer").build())]
    answerer: Predict<QASignature>,

    #[builder(default = Predict::<RateSignature>::builder().named("rater").build())]
    rater: Predict<RateSignature>,
}

/// Typed output of the composed pipeline. `Module::Output` types are typed
/// structs: `Facet` + serde derives satisfy the `Schema` bound.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, facet::Facet)]
struct RatedAnswer {
    question: String,
    answer: String,
    rating: i8,
}

impl Module for QARater {
    type Input = QASignatureInput;
    type Output = RatedAnswer;

    async fn forward(
        &self,
        input: QASignatureInput,
    ) -> Result<Predicted<RatedAnswer>, PredictError> {
        let answer_predicted = self.answerer.call(input.clone()).await?;
        let answer_usage = answer_predicted.metadata().lm_usage;
        let answer_output = answer_predicted.into_inner();

        let rating_predicted = self
            .rater
            .call(RateSignatureInput {
                question: input.question.clone(),
                answer: answer_output.answer.clone(),
            })
            .await?;
        let rating_usage = rating_predicted.metadata().lm_usage;
        let rating_output = rating_predicted.into_inner();

        let output = RatedAnswer {
            question: input.question,
            answer: answer_output.answer,
            rating: rating_output.rating,
        };
        let metadata = CallMetadata {
            lm_usage: answer_usage + rating_usage,
            ..CallMetadata::default()
        };

        Ok(Predicted::new(output, metadata))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing()?;

    configure(LM::builder()
            .model("openai:gpt-4o-mini".to_string())
            .build()
            .await?);

    let module = QARater::builder().build();

    println!("Starting capture...");
    let (result, trace) = trace::capture(|| async {
        module
            .call(QASignatureInput {
                question: "Hello".to_string(),
            })
            .await
    })
    .await;

    match result {
        Ok(predicted) => {
            let rated = predicted.into_inner();
            println!(
                "Result: question={:?} answer={:?} rating={}",
                rated.question, rated.answer, rated.rating
            );
        }
        Err(err) => println!("Error (expected without credentials/network): {err}"),
    }

    // Each Predict call records one span: component name, invocation seq,
    // typed input/output as JSON, a link to the previous span, and timing.
    // Failed calls stay visible with the prompt recorded and output absent.
    println!("Trace {} spans: {}", trace.meta.trace_id, trace.spans.len());
    for span in &trace.spans {
        println!(
            "Span {}: component={:?} seq={} links={:?} events={}",
            span.id.0,
            trace.component_name(span.component),
            span.seq,
            span.links,
            span.events.len(),
        );
        if let Some(input) = &span.input {
            println!("  recorded input: {input:?}");
        }
        match (&span.output, &span.error) {
            (Some(output), _) => println!("  recorded output: {output:?}"),
            (None, Some(error)) => println!("  error ({}): {}", error.kind.as_str(), error.message),
            (None, None) => {}
        }
    }

    // Spans are addressed by the same names an optimizer would mutate.
    for span in trace.for_component("rater") {
        println!(
            "rater call {} rendered a {}-message prompt",
            span.seq,
            trace.prompt(span).len()
        );
    }

    // The whole rollout serializes to JSONL (header + one line per span).
    let jsonl = trace.to_jsonl()?;
    println!("\nJSONL ({} lines):", jsonl.lines().count());
    for line in jsonl.lines() {
        let preview: String = line.chars().take(120).collect();
        println!("  {preview}...");
    }

    Ok(())
}
