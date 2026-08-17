//! Trainset rows are plain structs: `#[derive(Example)]` projects them into
//! any signature's input/output by field name at runtime; tuples stay the
//! compile-checked zero-conversion path.

use anyhow::Result;
use dspy_rs::*;

#[derive(Signature, Clone, Debug)]
/// Answer questions accurately.
struct RowQA {
    #[input]
    question: String,
    #[output]
    answer: String,
}

// No signature named, no field marks: the row is just data. `question` and
// `answer` are matched by name at the call site; `supporting_facts` is only
// ever seen by the metric.
#[derive(Example, Clone, Debug, serde::Serialize)]
struct HotpotRow {
    question: String,
    answer: String,
    supporting_facts: Vec<String>,
}

#[test]
fn derive_projects_by_field_name() {
    let row = HotpotRow {
        question: "What is 2+2?".to_string(),
        answer: "4".to_string(),
        supporting_facts: vec!["arithmetic".to_string()],
    };

    let input: RowQAInput = row.to_input().expect("question field lines up");
    let output: RowQAOutput = row.to_output().expect("answer field lines up");
    assert_eq!(input.question, "What is 2+2?");
    assert_eq!(output.answer, "4");
}

#[test]
fn one_row_type_serves_any_matching_signature() {
    #[derive(Signature, Clone, Debug)]
    /// Same input contract, different signature type.
    struct OtherQA {
        #[input]
        question: String,
        #[output]
        answer: String,
    }

    let row = HotpotRow {
        question: "q".to_string(),
        answer: "a".to_string(),
        supporting_facts: vec![],
    };
    let input: OtherQAInput = row.to_input().expect("rows are signature-independent");
    assert_eq!(input.question, "q");
}

#[test]
fn projection_mismatch_is_a_clear_runtime_error() {
    #[derive(Example, Clone, Debug, serde::Serialize)]
    struct WrongRow {
        prompt: String,
    }

    let row = WrongRow {
        prompt: "q".to_string(),
    };
    let projected: Result<RowQAInput> = row.to_input();
    let err = projected.expect_err("no `question` field to project");
    let message = format!("{err:#}");
    assert!(message.contains("WrongRow"), "names the row: {message}");
    assert!(message.contains("RowQAInput"), "names the target: {message}");
}

#[test]
fn tuple_rows_project_both_ways() {
    let row = (
        RowQAInput {
            question: "q".to_string(),
        },
        RowQAOutput {
            answer: "a".to_string(),
        },
    );
    let input: RowQAInput = row.to_input().expect("tuple input clones");
    let output: RowQAOutput = row.to_output().expect("tuple output clones");
    assert_eq!(input.question, "q");
    assert_eq!(output.answer, "a");
}

#[test]
fn rows_seed_labeled_demos() -> Result<()> {
    let row = HotpotRow {
        question: "q".to_string(),
        answer: "a".to_string(),
        supporting_facts: vec![],
    };
    let demo = Demo::<RowQA>::new(row.to_input()?, row.to_output()?);
    assert_eq!(demo.input.question, "q");
    assert_eq!(demo.output.answer, "a");
    Ok(())
}

/// An offline module: echoes the question back as the answer.
#[derive(facet::Facet)]
#[facet(crate = facet)]
struct EchoModule {
    predictor: Predict<RowQA>,
}

impl Module for EchoModule {
    type Input = RowQAInput;
    type Output = RowQAOutput;

    async fn forward(&self, input: RowQAInput) -> Result<Predicted<RowQAOutput>, PredictError> {
        Ok(Predicted::new(
            RowQAOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

/// The metric reads row fields the module never saw.
struct SupportingFactsMetric;

impl TypedMetric<HotpotRow, EchoModule> for SupportingFactsMetric {
    async fn evaluate(
        &self,
        example: &HotpotRow,
        prediction: &Predicted<RowQAOutput>,
        _trace: Option<&trace::Trace>,
    ) -> Result<Eval> {
        let mut score = if prediction.answer == example.question {
            0.5
        } else {
            0.0
        };
        // Gold data that is not part of the signature output.
        if example.supporting_facts.is_empty() {
            score += 0.5;
        }
        Ok(Eval::score(score))
    }
}

#[tokio::test]
async fn evaluate_trainset_over_plain_rows() {
    let module = EchoModule {
        predictor: Predict::<RowQA>::new(),
    };
    let trainset = vec![
        HotpotRow {
            question: "echo me".to_string(),
            answer: "echo me".to_string(),
            supporting_facts: vec![],
        },
        HotpotRow {
            question: "other".to_string(),
            answer: "gold".to_string(),
            supporting_facts: vec!["fact".to_string()],
        },
    ];

    let evals = evaluate_trainset(&module, &trainset, &SupportingFactsMetric)
        .await
        .expect("offline evaluation should succeed");
    assert_eq!(evals.len(), 2);
    assert_eq!(evals[0].score, 1.0);
    assert_eq!(evals[1].score, 0.5);
}
