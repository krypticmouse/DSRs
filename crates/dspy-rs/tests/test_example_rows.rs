//! Trainset rows are plain structs: `#[derive(Example)]`, tuple rows, and the
//! `ToInput`/`ToOutput` projections they feed into evaluation and optimization.

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

#[derive(Example, Clone, Debug, serde::Serialize)]
#[example(RowQA)]
struct HotpotRow {
    #[input]
    question: String,
    #[output]
    answer: String,
    // Metric-only: visible to TypedMetric, never sent to the module.
    supporting_facts: Vec<String>,
}

#[derive(Example, Clone, Debug)]
#[example(RowQA)]
struct DefaultPartitionRow {
    #[input]
    question: String,
    // No explicit #[output] anywhere: every non-input field is an output.
    answer: String,
}

#[test]
fn derive_projects_marked_fields_by_name() {
    let row = HotpotRow {
        question: "What is 2+2?".to_string(),
        answer: "4".to_string(),
        supporting_facts: vec!["arithmetic".to_string()],
    };

    let input: RowQAInput = row.to_input();
    let output: RowQAOutput = row.to_output();
    assert_eq!(input.question, "What is 2+2?");
    assert_eq!(output.answer, "4");
}

#[derive(Signature, Clone, Debug)]
/// Think step by step, then answer.
struct ReasonedQA {
    #[input]
    question: String,
    #[output]
    reasoning: String,
    #[output]
    answer: String,
}

// Gold answer doesn't cover ReasonedQA's `reasoning` output, so it stays
// `#[meta]`: the metric reads it from the row and no ToOutput is generated.
#[derive(Example, Clone, Debug)]
#[example(ReasonedQA)]
struct GoldOnlyRow {
    #[input]
    question: String,
    #[meta]
    gold_answer: String,
}

#[test]
fn meta_fields_skip_the_default_output_partition() {
    let row = GoldOnlyRow {
        question: "q".to_string(),
        gold_answer: "a".to_string(),
    };
    let input: ReasonedQAInput = row.to_input();
    assert_eq!(input.question, "q");
    assert_eq!(row.gold_answer, "a");
}

#[test]
fn derive_defaults_non_input_fields_to_output() {
    let row = DefaultPartitionRow {
        question: "q".to_string(),
        answer: "a".to_string(),
    };
    let output: RowQAOutput = row.to_output();
    assert_eq!(output.answer, "a");
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
    let input: RowQAInput = row.to_input();
    let output: RowQAOutput = row.to_output();
    assert_eq!(input.question, "q");
    assert_eq!(output.answer, "a");
}

#[test]
fn rows_seed_labeled_demos() {
    let row = HotpotRow {
        question: "q".to_string(),
        answer: "a".to_string(),
        supporting_facts: vec![],
    };
    let demo = Demo::<RowQA>::new(row.to_input(), row.to_output());
    assert_eq!(demo.input.question, "q");
    assert_eq!(demo.output.answer, "a");
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
