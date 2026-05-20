use anyhow::Result;
use dsrs_core::{CallMetadata, Example, Facet, Module, PredictError, Predicted, Signature};
use dsrs_evaluate::{FeedbackMetric, MetricOutcome, TypedMetric};
use dsrs_leaven::{DsrsEvaluator, DsrsModuleFactory, DsrsProgramArtifact};
use dsrs_predict::Predict;
use leaven_gepa::GepaCaseEvidence;
use leaven_kernel::CaseId;

#[derive(Signature, Clone, Debug)]
struct EvalSig {
    /// Echo the question.
    #[input]
    question: String,

    #[output]
    answer: String,
}

#[derive(Facet)]
struct EvalProgram {
    predictor: Predict<EvalSig>,
}

impl Default for EvalProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::<EvalSig>::builder()
                .instruction("echo exactly")
                .build(),
        }
    }
}

impl Module for EvalProgram {
    type Input = EvalSigInput;
    type Output = EvalSigOutput;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        Ok(Predicted::new(
            EvalSigOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}

#[derive(Clone)]
struct EvalFactory;

impl DsrsModuleFactory<EvalProgram> for EvalFactory {
    fn fresh_module(&self) -> EvalProgram {
        EvalProgram::default()
    }
}

struct ExactMetric;

impl TypedMetric<EvalSig, EvalProgram> for ExactMetric {
    async fn evaluate(
        &self,
        example: &Example<EvalSig>,
        prediction: &Predicted<EvalSigOutput>,
    ) -> Result<MetricOutcome> {
        let correct = prediction.answer == example.output.answer;
        Ok(MetricOutcome::with_feedback(
            if correct { 1.0 } else { 0.0 },
            FeedbackMetric::new(
                if correct { 1.0 } else { 0.0 },
                format!(
                    "expected {}, got {}",
                    example.output.answer, prediction.answer
                ),
            ),
        ))
    }
}

#[tokio::test]
async fn evaluator_uses_typed_metric_and_projects_gepa_scores() -> Result<()> {
    let mut seed = EvalProgram::default();
    let artifact =
        DsrsProgramArtifact::<EvalSig, EvalProgram, EvalFactory>::capture(EvalFactory, &mut seed)?;
    let cases = vec![
        Example::new(
            EvalSigInput {
                question: "alpha".to_string(),
            },
            EvalSigOutput {
                answer: "alpha".to_string(),
            },
        ),
        Example::new(
            EvalSigInput {
                question: "beta".to_string(),
            },
            EvalSigOutput {
                answer: "not beta".to_string(),
            },
        ),
    ];
    let evaluator =
        DsrsEvaluator::<EvalSig, EvalProgram, EvalFactory, ExactMetric>::new(cases, ExactMetric);

    let evidence = evaluator
        .evaluate_artifact_cases(&artifact, &[CaseId::new(0), CaseId::new(1)])
        .await?;

    assert_eq!(evidence.len(), 2);
    assert_eq!(evidence[0].case_id, CaseId::new(0));
    assert_eq!(evidence[0].score, 1.0);
    assert_eq!(evidence[0].output["answer"], "alpha");
    assert_eq!(
        evidence[0].feedback.as_ref().unwrap().feedback,
        "expected alpha, got alpha"
    );
    assert_eq!(evidence[1].case_id, CaseId::new(1));
    assert_eq!(evidence[1].score, 0.0);

    assert_eq!(evidence[0].scalar_score().unwrap().score(), 1.0);
    assert_eq!(evidence[1].scalar_score().unwrap().score(), 0.0);

    Ok(())
}
