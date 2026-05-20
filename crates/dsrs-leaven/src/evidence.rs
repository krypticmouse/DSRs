use dsrs_evaluate::FeedbackMetric;
use leaven_evidence::ScalarEvidence;
use leaven_kernel::CaseId;

/// Typed evidence for one DSRs evaluation case.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct DsrsCaseEvidence {
    pub case_id: CaseId,
    pub score: f32,
    pub output: serde_json::Value,
    pub feedback: Option<FeedbackMetric>,
}

impl leaven_core::Evidence for DsrsCaseEvidence {}

impl leaven_gepa::GepaCaseEvidence for DsrsCaseEvidence {
    fn scalar_score(&self) -> Option<ScalarEvidence> {
        ScalarEvidence::new(f64::from(self.score)).ok()
    }
}
