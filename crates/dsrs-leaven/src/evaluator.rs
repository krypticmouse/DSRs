use std::{marker::PhantomData, sync::Arc};

use anyhow::anyhow;
use dsrs_core::{BamlType, Example, Module, Signature};
use dsrs_evaluate::TypedMetric;
use leaven_core::{
    Assessment, AssessmentGranularity, AssessmentTarget, ResolvedEvaluationRequest,
    ResolvedRequestKind,
};
use leaven_engine::{CachePolicy, EvaluationContext, EvaluationError, Evaluator};
use leaven_kernel::{CaseId, Cost, EvaluationSetId, EvaluatorId, Fingerprint, Metered};

use crate::{DsrsCaseEvidence, DsrsModuleFactory, DsrsProgramArtifact};

/// Evaluator helper that runs immutable DSRs program artifacts through typed DSRs metrics.
///
#[derive(Clone)]
pub struct DsrsEvaluator<S, M, F, MT>
where
    S: Signature,
    M: Module<Input = S::Input>,
    F: DsrsModuleFactory<M>,
    MT: TypedMetric<S, M>,
{
    cases: Arc<Vec<Example<S>>>,
    metric: MT,
    fingerprint: Fingerprint,
    cache_policy: CachePolicy,
    _phantom: PhantomData<(M, F)>,
}

impl<S, M, F, MT> DsrsEvaluator<S, M, F, MT>
where
    S: Signature,
    M: Module<Input = S::Input> + for<'a> dsrs_core::Facet<'a>,
    F: DsrsModuleFactory<M>,
    MT: TypedMetric<S, M>,
    S::Input: Clone,
    M::Output: BamlType,
{
    #[must_use]
    pub fn new(cases: Vec<Example<S>>, metric: MT) -> Self {
        Self {
            cases: Arc::new(cases),
            metric,
            fingerprint: Fingerprint::from_bytes([41; 32]),
            cache_policy: CachePolicy::Never,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    pub fn with_fingerprint(mut self, fingerprint: Fingerprint) -> Self {
        self.fingerprint = fingerprint;
        self
    }

    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    #[must_use]
    pub fn with_cache_policy(mut self, cache_policy: CachePolicy) -> Self {
        self.cache_policy = cache_policy;
        self
    }

    #[must_use]
    pub fn cache_policy(&self) -> &CachePolicy {
        &self.cache_policy
    }

    /// Evaluates one artifact against concrete case ids using DSRs' typed metric seam.
    pub async fn evaluate_artifact_cases(
        &self,
        artifact: &DsrsProgramArtifact<S, M, F>,
        case_ids: &[CaseId],
    ) -> anyhow::Result<Vec<DsrsCaseEvidence>> {
        let module = artifact.materialize_module()?;
        let mut cases = Vec::with_capacity(case_ids.len());
        for case_id in case_ids {
            let index = usize::try_from(case_id.0)
                .map_err(|_| anyhow!("case id {case_id} does not fit usize"))?;
            let example = self
                .cases
                .get(index)
                .ok_or_else(|| anyhow!("case {case_id} is missing from evaluator cases"))?;
            let prediction = module.call(example.input.clone()).await?;
            let outcome = self.metric.evaluate(example, &prediction).await?;
            let output = serde_json::to_value(prediction.to_baml_value())?;
            cases.push(DsrsCaseEvidence {
                case_id: *case_id,
                score: outcome.score,
                output,
                feedback: outcome.feedback,
            });
        }
        Ok(cases)
    }
}

#[derive(Clone, Debug)]
pub struct DsrsLeavenProblem<S, M, F>
where
    S: Signature,
    M: Module,
    F: DsrsModuleFactory<M>,
{
    _phantom: PhantomData<(S, M, F)>,
}

impl<S, M, F> leaven_core::OptimizationProblem for DsrsLeavenProblem<S, M, F>
where
    S: Signature,
    M: Module<Input = S::Input> + for<'a> dsrs_core::Facet<'a> + Send + Sync + 'static,
    F: DsrsModuleFactory<M>,
{
    type Artifact = DsrsProgramArtifact<S, M, F>;
    type Case = Example<S>;
    type Evidence = DsrsCaseEvidence;
    type ProposalAnnotations = ();
}

impl<S, M, F, MT> Evaluator<DsrsLeavenProblem<S, M, F>> for DsrsEvaluator<S, M, F, MT>
where
    S: Signature,
    M: Module<Input = S::Input> + for<'a> dsrs_core::Facet<'a> + Send + Sync + 'static,
    F: DsrsModuleFactory<M>,
    MT: TypedMetric<S, M>,
    S::Input: Clone,
    M::Output: BamlType,
{
    fn id(&self) -> EvaluatorId {
        EvaluatorId::PRIMARY
    }

    fn fingerprint(&self) -> Fingerprint {
        self.fingerprint
    }

    fn cache_policy(&self, _request: &ResolvedEvaluationRequest) -> CachePolicy {
        self.cache_policy.clone()
    }

    async fn evaluate(
        &self,
        request: ResolvedEvaluationRequest,
        ctx: EvaluationContext<'_, DsrsLeavenProblem<S, M, F>>,
    ) -> Result<Metered<Vec<Assessment<DsrsLeavenProblem<S, M, F>>>>, EvaluationError> {
        if request.granularity != AssessmentGranularity::PerCase {
            return Err(EvaluationError::Message(
                "dsrs-leaven evaluator requires per-case granularity".to_string(),
            ));
        }
        let ResolvedRequestKind::Independent { candidates } = request.kind else {
            return Err(EvaluationError::Message(
                "dsrs-leaven evaluator requires independent requests".to_string(),
            ));
        };

        let case_count = request.set.case_ids.len();
        let mut assessments = Vec::with_capacity(candidates.len() * case_count);
        let mut total_cost = Cost::zero();
        for candidate in candidates {
            let artifact = ctx.graph().artifact(candidate).ok_or_else(|| {
                EvaluationError::Message(format!("candidate {candidate} is missing"))
            })?;
            let cases = self
                .evaluate_artifact_cases(artifact, &request.set.case_ids)
                .await
                .map_err(|err| {
                    EvaluationError::Message(format!("DSRs evaluation failed: {err:#}"))
                })?;
            let set = EvaluationSetId::new();
            for evidence in cases {
                let case_id = evidence.case_id;
                let cost = Cost::metric_calls(1);
                total_cost = total_cost.combine(&cost);
                assessments.push(Assessment::Independent {
                    candidate,
                    target: AssessmentTarget::Case { set, case: case_id },
                    evidence,
                    cost,
                    metadata: leaven_kernel::MetadataBag::new(),
                });
            }
        }

        Ok(Metered::new(assessments, total_cost))
    }
}
