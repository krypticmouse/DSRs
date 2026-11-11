use dspy_rs::optimizer::engine::RolloutError;
use dspy_rs::{
    CallMetadata, Candidate, Engine, EngineConfig, Eval, Module, OptimizeTarget, Predict,
    PredictError, Predicted, Signature, Trace, TypedMetric,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tracing::instrument::WithSubscriber;
use tracing::{
    Subscriber,
    field::{Field, Visit},
};
use tracing_subscriber::{
    Layer,
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
};

type Records = Arc<Mutex<Vec<(String, BTreeMap<String, String>)>>>;
#[derive(Clone)]
struct Recorder(Records);
#[derive(Default)]
struct Fields(BTreeMap<String, String>);
impl Visit for Fields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }
}
impl<S> Layer<S> for Recorder
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _: &tracing::Id,
        _: Context<'_, S>,
    ) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        self.0
            .lock()
            .unwrap()
            .push((attrs.metadata().name().into(), fields.0));
    }
    fn on_event(&self, event: &tracing::Event<'_>, ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let scope = ctx
            .event_scope(event)
            .map(|scope| {
                scope
                    .from_root()
                    .map(|span| span.metadata().name())
                    .collect::<Vec<_>>()
                    .join("/")
            })
            .unwrap_or_default();
        self.0.lock().unwrap().push((scope, fields.0));
    }
}

#[derive(Signature, Clone, Debug)]
struct QA {
    #[input]
    question: String,
    #[output]
    answer: String,
}
struct Echo {
    leaf: Predict<QA>,
    fail: bool,
}
dspy_rs::predictors!(Echo { leaf });
impl Module for Echo {
    type Input = QAInput;
    type Output = QAOutput;
    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        if self.fail {
            return Err(PredictError::Params {
                name: "leaf".into(),
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "bad fixture parameter",
                )),
            });
        }
        Ok(Predicted::new(
            QAOutput {
                answer: input.question,
            },
            CallMetadata::default(),
        ))
    }
}
struct Exact;

struct Parent {
    leaf: Predict<QA>,
}
impl Module for Parent {
    type Input = QAInput;
    type Output = QAOutput;
    async fn forward(&self, input: QAInput) -> Result<Predicted<QAOutput>, PredictError> {
        self.leaf.call(input).await
    }
}
impl TypedMetric<(QAInput, QAOutput), Echo> for Exact {
    async fn evaluate(
        &self,
        example: &(QAInput, QAOutput),
        prediction: &Predicted<QAOutput>,
        _: Option<&Trace>,
    ) -> anyhow::Result<Eval> {
        Ok(Eval::score(
            (example.1.answer == prediction.answer) as u8 as f64,
        ))
    }
}

#[tokio::test]
async fn optimizer_emits_candidate_progress_scores_and_budget_outcomes() {
    let records = Records::default();
    let subscriber = tracing_subscriber::registry().with(Recorder(records.clone()));
    async {
        let mut module = Echo {
            leaf: Predict::new(),
            fail: false,
        };
        let rows = vec![(
            QAInput {
                question: "a".into(),
            },
            QAOutput { answer: "a".into() },
        )];
        let target = OptimizeTarget::module(&mut module, &rows, &Exact);
        let mut engine = Engine::new(EngineConfig::default());
        let candidate = engine.register(Candidate::new());
        engine.evaluate(&target, candidate, None).await.unwrap();
        engine.evaluate(&target, candidate, None).await.unwrap();
        let mut blocked = Engine::new(EngineConfig {
            budget: dspy_rs::Budget {
                max_metric_calls: Some(0),
                ..dspy_rs::Budget::unlimited()
            },
            ..EngineConfig::default()
        });
        let candidate = blocked.register(Candidate::new());
        blocked.evaluate(&target, candidate, None).await.unwrap();
    }
    .with_subscriber(subscriber)
    .await;
    let records = records.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|(name, fields)| name == "dsrs.optimizer.batch"
                && fields.contains_key("candidates"))
    );
    assert!(
        records
            .iter()
            .any(|(name, fields)| name == "dsrs.optimizer.rollout"
                && fields.get("candidate").map(String::as_str) == Some("0")
                && fields.get("example").map(String::as_str) == Some("0")
                && fields.get("rollout").map(String::as_str) == Some("1")
                && fields.get("total").map(String::as_str) == Some("1"))
    );
    assert!(
        records
            .iter()
            .any(|(scope, fields)| scope.contains("dsrs.optimizer.rollout")
                && fields.contains_key("score"))
    );
    assert!(
        records.iter().any(
            |(_, fields)| fields.get("cached").map(String::as_str) == Some("1")
                && fields.contains_key("best_score")
                && fields.contains_key("average_score")
        )
    );
    assert!(
        records
            .iter()
            .any(|(_, fields)| fields.get("needed").map(String::as_str) == Some("1"))
    );
}

#[tokio::test]
async fn optimizer_keeps_typed_error_and_candidate_example_context() {
    let mut module = Echo {
        leaf: Predict::new(),
        fail: true,
    };
    let rows = vec![(
        QAInput {
            question: "a".into(),
        },
        QAOutput { answer: "a".into() },
    )];
    let target = OptimizeTarget::module(&mut module, &rows, &Exact);
    let mut engine = Engine::new(EngineConfig::default());
    let candidate = engine.register(Candidate::new());
    let error = engine.evaluate(&target, candidate, None).await.unwrap_err();
    let context = error.downcast_ref::<RolloutError>().unwrap();
    assert_eq!((context.candidate, context.example), (0, 0));
    assert!(error.chain().any(|cause|matches!(cause.downcast_ref::<PredictError>(),Some(PredictError::Params {name,..}) if name=="leaf")));
    assert!(format!("{error:#}").contains("bad fixture parameter"));
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn full_prompt_and_response_logs_require_the_content_target() {
    use dspy_rs::{LM, LMClient, TestCompletionModel};
    use rig::completion::AssistantContent;
    let client = TestCompletionModel::new((0..2).map(|_| {
        AssistantContent::text("[[ ## answer ## ]]\nunique-response\n\n[[ ## completed ## ]]")
    }));
    let lm = LM::builder()
        .base_url("http://localhost:1/v1".into())
        .model("logs".into())
        .cache(false)
        .build()
        .await
        .unwrap()
        .with_client(LMClient::Test(client))
        .await
        .unwrap();
    let predictor = Parent {
        leaf: Predict::<QA>::builder().lm(lm).build(),
    };
    for enabled in [false, true] {
        let records = Records::default();
        let filter = if enabled {
            "dspy_rs=debug,dsrs::content=debug"
        } else {
            "dspy_rs=debug"
        };
        let subscriber = tracing_subscriber::registry().with(
            Recorder(records.clone()).with_filter(tracing_subscriber::EnvFilter::new(filter)),
        );
        predictor
            .call(QAInput {
                question: "unique-prompt".into(),
            })
            .with_subscriber(subscriber)
            .await
            .unwrap();
        let records = records.lock().unwrap();
        assert!(records.iter().any(|(name, fields)| {
            name == "dsrs.module.call"
                && fields
                    .get("module")
                    .is_some_and(|name| name.contains("Parent"))
        }));
        let content = records.iter().any(|(_, fields)| {
            fields.contains_key("prompt")
                || fields.contains_key("response")
                || fields.contains_key("choice")
        });
        assert_eq!(content, enabled);
        if enabled {
            assert!(records.iter().any(|(_, fields)| {
                fields
                    .get("prompt")
                    .is_some_and(|value| value.contains("unique-prompt"))
            }));
            assert!(records.iter().any(|(_, fields)| {
                fields
                    .get("response")
                    .is_some_and(|value| value.contains("unique-response"))
            }));
        }
    }
}
