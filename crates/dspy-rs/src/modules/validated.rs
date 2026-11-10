use std::num::NonZeroU32;

use crate::{
    LmError, LmUsage, Message, Module, ParseError, Predict, PredictError, Predicted, PredictorInfo,
    Predictors, Signature,
};

type Validator<I, O> = Box<dyn Fn(&I, &O) -> Result<(), String> + Send + Sync>;

/// A predictor with application validators and a bounded corrective conversation.
///
/// Validators receive the complete typed input and output, so the same interface supports
/// individual-field rules and cross-field invariants. Their error strings become
/// feedback to the LM. The default is one attempt and nonfatal soft checks.
///
/// ```no_run
/// use dspy_rs::{Module, Predict};
/// use dspy_rs::doctest::QA;
/// use std::num::NonZeroU32;
/// # async fn example() -> Result<(), dspy_rs::PredictError> {
/// let module = Predict::<QA>::new().validated()
///     .max_attempts(NonZeroU32::new(3).unwrap())
///     .validator(|_, output| {
///         if output.answer.trim().is_empty() {
///             Err("Give a nonempty answer.".into())
///         } else {
///             Ok(())
///         }
///     });
/// # Ok(())
/// # }
/// ```
pub struct ValidatedPredict<S: Signature> {
    pub predictor: Predict<S>,
    validators: Vec<Validator<S::Input, S::Output>>,
    max_attempts: NonZeroU32,
    retry_checks: bool,
}

impl<S: Signature> ValidatedPredict<S> {
    pub fn new(predictor: Predict<S>) -> Self {
        Self {
            predictor,
            validators: Vec::new(),
            max_attempts: NonZeroU32::MIN,
            retry_checks: false,
        }
    }

    /// Registers a field rule or a whole-output rule. All failures are reported.
    pub fn validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&S::Input, &S::Output) -> Result<(), String> + Send + Sync + 'static,
    {
        self.validators.push(Box::new(validator));
        self
    }

    /// Total attempts, including the initial call. One disables corrective retries.
    pub fn max_attempts(mut self, attempts: NonZeroU32) -> Self {
        self.max_attempts = attempts;
        self
    }

    /// Treats failed `#[check]` constraints as validation failures. By default,
    /// soft checks remain informational; `#[assert]` failures always reject output.
    pub fn retry_failed_checks(mut self, enabled: bool) -> Self {
        self.retry_checks = enabled;
        self
    }
}

impl<S: Signature> Module for ValidatedPredict<S> {
    type Input = S::Input;
    type Output = S::Output;

    async fn forward(&self, input: Self::Input) -> Result<Predicted<Self::Output>, PredictError> {
        if self.max_attempts.get() > 1 && self.predictor.has_tools() {
            return Err(PredictError::Lm {
                source: LmError::Provider {
                    provider: "validation configuration".into(),
                    message: "validation retries cannot repeat predictors with attached tools; \
                              use one attempt or the agent loop's repair policy"
                        .into(),
                    source: None,
                },
            });
        }
        let mut chat = self.predictor.build_chat(&input).await?;
        let mut usage = LmUsage::default();
        let mut remaining = self.max_attempts.get();
        loop {
            remaining -= 1;
            let feedback = match self.predictor.call_and_parse(chat.clone()).await {
                Ok((prediction, next_chat)) => {
                    let (output, mut metadata) = prediction.into_parts();
                    usage = usage + metadata.lm_usage;
                    let mut failures: Vec<String> = self
                        .validators
                        .iter()
                        .filter_map(|validator| validator(&input, &output).err())
                        .collect();
                    if self.retry_checks {
                        for (field, meta) in &metadata.field_meta {
                            for check in meta.checks.iter().filter(|check| !check.passed) {
                                failures.push(format!(
                                    "Check `{}` failed for `{field}`: {}",
                                    check.label, check.expression
                                ));
                            }
                        }
                    }
                    if failures.is_empty() {
                        metadata.lm_usage = usage;
                        return Ok(Predicted::new(output, metadata));
                    }
                    let feedback = failures.join("\n");
                    if remaining == 0 {
                        return Err(PredictError::Parse {
                            source: ParseError::ValidationFailed { feedback },
                            raw_response: metadata.raw_response,
                            lm_usage: usage,
                        });
                    }
                    chat = next_chat;
                    feedback
                }
                Err(PredictError::Parse {
                    source,
                    raw_response,
                    lm_usage,
                }) => {
                    usage = usage + lm_usage;
                    if remaining == 0 {
                        return Err(PredictError::Parse {
                            source,
                            raw_response,
                            lm_usage: usage,
                        });
                    }
                    let feedback = parse_feedback(&source);
                    chat.push_message(Message::assistant(raw_response));
                    feedback
                }
                Err(other) => return Err(other),
            };
            chat.push_message(Message::user(format!(
                "Your previous response failed validation:\n{feedback}\nReturn a corrected response using the required output fields and format."
            )));
        }
    }
}

fn parse_feedback(error: &ParseError) -> String {
    match error {
        ParseError::Multiple { errors, .. } => errors
            .iter()
            .map(parse_feedback)
            .collect::<Vec<_>>()
            .join("\n"),
        ParseError::AssertFailed {
            field,
            label,
            expression,
            ..
        } => format!("Assertion `{label}` failed for `{field}`: {expression}"),
        other => other.to_string(),
    }
}

impl<S: Signature> Predictors for ValidatedPredict<S> {
    fn predictors(&self) -> Vec<(String, &dyn PredictorInfo)> {
        self.predictor.predictors()
    }

    fn predictors_mut(&mut self) -> Vec<(String, &mut dyn PredictorInfo)> {
        self.predictor.predictors_mut()
    }
}
