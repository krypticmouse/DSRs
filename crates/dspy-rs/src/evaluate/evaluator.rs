use anyhow::{Result, anyhow};
use bamltype::baml_types::BamlMap;

use crate::core::Module;
use crate::data::example::Example;
use crate::{BamlType, BamlValue, Predicted};

use super::FeedbackMetric;

#[derive(Debug, Clone, PartialEq)]
pub struct MetricOutcome {
    pub score: f32,
    pub feedback: Option<FeedbackMetric>,
}

impl MetricOutcome {
    pub fn score(score: f32) -> Self {
        Self {
            score,
            feedback: None,
        }
    }

    pub fn with_feedback(score: f32, feedback: FeedbackMetric) -> Self {
        Self {
            score,
            feedback: Some(feedback),
        }
    }
}

#[allow(async_fn_in_trait)]
pub trait TypedMetric<M: Module>: Send + Sync {
    async fn evaluate(
        &self,
        example: &Example,
        prediction: &Predicted<M::Output>,
    ) -> Result<MetricOutcome>;
}

fn baml_map_from_example_keys(example: &Example, keys: &[String]) -> Result<BamlMap<String, BamlValue>> {
    let mut map = BamlMap::new();
    for key in keys {
        if let Some(value) = example.data.get(key) {
            let baml_value =
                BamlValue::try_from(value.clone()).map_err(|err| anyhow!("{err}"))?;
            map.insert(key.clone(), baml_value);
        }
    }
    Ok(map)
}

pub fn input_keys_from_example(example: &Example) -> Vec<String> {
    if !example.input_keys.is_empty() {
        return example.input_keys.clone();
    }

    if !example.output_keys.is_empty() {
        return example
            .data
            .keys()
            .filter(|key| !example.output_keys.contains(*key))
            .cloned()
            .collect();
    }

    example.data.keys().cloned().collect()
}

pub fn input_from_example<I>(example: &Example) -> Result<I>
where
    I: BamlType,
{
    let keys = input_keys_from_example(example);
    let map = baml_map_from_example_keys(example, &keys)?;
    I::try_from_baml_value(BamlValue::Map(map)).map_err(|err| anyhow!("{err}"))
}

pub async fn evaluate_trainset<M, MT>(
    module: &M,
    trainset: &[Example],
    metric: &MT,
) -> Result<Vec<MetricOutcome>>
where
    M: Module,
    MT: TypedMetric<M>,
{
    let mut outcomes = Vec::with_capacity(trainset.len());

    for example in trainset {
        let input = input_from_example::<M::Input>(example)?;
        let predicted = module.call(input).await.map_err(|err| anyhow!("{err}"))?;
        outcomes.push(metric.evaluate(example, &predicted).await?);
    }

    Ok(outcomes)
}

pub fn average_score(outcomes: &[MetricOutcome]) -> f32 {
    if outcomes.is_empty() {
        return 0.0;
    }

    outcomes.iter().map(|o| o.score).sum::<f32>() / outcomes.len() as f32
}
