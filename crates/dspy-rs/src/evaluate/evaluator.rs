use crate::core::Module;
use crate::data::{example::Example, prediction::Prediction};
use futures::future::join_all;

#[allow(async_fn_in_trait)]
pub trait Evaluator: Module {
    async fn predict(&self, examples: Vec<Example>) -> Vec<Prediction> {
        let futures: Vec<_> = examples
            .iter()
            .map(|example| self.forward(example.clone()))
            .collect();

        join_all(futures)
            .await
            .into_iter()
            .map(|x| x.unwrap())
            .collect()
    }

    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32;

    async fn evaluate(&self, examples: Vec<Example>) -> f32 {
        let predictions = self.predict(examples.clone()).await;

        let futures: Vec<_> = examples
            .iter()
            .zip(predictions.iter())
            .map(|(example, prediction)| {
                let prediction = prediction.clone();
                async move { self.metric(example, &prediction).await }
            })
            .collect();

        let metrics = join_all(futures).await;
        metrics.iter().sum::<f32>() / examples.len() as f32
    }
}
