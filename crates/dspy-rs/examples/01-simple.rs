use anyhow::Result;
use dspy_rs::{example, Prediction, Signature, Predict, Module, Example, hashmap, LM, configure, ChatAdapter};
use secrecy::SecretString;

#[Signature(cot)]
struct QASignature {
    #[input]
    pub question: String,

    #[output]
    pub answer: String,
}

#[Signature]
struct RateSignature {
    #[input]
    pub answer: String,
    #[output]
    pub rating: i8,
}

pub struct QARater {
    pub answerer: Predict,
    pub rater: Predict,
}

impl QARater {
    pub fn new() -> Self {
        Self {
            answerer: Predict::new(QASignature::new()),
            rater: Predict::new(RateSignature::new()),
        }
    }
}

impl Module for QARater {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        let answerer_prediction = self.answerer.forward(inputs.clone()).await?;

        let answer = answerer_prediction.data.get("answer").unwrap().clone();

        let inputs = Example::new(
            hashmap! {
                "answer".to_string() => answer.clone()
            },
            vec!["answer".to_string()],
            vec![],
        );
        let rating_prediction = self.rater.forward(inputs).await?;
        Ok(Prediction::new(
            hashmap! {
                "answer".to_string() => answer,
                "rating".to_string() => rating_prediction.data.get("rating").unwrap().clone()
            },
            rating_prediction.lm_usage
        ))
    }
}

#[tokio::main]
async fn main() {
    configure(
        LM::builder()
            .api_key(SecretString::from(std::env::var("OPENAI_API_KEY").unwrap()))
            .build(),
        ChatAdapter {},
    );

    let example = example! {
        "question": "What is the capital of France?",
    };

    let qa_rater = QARater::new();
    let prediction = qa_rater.forward(example).await.unwrap();
    println!("{:?}", prediction);
}