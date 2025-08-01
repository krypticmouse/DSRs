use smart_default::SmartDefault;
use std::collections::HashMap;
use std::env;
use tokio;

use dspy_rs::adapter::chat_adapter::ChatAdapter;
use dspy_rs::clients::lm::LM;
use dspy_rs::data::example::Example;
use dspy_rs::data::prediction::Prediction;
use dspy_rs::module::Module;
use dspy_rs::programs::cot::ChainofThought;
use dspy_rs::signature::Signature;
use dspy_rs::utils::settings::configure_settings;

#[derive(SmartDefault)]
pub struct QQAModule {
    #[default(ChainofThought::new(&mut Signature::from("question->answer"), false))]
    pub qa1: ChainofThought,
    #[default(ChainofThought::new(&mut Signature::from("question->answer"), false))]
    pub qa2: ChainofThought,
}

impl Module for QQAModule {
    async fn forward(
        &self,
        inputs: Example,
        lm: Option<LM>,
        adapter: Option<ChatAdapter>,
    ) -> Prediction {
        let q1 = self.qa1.forward(inputs.clone(), None, None).await;
        let q2 = self.qa2.forward(inputs.clone(), None, None).await;

        let q1_answer = q1.get("answer", Some(""));
        let q2_answer = q2.get("answer", Some(""));

        let prediction = Prediction::new(HashMap::from([
            ("answer1".to_string(), q1_answer),
            ("answer2".to_string(), q2_answer),
        ]));

        prediction
    }
}

#[tokio::main]
async fn main() {
    let lm = LM::builder()
        .api_key(env::var("OPENROUTER_API_KEY").unwrap())
        .model("openai/gpt-4o-mini".to_string())
        .build()
        .unwrap();

    configure_settings(Some(lm), Some(ChatAdapter::default()));

    let inputs = Example::new(
        HashMap::from([(
            "question".to_string(),
            "What is the capital of France?".to_string(),
        )]),
        vec!["question".to_string()],
        vec!["answer".to_string()],
    );

    let qa_module = QQAModule::default();

    let output = qa_module.forward(inputs, None, None).await;
    println!("{:?}", output);
}
